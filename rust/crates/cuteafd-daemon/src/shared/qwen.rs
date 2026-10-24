//! Owner-local recurrent state and exact verify replay for hybrid Qwen families.
pub(crate) mod weights;
pub(crate) mod scheduler;
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::qwen::{GdnGeometry, HybridAttention};
use std::{ffi::c_void, rc::Rc};

pub(crate) enum Encoder {
    Local(crate::shared::vision::local::LocalEncoder),
    Remote(crate::shared::vision::remote::RemoteEncoder),
    Off,
}
impl Encoder {
    pub fn health_handle(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        match self { Self::Remote(client) => Some(client.health_handle()), Self::Local(client) => Some(client.health_handle()), Self::Off => None }
    }
    pub fn available(&self) -> bool {
        match self { Self::Remote(client) => client.healthy(), Self::Local(client) => client.healthy(), Self::Off => false }
    }
}
impl cuteafd_engine::media::EncoderClient for Encoder {
    fn submit(&mut self, job: cuteafd_engine::media::EncodeJob) -> std::result::Result<cuteafd_engine::media::EncoderTicket, cuteafd_engine::media::MediaError> {
        use cuteafd_engine::media::MediaError;
        match self { Self::Local(client) => client.submit(job), Self::Remote(client) => client.submit(job),
            Self::Off => Err(MediaError::Encoder("encoder not loaded".into())) }
    }
    fn poll(&mut self, ticket: cuteafd_engine::media::EncoderTicket) -> Option<std::result::Result<cuteafd_engine::media::EncodeOutput, cuteafd_engine::media::MediaError>> {
        match self { Self::Local(client) => client.poll(ticket), Self::Remote(client) => client.poll(ticket), Self::Off => None }
    }
    fn cancel(&mut self, ticket: cuteafd_engine::media::EncoderTicket) {
        match self { Self::Local(client) => client.cancel(ticket), Self::Remote(client) => client.cancel(ticket), Self::Off => () }
    }
}
impl scheduler::Encoder for Encoder {
    fn available(&self) -> bool { self.available() }
}

type Dev<'a> = Rc<Allocation<'a>>;

/// Rows recorded by the shared GDN replay ABI in the fork.
pub(crate) const REPLAY_ROWS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerStateHome {
    pub(crate) owner: usize,
    pub(crate) gdn_ordinal: Option<usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct LayerStateMap {
    pub(crate) layers: Vec<LayerStateHome>,
    pub(crate) gdn_layers: Vec<usize>,
}

impl LayerStateMap {
    pub(crate) fn new(kinds: &[HybridAttention], owners: &[usize], devices: usize) -> Result<Self> {
        ensure!(devices > 0 && kinds.len() == owners.len(), "invalid Qwen layer ownership map");
        let mut gdn_layers = vec![0; devices];
        let mut layers = Vec::with_capacity(kinds.len());
        for (&kind, &owner) in kinds.iter().zip(owners) {
            ensure!(owner < devices, "Qwen layer owner {owner} outside {devices} devices");
            let gdn_ordinal = if kind == HybridAttention::Gdn {
                let ordinal = gdn_layers[owner];
                gdn_layers[owner] += 1;
                Some(ordinal)
            } else { None };
            layers.push(LayerStateHome { owner, gdn_ordinal });
        }
        Ok(Self { layers, gdn_layers })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GdnLayout {
    pub(crate) conv_slot_bytes: usize,
    pub(crate) state_slot_bytes: usize,
    pub(crate) replay_layer_bytes: usize,
}

impl GdnLayout {
    pub(crate) fn new(g: GdnGeometry) -> Result<Self> {
        let mul = |a: usize, b: usize| a.checked_mul(b).context("Qwen GDN layout size overflow");
        let add = |a: usize, b: usize| a.checked_add(b).context("Qwen GDN layout size overflow");
        ensure!(g.key_heads > 0 && g.value_heads > 0 && g.key_dim > 0 && g.value_dim > 0
            && g.conv_kernel > 0 && g.value_heads % g.key_heads == 0, "invalid Qwen GDN geometry");
        let key = mul(g.key_heads, g.key_dim)?;
        let value = mul(g.value_heads, g.value_dim)?;
        let conv = add(mul(2, key)?, value)?;
        let conv_slot_bytes = mul(mul(g.conv_kernel - 1, conv)?, 2)?;
        let state_slot_bytes = mul(mul(value, g.key_dim)?, 4)?;
        // Normalized K, V, decay/beta and native BF16 convolution inputs.
        let replay_row = add(add(mul(key, 4)?, mul(value, 4)?)?,
            add(mul(g.value_heads, 8)?, mul(conv, 2)?)?)?;
        let replay_layer_bytes = mul(add(mul(REPLAY_ROWS, replay_row)?, 1023)? / 1024, 1024)?;
        Ok(Self { conv_slot_bytes, state_slot_bytes, replay_layer_bytes })
    }

    fn pool_bytes(self, layers: usize, slots: usize) -> Result<[usize; 3]> {
        let records = layers.checked_mul(slots).context("Qwen GDN pool size overflow")?;
        Ok([records.checked_mul(self.conv_slot_bytes).context("Qwen GDN pool size overflow")?,
            records.checked_mul(self.state_slot_bytes).context("Qwen GDN pool size overflow")?,
            layers.checked_mul(self.replay_layer_bytes).context("Qwen GDN pool size overflow")?])
    }
}

/// Exact prefix marks retain global state order while packing each owner's arena.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct MarkLayout {
    pub(crate) owners: Vec<(i32, usize)>,
    /// Owner arena, byte offset, byte length, in global state-region order.
    pub(crate) regions: Vec<(usize, usize, usize)>,
    pub(crate) bytes: usize,
}

impl MarkLayout {
    pub(crate) fn new(regions: &[(i32, usize)]) -> Result<Self> {
        let mut layout = Self { owners: Vec::new(), regions: Vec::new(), bytes: 0 };
        for &(device, bytes) in regions {
            let owner = layout.owners.iter().position(|&(id, _)| id == device).unwrap_or_else(|| {
                layout.owners.push((device, 0));
                layout.owners.len() - 1
            });
            let offset = layout.owners[owner].1;
            layout.owners[owner].1 = offset.checked_add(bytes).context("Qwen owner mark size overflow")?;
            layout.bytes = layout.bytes.checked_add(bytes).context("Qwen logical mark size overflow")?;
            layout.regions.push((owner, offset, bytes));
        }
        Ok(layout)
    }
}

/// Compact pools on one owner; global layer ids never index them directly.
/// The enclosing engine drains its streams before releasing this bank.
pub(crate) struct GdnBank<'a> {
    library: &'a NativeLibrary,
    pub(crate) programs: &'a Programs<'a>,
    pub(crate) stream: *mut c_void,
    layers: usize,
    slots: usize,
    layout: GdnLayout,
    commit_program: &'static str,
    pub(crate) conv: Option<Dev<'a>>,
    pub(crate) state: Option<Dev<'a>>,
    replay: Option<Dev<'a>>,
    pub(crate) commit_tables: Dev<'a>,
}

impl<'a> GdnBank<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, stream: *mut c_void,
        geometry: GdnGeometry, layers: usize, slots: usize, commit_program: &'static str) -> Result<Self> {
        let layout = GdnLayout::new(geometry)?;
        let [conv_bytes, state_bytes, replay_bytes] = layout.pool_bytes(layers, slots)?;
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = Rc::new(Allocation::new(Device { library, id: library.cuda_get_device()? }, bytes.max(256))?);
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let (conv, state, replay) = if layers > 0 {
            (Some(zeroed(conv_bytes)?), Some(zeroed(state_bytes)?), Some(zeroed(replay_bytes)?))
        } else { (None, None, None) };
        let commit_tables = zeroed(3 * REPLAY_ROWS * 4)?;
        // SAFETY: the bank's live table and stream; reserving before readiness keeps
        // every later ordered commit upload free of pinned allocation.
        unsafe { library.reserve_h2d_ordered(commit_tables.buffer, stream)? };
        Ok(Self { library, programs, stream, layers, slots, layout, commit_program, conv, state, replay,
            commit_tables })
    }

    pub(crate) fn allocated_bytes(&self) -> [usize; 3] {
        [self.conv.as_ref().map_or(0, |a| a.buffer.bytes)
            + self.state.as_ref().map_or(0, |a| a.buffer.bytes),
            self.replay.as_ref().map_or(0, |a| a.buffer.bytes), self.commit_tables.buffer.bytes]
    }

    fn region(pool: &Dev<'_>, offset: usize, bytes: usize) -> CuteafdDeviceBuffer {
        debug_assert!(offset + bytes <= pool.buffer.bytes);
        CuteafdDeviceBuffer { ptr: pool.buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..pool.buffer }
    }

    pub(crate) fn slot_regions(&self, ordinal: usize, slot: usize) -> [CuteafdDeviceBuffer; 2] {
        assert!(ordinal < self.layers && slot < self.slots);
        let record = ordinal * self.slots + slot;
        [Self::region(self.conv.as_ref().unwrap(), record * self.layout.conv_slot_bytes, self.layout.conv_slot_bytes),
            Self::region(self.state.as_ref().unwrap(), record * self.layout.state_slot_bytes, self.layout.state_slot_bytes)]
    }

    pub(crate) fn pools(&self, ordinal: usize) -> Result<[*mut c_void; 3]> {
        ensure!(ordinal < self.layers, "GDN owner-local layer out of range");
        let (Some(conv), Some(state), Some(replay)) = (&self.conv, &self.state, &self.replay) else {
            anyhow::bail!("GDN layer without state pools")
        };
        Ok([Self::region(conv, ordinal * self.slots * self.layout.conv_slot_bytes, 0).ptr,
            Self::region(state, ordinal * self.slots * self.layout.state_slot_bytes, 0).ptr,
            Self::region(replay, ordinal * self.layout.replay_layer_bytes, 0).ptr])
    }

    pub(crate) fn commit(&self, table: &[i32], sequences: usize) -> Result<()> {
        ensure!(sequences <= REPLAY_ROWS && table.len() == 3 * sequences, "invalid Qwen GDN commit table");
        for i in 0..sequences {
            let slot = usize::try_from(table[i])?;
            let first = usize::try_from(table[sequences + i])?;
            let keep = usize::try_from(table[2 * sequences + i])?;
            ensure!(slot < self.slots && first.checked_add(keep).is_some_and(|end| end <= REPLAY_ROWS),
                "Qwen GDN commit outside replay or state pool");
        }
        let bytes: Vec<u8> = table.iter().flat_map(|value| value.to_ne_bytes()).collect();
        // SAFETY: every reader of the commit table (this commit, Qwen4's PLE commit)
        // runs on this bank's stream, so the ordered upload follows the previous
        // step's readers and precedes this launch without a host wait.
        unsafe { self.library.copy_h2d_ordered(self.commit_tables.buffer, &bytes, self.stream)? };
        if let (Some(conv), Some(state), Some(replay)) = (&self.conv, &self.state, &self.replay) {
            let program = self.programs.program(self.commit_program, &["state", "conv_state", "replay", "tables"])?;
            let scalars = [sequences, self.layers, self.slots].map(|n| i32::try_from(n).map(Scalar::I32));
            let scalars = scalars.into_iter().collect::<std::result::Result<Vec<_>, _>>()?;
            // SAFETY: live owner-local pools match the compact layer/slot geometry.
            unsafe { program.launch(&[state.buffer.ptr, conv.buffer.ptr, replay.buffer.ptr,
                self.commit_tables.buffer.ptr], &scalars, self.stream)? };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::families::qwen::GdnOutputGate;

    #[test]
    fn global_layers_use_owner_local_gdn_ordinals() -> Result<()> {
        use HybridAttention::{Full, Gdn};
        let kinds = [Gdn, Gdn, Full, Gdn, Full, Gdn, Gdn, Full];
        for cutover in 1..kinds.len() {
            let owners: Vec<_> = (0..kinds.len()).map(|i| usize::from(i >= cutover)).collect();
            let map = LayerStateMap::new(&kinds, &owners, 2)?;
            let mut counts = [0; 2];
            for (global, home) in map.layers.iter().enumerate() {
                assert_eq!(home.owner, owners[global]);
                if kinds[global] == Gdn {
                    assert_eq!(home.gdn_ordinal, Some(counts[home.owner]));
                    counts[home.owner] += 1;
                } else { assert_eq!(home.gdn_ordinal, None); }
            }
            assert_eq!(map.gdn_layers, counts);
            assert_eq!(counts.iter().sum::<usize>(), 5);
        }
        assert!(LayerStateMap::new(&kinds, &[0], 2).is_err());
        assert!(LayerStateMap::new(&[Gdn], &[2], 2).is_err());
        Ok(())
    }

    #[test]
    fn invalid_and_overflowing_gdn_geometry_is_rejected() {
        let geometry = GdnGeometry { key_heads: 16, value_heads: 32, key_dim: 128, value_dim: 128,
            conv_kernel: 4, output_gate: GdnOutputGate::Silu };
        for invalid in [GdnGeometry { key_heads: 0, ..geometry },
            GdnGeometry { value_heads: 31, ..geometry }, GdnGeometry { key_dim: 0, ..geometry },
            GdnGeometry { value_dim: 0, ..geometry }, GdnGeometry { conv_kernel: 0, ..geometry },
            GdnGeometry { key_dim: usize::MAX, ..geometry }] {
            assert!(GdnLayout::new(invalid).is_err());
        }
    }

    #[test]
    fn dense_and_flash_next_layouts_preserve_native_replay_abi() -> Result<()> {
        for (kh, vh, gate) in [(16, 32, GdnOutputGate::Silu), (16, 48, GdnOutputGate::Silu),
            (16, 48, GdnOutputGate::Sigmoid)] {
            let g = GdnGeometry { key_heads: kh, value_heads: vh, key_dim: 128, value_dim: 128,
                conv_kernel: 4, output_gate: gate };
            let layout = GdnLayout::new(g)?;
            assert_eq!(layout.conv_slot_bytes + layout.state_slot_bytes, g.state_bytes());
            let old_replay = (REPLAY_ROWS * (kh * 128 * 4 + vh * 128 * 4 + vh * 8 + g.conv_width() * 2)).div_ceil(1024) * 1024;
            assert_eq!(layout.replay_layer_bytes, old_replay);
            assert_eq!(layout.pool_bytes(5, 7)?, [35 * layout.conv_slot_bytes,
                35 * layout.state_slot_bytes, 5 * old_replay]);
            assert!(layout.pool_bytes(usize::MAX, 2).is_err());
        }
        Ok(())
    }
}
