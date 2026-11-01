use super::*;
use crate::shared::memory::DeviceAllocation;
use cuteafd_loader::placement::families::deepseek_v41::owners;

/// One retained row (residual + pre-norm scales) and a prompt's rows: the
/// planner's constants, so the startup pool is exactly what it charges.
const ROW_BYTES: usize = owners::ENCODER_SUFFIX_ROW_BYTES as usize;
const SUFFIX_ROWS: usize = owners::ENCODER_SUFFIX_ROWS as usize;
const _: () = assert!(ROW_BYTES == 40960 + 16);

/// Source rows and request-local positions of each member's retained CED tail.
fn packed_tails(segments: &[crate::shared::prefill_packing::Segment])
    -> Result<Vec<(std::ops::Range<usize>, std::ops::Range<u64>)>> {
    ensure!(segments.len() > 1 && segments.len() <= 16, "packed suffix needs 2..=16 members");
    let mut first = 0;
    segments.iter().map(|segment| {
        ensure!(segment.first == first && segment.rows > 0, "packed suffix segments differ");
        let end_row = first.checked_add(segment.rows).context("packed suffix row overflow")?;
        let end = segment.start.checked_add(segment.rows).context("packed suffix position overflow")?;
        ensure!(end <= 1048576, "packed suffix exceeds context");
        let count = segment.rows.min(128);
        first = end_row;
        Ok((end_row - count..end_row, (end - count) as u64..end as u64))
    }).collect()
}

/// One admitted prompt's final encoder rows, retained across encoder chunk reuse.
/// Allocate at admission; capture copies only the final window's intersecting rows.
pub(crate) struct EncoderSuffix<'a> {
    residual: std::rc::Rc<DeviceAllocation<'a>>,
    pre: std::rc::Rc<DeviceAllocation<'a>>,
    tokens: Vec<u64>,
    next: u64,
    end: u64,
    binding: QueryBinding,
    invalid: bool,
    packed: Option<Vec<crate::shared::prefill_packing::Segment>>,
}
impl<'a> EncoderSuffix<'a> {
    #[cfg(test)]
    pub fn device_bytes(prompt_end: u64) -> Result<usize> {
        ensure!((1..=1048576).contains(&prompt_end), "invalid encoder suffix extent");
        Ok(prompt_end.min(SUFFIX_ROWS as u64) as usize * ROW_BYTES)
    }
    #[cfg(test)]
    pub fn new(library: &'a NativeLibrary, prompt_end: u64, budget: usize) -> Result<Self> {
        ensure!(Self::device_bytes(prompt_end)? <= budget, "encoder suffix exceeds budget");
        let rows = prompt_end.min(SUFFIX_ROWS as u64) as usize;
        Self::prompt(&Self::arena(library, rows)?, prompt_end)
    }
    /// A startup arena of `rows` suffix rows, at rest (no prompt bound).
    fn arena(library: &'a NativeLibrary, rows: usize) -> Result<Self> {
        Ok(Self { residual: std::rc::Rc::new(DeviceAllocation::new(library, rows * 40960)?),
            pre: std::rc::Rc::new(DeviceAllocation::new(library, rows * 16)?), tokens: Vec::new(),
            next: 0, end: 0, binding: QueryBinding::new(19)?, invalid: true, packed: None })
    }
    /// One serially reused arena, charged before KV pool admission.
    pub(crate) fn packed_storage(library: &'a NativeLibrary, unit: usize) -> Result<Self> {
        ensure!(unit > 0 && unit <= 4096, "invalid packed suffix unit");
        Self::arena(library, owners::packed_suffix_rows(unit as u64) as usize)
    }
    /// One prompt's suffix arena (the final window's rows), allocated at
    /// startup: the prefill slots hold one per request slot, charged with them.
    pub(crate) fn prompt_storage(library: &'a NativeLibrary) -> Result<Self> {
        Self::arena(library, SUFFIX_ROWS)
    }
    /// Whether a live prompt or packed group holds this arena.
    pub(crate) fn in_use(&self) -> bool {
        std::rc::Rc::strong_count(&self.residual) > 1 || std::rc::Rc::strong_count(&self.pre) > 1
    }
    /// Bind a prompt ending at `prompt_end` to `storage`, with no allocation:
    /// it holds the arena until it drops with its request.
    pub(crate) fn prompt(storage: &Self, prompt_end: u64) -> Result<Self> {
        ensure!((1..=1048576).contains(&prompt_end), "invalid encoder suffix extent");
        ensure!(!storage.in_use(), "encoder suffix arena is already held by a live prompt");
        let start = prompt_end.saturating_sub(SUFFIX_ROWS as u64);
        let rows = (prompt_end - start) as usize;
        ensure!(rows * 40960 <= storage.residual.buffer.bytes && rows * 16 <= storage.pre.buffer.bytes,
            "prompt suffix exceeds its arena");
        Ok(Self { residual: storage.residual.clone(), pre: storage.pre.clone(), tokens: (start..prompt_end).collect(),
            next: start, end: prompt_end, binding: QueryBinding::new(19)?, invalid: false, packed: None })
    }
    /// Retain each member's CED tail in the startup arena, with no runtime allocation.
    pub(crate) fn packed(storage: &Self,
        segments: &[crate::shared::prefill_packing::Segment]) -> Result<Self> {
        ensure!(!storage.in_use(), "packed suffix arena is already borrowed by a live group");
        let tails = packed_tails(segments)?;
        let tokens: Vec<_> = tails.into_iter().flat_map(|(_, positions)| positions).collect();
        let rows = tokens.len();
        ensure!(rows * 40960 <= storage.residual.buffer.bytes && rows * 16 <= storage.pre.buffer.bytes,
            "packed suffix exceeds admitted arena");
        Ok(Self { residual: storage.residual.clone(), pre: storage.pre.clone(), tokens,
            next: 0, end: rows as u64, binding: QueryBinding::new(19)?, invalid: false,
            packed: Some(segments.to_vec()) })
    }
    /// The producer is a completed encoder chunk from this admitted request.
    /// Incomplete/failed captures never expose an output; discard on request failure.
    pub fn capture(&mut self, output: &BlockOutput<'_>) -> Result<()> {
        let library = self.residual.library;
        self.capture_with(output, |dst, src, bytes| library.copy_d2d(dst, src, bytes))
    }
    /// Device-ordered prefill units (#815 option (i)): the copies join the
    /// stage chain on `stream` and record its head, without a host wait. The
    /// unit's commit drains the chain before the suffix is read.
    /// # Safety
    /// `stream` is a live stream on this suffix's GPU, inside a chain scope;
    /// the output's producer stays live until the chain drains.
    pub unsafe fn capture_chained(&mut self, output: &BlockOutput<'_>, stream: *mut std::ffi::c_void) -> Result<()> {
        let library = self.residual.library;
        unsafe { crate::shared::memory::chain::join(library, stream)?; }
        self.capture_with(output, |dst, src, bytes| unsafe { library.copy_d2d_async(dst, src, bytes, stream) })?;
        unsafe { crate::shared::memory::chain::finish(library, stream) }
    }
    /// # Safety
    /// The completed output and stream are on this suffix's GPU. Retain the
    /// producer through completion; cancellation drains queued copies and leaves
    /// this suffix invalid until it is dropped with its failed request.
    pub async unsafe fn capture_cooperative(&mut self, output: &BlockOutput<'_>,
        stream: &crate::shared::memory::LoadStream<'_>) -> Result<()> {
        stream.require_complete()?;
        let queued = self.capture_with(output, |dst, src, bytes| unsafe {
            stream.library.copy_d2d_async(dst, src, bytes, stream.raw)
        });
        self.invalid = true;
        stream.wait().await?;
        queued?;
        self.invalid = false;
        Ok(())
    }
    fn capture_with(&mut self, output: &BlockOutput<'_>,
        mut copy: impl FnMut(CuteafdDeviceBuffer, CuteafdDeviceBuffer, usize) -> Result<()>) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        let rows = output.tokens.len();
        if let Some(segments) = &self.packed {
            ensure!(valid && self.next == 0 && output.layer == 19 && output.binding.layer() == 19
                && segments.last().is_some_and(|s| s.end_row() == rows)
                && output.residual.bytes == rows * 40960 && output.pre.bytes == rows * 16,
                "packed encoder suffix producer differs");
            let mut destination_row = 0;
            for (source_rows, _) in packed_tails(segments)? {
                let count = source_rows.len();
                let source_row = source_rows.start;
                ensure!(output.tokens[source_rows]
                    == self.tokens[destination_row..destination_row + count],
                    "packed encoder suffix positions differ");
                for (source, destination, stride) in [(output.residual, self.residual.buffer, 40960),
                    (output.pre, self.pre.buffer, 16)] {
                    ensure!(source.device_id == destination.device_id, "packed encoder suffix device differs");
                    let bytes = count * stride;
                    // SAFETY: the validated segment and retained-tail extents bound both slices.
                    let src = CuteafdDeviceBuffer { ptr: unsafe { source.ptr.cast::<u8>().add(source_row * stride).cast() }, bytes, ..source };
                    // SAFETY: destination_row advances through the allocated concatenated tails.
                    let dst = CuteafdDeviceBuffer { ptr: unsafe { destination.ptr.cast::<u8>().add(destination_row * stride).cast() }, bytes, ..destination };
                    copy(dst, src, bytes)?;
                }
                destination_row += count;
            }
            self.next = self.end;
            self.invalid = false;
            return Ok(());
        }
        ensure!(valid && output.layer == 19 && output.binding.layer() == 19
            && rows > 0 && output.tokens.windows(2).all(|p| p[0].checked_add(1) == Some(p[1]))
            && output.tokens[rows - 1] < self.end
            && output.residual.bytes == rows * 40960 && output.pre.bytes == rows * 16,
            "encoder suffix producer layer, positions or extents differ");
        let first = output.tokens.partition_point(|&position| position < self.tokens[0]);
        if first < rows {
            ensure!(output.tokens[first] == self.next, "encoder suffix has a gap or duplicate rows");
            let offset = (self.next - self.tokens[0]) as usize;
            for (source, destination, stride) in [(output.residual, self.residual.buffer, 40960),
                (output.pre, self.pre.buffer, 16)] {
                ensure!(source.device_id == destination.device_id, "encoder suffix device differs");
                let bytes = (rows - first) * stride;
                let src = CuteafdDeviceBuffer { ptr: unsafe { source.ptr.cast::<u8>().add(first * stride).cast() }, bytes, ..source };
                let dst = CuteafdDeviceBuffer { ptr: unsafe { destination.ptr.cast::<u8>().add(offset * stride).cast() }, bytes, ..destination };
                copy(dst, src, bytes)?;
            }
            self.next = output.tokens[rows - 1] + 1;
        }
        self.invalid = false;
        Ok(())
    }
    pub fn output(&self) -> Result<BlockOutput<'_>> {
        ensure!(!self.invalid && self.next == self.end, "encoder suffix is incomplete or invalid");
        let (mut residual, mut pre) = (self.residual.buffer, self.pre.buffer);
        residual.bytes = self.tokens.len() * 40960;
        pre.bytes = self.tokens.len() * 16;
        Ok(BlockOutput { binding: self.binding, residual, pre, layer: 19, tokens: &self.tokens })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_ced_tails_equal_singletons_and_keep_partner_rows_private() -> Result<()> {
        use crate::shared::prefill_packing::{plan, Limits};
        let lengths = [1, 29, 127, 128, 129, 257];
        let segments = plan(&lengths.map(|rows| (0, rows)), &Limits {
            rows: 4096, chunk_rows: 4096, max_context: 1048576, taps: None })?;
        let tails = packed_tails(&segments)?;
        let input: Vec<_> = lengths.iter().enumerate().flat_map(|(member, &rows)|
            (0..rows).map(move |position| (member, position))).collect();
        for (member, (source, positions)) in tails.iter().enumerate() {
            let expected: Vec<_> = (lengths[member].saturating_sub(128)..lengths[member])
                .map(|position| (member, position)).collect();
            assert_eq!(&input[source.clone()], expected);
            assert_eq!(positions.clone().collect::<Vec<_>>(), expected.iter().map(|r| r.1 as u64).collect::<Vec<_>>());
        }
        let mut changed = input.clone();
        changed[..lengths[0]].fill((999, 999));
        for (source, _) in &tails[1..] { assert_eq!(input[source.clone()], changed[source.clone()]); }
        let mut invalid = segments.clone();
        invalid[1].first -= 1;
        assert!(packed_tails(&invalid).is_err());
        invalid = segments.clone(); invalid[1].start = usize::MAX;
        assert!(packed_tails(&invalid).is_err());
        assert!(packed_tails(&segments[..1]).is_err());
        Ok(())
    }

    #[test]
    fn retained_encoder_suffix_crosses_chunk_boundaries() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_ENCODER_SUFFIX_LIBRARY") else {
            eprintln!("skip GPU suffix test: CUTEAFD_ENCODER_SUFFIX_LIBRARY unset"); return Ok(());
        };
        let lib = unsafe { NativeLibrary::load(path)? };
        let residual = DeviceAllocation::new(&lib, 80 * 40960)?;
        let pre = DeviceAllocation::new(&lib, 80 * 16)?;
        for end in [1, 127, 128, 129, 257, 2049] {
            let mut suffix = EncoderSuffix::new(&lib, end, EncoderSuffix::device_bytes(end)?)?;
            assert!(suffix.output().is_err());
            for start in (0..end).step_by(80) {
                let tokens: Vec<u64> = (start..end.min(start + 80)).collect();
                let rows = tokens.len();
                for (allocation, stride, salt) in [(&residual, 40960, 0), (&pre, 16, 79)] {
                    let values: Vec<u8> = tokens.iter().flat_map(|&p| std::iter::repeat_n(((p + salt) % 251) as u8, stride)).collect();
                    lib.copy_h2d(allocation.buffer, &values)?;
                }
                let output = BlockOutput { binding: QueryBinding::new(19)?, layer: 19, tokens: &tokens,
                    residual: CuteafdDeviceBuffer { bytes: rows * 40960, ..residual.buffer },
                    pre: CuteafdDeviceBuffer { bytes: rows * 16, ..pre.buffer } };
                suffix.capture(&output)?;
                if start + (rows as u64) < end { assert!(suffix.output().is_err()); }
            }
            let output = suffix.output()?;
            assert_eq!(output.tokens, &(end.saturating_sub(128)..end).collect::<Vec<_>>());
            for (buffer, stride, salt) in [(output.residual, 40960, 0), (output.pre, 16, 79)] {
                let mut actual = vec![0; buffer.bytes]; lib.copy_d2h(&mut actual, buffer)?;
                let expected: Vec<u8> = output.tokens.iter().flat_map(|&p| std::iter::repeat_n(((p + salt) % 251) as u8, stride)).collect();
                assert_eq!(actual, expected);
            }
        }
        let mut gap = EncoderSuffix::new(&lib, 257, EncoderSuffix::device_bytes(257)?)?;
        let output = BlockOutput { binding: QueryBinding::new(19)?, layer: 19, tokens: &[130],
            residual: CuteafdDeviceBuffer { bytes: 40960, ..residual.buffer },
            pre: CuteafdDeviceBuffer { bytes: 16, ..pre.buffer } };
        assert!(gap.capture(&output).is_err()); assert!(gap.output().is_err());
        assert!(EncoderSuffix::device_bytes(0).is_err());
        assert!(EncoderSuffix::device_bytes(1048577).is_err());
        eprintln!("PASS encoder suffix 1/127/128/129/257/2049 tokens: residual/pre bytes, chunk crossings, incomplete and gap guards");
        Ok(())
    }
}
