//! Admitted K1 workspace: graph-visible routing storage never moves.
use super::{view, Exlr8PrefillActivations, Exlr8Weights};
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Program, ProgramSpec, Programs, Scalar};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::plan::exlr8::residency::ExpertSet;
use std::ffi::c_void;

/// Measured small-row dispatch: grouped is faster at every measured row once a
/// rank owns four units (TP4: 1.59x at one row, 1.97x at 64 rows). With three
/// owned units (TP6) K1 still wins at one row and grouped wins from four rows
/// (realistic routes 1.19x at four, 1.26x at 64), so the threshold keys on
/// owned units and never on the SM count.
pub(crate) fn grouped_decode_from_row(unit_limit: usize) -> usize {
    if unit_limit >= 4 { 1 } else { 4 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    /// Serving kernel below the grouped threshold (one row here at TP6).
    DecodeK1,
    /// 64-row grouped instance with F32 FC2 partials: bit-identical to K1.
    DecodeGrouped,
    /// 4096-row grouped prefill.
    Prefill,
}

impl Variant {
    fn grouped(self) -> bool {
        !matches!(self, Self::DecodeK1)
    }
}

fn program_names(
    set: ExpertSet,
    variant: Variant,
    activations: Exlr8PrefillActivations,
) -> (&'static str, &'static str) {
    match (set, variant) {
        (ExpertSet::Routed, Variant::Prefill) => (
            match activations {
                Exlr8PrefillActivations::A16 => "exlr8_a16_grouped",
                Exlr8PrefillActivations::A8 => "exlr8_a8_grouped",
            },
            "exlr8_grouped_group",
        ),
        (ExpertSet::Routed, Variant::DecodeGrouped) => ("exlr8_a16_grouped64", "exlr8_group"),
        (ExpertSet::Routed, Variant::DecodeK1) => ("exlr8_a16_serving", "exlr8_group"),
        (ExpertSet::Shared, _) => ("exlr8_shared_a16_serving", "exlr8_shared_group"),
    }
}

const K1_POINTERS: &[&str] = &[
    "x",
    "pointers",
    "bits",
    "group_expert",
    "route_slots",
    "route_group_lane",
    "route_weights",
    "fc1",
    "intermediate",
    "fc2",
    "signs_out",
    "output",
    "group_count",
];

fn pointers(set: ExpertSet, variant: Variant) -> Vec<&'static str> {
    let mut names = K1_POINTERS.to_vec();
    if set == ExpertSet::Routed && variant.grouped() {
        names.extend(["unit_count", "unit_map"]);
    }
    names
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RoutingIdentity {
    addresses: [usize; 5],
    rows: usize,
    top_k: usize,
    max_groups: usize,
}

#[derive(Default)]
struct RoutingLifetime {
    captured: Option<RoutingIdentity>,
}
impl RoutingLifetime {
    fn check(&self, candidate: RoutingIdentity) -> Result<()> {
        ensure!(
            self.captured.is_none_or(|bound| bound == candidate),
            "EXLR8 routing storage or shape changed while a graph retains it"
        );
        Ok(())
    }
    fn capture(&mut self, identity: RoutingIdentity) -> Result<()> {
        self.check(identity)?;
        self.captured = Some(identity);
        Ok(())
    }
}

#[derive(Debug)]
struct Workspace {
    ranges: [(usize, usize); 11],
    bytes: usize,
    rows: usize,
    top_k: usize,
    max_groups: usize,
    hidden: usize,
    units: usize,
    unit_limit: usize,
    intermediate_bytes: usize,
    /// Owned-unit tables and unit-scaled scratch: every grouped instance.
    compact: bool,
}
impl Workspace {
    fn plan(
        spec: &ProgramSpec,
        rows: usize,
        set: ExpertSet,
        variant: Variant,
        unit_limit: usize,
        activations: Exlr8PrefillActivations,
    ) -> Result<Self> {
        cuteafd_ffi::exlr8::validate(spec)?;
        ensure!(
            spec.name == program_names(set, variant, activations).0 && spec.geometry["abi_version"] == 3,
            "EXLR8 execution requires fused rotation ABI v3"
        );
        ensure!(
            rows > 0 && rows <= spec.capacity_rows as usize,
            "EXLR8 rows outside admitted K1 capacity"
        );
        // Decode rows route to the K1 or the 64-row grouped instance; the
        // 4096-row prefill instance is only admitted above the K1 row limit.
        ensure!(
            variant != Variant::Prefill
                || rows > cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS,
            "EXLR8 prefill instance is only admitted above the K1 row limit"
        );
        let dimension = |name| -> Result<usize> {
            usize::try_from(
                spec.geometry[name]
                    .as_u64()
                    .context("missing K1 geometry")?,
            )
            .context("K1 geometry exceeds address space")
        };
        let hidden = dimension("hidden")?;
        let top_k = dimension("top_k")?;
        let max_groups = dimension("max_groups")?;
        let units = dimension("units")?;
        let scratch = |name: &str| -> Result<usize> {
            usize::try_from(*spec.scratch.get(name).context("missing K1 scratch")?)
                .context("K1 scratch exceeds address space")
        };
        let compact = set == ExpertSet::Routed && variant.grouped();
        ensure!(!compact || (1..=units).contains(&unit_limit), "invalid compact unit limit");
        let scratch = |name: &str| -> Result<usize> {
            let bytes = scratch(name)?;
            if compact && name != "fc2" {
                ensure!(bytes % units == 0, "compact scratch must divide by global units");
                (bytes / units).checked_mul(unit_limit).context("compact scratch overflow")
            } else { Ok(bytes) }
        };
        let intermediate_bytes = scratch("intermediate")?;
        let (ranges, bytes) = cuteafd_loader::plan::exlr8::residency::execution_workspace_layout(
            rows, spec.capacity_rows as usize,
            spec.geometry["group_rows"].as_u64().unwrap_or(16) as usize,
            hidden, top_k, max_groups, scratch("fc1")?,
            scratch("intermediate")?, scratch("fc2")?, set)?;
        Ok(Self {
            ranges,
            bytes,
            rows,
            top_k,
            max_groups,
            hidden,
            units,
            unit_limit: if compact { unit_limit } else { units },
            intermediate_bytes,
            compact,
        })
    }
}

pub(crate) struct Exlr8Execution<'a> {
    group: Program<'a>,
    k1: Program<'a>,
    // Private, fixed storage: no request can resize or replace captured pointers.
    arena: DeviceAllocation<'a>,
    layout: Workspace,
    routing: RoutingLifetime,
    intermediate_bytes: usize,
    cta_limit: i32,
    group_cta_limit: i32,
    set: ExpertSet,
}
impl<'a> Exlr8Execution<'a> {
    pub(crate) fn workspace_bytes(
        programs: &'a Programs<'a>,
        rows: usize,
        set: ExpertSet,
    ) -> Result<usize> {
        Self::workspace_bytes_for(programs, rows, set, Variant::DecodeK1, 16,
            Exlr8PrefillActivations::A16)
    }

    pub(crate) fn workspace_bytes_with_limit(
        programs: &'a Programs<'a>,
        rows: usize,
        set: ExpertSet,
        unit_limit: usize,
        activations: Exlr8PrefillActivations,
    ) -> Result<usize> {
        Self::workspace_bytes_for(programs, rows, set, Variant::Prefill, unit_limit, activations)
    }

    /// 64-row grouped decode arena; F32 FC2 partials keep it bit-identical to K1.
    pub(crate) fn decode_grouped_workspace_bytes(
        programs: &'a Programs<'a>,
        rows: usize,
        unit_limit: usize,
    ) -> Result<usize> {
        Self::workspace_bytes_for(programs, rows, ExpertSet::Routed, Variant::DecodeGrouped,
            unit_limit, Exlr8PrefillActivations::A16)
    }

    fn workspace_bytes_for(
        programs: &'a Programs<'a>,
        rows: usize,
        set: ExpertSet,
        variant: Variant,
        unit_limit: usize,
        activations: Exlr8PrefillActivations,
    ) -> Result<usize> {
        let core = programs.program(program_names(set, variant, activations).0,
            &pointers(set, variant))?;
        Ok(Workspace::plan(core.spec(), rows, set, variant, unit_limit, activations)?.bytes)
    }

    pub(crate) fn new(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        rows: usize,
        available: usize,
        set: ExpertSet,
    ) -> Result<Self> {
        Self::new_for(library, programs, rows, available, set, Variant::DecodeK1, 16,
            Exlr8PrefillActivations::A16)
    }

    pub(crate) fn new_with_limit(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        rows: usize,
        available: usize,
        set: ExpertSet,
        unit_limit: usize,
        activations: Exlr8PrefillActivations,
    ) -> Result<Self> {
        Self::new_for(library, programs, rows, available, set, Variant::Prefill, unit_limit, activations)
    }

    pub(crate) fn new_decode_grouped(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        rows: usize,
        available: usize,
        unit_limit: usize,
    ) -> Result<Self> {
        Self::new_for(library, programs, rows, available, ExpertSet::Routed, Variant::DecodeGrouped,
            unit_limit, Exlr8PrefillActivations::A16)
    }

    fn new_for(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        rows: usize,
        available: usize,
        set: ExpertSet,
        variant: Variant,
        unit_limit: usize,
        activations: Exlr8PrefillActivations,
    ) -> Result<Self> {
        let (k1_name, group_name) = program_names(set, variant, activations);
        let k1 = programs.program(k1_name, &pointers(set, variant))?;
        let group = programs.program(
            group_name,
            &[
                "ids",
                "x",
                "signs_in",
                "group_expert",
                "route_slots",
                "route_group_lane",
                "group_count",
                "activation",
            ],
        )?;
        let layout = Workspace::plan(k1.spec(), rows, set, variant, unit_limit, activations)?;
        ensure!(
            group.spec().geometry["abi_version"] == 2
                && group.spec().geometry["top_k"] == k1.spec().geometry["top_k"]
                && group.spec().geometry["max_groups"] == k1.spec().geometry["max_groups"]
                && group.spec().geometry["hidden"] == k1.spec().geometry["hidden"]
                && group.spec().capacity_rows == k1.spec().capacity_rows
                && group.spec().geometry["group_rows"].as_u64().unwrap_or(16)
                    == k1.spec().geometry["group_rows"].as_u64().unwrap_or(16)
                && group.spec().geometry["input_format"] == match set {
                    ExpertSet::Routed => "fp8_k32",
                    ExpertSet::Shared => "bf16",
                }
                && group.spec().geometry["input_row_bytes"].as_u64()
                    == Some((layout.ranges[0].1 / rows) as u64),
            "EXLR8 grouping and K1 geometry differ"
        );
        ensure!(
            layout.bytes <= available,
            "EXLR8 workspace exceeds admitted device bytes"
        );
        let intermediate_bytes = layout.intermediate_bytes;
        let sms = i32::try_from(library.sm_count()?)?;
        let limit = |spec: &ProgramSpec| -> Result<i32> {
            let ctas = i32::try_from(spec.geometry["ctas_per_sm"].as_u64().context("CTA hint")?)?;
            ensure!((1..=32).contains(&ctas) && sms > 0, "invalid runtime CTA geometry");
            sms.checked_mul(ctas).context("CTA limit overflow")
        };
        let cta_limit = limit(k1.spec())?;
        let group_cta_limit = limit(group.spec())?;
        let arena = DeviceAllocation::named(library, layout.bytes, "experts/workspace/exlr8")?;
        Ok(Self {
            group,
            k1,
            arena,
            layout,
            routing: RoutingLifetime::default(),
            intermediate_bytes,
            cta_limit,
            group_cta_limit,
            set,
        })
    }
    fn buffer(&self, index: usize) -> Result<CuteafdDeviceBuffer> {
        let (offset, bytes) = self.layout.ranges[index];
        view(self.arena.buffer, offset, bytes)
    }
    fn identity(&self) -> Result<RoutingIdentity> {
        Ok(RoutingIdentity {
            addresses: [
                self.buffer(4)?.ptr as usize,
                self.buffer(5)?.ptr as usize,
                self.buffer(6)?.ptr as usize,
                self.buffer(7)?.ptr as usize,
                self.buffer(10)?.ptr as usize,
            ],
            rows: self.layout.rows,
            top_k: self.layout.top_k,
            max_groups: self.layout.max_groups,
        })
    }
    /// Called before capture. Retain this owner until every graph is destroyed
    /// and queued replay work has drained; binding or shape changes are refused.
    pub(crate) fn seal_graph_storage(&mut self) -> Result<()> {
        self.routing.capture(self.identity()?)
    }
    pub(crate) fn input(&self) -> Result<CuteafdDeviceBuffer> {
        self.buffer(0)
    }
    pub(crate) fn ids(&self) -> Result<CuteafdDeviceBuffer> {
        self.buffer(1)
    }
    pub(crate) fn route_weights(&self) -> Result<CuteafdDeviceBuffer> {
        self.buffer(2)
    }
    pub(crate) fn output(&self) -> Result<CuteafdDeviceBuffer> {
        self.buffer(3)
    }

    /// # Safety
    /// Inputs and weights are retained on this device until the caller drains
    /// `stream`. The caller also retains this workspace through graph replay.
    pub(crate) unsafe fn run(
        &self,
        weights: &Exlr8Weights<'_>,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.layout.rows,
            "EXLR8 live rows exceed admission"
        );
        ensure!(
            weights.layout.hidden == self.layout.hidden
                && weights.layout.units == self.layout.units
                && weights.layout.set == self.set,
            "EXLR8 weights and workspace geometry differ"
        );
        ensure!(
            self.group.spec().geometry["experts"].as_u64() == Some(weights.layout.experts as u64),
            "EXLR8 grouping expert count differs from residency"
        );
        self.routing.check(self.identity()?)?;
        let buffers = [
            self.buffer(0)?,
            self.buffer(1)?,
            self.buffer(2)?,
            self.buffer(3)?,
            self.buffer(4)?,
            self.buffer(5)?,
            self.buffer(6)?,
            self.buffer(7)?,
            self.buffer(8)?,
            self.buffer(9)?,
            self.buffer(10)?,
        ];
        let p = |i: usize| buffers[i].ptr;
        let intermediate = view(buffers[9], 0, self.intermediate_bytes)?;
        let fc2 = view(
            buffers[9],
            self.intermediate_bytes,
            buffers[9].bytes - self.intermediate_bytes,
        )?;
        let compact = self.layout.compact;
        ensure!(!compact || weights.layout.compact_unit_limit() <= self.layout.unit_limit,
            "EXLR8 layer ownership exceeds compact arena stride");
        let scalars = [Scalar::I32(rows as i32), Scalar::I32(self.cta_limit),
            Scalar::I32(self.layout.unit_limit as i32)];
        let mut operands = [p(10), weights.pointers()?.ptr, weights.bits()?.ptr,
            p(4), p(5), p(6), p(2), p(8), intermediate.ptr, fc2.ptr,
            weights.signs(true)?.ptr, p(3), p(7), std::ptr::null_mut(), std::ptr::null_mut()];
        if compact {
            operands[13..].copy_from_slice(&weights.compact()?.map(|buffer| buffer.ptr));
        }
        // SAFETY: private admitted arena views retain the manifest's exact sizes;
        // weights own all referenced descriptors, and caller retains the stream.
        unsafe {
            self.group.launch(
                &[p(1), p(0), weights.signs(false)?.ptr, p(4), p(5), p(6), p(7), p(10)],
                &[Scalar::I32(rows as i32), Scalar::I32(self.group_cta_limit)],
                stream,
            )?;
            self.k1.launch(&operands[..if compact { 15 } else { 13 }],
                &scalars[..if compact { 3 } else { 2 }], stream)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(set: ExpertSet) -> ProgramSpec {
        let (top_k, groups) = match set {
            ExpertSet::Routed => (8, 512),
            ExpertSet::Shared => (1, 4),
        };
        ProgramSpec {
            name: program_names(set, Variant::DecodeK1, Exlr8PrefillActivations::A16).0.into(),
            pointers: K1_POINTERS.iter().map(|name| (*name).into()).collect(),
            scalar_kinds: b"ii".to_vec(),
            scratch: [
                ("fc1".into(), groups * 16 * 2 * 6 * 16 * 128 * 4),
                ("intermediate".into(), groups * 16 * 16 * 128 * 2),
                ("fc2".into(), groups * 16 * 6144 * 4),
            ]
            .into(),
            capacity_rows: 64,
            geometry: serde_json::json!({"abi_version": 3, "hidden": 6144, "units": 16,
                "top_k": top_k, "max_groups": groups, "max_rows": 64,
                "split_k_blocks": 8, "fc1_splits": 6, "ctas_per_sm": 4}),
        }
    }

    /// The shipped 64-row grouped instance: 16-row groups, F32 FC2 partials.
    fn grouped64_spec() -> ProgramSpec {
        let mut spec = spec(ExpertSet::Routed);
        spec.name = program_names(ExpertSet::Routed, Variant::DecodeGrouped,
            Exlr8PrefillActivations::A16).0.into();
        spec.pointers = pointers(ExpertSet::Routed, Variant::DecodeGrouped)
            .into_iter().map(String::from).collect();
        spec.scalar_kinds = b"iii".to_vec();
        spec.geometry["group_rows"] = serde_json::json!(16);
        spec.geometry["fc2_partials"] = serde_json::json!("f32");
        spec.geometry["ctas_per_sm"] = serde_json::json!(2);
        spec
    }

    #[test]
    fn exlr8_grouped_decode_threshold_follows_owned_units() {
        // Measured on GB10: TP4 (4 owned units) wins at every row, TP6 (3)
        // keeps K1 at one row, and narrower ranks have no measured K1 win.
        assert_eq!(grouped_decode_from_row(1), 4);
        assert_eq!(grouped_decode_from_row(2), 4);
        assert_eq!(grouped_decode_from_row(3), 4);
        assert_eq!(grouped_decode_from_row(4), 1);
        assert_eq!(grouped_decode_from_row(8), 1);
        assert_eq!(grouped_decode_from_row(16), 1);
    }

    #[test]
    fn exlr8_shared_workspace_keeps_exact_aligned_admission() {
        for set in [ExpertSet::Routed, ExpertSet::Shared] {
            let spec = spec(set);
            let wide = Workspace::plan(&spec, 64, set, Variant::DecodeK1, 16,
                Exlr8PrefillActivations::A16).unwrap();
            let single = Workspace::plan(&spec, 1, set, Variant::DecodeK1, 16,
                Exlr8PrefillActivations::A16).unwrap();
            for layout in [&single, &wide] {
                let mut end = 0;
                for (offset, bytes) in layout.ranges {
                    assert_eq!(offset % 256, 0);
                    assert!(offset >= end);
                    end = offset + bytes;
                }
                assert_eq!(layout.bytes, end);
            }
            assert_eq!(single.ranges[8].1, wide.ranges[8].1);
            assert_eq!(single.ranges[9].1, wide.ranges[9].1);
            assert_eq!(single.ranges[0].1 * 64, wide.ranges[0].1);
            assert_eq!(single.ranges[3].1 * 64, wide.ranges[3].1);
            assert_eq!(single.ranges[10].1 * 64, wide.ranges[10].1);
            assert_eq!(wide.ranges[3].1, 64 * 6144 * 2);
            assert_eq!(wide.ranges[10].1, 64 * 6144 * 2);
            assert_eq!(wide.ranges[0].1, 64 * match set {
                ExpertSet::Routed => 6336,
                ExpertSet::Shared => 12288,
            });
            let mut stale = spec.clone();
            stale.geometry["abi_version"] = serde_json::json!(2);
            assert!(Workspace::plan(&stale, 64, set, Variant::DecodeK1, 16,
                Exlr8PrefillActivations::A16).is_err());
            assert!(Workspace::plan(&spec, 0, set, Variant::DecodeK1, 16,
                Exlr8PrefillActivations::A16).is_err());
            assert!(Workspace::plan(&spec, 65, set, Variant::DecodeK1, 16,
                Exlr8PrefillActivations::A16).is_err());
            let other = if set == ExpertSet::Shared {
                ExpertSet::Routed
            } else {
                ExpertSet::Shared
            };
            assert!(Workspace::plan(&spec, 64, other, Variant::DecodeK1, 16,
                Exlr8PrefillActivations::A16).is_err());
        }
        assert!(
            Workspace::plan(&spec(ExpertSet::Shared), 64, ExpertSet::Shared, Variant::DecodeK1,
                16, Exlr8PrefillActivations::A16).unwrap().bytes
                < Workspace::plan(&spec(ExpertSet::Routed), 64, ExpertSet::Routed,
                    Variant::DecodeK1, 16, Exlr8PrefillActivations::A16).unwrap().bytes
        );
    }

    #[test]
    fn grouped_decode_instance_compacts_and_keeps_k1_route_slots() {
        let grouped = grouped64_spec();
        for rows in [1, 4, 17, 64] {
            let full = Workspace::plan(&grouped, rows, ExpertSet::Routed, Variant::DecodeGrouped,
                16, Exlr8PrefillActivations::A16).unwrap();
            let owned = Workspace::plan(&grouped, rows, ExpertSet::Routed, Variant::DecodeGrouped,
                3, Exlr8PrefillActivations::A16).unwrap();
            assert!(full.compact && owned.compact && owned.unit_limit == 3);
            // fc1 and the activation scratch follow owned units; FC2 keeps the
            // full group capacity exactly as K1's arena does.
            assert_eq!(owned.ranges[8].1 * 16, full.ranges[8].1 * 3);
            assert_eq!(owned.intermediate_bytes * 16, full.intermediate_bytes * 3);
            assert_eq!(owned.ranges[9].1 - owned.intermediate_bytes,
                full.ranges[9].1 - full.intermediate_bytes);
            assert!(owned.bytes < full.bytes);
            assert_eq!(full.ranges[4].1, 512 * 4);
            assert_eq!(full.ranges[5].1, 512 * 16 * 4);
            assert_eq!(full.ranges[0].1, rows * 6336);
            // Serving K1 does not compact at these rows, so the grouped arena
            // is strictly smaller once a rank owns three units.
            let k1 = spec(ExpertSet::Routed);
            let decode = Workspace::plan(&k1, rows, ExpertSet::Routed, Variant::DecodeK1, 16,
                Exlr8PrefillActivations::A16).unwrap();
            assert!(!decode.compact);
            assert_eq!(decode.ranges[8].1, full.ranges[8].1);
        }
        for rows in [0, 65] {
            assert!(Workspace::plan(&grouped, rows, ExpertSet::Routed, Variant::DecodeGrouped,
                3, Exlr8PrefillActivations::A16).is_err());
        }
        let mut missing_partials = grouped.clone();
        missing_partials.geometry["fc2_partials"] = serde_json::json!("f16");
        assert!(Workspace::plan(&missing_partials, 64, ExpertSet::Routed,
            Variant::DecodeGrouped, 3, Exlr8PrefillActivations::A16).is_err());
        // A K1 spec cannot masquerade as the grouped decode instance.
        assert!(Workspace::plan(&spec(ExpertSet::Routed), 64, ExpertSet::Routed,
            Variant::DecodeGrouped, 3, Exlr8PrefillActivations::A16).is_err());
    }

    #[test]
    fn grouped_workspace_uses_wide_route_slots_without_changing_decode() {
        let mut grouped = spec(ExpertSet::Routed);
        grouped.name = program_names(ExpertSet::Routed, Variant::Prefill,
            Exlr8PrefillActivations::A16).0.into();
        grouped.capacity_rows = 4096;
        grouped.pointers = pointers(ExpertSet::Routed, Variant::Prefill)
            .into_iter().map(String::from).collect();
        grouped.scalar_kinds = b"iii".to_vec();
        grouped.geometry["max_rows"] = serde_json::json!(4096);
        grouped.geometry["group_rows"] = serde_json::json!(64);
        grouped.geometry["ctas_per_sm"] = serde_json::json!(2);
        for bytes in grouped.scratch.values_mut() { *bytes *= 4; }
        for rows in [65, 128, 4095, 4096] {
            let workspace = Workspace::plan(&grouped, rows, ExpertSet::Routed, Variant::Prefill,
                16, Exlr8PrefillActivations::A16).unwrap();
            let compact = Workspace::plan(&grouped, rows, ExpertSet::Routed, Variant::Prefill,
                3, Exlr8PrefillActivations::A16).unwrap();
            assert_eq!(compact.ranges[8].1 * 16, workspace.ranges[8].1 * 3);
            assert_eq!(compact.intermediate_bytes * 16, workspace.intermediate_bytes * 3);
            assert!(compact.bytes < workspace.bytes);
            assert_eq!(workspace.ranges[5].1, 512 * 64 * 4);
            assert_eq!(workspace.ranges[0].1, rows * 6336);
            assert_eq!(workspace.ranges[10].1, rows * 6144 * 2);
            assert!(Workspace::plan(&grouped, rows, ExpertSet::Routed, Variant::Prefill, 16,
                Exlr8PrefillActivations::A8).is_err());
            let mut grouped_a8 = grouped.clone();
            grouped_a8.name = program_names(ExpertSet::Routed, Variant::Prefill,
                Exlr8PrefillActivations::A8).0.into();
            grouped_a8.geometry["activations"] = serde_json::json!("a8");
            let workspace_a8 = Workspace::plan(&grouped_a8, rows, ExpertSet::Routed,
                Variant::Prefill, 16, Exlr8PrefillActivations::A8).unwrap();
            assert_eq!(workspace_a8.bytes, workspace.bytes);
            assert!(Workspace::plan(&grouped_a8, rows, ExpertSet::Routed, Variant::Prefill,
                16, Exlr8PrefillActivations::A16).is_err());
        }
        for invalid in [0, 17, usize::MAX] {
            assert!(Workspace::plan(&grouped, 1024, ExpertSet::Routed, Variant::Prefill,
                invalid, Exlr8PrefillActivations::A16).is_err());
        }
        assert!(Workspace::plan(&grouped, 64, ExpertSet::Routed, Variant::Prefill, 16,
            Exlr8PrefillActivations::A16).is_err());
        assert!(Workspace::plan(&grouped, 4097, ExpertSet::Routed, Variant::Prefill, 16,
            Exlr8PrefillActivations::A16).is_err());
        assert!(Workspace::plan(&grouped, 4096, ExpertSet::Shared, Variant::Prefill, 16,
            Exlr8PrefillActivations::A16).is_err());
        for activations in [Exlr8PrefillActivations::A16, Exlr8PrefillActivations::A8] {
            assert_eq!(program_names(ExpertSet::Routed, Variant::DecodeK1, activations).0,
                "exlr8_a16_serving");
            assert_eq!(program_names(ExpertSet::Shared, Variant::DecodeK1, activations).0,
                "exlr8_shared_a16_serving");
            assert_eq!(program_names(ExpertSet::Routed, Variant::DecodeGrouped, activations).0,
                "exlr8_a16_grouped64");
        }
        assert_eq!(program_names(ExpertSet::Routed, Variant::Prefill,
            Exlr8PrefillActivations::A16).0, "exlr8_a16_grouped");
        assert_eq!(program_names(ExpertSet::Routed, Variant::Prefill,
            Exlr8PrefillActivations::A8).0, "exlr8_a8_grouped");
        assert_eq!(program_names(ExpertSet::Routed, Variant::DecodeGrouped,
            Exlr8PrefillActivations::A16).1, "exlr8_group");
        assert_eq!(program_names(ExpertSet::Routed, Variant::Prefill,
            Exlr8PrefillActivations::A16).1, "exlr8_grouped_group");
    }

    #[test]
    fn refuses_reallocated_route_storage_or_shape_after_capture() {
        let identity = RoutingIdentity {
            addresses: [256, 512, 768, 1024, 1280],
            rows: 64,
            top_k: 8,
            max_groups: 512,
        };
        let mut lifetime = RoutingLifetime::default();
        lifetime.capture(identity).unwrap();
        lifetime.check(identity).unwrap();
        for index in 0..5 {
            let mut replacement = identity;
            replacement.addresses[index] += 4096;
            assert!(lifetime.check(replacement).is_err());
            assert!(lifetime.capture(replacement).is_err());
        }
        let mut changed = identity;
        changed.rows = 32;
        assert!(lifetime.check(changed).is_err());
        let mut changed = identity;
        changed.top_k = 4;
        assert!(lifetime.check(changed).is_err());
        let mut changed = identity;
        changed.max_groups = 1024;
        assert!(lifetime.check(changed).is_err());
        lifetime.check(identity).unwrap();
    }
}
