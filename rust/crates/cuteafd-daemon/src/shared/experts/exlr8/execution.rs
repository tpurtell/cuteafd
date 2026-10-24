//! Admitted K1 workspace: graph-visible routing storage never moves.
use super::{view, Exlr8PrefillActivations, Exlr8Weights};
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Program, ProgramSpec, Programs, Scalar};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::plan::exlr8::residency::ExpertSet;
use std::ffi::c_void;

/// Measured first decode row at which the 64-row grouped instance beats the
/// compact-unit routed serving core, keyed on this rank's owned units per
/// expert and never on the SM count.
///
/// Anchors on GB10, one interleaved session per width (rank 0, random and
/// realistic routes, medians of three): TP6 (three owned units) keeps the
/// routed core through 8 rows (1.22-1.26x over K1 against 1.17-1.20x for the
/// grouped instance) and switches at 16 (grouped 1.24-1.28x against
/// 1.12-1.14x); TP4 (four owned units) is a 1-2% tie at 8 rows and the
/// grouped instance leads by 4-8% from 16 rows on.
///
/// Widths outside those anchors take the nearest measured one: two owned
/// units take the three-unit threshold and five or more take the four-unit
/// threshold. A narrow rank wastes more of a fixed grid, which argues for a
/// later switch below four units, so the nearest anchor stays conservative.
pub(crate) fn grouped_decode_from_row(unit_limit: usize) -> usize {
    if unit_limit <= 3 { 16 } else { 8 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    /// Shared-expert serving kernel: one K1 launch over all units.
    SharedDecode,
    /// Routed decode below the grouped threshold: the compact-unit routed
    /// serving core, whose grids and scratch cover only this rank's owned
    /// units.
    DecodeCompact,
    /// 64-row grouped instance with F32 FC2 partials: bit-identical to K1.
    DecodeGrouped,
    /// 4096-row grouped prefill.
    Prefill,
}

impl Variant {
    /// Owned-unit grids and unit-scaled scratch.
    fn owned_units(self) -> bool {
        matches!(self, Self::DecodeCompact | Self::DecodeGrouped | Self::Prefill)
    }
    /// Carries the `unit_count`/`unit_map` tables; the routed serving core
    /// derives its live units on device instead.
    fn unit_tables(self) -> bool {
        matches!(self, Self::DecodeGrouped | Self::Prefill)
    }
}

fn program_names(
    set: ExpertSet,
    variant: Variant,
    activations: Exlr8PrefillActivations,
) -> (&'static str, &'static str) {
    match set {
        ExpertSet::Shared => ("exlr8_shared_a16_serving", "exlr8_shared_group"),
        ExpertSet::Routed => match variant {
            Variant::Prefill => (
                match activations {
                    Exlr8PrefillActivations::A16 => "exlr8_a16_grouped",
                    Exlr8PrefillActivations::A8 => "exlr8_a8_grouped",
                },
                "exlr8_grouped_group",
            ),
            Variant::DecodeGrouped => ("exlr8_a16_grouped64", "exlr8_group"),
            Variant::DecodeCompact => ("exlr8_a16_serving", "exlr8_group"),
            // `Workspace::plan` refuses a shared variant on the routed set
            // before this name reaches a program lookup.
            Variant::SharedDecode => ("exlr8_shared_a16_serving", "exlr8_shared_group"),
        },
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
    if set == ExpertSet::Routed && variant.unit_tables() {
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
    /// Unit-scaled scratch under a compact arena: every owned-unit instance.
    compact: bool,
    /// Registers `unit_count`/`unit_map` and the `unit_limit` scalar.
    unit_tables: bool,
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
            (set == ExpertSet::Shared) == (variant == Variant::SharedDecode),
            "EXLR8 shared decode requires the shared expert set"
        );
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
        let compact = set == ExpertSet::Routed && variant.owned_units();
        let unit_tables = set == ExpertSet::Routed && variant.unit_tables();
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
            unit_tables,
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
    pub(crate) fn shared_workspace_bytes(
        programs: &'a Programs<'a>,
        rows: usize,
    ) -> Result<usize> {
        Self::workspace_bytes_for(programs, rows, ExpertSet::Shared, Variant::SharedDecode, 16,
            Exlr8PrefillActivations::A16)
    }

    /// Routed decode arena: the compact-unit serving core over owned units.
    pub(crate) fn decode_compact_workspace_bytes(
        programs: &'a Programs<'a>,
        rows: usize,
        unit_limit: usize,
    ) -> Result<usize> {
        Self::workspace_bytes_for(programs, rows, ExpertSet::Routed, Variant::DecodeCompact,
            unit_limit, Exlr8PrefillActivations::A16)
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

    pub(crate) fn new_shared(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        rows: usize,
        available: usize,
    ) -> Result<Self> {
        Self::new_for(library, programs, rows, available, ExpertSet::Shared, Variant::SharedDecode,
            16, Exlr8PrefillActivations::A16)
    }

    pub(crate) fn new_decode_compact(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        rows: usize,
        available: usize,
        unit_limit: usize,
    ) -> Result<Self> {
        Self::new_for(library, programs, rows, available, ExpertSet::Routed,
            Variant::DecodeCompact, unit_limit, Exlr8PrefillActivations::A16)
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
        ensure!(!self.layout.compact
                || weights.layout.compact_unit_limit() <= self.layout.unit_limit,
            "EXLR8 layer ownership exceeds compact arena stride");
        let scalars = [Scalar::I32(rows as i32), Scalar::I32(self.cta_limit),
            Scalar::I32(self.layout.unit_limit as i32)];
        let mut operands = [p(10), weights.pointers()?.ptr, weights.bits()?.ptr,
            p(4), p(5), p(6), p(2), p(8), intermediate.ptr, fc2.ptr,
            weights.signs(true)?.ptr, p(3), p(7), std::ptr::null_mut(), std::ptr::null_mut()];
        if self.layout.unit_tables {
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
            self.k1.launch(&operands[..if self.layout.unit_tables { 15 } else { 13 }],
                &scalars[..if self.layout.unit_tables { 3 } else { 2 }], stream)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared_spec() -> ProgramSpec {
        ProgramSpec {
            name: program_names(ExpertSet::Shared, Variant::SharedDecode,
                Exlr8PrefillActivations::A16).0.into(),
            pointers: K1_POINTERS.iter().map(|name| (*name).into()).collect(),
            scalar_kinds: b"ii".to_vec(),
            scratch: [
                ("fc1".into(), 4 * 16 * 2 * 6 * 16 * 128 * 4),
                ("intermediate".into(), 4 * 16 * 16 * 128 * 2),
                ("fc2".into(), 4 * 16 * 6144 * 4),
            ]
            .into(),
            capacity_rows: 64,
            geometry: serde_json::json!({"abi_version": 3, "hidden": 6144, "units": 16,
                "top_k": 1, "max_groups": 4, "max_rows": 64,
                "split_k_blocks": 8, "fc1_splits": 6, "ctas_per_sm": 4}),
        }
    }

    /// The routed decode core: the plain serving ABI over owned-unit scratch.
    fn compact_spec() -> ProgramSpec {
        let mut spec = shared_spec();
        spec.name = program_names(ExpertSet::Routed, Variant::DecodeCompact,
            Exlr8PrefillActivations::A16).0.into();
        spec.pointers = pointers(ExpertSet::Routed, Variant::DecodeCompact)
            .into_iter().map(String::from).collect();
        assert_eq!(spec.pointers.len(), 13);
        assert_eq!(spec.scalar_kinds, b"ii");
        spec.geometry["top_k"] = serde_json::json!(8);
        spec.geometry["max_groups"] = serde_json::json!(512);
        spec.geometry["group_rows"] = serde_json::json!(16);
        spec.geometry["serving_variant"] = serde_json::json!("k1_reg_pair_b2");
        spec.scratch = [
            ("fc1".into(), 512 * 16 * 2 * 6 * 16 * 128 * 4),
            ("intermediate".into(), 512 * 16 * 16 * 128 * 2),
            ("fc2".into(), 512 * 16 * 6144 * 4),
        ]
        .into();
        spec
    }

    /// The shipped 64-row grouped instance: 16-row groups, F32 FC2 partials.
    fn grouped64_spec() -> ProgramSpec {
        let mut spec = compact_spec();
        spec.name = program_names(ExpertSet::Routed, Variant::DecodeGrouped,
            Exlr8PrefillActivations::A16).0.into();
        spec.pointers = pointers(ExpertSet::Routed, Variant::DecodeGrouped)
            .into_iter().map(String::from).collect();
        spec.scalar_kinds = b"iii".to_vec();
        spec.geometry.as_object_mut().unwrap().remove("serving_variant");
        spec.geometry["fc2_partials"] = serde_json::json!("f32");
        spec.geometry["ctas_per_sm"] = serde_json::json!(2);
        spec
    }

    #[test]
    fn decode_dispatch_switches_to_grouped_on_the_measured_owned_unit_anchor() {
        // Measured on GB10: TP6 (three owned units) keeps the routed core
        // through 8 rows and switches at 16; TP4 (four owned units) switches
        // at 8. Widths outside the anchors take the nearest one, so no SM
        // count or card identity enters the decision.
        assert_eq!(grouped_decode_from_row(1), 16);
        assert_eq!(grouped_decode_from_row(3), 16);
        assert_eq!(grouped_decode_from_row(4), 8);
        assert_eq!(grouped_decode_from_row(8), 8);
        for variant in [Variant::DecodeCompact, Variant::DecodeGrouped] {
            assert_eq!(program_names(ExpertSet::Routed, variant,
                Exlr8PrefillActivations::A16).1, "exlr8_group");
        }
        assert_eq!(program_names(ExpertSet::Routed, Variant::DecodeCompact,
            Exlr8PrefillActivations::A16).0, "exlr8_a16_serving");
        assert!(!Variant::DecodeCompact.unit_tables() && Variant::DecodeCompact.owned_units());
        assert!(Variant::DecodeGrouped.unit_tables() && Variant::Prefill.unit_tables());
    }

    #[test]
    fn exlr8_shared_workspace_keeps_exact_aligned_admission() {
        let spec = shared_spec();
        let wide = Workspace::plan(&spec, 64, ExpertSet::Shared, Variant::SharedDecode, 16,
            Exlr8PrefillActivations::A16).unwrap();
        let single = Workspace::plan(&spec, 1, ExpertSet::Shared, Variant::SharedDecode, 16,
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
        assert_eq!(wide.ranges[0].1, 64 * 12288);
        let mut stale = spec.clone();
        stale.geometry["abi_version"] = serde_json::json!(2);
        assert!(Workspace::plan(&stale, 64, ExpertSet::Shared, Variant::SharedDecode, 16,
            Exlr8PrefillActivations::A16).is_err());
        assert!(Workspace::plan(&spec, 0, ExpertSet::Shared, Variant::SharedDecode, 16,
            Exlr8PrefillActivations::A16).is_err());
        assert!(Workspace::plan(&spec, 65, ExpertSet::Shared, Variant::SharedDecode, 16,
            Exlr8PrefillActivations::A16).is_err());
        assert!(Workspace::plan(&spec, 64, ExpertSet::Routed, Variant::SharedDecode, 16,
            Exlr8PrefillActivations::A16).is_err());
    }

    #[test]
    fn compact_decode_instances_share_streams_and_scale_scratch_with_owned_units() {
        let compact = compact_spec();
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
            // The routed decode core admits the same streams and owned-unit
            // scratch as the grouped instance at the same rows, without the
            // unit tables.
            let routed = Workspace::plan(&compact, rows, ExpertSet::Routed,
                Variant::DecodeCompact, 3, Exlr8PrefillActivations::A16).unwrap();
            assert!(routed.compact && !routed.unit_tables && routed.unit_limit == 3);
            assert_eq!((routed.ranges, routed.bytes), (owned.ranges, owned.bytes));
        }
        for rows in [0, 65] {
            assert!(Workspace::plan(&grouped, rows, ExpertSet::Routed, Variant::DecodeGrouped,
                3, Exlr8PrefillActivations::A16).is_err());
            assert!(Workspace::plan(&compact, rows, ExpertSet::Routed, Variant::DecodeCompact,
                3, Exlr8PrefillActivations::A16).is_err());
        }
        let mut missing_partials = grouped.clone();
        missing_partials.geometry["fc2_partials"] = serde_json::json!("f16");
        assert!(Workspace::plan(&missing_partials, 64, ExpertSet::Routed,
            Variant::DecodeGrouped, 3, Exlr8PrefillActivations::A16).is_err());
        // Neither decode instance accepts the other's program.
        assert!(Workspace::plan(&compact, 64, ExpertSet::Routed,
            Variant::DecodeGrouped, 3, Exlr8PrefillActivations::A16).is_err());
        assert!(Workspace::plan(&grouped, 64, ExpertSet::Routed,
            Variant::DecodeCompact, 3, Exlr8PrefillActivations::A16).is_err());
        // The shared set never admits a routed decode variant.
        assert!(Workspace::plan(&compact, 64, ExpertSet::Shared,
            Variant::DecodeCompact, 3, Exlr8PrefillActivations::A16).is_err());
    }

    #[test]
    fn grouped_workspace_uses_wide_route_slots_without_changing_decode() {
        let mut grouped = compact_spec();
        grouped.name = program_names(ExpertSet::Routed, Variant::Prefill,
            Exlr8PrefillActivations::A16).0.into();
        grouped.capacity_rows = 4096;
        grouped.pointers = pointers(ExpertSet::Routed, Variant::Prefill)
            .into_iter().map(String::from).collect();
        grouped.scalar_kinds = b"iii".to_vec();
        grouped.geometry["max_rows"] = serde_json::json!(4096);
        grouped.geometry["group_rows"] = serde_json::json!(64);
        grouped.geometry.as_object_mut().unwrap().remove("serving_variant");
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
                assert_eq!(program_names(ExpertSet::Routed, Variant::DecodeCompact, activations).0,
                "exlr8_a16_serving");
            assert_eq!(program_names(ExpertSet::Shared, Variant::SharedDecode, activations).0,
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
