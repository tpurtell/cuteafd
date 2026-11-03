//! Admitted K1 workspace: graph-visible routing storage never moves.
use super::{view, Exlr8PrefillActivations, Exlr8Weights};
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Program, ProgramSpec, Programs, Scalar};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::plan::exlr8::residency::ExpertSet;
use std::ffi::c_void;
use cuteafd_ffi::exlr8::KernelFamily;
use cuteafd_loader::plan::exlr8::ExecutionVariant;

const DECODE_CAPACITY: usize = cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS;

/// Measured first decode row at which the 64-row grouped instance beats the
/// compact-unit routed serving core, keyed on runtime compute capability and
/// this rank's owned units per expert, never on the SM count.
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
/// SM120 TP2 (eight owned units): the routed core wins at every measured row
/// from 8 through 64 (1.042-1.303x); all live rows 1..64 are bit-exact. Keep
/// the grouped switch outside decode capacity. Other widths retain GB10 policy.
pub(crate) fn grouped_decode_from_row(unit_limit: usize, capability: (i32, i32)) -> usize {
    if capability == (12, 0) && unit_limit == 8 { DECODE_CAPACITY + 1 }
    else if unit_limit <= 3 { 16 } else { 8 }
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
    fn owned_units(self) -> bool { self != Self::SharedDecode }
    fn unit_tables(self) -> bool { matches!(self, Self::DecodeGrouped | Self::Prefill) }
}

fn program_names(
    family: KernelFamily,
    set: ExpertSet,
    variant: Variant,
    activations: Exlr8PrefillActivations,
) -> (&'static str, &'static str) {
    let variant = if set == ExpertSet::Shared { ExecutionVariant::SharedDecode } else {
        match variant {
            Variant::SharedDecode => ExecutionVariant::SharedDecode,
            Variant::DecodeCompact => ExecutionVariant::DecodeCompact,
            Variant::DecodeGrouped => ExecutionVariant::DecodeGrouped,
            Variant::Prefill => ExecutionVariant::Prefill,
        }
    };
    (variant.program(family, activations), variant.group(family))
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
    /// Owned-unit instance: FC1 (and, for grouped instances, the activation
    /// scratch) scale with `unit_limit`; the loader owns the exact sizing rule.
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
        use cuteafd_loader::plan::exlr8::{execution_workspace, ExecutionVariant};
        let variant = match variant {
            Variant::SharedDecode => ExecutionVariant::SharedDecode,
            Variant::DecodeCompact => ExecutionVariant::DecodeCompact,
            Variant::DecodeGrouped => ExecutionVariant::DecodeGrouped,
            Variant::Prefill => ExecutionVariant::Prefill,
        };
        let layout = execution_workspace(spec, rows, set, variant, unit_limit, activations)?;
        Ok(Self { ranges: layout.ranges, bytes: layout.bytes, rows: layout.rows,
            top_k: layout.top_k, max_groups: layout.max_groups, hidden: layout.hidden,
            units: layout.units, unit_limit: layout.unit_limit,
            intermediate_bytes: layout.intermediate_bytes, compact: layout.compact,
            unit_tables: layout.unit_tables })
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
    l2_bytes: i32,
    set: ExpertSet,
}
impl<'a> Exlr8Execution<'a> {
    pub(crate) fn routed_hidden(programs: &'a Programs<'a>, family: KernelFamily) -> Result<usize> {
        let core = programs.program(family.program(cuteafd_ffi::exlr8::KernelKind::Routed),
            &pointers(ExpertSet::Routed, Variant::DecodeCompact))?;
        cuteafd_ffi::exlr8::validate(core.spec())?;
        usize::try_from(core.spec().geometry["hidden"].as_u64().context("EXLR8 hidden missing")?)
            .context("EXLR8 hidden exceeds address space")
    }

    pub(crate) fn shared_workspace_bytes(
        programs: &'a Programs<'a>,
        family: KernelFamily,
        rows: usize,
    ) -> Result<usize> {
        Self::workspace_bytes_for(programs, family, rows, ExpertSet::Shared, Variant::SharedDecode, 16,
            Exlr8PrefillActivations::A16)
    }

    /// Routed decode arena: the compact-unit serving core over owned units.
    pub(crate) fn decode_compact_workspace_bytes(
        programs: &'a Programs<'a>,
        family: KernelFamily,
        rows: usize,
        unit_limit: usize,
    ) -> Result<usize> {
        Self::workspace_bytes_for(programs, family, rows, ExpertSet::Routed, Variant::DecodeCompact,
            unit_limit, Exlr8PrefillActivations::A16)
    }

    pub(crate) fn workspace_bytes_with_limit(
        programs: &'a Programs<'a>,
        family: KernelFamily,
        rows: usize,
        set: ExpertSet,
        unit_limit: usize,
        activations: Exlr8PrefillActivations,
    ) -> Result<usize> {
        Self::workspace_bytes_for(programs, family, rows, set, Variant::Prefill, unit_limit, activations)
    }

    /// 64-row grouped decode arena; F32 FC2 partials keep it bit-identical to K1.
    pub(crate) fn decode_grouped_workspace_bytes(
        programs: &'a Programs<'a>,
        family: KernelFamily,
        rows: usize,
        unit_limit: usize,
    ) -> Result<usize> {
        Self::workspace_bytes_for(programs, family, rows, ExpertSet::Routed, Variant::DecodeGrouped,
            unit_limit, Exlr8PrefillActivations::A16)
    }

    fn workspace_bytes_for(
        programs: &'a Programs<'a>,
        family: KernelFamily,
        rows: usize,
        set: ExpertSet,
        variant: Variant,
        unit_limit: usize,
        activations: Exlr8PrefillActivations,
    ) -> Result<usize> {
        let core = programs.program(program_names(family, set, variant, activations).0,
            &pointers(set, variant))?;
        Ok(Workspace::plan(core.spec(), rows, set, variant, unit_limit, activations)?.bytes)
    }

    pub(crate) fn new_shared(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        family: KernelFamily,
        rows: usize,
        available: usize,
    ) -> Result<Self> {
        Self::new_for(library, programs, family, rows, available, ExpertSet::Shared, Variant::SharedDecode,
            16, Exlr8PrefillActivations::A16)
    }

    pub(crate) fn new_decode_compact(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        family: KernelFamily,
        rows: usize,
        available: usize,
        unit_limit: usize,
    ) -> Result<Self> {
        Self::new_for(library, programs, family, rows, available, ExpertSet::Routed,
            Variant::DecodeCompact, unit_limit, Exlr8PrefillActivations::A16)
    }

    pub(crate) fn new_with_limit(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        family: KernelFamily,
        rows: usize,
        available: usize,
        set: ExpertSet,
        unit_limit: usize,
        activations: Exlr8PrefillActivations,
    ) -> Result<Self> {
        Self::new_for(library, programs, family, rows, available, set, Variant::Prefill, unit_limit, activations)
    }

    pub(crate) fn new_decode_grouped(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        family: KernelFamily,
        rows: usize,
        available: usize,
        unit_limit: usize,
    ) -> Result<Self> {
        Self::new_for(library, programs, family, rows, available, ExpertSet::Routed, Variant::DecodeGrouped,
            unit_limit, Exlr8PrefillActivations::A16)
    }

    fn new_for(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        family: KernelFamily,
        rows: usize,
        available: usize,
        set: ExpertSet,
        variant: Variant,
        unit_limit: usize,
        activations: Exlr8PrefillActivations,
    ) -> Result<Self> {
        let (k1_name, group_name) = program_names(family, set, variant, activations);
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
                && cuteafd_ffi::exlr8::group_rows(group.spec())?
                    == cuteafd_ffi::exlr8::group_rows(k1.spec())?
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
        let l2_bytes = i32::try_from(library.l2_cache_bytes()?).context("EXLR8 runtime L2 exceeds int32")?;
        ensure!(l2_bytes > 0, "EXLR8 runtime L2 must be positive");
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
            l2_bytes,
            set,
        })
    }
    pub(crate) fn allocation_buffer(&self) -> CuteafdDeviceBuffer { self.arena.buffer }

    /// Census retained storage and the actual kernel bindings, not forecast records.
    pub(crate) fn workspace_receipts(&self, prefix: &str,
        location: cuteafd_loader::placement::planner::Location, physical_device: i32,
        lifetime: cuteafd_loader::placement::planner::Lifetime)
        -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::placement::planner::{Allocation, Domain, Receipt, View};
        ensure!(!prefix.is_empty() && location.domain == Domain::Device,
            "invalid EXLR8 workspace census identity");
        let arena = self.arena.buffer;
        let receipt = |name: &str, buffer: CuteafdDeviceBuffer| -> Result<Receipt> {
            ensure!(buffer.device_id == physical_device && !buffer.ptr.is_null() && buffer.bytes > 0,
                "EXLR8 workspace has wrong physical device or empty buffer");
            let mut allocation = Allocation::buffer(format!("{prefix}.{name}"), location,
                u64::try_from(buffer.bytes)?, lifetime, Category::Workspace, "EXLR8 expert workspace");
            allocation.alignment = 256;
            Ok(Receipt { allocation, address: buffer.ptr as u64 })
        };
        let mut out = vec![receipt("arena", arena)?];
        let mut binding = |name: &str, buffer: CuteafdDeviceBuffer| -> Result<()> {
            let mut actual = receipt(name, buffer)?;
            let offset = (buffer.ptr as usize).checked_sub(arena.ptr as usize)
                .context("EXLR8 workspace binding precedes physical owner")?;
            ensure!(offset.checked_add(buffer.bytes).is_some_and(|end| end <= arena.bytes),
                "EXLR8 workspace binding exceeds physical owner");
            actual.allocation.view = Some(View { backing: format!("{prefix}.arena"), offset: u64::try_from(offset)? });
            out.push(actual);
            Ok(())
        };
        for (name, index) in [("input", 0), ("ids", 1), ("route_weights", 2), ("output", 3),
            ("group_expert", 4), ("route_slots", 5), ("route_group_lane", 6), ("group_count", 7),
            ("fc1", 8), ("intermediate_fc2", 9), ("rotated", 10)] {
            binding(name, self.buffer(index)?)?;
        }
        let combined = self.buffer(9)?;
        binding("intermediate", view(combined, 0, self.intermediate_bytes)?)?;
        binding("fc2", view(combined, self.intermediate_bytes,
            combined.bytes.checked_sub(self.intermediate_bytes).context("EXLR8 FC2 split exceeds buffer")?)?)?;
        Ok(out)
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
            Scalar::I32(self.layout.unit_limit as i32), Scalar::I32(self.l2_bytes)];
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
                &scalars[..if self.layout.unit_tables { 4 } else { 2 }], stream)?;
        }
        Ok(())
    }
}

/// Small group capacity bounds CPU memory; the compiled family/ABI remains exact.
#[cfg(test)]
pub(crate) fn owner_manifest(family: KernelFamily) -> serde_json::Value {
    use cuteafd_ffi::exlr8::KernelKind;
    use serde_json::json;
    let h = family.geometry().hidden;
    let mut entries = Vec::new();
    for (variant, activations) in [(Variant::SharedDecode, Exlr8PrefillActivations::A16),
        (Variant::DecodeCompact, Exlr8PrefillActivations::A16),
        (Variant::DecodeGrouped, Exlr8PrefillActivations::A16),
        (Variant::Prefill, Exlr8PrefillActivations::A16),
        (Variant::Prefill, Exlr8PrefillActivations::A8)] {
        let shared = variant == Variant::SharedDecode;
        let grouped = variant.unit_tables();
        let rows = if variant == Variant::Prefill { 4096 } else { 64 };
        let top = if shared { 1 } else { 8 };
        let g = 4usize;
        let r = 16usize;
        let s = h.div_ceil(1024);
        let mut geometry = json!({"abi_version":3,"hidden":h,"units":16,"top_k":top,
            "max_groups":g,"max_rows":rows,"split_k_blocks":8,"fc1_splits":s,"ctas_per_sm":4});
        if family == KernelFamily::GlmFlash { geometry["swiglu_limit"] = json!(10.0); }
        if grouped { geometry["group_rows"] = json!(r); }
        if variant == Variant::DecodeCompact { geometry["serving_variant"] = json!("k1_k23_coalesced"); }
        if variant == Variant::DecodeGrouped { geometry["fc2_partials"] = json!("f32"); }
        if activations == Exlr8PrefillActivations::A8 { geometry["activations"] = json!("a8"); }
        let shapes = [format!("[rows,{h}]"), "[E,16,3,3]".into(), "[E,16,3]".into(),
            format!("[{g}]"), format!("[{g},{r}]"), format!("[rows,{top}]"), format!("[rows,{top}]"),
            format!("[{g},16,2,{s},{r},128]"), format!("[{g},16,{r},128]"),
            format!("[{g},{r},{h}]"), format!("[{h}]"), format!("[rows,{h}]"), "[1]".into(),
            "[E]".into(), "[E,16]".into()];
        let dtypes = ["float16", "int64", "int32", "int32", "int32", "int32", "float32",
            "float32", "float16", "float32", "int8", "bfloat16", "int32", "int32", "int32"];
        entries.push(json!({"name":program_names(family, if shared {ExpertSet::Shared} else {ExpertSet::Routed},
            variant, activations).0,"capacity_rows":rows,"geometry":geometry,
            "pointers":pointers(if shared {ExpertSet::Shared} else {ExpertSet::Routed}, variant).iter()
                .enumerate().map(|(i,name)| json!({"name":name,"dtype":dtypes[i],"shape":shapes[i],
                    "role":if (7..=9).contains(&i) {"scratch"} else if i==11 {"out"} else {"in"}})).collect::<Vec<_>>(),
            "scalars":(["rows", "cta_limit", "unit_limit", "l2_bytes"][..if grouped {4} else {2}].iter()
                .map(|name| json!({"name":name,"type":"int32"})).collect::<Vec<_>>()),
            "scratch_bytes_at_capacity":{"fc1":g*16*2*s*r*128*4,
                "intermediate":g*16*r*128*2,"fc2":g*r*h*4}}));
    }
    for kind in [KernelKind::SharedGroup, KernelKind::Group, KernelKind::GroupedGroup] {
        let shared = kind == KernelKind::SharedGroup;
        let rows = if kind == KernelKind::GroupedGroup {4096} else {64};
        let top = if shared {1} else {8};
        let row = if shared {h} else {h+h/32};
        let descriptors = [("ids", "int32", format!("[rows,{top}]"), "in"),
            ("x", if shared {"bfloat16"} else {"uint8"}, format!("[rows,{row}]"), "in"),
            ("signs_in", "int8", format!("[{h}]"), "in"),
            ("group_expert", "int32", "[4]".into(), "out"),
            ("route_slots", "int32", "[4,16]".into(), "out"),
            ("route_group_lane", "int32", format!("[rows,{top}]"), "out"),
            ("group_count", "int32", "[1]".into(), "out"),
            ("activation", "float16", format!("[rows,{h}]"), "out")];
        let mut geometry = json!({"abi_version":2,"experts":if shared {1} else {family.geometry().experts},
            "top_k":top,"max_groups":4,"max_rows":rows,"hidden":h,"ctas_per_sm":4,"ingress_warps":8,
            "input_format":if shared {"bf16"} else {"fp8_k32"},"input_row_bytes":if shared {h*2} else {row}});
        if kind == KernelKind::GroupedGroup { geometry["group_rows"] = json!(16); }
        entries.push(json!({"name":family.program(kind),"capacity_rows":rows,"geometry":geometry,
            "pointers":descriptors.iter().map(|(name,dtype,shape,role)|
                json!({"name":name,"dtype":dtype,"shape":shape,"role":role})).collect::<Vec<_>>(),
            "scalars":[{"name":"rows","type":"int32"},{"name":"cta_limit","type":"int32"}],
            "scratch_bytes_at_capacity":{}}));
    }
    json!({"programs":entries})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn retained_execution_arenas_reconcile_actual_kernel_bindings() -> Result<()> {
        execution_owner_fixture(KernelFamily::Glm)
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn retained_flash_execution_arenas_reconcile_actual_kernel_bindings() -> Result<()> {
        execution_owner_fixture(KernelFamily::GlmFlash)
    }

    fn execution_owner_fixture(family: KernelFamily) -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::placement::planner::{self, *};
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        {
            fixture.exlr8_programs(&library, family == KernelFamily::GlmFlash)?;
            let manifest = fixture.directory().join("PROGRAMS.json");
            std::fs::write(&manifest, serde_json::to_vec(&owner_manifest(family))?)?;
            let programs = library.programs()?.with_manifest(&manifest)?;
            for (physical, location) in [(1, Location::coordinator(0)),
                (0, Location { device: planner::Device::Spark(3), domain: Domain::Device })] {
                library.cuda_set_device(physical)?;
                fixture.configure_memory(&library, physical, 64 << 20)?;
                for (variant, activations) in [(Variant::SharedDecode, Exlr8PrefillActivations::A16),
                    (Variant::DecodeCompact, Exlr8PrefillActivations::A16),
                    (Variant::DecodeGrouped, Exlr8PrefillActivations::A16),
                    (Variant::Prefill, Exlr8PrefillActivations::A16),
                    (Variant::Prefill, Exlr8PrefillActivations::A8)] {
                    let shared = variant == Variant::SharedDecode;
                    let set = if shared {ExpertSet::Shared} else {ExpertSet::Routed};
                    let public_variant = match variant {
                        Variant::SharedDecode => ExecutionVariant::SharedDecode,
                        Variant::DecodeCompact => ExecutionVariant::DecodeCompact,
                        Variant::DecodeGrouped => ExecutionVariant::DecodeGrouped,
                        Variant::Prefill => ExecutionVariant::Prefill,
                    };
                    for rows in if variant == Variant::Prefill { [65, 128] } else { [1, 64] } {
                        let spec = programs.spec(program_names(family, set, variant, activations).0)?;
                        let layout = cuteafd_loader::plan::exlr8::execution_workspace(spec, rows, set,
                            public_variant, 3, activations)?;
                        let allocations = layout.allocations("test.execution", location, Lifetime::RESIDENT)?;
                        let facts = Facts {phase_order: cuteafd_loader::placement::planner::PhaseOrder::TARGET_THEN_DRAFT, allocations, deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![] };
                        let budgets = vec![Budget { id:"CPU fixture".into(),locations:vec![location],
                            physical_bytes:64<<20,logical_bytes:64<<20,headroom:0,
                            baseline:Baseline::Forecast {occupied:0,source:"CPU fixture".into(),assumed:false} }];
                        let phases = planner::evaluate(&facts, &budgets)?;
                        let plan = ResolvedPlan { request:Request {context:64,pool:Pin::Exact(64),onboard:Pin::Auto,
                                prefill:Pin::Auto,spark_reply_format: None, operation:Operation::Serving},
                            decisions:Decisions {spark_reply_format: None, attention:vec![],onboard_layers:vec![],expert_homes:Default::default(),
                                embedding:None,prefill:PrefillShape {rows:64,waves:1},lanes:1},
                            pool_tokens:64,facts,budgets,phases };
                        assert!(Exlr8Execution::new_for(&library, &programs, family, rows, layout.bytes-1,
                            set, variant, 3, activations).is_err());
                        assert_eq!(library.cuda_memory_info()?, (64 << 20, 64 << 20));
                        let mut execution = Exlr8Execution::new_for(&library, &programs, family, rows,
                            layout.bytes, set, variant, 3, activations)?;
                        execution.seal_graph_storage()?;
                        let receipts = execution.workspace_receipts("test.execution", location, physical, Lifetime::RESIDENT)?;
                        planner::reconcile(&plan, Phase::Load, &receipts)?;
                        assert!(execution.workspace_receipts("test.execution", location, 1-physical, Lifetime::RESIDENT).is_err());
                        assert_eq!(library.cuda_memory_info()?.0, (64 << 20) - layout.bytes);
                        for negative in 0..11 {
                            let mut bad = receipts.clone();
                            match negative {
                                0 => { bad.pop(); },
                                1 => { let mut extra=bad[0].clone();extra.allocation.id.push_str(".extra");bad.push(extra); },
                                2 => bad[0].allocation.bytes += 256,
                                3 => { bad[1].allocation.bytes += 1;bad[2].allocation.bytes -= 1; },
                                4 => bad[0].allocation.location=Location::coordinator(7),
                                5 => bad[0].address=0,
                                6 => bad[1].address += 256,
                                7 => bad[1].allocation.view.as_mut().unwrap().offset += 256,
                                8 => bad[1].allocation.view.as_mut().unwrap().backing=bad[2].allocation.id.clone(),
                                9 => bad[1].allocation.dtype="bf16".into(),
                                10 => bad[0].allocation.lifetime.first=Phase::PostTargetKv,
                                _ => unreachable!(),
                            }
                            assert!(planner::reconcile(&plan, Phase::Load, &bad).is_err(), "negative {negative}");
                        }
                        let range = execution.layout.ranges[0];
                        execution.layout.ranges[0].0 = execution.arena.buffer.bytes;
                        assert!(execution.workspace_receipts("test.execution", location, physical, Lifetime::RESIDENT).is_err());
                        execution.layout.ranges[0] = range;
                        let split = execution.intermediate_bytes;
                        execution.intermediate_bytes = execution.layout.ranges[9].1 + 1;
                        assert!(execution.workspace_receipts("test.execution", location, physical, Lifetime::RESIDENT).is_err());
                        execution.intermediate_bytes = split;
                        execution.routing.check(execution.identity()?)?;
                        drop(execution);
                        assert_eq!(library.cuda_memory_info()?, (64 << 20, 64 << 20));
                    }
                }
            }
        }
        Ok(())
    }

    fn shared_spec() -> ProgramSpec {
        ProgramSpec {
            name: program_names(KernelFamily::Glm, ExpertSet::Shared, Variant::SharedDecode,
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
        spec.name = program_names(KernelFamily::Glm, ExpertSet::Routed, Variant::DecodeCompact,
            Exlr8PrefillActivations::A16).0.into();
        spec.pointers = pointers(ExpertSet::Routed, Variant::DecodeCompact)
            .into_iter().map(String::from).collect();
        assert_eq!(spec.pointers.len(), 13);
        assert_eq!(spec.scalar_kinds, b"ii");
        spec.geometry["top_k"] = serde_json::json!(8);
        spec.geometry["max_groups"] = serde_json::json!(512);
        spec.geometry["serving_variant"] = serde_json::json!("k1_k23_coalesced");
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
        spec.name = program_names(KernelFamily::Glm, ExpertSet::Routed, Variant::DecodeGrouped,
            Exlr8PrefillActivations::A16).0.into();
        spec.pointers = pointers(ExpertSet::Routed, Variant::DecodeGrouped)
            .into_iter().map(String::from).collect();
        spec.scalar_kinds = b"iiii".to_vec();
        spec.geometry.as_object_mut().unwrap().remove("serving_variant");
        spec.geometry["group_rows"] = serde_json::json!(cuteafd_ffi::exlr8::COMPACT_GROUP_ROWS);
        spec.geometry["fc2_partials"] = serde_json::json!("f32");
        spec.geometry["ctas_per_sm"] = serde_json::json!(2);
        spec
    }

    #[test]
    fn decode_dispatch_switches_to_grouped_on_the_measured_owned_unit_anchor() {
        for capability in [(12, 1), (11, 0), (13, 0)] {
            assert_eq!(grouped_decode_from_row(1, capability), 16);
            assert_eq!(grouped_decode_from_row(3, capability), 16);
            assert_eq!(grouped_decode_from_row(4, capability), 8);
            assert_eq!(grouped_decode_from_row(8, capability), 8);
        }
        for units in 1..=16 {
            let expected = if units == 8 { DECODE_CAPACITY + 1 } else if units <= 3 { 16 } else { 8 };
            assert_eq!(grouped_decode_from_row(units, (12, 0)), expected);
        }
        for rows in 1..=DECODE_CAPACITY {
            assert!(rows < grouped_decode_from_row(8, (12, 0)));
        }
        for variant in [Variant::DecodeCompact, Variant::DecodeGrouped] {
            assert_eq!(program_names(KernelFamily::Glm, ExpertSet::Routed, variant,
                Exlr8PrefillActivations::A16).1, "exlr8_group");
        }
        assert_eq!(program_names(KernelFamily::Glm, ExpertSet::Routed, Variant::DecodeCompact,
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
            // The routed decode core admits the same streams and owned-unit FC1
            // scratch as the grouped instance at the same rows, without the
            // unit tables; its activation scratch keeps the global unit stride
            // its kernels address.
            let routed = Workspace::plan(&compact, rows, ExpertSet::Routed,
                Variant::DecodeCompact, 3, Exlr8PrefillActivations::A16).unwrap();
            assert!(routed.compact && !routed.unit_tables && routed.unit_limit == 3);
            assert_eq!(routed.ranges[..9], owned.ranges[..9]);
            assert_eq!(routed.intermediate_bytes, full.intermediate_bytes);
            assert_eq!(routed.ranges[9].1, full.ranges[9].1);
            assert!(routed.bytes > owned.bytes && routed.bytes < full.bytes);
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

    /// Highest byte each decode kernel addresses in FC1 and in the activation
    /// `intermediate`, as the fork's kernels index them (475ceec6 `exlr8.py`
    /// `Exlr8RoutedServingKernel`, `exlr8_grouped.py`): the routed core stores
    /// FC1 by live owned-unit task (`task < live_units * 2 * splits`) but the
    /// activation write and FC2 read use `(group * units + unit)` over all
    /// global units; the grouped instance uses `group * unit_limit + local`.
    fn decode_scratch_extent(variant: Variant, groups: usize, live_units: usize,
        unit_limit: usize, last_global_unit: usize) -> (usize, usize) {
        const TILE_F32: usize = 16 * 128 * 4;
        const TILE_F16: usize = 16 * 128 * 2;
        let (units, splits) = (16, 6);
        let fc1 = live_units * 2 * splits * TILE_F32;
        let intermediate = match variant {
            Variant::DecodeCompact => ((groups - 1) * units + last_global_unit + 1) * TILE_F16,
            Variant::DecodeGrouped => groups * unit_limit * TILE_F16,
            _ => unreachable!(),
        };
        (fc1, intermediate)
    }

    #[test]
    fn decode_arenas_cover_every_tp4_and_tp6_owned_unit_kernel_address() {
        // Fails before the fix: the routed core's activation scratch was cut to
        // three owned units at TP6, so 96+ live groups (rows 12-15 below the
        // 16-row grouped switch) wrote over FC2 partials and the next layer's
        // router saw a non-finite gate weight.
        let compact = compact_spec();
        let grouped = grouped64_spec();
        for (world, unit_limit, capability) in [
            (4usize, 4usize, (12, 1)), (6, 3, (12, 1)), (6, 2, (12, 1)),
            (2, 8, (12, 0)),
        ] {
            let switch = grouped_decode_from_row(unit_limit, capability);
            for rows in 1..=64usize {
                let variant = if rows >= switch { Variant::DecodeGrouped } else { Variant::DecodeCompact };
                let spec = if variant == Variant::DecodeGrouped { &grouped } else { &compact };
                let layout = Workspace::plan(spec, 64, ExpertSet::Routed, variant, unit_limit,
                    Exlr8PrefillActivations::A16).unwrap();
                // Distinct experts per route: the most groups this row count makes,
                // each owning `unit_limit` units up to the highest global unit.
                let groups = (rows * 8).min(512);
                let (fc1, intermediate) = decode_scratch_extent(variant, groups,
                    groups * unit_limit, unit_limit, 15);
                assert!(fc1 <= layout.ranges[8].1,
                    "TP{world} u={unit_limit} rows={rows}: FC1 {fc1} > {}", layout.ranges[8].1);
                assert!(intermediate <= layout.intermediate_bytes,
                    "TP{world} u={unit_limit} rows={rows} {variant:?}: intermediate {intermediate} > {}",
                    layout.intermediate_bytes);
            }
        }
    }

    #[test]
    fn grouped_workspace_uses_wide_route_slots_without_changing_decode() {
        let mut grouped = compact_spec();
        grouped.name = program_names(KernelFamily::Glm, ExpertSet::Routed, Variant::Prefill,
            Exlr8PrefillActivations::A16).0.into();
        grouped.capacity_rows = 4096;
        grouped.pointers = pointers(ExpertSet::Routed, Variant::Prefill)
            .into_iter().map(String::from).collect();
        grouped.scalar_kinds = b"iiii".to_vec();
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
            grouped_a8.name = program_names(KernelFamily::Glm, ExpertSet::Routed, Variant::Prefill,
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
                assert_eq!(program_names(KernelFamily::Glm, ExpertSet::Routed, Variant::DecodeCompact, activations).0,
                "exlr8_a16_serving");
            assert_eq!(program_names(KernelFamily::Glm, ExpertSet::Shared, Variant::SharedDecode, activations).0,
                "exlr8_shared_a16_serving");
            assert_eq!(program_names(KernelFamily::Glm, ExpertSet::Routed, Variant::DecodeGrouped, activations).0,
                "exlr8_a16_grouped64");
        }
        assert_eq!(program_names(KernelFamily::Glm, ExpertSet::Routed, Variant::Prefill,
            Exlr8PrefillActivations::A16).0, "exlr8_a16_grouped");
        assert_eq!(program_names(KernelFamily::Glm, ExpertSet::Routed, Variant::Prefill,
            Exlr8PrefillActivations::A8).0, "exlr8_a8_grouped");
        assert_eq!(program_names(KernelFamily::Glm, ExpertSet::Routed, Variant::DecodeGrouped,
            Exlr8PrefillActivations::A16).1, "exlr8_group");
        assert_eq!(program_names(KernelFamily::Glm, ExpertSet::Routed, Variant::Prefill,
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
