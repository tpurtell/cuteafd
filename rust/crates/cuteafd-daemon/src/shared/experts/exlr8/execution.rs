//! Admitted K1 workspace: graph-visible routing storage never moves.
use super::{view, Exlr8Weights};
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Program, ProgramSpec, Programs, Scalar};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::plan::exlr8::residency::ExpertSet;
use std::ffi::c_void;

fn program_names(set: ExpertSet) -> (&'static str, &'static str) {
    match set {
        ExpertSet::Routed => ("exlr8_a16_serving", "exlr8_group"),
        ExpertSet::Shared => ("exlr8_shared_a16_serving", "exlr8_shared_group"),
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
    "output",
    "group_count",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RoutingIdentity {
    addresses: [usize; 4],
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
    ranges: [(usize, usize); 10],
    bytes: usize,
    rows: usize,
    top_k: usize,
    max_groups: usize,
    hidden: usize,
    units: usize,
}
impl Workspace {
    fn plan(spec: &ProgramSpec, rows: usize, set: ExpertSet) -> Result<Self> {
        cuteafd_ffi::exlr8::validate(spec)?;
        ensure!(
            spec.name == program_names(set).0 && spec.geometry["abi_version"] == 2,
            "EXLR8 execution requires device-count ABI v2"
        );
        ensure!(
            rows > 0 && rows <= 64 && rows <= spec.capacity_rows as usize,
            "EXLR8 rows outside admitted K1 capacity"
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
        let (ranges, bytes) = cuteafd_loader::plan::exlr8::residency::execution_workspace_layout(
            rows, hidden, top_k, max_groups, scratch("fc1")?,
            scratch("intermediate")?, scratch("fc2")?)?;
        Ok(Self {
            ranges,
            bytes,
            rows,
            top_k,
            max_groups,
            hidden,
            units,
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
    set: ExpertSet,
}
impl<'a> Exlr8Execution<'a> {
    pub(crate) fn workspace_bytes(
        programs: &'a Programs<'a>,
        rows: usize,
        set: ExpertSet,
    ) -> Result<usize> {
        let k1 = programs.program(program_names(set).0, K1_POINTERS)?;
        Ok(Workspace::plan(k1.spec(), rows, set)?.bytes)
    }

    pub(crate) fn new(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        rows: usize,
        available: usize,
        set: ExpertSet,
    ) -> Result<Self> {
        cuteafd_loader::plan::exlr8::require_spark_rotation()?;
        let (k1_name, group_name) = program_names(set);
        let k1 = programs.program(k1_name, K1_POINTERS)?;
        let group = programs.program(
            group_name,
            &[
                "ids",
                "group_expert",
                "route_slots",
                "route_group_lane",
                "group_count",
            ],
        )?;
        let layout = Workspace::plan(k1.spec(), rows, set)?;
        ensure!(
            group.spec().geometry["top_k"] == k1.spec().geometry["top_k"]
                && group.spec().geometry["max_groups"] == k1.spec().geometry["max_groups"],
            "EXLR8 grouping and K1 geometry differ"
        );
        ensure!(
            layout.bytes <= available,
            "EXLR8 workspace exceeds admitted device bytes"
        );
        let intermediate_bytes = usize::try_from(k1.spec().scratch["intermediate"])?;
        let ctas = i32::try_from(
            k1.spec().geometry["ctas_per_sm"]
                .as_u64()
                .context("CTA hint")?,
        )?;
        let cta_limit = i32::try_from(library.sm_count()?)?
            .checked_mul(ctas)
            .context("CTA limit overflow")?;
        ensure!(cta_limit > 0, "runtime SM count must be positive");
        let arena = DeviceAllocation::named(library, layout.bytes, "experts/workspace/exlr8")?;
        Ok(Self {
            group,
            k1,
            arena,
            layout,
            routing: RoutingLifetime::default(),
            intermediate_bytes,
            cta_limit,
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
        ];
        let p = |i: usize| buffers[i].ptr;
        let intermediate = view(buffers[9], 0, self.intermediate_bytes)?;
        let fc2 = view(
            buffers[9],
            self.intermediate_bytes,
            buffers[9].bytes - self.intermediate_bytes,
        )?;
        // SAFETY: private admitted arena views retain the manifest's exact sizes;
        // weights own all referenced descriptors, and caller retains the stream.
        unsafe {
            self.group.launch(
                &[p(1), p(4), p(5), p(6), p(7)],
                &[Scalar::I32(rows as i32)],
                stream,
            )?;
            self.k1.launch(
                &[
                    p(0),
                    weights.pointers()?.ptr,
                    weights.bits()?.ptr,
                    p(4),
                    p(5),
                    p(6),
                    p(2),
                    p(8),
                    intermediate.ptr,
                    fc2.ptr,
                    p(3),
                    p(7),
                ],
                &[Scalar::I32(rows as i32), Scalar::I32(self.cta_limit)],
                stream,
            )?;
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
            name: program_names(set).0.into(),
            pointers: K1_POINTERS.iter().map(|name| (*name).into()).collect(),
            scalar_kinds: b"ii".to_vec(),
            scratch: [
                ("fc1".into(), groups * 16 * 2 * 6 * 16 * 128 * 4),
                ("intermediate".into(), groups * 16 * 16 * 128 * 2),
                ("fc2".into(), groups * 16 * 6144 * 4),
            ]
            .into(),
            capacity_rows: 64,
            geometry: serde_json::json!({"abi_version": 2, "hidden": 6144, "units": 16,
                "top_k": top_k, "max_groups": groups, "max_rows": 64,
                "split_k_blocks": 8, "fc1_splits": 6, "ctas_per_sm": 4}),
        }
    }

    #[test]
    fn exlr8_shared_workspace_keeps_exact_aligned_admission() {
        for set in [ExpertSet::Routed, ExpertSet::Shared] {
            let spec = spec(set);
            let wide = Workspace::plan(&spec, 64, set).unwrap();
            let single = Workspace::plan(&spec, 1, set).unwrap();
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
            assert!(Workspace::plan(&spec, 0, set).is_err());
            assert!(Workspace::plan(&spec, 65, set).is_err());
            let other = if set == ExpertSet::Shared {
                ExpertSet::Routed
            } else {
                ExpertSet::Shared
            };
            assert!(Workspace::plan(&spec, 64, other).is_err());
        }
        assert!(
            Workspace::plan(&spec(ExpertSet::Shared), 64, ExpertSet::Shared)
                .unwrap()
                .bytes
                < Workspace::plan(&spec(ExpertSet::Routed), 64, ExpertSet::Routed)
                    .unwrap()
                    .bytes
        );
    }

    #[test]
    fn refuses_reallocated_route_storage_or_shape_after_capture() {
        let identity = RoutingIdentity {
            addresses: [256, 512, 768, 1024],
            rows: 64,
            top_k: 8,
            max_groups: 512,
        };
        let mut lifetime = RoutingLifetime::default();
        lifetime.capture(identity).unwrap();
        lifetime.check(identity).unwrap();
        for index in 0..4 {
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
