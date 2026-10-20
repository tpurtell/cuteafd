//! Routed-expert pieces every family uses: layer identity and native resident
//! weights, per-wave execution and the host exchange, the EXL3 package path,
//! the expertd-native service, the exact FP8 expert packages and the TP2 RTX
//! expert halves (`rtx`).

pub(crate) mod execution;
pub(crate) mod exl3;
pub(crate) mod exlr8;
pub(crate) mod fp8;
pub(crate) mod layer;
pub(crate) mod local;
pub(crate) mod paired_load;
pub(crate) mod rtx;
pub(crate) mod service;
pub(crate) mod staging;
#[cfg(test)]
mod tp_gate;
#[cfg(test)]
mod tp_reference;

/// Frozen compatibility set shared by native and EXL3 expert execution.
/// Launch capacity may vary; every decode variant must match canonical m1 sums.
#[derive(Debug)]
pub(crate) struct ExpertReductionPlan {
    decode: Vec<(usize, usize)>,
}

impl ExpertReductionPlan {
    pub(crate) fn new<S: Eq + std::fmt::Debug>(variants: &[(usize, S)]) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        let canonical = &variants
            .iter()
            .find(|(capacity, _)| *capacity == 1)
            .context("expert package needs canonical m1 reduction")?
            .1;
        ensure!(
            variants.iter().all(|(capacity, _)| *capacity > 0),
            "expert export capacity must be positive"
        );
        let maximum = variants
            .iter()
            .map(|(capacity, _)| *capacity)
            .max()
            .context("empty expert reduction package")?;
        let mut decode: Vec<_> = variants
            .iter()
            .enumerate()
            .filter(|(_, (capacity, signature))| *capacity <= 80 && signature == canonical)
            .map(|(index, (capacity, _))| (*capacity, index))
            .collect();
        decode.sort_unstable();
        ensure!(decode.last().is_some_and(|&(capacity, _)| capacity >= maximum.min(80)),
            "expert package has no export covering {} decode rows with canonical m1 reduction {canonical:?}; admitted capacity/recipes: {variants:?}; export a matching capacity",
            maximum.min(80));
        Ok(Self { decode })
    }

    /// One compiled-recipe check for all native families and routed-expert owners.
    pub(crate) fn native(
        library: &cuteafd_ffi::NativeLibrary,
        infos: impl IntoIterator<Item = cuteafd_ffi::V41ExpertInfo>,
    ) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        let variants = infos.into_iter().map(|info| {
            let recipe = library.native_expert_reduction(&info).with_context(|| {
                format!("native MXFP4 role {} m{} reduction export", info.role, info.capacity_rows)
            })?;
            if info.capacity_rows <= 80 {
                ensure!(recipe.plane_fold == 1 && recipe.implementation == 1,
                    "native MXFP4 role {} m{} decode requires ordered FP32 slice planes, got {recipe:?}",
                    info.role, info.capacity_rows);
            }
            Ok((info.capacity_rows as usize, recipe))
        }).collect::<anyhow::Result<Vec<_>>>()?;
        Self::new(&variants)
            .context("native MXFP4 decode differs from canonical m1 slice/fold recipe")
    }

    pub(crate) fn execution_index(
        &self,
        rows: usize,
        required: usize,
        capacities: impl Iterator<Item = usize>,
    ) -> anyhow::Result<usize> {
        use anyhow::{ensure, Context};
        ensure!(rows > 0, "expert live rows must be positive");
        if rows <= 80 {
            return self
                .decode
                .iter()
                .find(|&&(capacity, _)| capacity >= rows)
                .map(|&(_, index)| index)
                .context("decode exceeds fixed expert reduction plan");
        }
        capacities
            .enumerate()
            .find(|&(_, capacity)| capacity >= required)
            .map(|(index, _)| index)
            .context("no expert execution for admitted rows")
    }
}

#[cfg(test)]
mod reduction_tests {
    use super::ExpertReductionPlan;

    #[test]
    fn native_owned_state_order_freezes_serial_and_grouped_decode() {
        // Execution owns main, m1, then m80 (not ascending capacities).
        let variants = [(4096, 192), (1, 64), (80, 64)];
        let plan = ExpertReductionPlan::new(&variants).unwrap();
        for rows in 1..=80 {
            assert_eq!(
                plan.execution_index(rows, rows, variants.iter().map(|v| v.0))
                    .unwrap(),
                if rows == 1 { 1 } else { 2 }
            );
        }
        assert_eq!(
            plan.execution_index(81, 81, variants.iter().map(|v| v.0))
                .unwrap(),
            0
        );
        let small = [(16, 64), (1, 64)];
        let plan = ExpertReductionPlan::new(&small).unwrap();
        for rows in 1..=16 {
            assert_eq!(
                plan.execution_index(rows, rows, small.iter().map(|v| v.0))
                    .unwrap(),
                if rows == 1 { 1 } else { 0 }
            );
        }
        assert!(plan
            .execution_index(17, 17, small.iter().map(|v| v.0))
            .is_err());
        let serial = ExpertReductionPlan::new(&[(1, 64)]).unwrap();
        assert_eq!(serial.execution_index(1, 1, [1].into_iter()).unwrap(), 0);
    }

    #[test]
    fn shared_decode_selector_is_width_invariant_and_ignores_prefill_numerics() {
        let variants = [(1, 64), (16, 192), (80, 64), (4096, 192)];
        let plan = ExpertReductionPlan::new(&variants).unwrap();
        for rows in 1..=80 {
            let index = plan
                .execution_index(rows, rows, variants.iter().map(|v| v.0))
                .unwrap();
            assert_eq!(variants[index].1, 64);
        }
        assert_eq!(
            plan.execution_index(81, 81, variants.iter().map(|v| v.0))
                .unwrap(),
            3
        );
        assert!(plan
            .execution_index(0, 1, variants.iter().map(|v| v.0))
            .is_err());
        assert!(ExpertReductionPlan::new(&[(1, 64), (80, 192), (4096, 192)]).is_err());
        assert!(ExpertReductionPlan::new(&[(80, 64)]).is_err());
        // Metadata changes after admission cannot mutate the owned compatibility set.
        let mut changed = variants;
        changed[2].1 = 192;
        assert_eq!(
            plan.execution_index(80, 80, changed.iter().map(|v| v.0))
                .unwrap(),
            2
        );
        assert!(ExpertReductionPlan::new(&changed).is_err());
    }
}
