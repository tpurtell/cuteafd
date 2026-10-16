//! Immutable per-layer expert binding beside P3's attention execution plan.
//! An installed backend is not a default for unbound layers: every routed
//! layer must match the solver, even when local and Spark backends coexist.
use cuteafd_loader::placement::{ExpertHome, LayerAssignment};
use thiserror::Error;

/// Attention residency is independent of a layer's FFN width. A whole DSA
/// owner runs full-width attention while both ranks retain their FFN shares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionShare {
    Absent,
    Whole,
    Split,
}

impl AttentionShare {
    pub fn bind(mode: cuteafd_loader::placement::LayerMode, ranks: usize) -> Result<Vec<Self>, BindingError> {
        use cuteafd_loader::placement::LayerMode;
        match mode {
            LayerMode::HeadSplit | LayerMode::ContextSplit if ranks == 2 => Ok(vec![Self::Split; ranks]),
            LayerMode::Whole { gpu, .. } if (1..=2).contains(&ranks) && usize::from(gpu) < ranks => {
                let mut shares = vec![Self::Absent; ranks];
                shares[usize::from(gpu)] = Self::Whole;
                Ok(shares)
            }
            _ => Err(BindingError::Attention { mode, ranks }),
        }
    }

    pub fn present(self) -> bool { self != Self::Absent }
    pub fn split(self) -> bool { self == Self::Split }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertRoute {
    Dense,
    Skipped,
    Backend(ExpertHome),
}

pub(crate) struct BackendLayers<'a> {
    pub home: ExpertHome,
    /// Backbone layer ids, not a bounding range; holes belong to other backends.
    pub layers: &'a [usize],
}

#[derive(Debug, PartialEq, Eq, Error)]
pub(crate) enum BindingError {
    #[error("attention mode {mode:?} cannot bind to {ranks} ranks")]
    Attention { mode: cuteafd_loader::placement::LayerMode, ranks: usize },
    #[error("expert binding needs matching nonempty layer and dense inventories")]
    Geometry,
    #[error("backend {home:?} binds layer {layer} outside the routed inventory")]
    Layer { layer: usize, home: ExpertHome },
    #[error("layer {layer} has multiple expert backend bindings")]
    Duplicate { layer: usize },
    #[error("layer {layer} has no installed expert backend (planned {expected:?})")]
    Missing { layer: usize, expected: ExpertHome },
    #[error("layer {layer} installed {actual:?}, but placement requires {expected:?}")]
    Home { layer: usize, actual: ExpertHome, expected: ExpertHome },
}

#[derive(Debug, Clone)]
pub(crate) struct ExpertRouting {
    routes: Vec<ExpertRoute>,
}

impl ExpertRouting {
    /// Called after loading and worker selection, before graph capture or enqueue.
    pub fn bind(layers: &[LayerAssignment], dense: &[bool], installed: &[BackendLayers<'_>])
        -> Result<Self, BindingError> {
        if layers.is_empty() || layers.len() != dense.len() { return Err(BindingError::Geometry); }
        let mut homes = vec![None; layers.len()];
        for backend in installed {
            for &layer in backend.layers {
                if dense.get(layer) != Some(&false) { return Err(BindingError::Layer { layer, home: backend.home }); }
                if homes[layer].replace(backend.home).is_some() { return Err(BindingError::Duplicate { layer }); }
            }
        }
        let routes = layers.iter().zip(dense).enumerate().map(|(layer, (assignment, &dense))| {
            if dense { return Ok(ExpertRoute::Dense); }
            let expected = assignment.experts;
            let actual = homes[layer].ok_or(BindingError::Missing { layer, expected })?;
            if actual != expected { return Err(BindingError::Home { layer, actual, expected }); }
            Ok(ExpertRoute::Backend(actual))
        }).collect::<Result<_, _>>()?;
        Ok(Self { routes })
    }

    /// Profiling-only skip is explicit, never a substitute for an absent backend.
    pub fn skipped(dense: &[bool]) -> Self {
        Self { routes: dense.iter().map(|&dense| if dense { ExpertRoute::Dense } else { ExpertRoute::Skipped }).collect() }
    }

    pub fn route(&self, layer: usize) -> Option<ExpertRoute> { self.routes.get(layer).copied() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::placement::LayerMode;

    #[test]
    fn whole_attention_residency_is_independent_of_split_ffns() {
        use cuteafd_loader::placement::FfnMode;
        for ffn in [FfnMode::Split, FfnMode::Owner] {
            for gpu in 0..2 {
                let shares = AttentionShare::bind(LayerMode::Whole { gpu, ffn }, 2).unwrap();
                assert_eq!(shares[usize::from(gpu)], AttentionShare::Whole);
                assert_eq!(shares[usize::from(1 - gpu)], AttentionShare::Absent);
                assert!(shares.iter().all(|share| !share.split()));
            }
        }
        for mode in [LayerMode::HeadSplit, LayerMode::ContextSplit] {
            assert_eq!(AttentionShare::bind(mode, 2).unwrap(), vec![AttentionShare::Split; 2]);
            assert!(AttentionShare::bind(mode, 1).is_err());
        }
        for ranks in [0, 3] {
            assert!(AttentionShare::bind(LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }, ranks).is_err());
        }
        assert!(AttentionShare::bind(LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }, 1).is_err());
        assert!(AttentionShare::bind(LayerMode::Whole { gpu: 2, ffn: FfnMode::Split }, 2).is_err());
    }

    fn assignments(homes: &[ExpertHome]) -> Vec<LayerAssignment> {
        homes.iter().map(|&experts| LayerAssignment { mode: LayerMode::HeadSplit, experts }).collect()
    }

    #[test]
    fn noncontiguous_local_and_spark_layers_bind_without_a_global_default() {
        use ExpertHome::{RtxTp2, Spark};
        let layers = assignments(&[Spark, RtxTp2, Spark, RtxTp2, Spark]);
        let routes = ExpertRouting::bind(&layers, &[true, false, false, false, false], &[
            BackendLayers { home: RtxTp2, layers: &[1, 3] },
            BackendLayers { home: Spark, layers: &[2, 4] },
        ]).unwrap();
        assert_eq!(routes.route(0), Some(ExpertRoute::Dense));
        for layer in 1..5 { assert_eq!(routes.route(layer), Some(ExpertRoute::Backend(layers[layer].experts))); }
        assert_eq!(routes.route(5), None);
    }

    #[test]
    fn installed_spark_cannot_cover_a_missing_local_layer() {
        use ExpertHome::{RtxTp2, Spark};
        let layers = assignments(&[RtxTp2, Spark]);
        assert_eq!(ExpertRouting::bind(&layers, &[false; 2], &[
            BackendLayers { home: Spark, layers: &[1] },
        ]).unwrap_err(), BindingError::Missing { layer: 0, expected: RtxTp2 });
        assert_eq!(ExpertRouting::bind(&layers, &[false; 2], &[
            BackendLayers { home: Spark, layers: &[0, 1] },
        ]).unwrap_err(), BindingError::Home { layer: 0, actual: Spark, expected: RtxTp2 });
    }

    #[test]
    fn rejects_duplicates_dense_backends_and_out_of_bounds_layers() {
        use ExpertHome::Spark;
        let layers = assignments(&[Spark; 2]);
        for (dense, ids, expected) in [
            ([false; 2], vec![0, 0], BindingError::Duplicate { layer: 0 }),
            ([true, false], vec![0], BindingError::Layer { layer: 0, home: Spark }),
            ([false; 2], vec![2], BindingError::Layer { layer: 2, home: Spark }),
        ] {
            assert_eq!(ExpertRouting::bind(&layers, &dense, &[BackendLayers { home: Spark, layers: &ids }])
                .unwrap_err(), expected);
        }
    }

    #[test]
    fn whole_local_owner_is_part_of_the_binding() {
        let layers = assignments(&[ExpertHome::RtxWhole { gpu: 1 }]);
        assert!(matches!(ExpertRouting::bind(&layers, &[false], &[
            BackendLayers { home: ExpertHome::RtxWhole { gpu: 0 }, layers: &[0] },
        ]), Err(BindingError::Home { .. })));
    }

    #[test]
    fn skip_and_invalid_geometry_are_not_normal_bindings() {
        assert_eq!(ExpertRouting::bind(&[], &[], &[]).unwrap_err(), BindingError::Geometry);
        assert_eq!(ExpertRouting::bind(&assignments(&[ExpertHome::Spark]), &[], &[]).unwrap_err(), BindingError::Geometry);
        let routes = ExpertRouting::skipped(&[true, false]);
        assert_eq!(routes.route(0), Some(ExpertRoute::Dense));
        assert_eq!(routes.route(1), Some(ExpertRoute::Skipped));
    }
}
