//! Coordinator (RTX) ranks of a push group: the two coordinator GPUs join the
//! Spark group as ranks `sparks` and `sparks + 1`, each with its own ring and
//! admitted device arena. Plans reuse the shared builder; delivery reuses the
//! shared consumer and post-Done initialization. Nothing publishes before the
//! whole group commits; any failure drops every unpublished owner and the
//! whole group (Sparks included) reloads through the mount.
use super::{weight_arena::ArenaLayout, weight_validation::ValidationLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{push_load::builder::{Builder, CheckedPlan, Destination}, OfficialV41Catalog, V41Exl3Layer,
    V41Exl3Residency};

/// One coordinator GPU's admitted TP2 expert halves: the coordinator's legacy
/// (non-balanced) residency, laid out as today's mount arenas, one region per layer.
pub(crate) struct Tp2ExpertPlan {
    pub residencies: Vec<V41Exl3Residency>,
    pub layout: ArenaLayout,
    pub validation: ValidationLayout,
}

impl Tp2ExpertPlan {
    pub fn new(catalog: &OfficialV41Catalog, layers: &[usize], gpu: usize, admitted: usize) -> Result<Self> {
        ensure!(gpu < 2, "coordinator TP2 has two GPUs");
        let manifest = catalog.exl3().context("coordinator TP2 push requires EXL3 checkpoint headers")?;
        let residencies = layers.iter().map(|&layer| manifest.residency(V41Exl3Layer::Backbone(layer), 2, gpu))
            .collect::<Result<Vec<_>>>()?;
        ensure!(residencies.iter().all(|r| r.unit_map.is_none()), "coordinator TP2 keeps the legacy residency");
        let sizes = residencies.iter().map(|r| r.device_arena_layout().map(|(_, bytes)| bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let layout = ArenaLayout::new(sizes, admitted)?;
        let validation = ValidationLayout::exl3(&residencies, layout.bytes() as u64, usize::MAX)?;
        let layout = layout.with_validation(validation.bytes(), validation.bytes())?;
        Ok(Self { residencies, layout, validation })
    }

    /// Entries for group rank `rank` (the coordinator GPU's rank in the group).
    pub fn append(&self, builder: &mut Builder, catalog: &OfficialV41Catalog, rank: u32) -> Result<()> {
        for (index, residency) in self.residencies.iter().enumerate() {
            let range = self.layout.region(index)?;
            let (offsets, _) = residency.device_arena_layout()?;
            builder.exl3_residency(catalog, residency, &offsets, Destination {
                rank, offset: range.start as u64, bytes: range.len() as u64 })?;
        }
        self.validation.append(builder, catalog, rank)
    }

    /// This rank's checked entries alone, against a group of `world` ranks.
    pub fn checked(&self, catalog: &OfficialV41Catalog, root: &std::path::Path, world: usize, rank: usize)
        -> Result<CheckedPlan> {
        let mut arenas = vec![0; world];
        *arenas.get_mut(rank).context("coordinator rank outside group")? = self.layout.logical_bytes() as u64;
        let mut builder = Builder::new(root, arenas);
        self.append(&mut builder, catalog, u32::try_from(rank)?)?;
        Ok(builder.finish()?)
    }
}

