//! Retained peer/context owners, composed separately from target rows and hooks.
//! These fixed facts do not establish startup history or authorize Ready.
use super::{GlmContext, INDEX_PAGE_BYTES, PAGE_ROWS, RECORD_PAGE_BYTES};
use crate::families::glm5::GlmDsaConfig;
use crate::placement::planner::{
    Allocation, ByteCurve, Component, CoordinatorDevices, Lifetime, Location, PrefillShape, Refusal,
};
use cuteafd_core::memory_layout::Category;

const FLOOR: u64 = 256;

#[derive(Debug, Clone)]
pub struct ExchangeBuffers {
    /// Rank identity is independent of the physical device that owns its storage.
    pub physical_devices: [i32; 2],
    pub maximum_pages: u64,
    pub receive_bytes: Vec<u64>,
    pub terminal_abort: bool,
    pub diagnostics: bool,
    pub context: Option<ContextOwners>,
}

#[derive(Debug, Clone, Copy)]
pub struct ContextOwners {
    pub extent: u64,
    pub prefill: PrefillShape,
    pub scratch: GlmContext,
}

fn mul(a: u64, b: u64) -> Result<u64, Refusal> {
    a.checked_mul(b).ok_or(Refusal::Overflow("GLM exchanges"))
}

impl ExchangeBuffers {
    pub fn components(&self, cfg: &GlmDsaConfig) -> Result<Vec<Component>, Refusal> {
        let devices = CoordinatorDevices::new(&self.physical_devices)?;
        if self.maximum_pages == 0
            || self.receive_bytes.is_empty()
            || self
                .receive_bytes
                .iter()
                .any(|&bytes| bytes == 0 || bytes % 16 != 0)
        {
            return Err(Refusal::InvalidFact("GLM peer exchange geometry".into()));
        }
        let flags = (self.receive_bytes.len() as u64)
            .checked_add(1)
            .ok_or(Refusal::Overflow("GLM peer flags"))?;
        let control = mul(flags, 16)?.max(FLOOR);
        let progress = mul(
            flags,
            std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>() as u64,
        )?
        .max(FLOOR);
        let mut out = Vec::new();
        for &device in &self.physical_devices {
            let rank = devices.logical_rank(device)?;
            let prefix = format!("glm.exchange.rank.{rank}");
            let mut add = |name: &str, location: Location, bytes: u64, category: Category| {
                out.push(Component::allocation(
                    Allocation::buffer(
                        format!("{prefix}.{name}"),
                        location,
                        bytes,
                        Lifetime::TARGET,
                        category,
                        "GLM exchanges",
                    ),
                    ByteCurve::fixed(bytes, self.maximum_pages),
                    false,
                    self.maximum_pages,
                ));
            };
            let location = devices.location(device)?;
            for (slot, &bytes) in self.receive_bytes.iter().enumerate() {
                add(
                    &format!("peer.recv.{slot}"),
                    location,
                    bytes.max(FLOOR),
                    Category::Transport,
                );
            }
            add("peer.control", location, control, Category::Transport);
            if self.diagnostics {
                add("peer.progress", location, progress, Category::Transport);
            }
            if self.terminal_abort {
                add("peer.abort", location, FLOOR, Category::Transport);
            }
            if let Some(context) = self.context {
                let shape = context.prefill;
                if context.extent == 0
                    || cfg.heads == 0
                    || cfg.heads % 8 != 0
                    || cfg.index_topk == 0
                    || cfg.index_topk % 2 != 0
                    || !(64..=4096).contains(&shape.rows)
                    || shape.rows % PAGE_ROWS != 0
                    || !(1..=4).contains(&shape.waves)
                {
                    return Err(Refusal::InvalidFact("GLM context owner geometry".into()));
                }
                // Check every product before using the executor's shared geometry helpers.
                mul(cfg.index_topk as u64, super::DECODE_ROWS * 8)?;
                mul(cfg.heads as u64 / 2, super::DECODE_ROWS * (512 * 2 + 4))?;
                mul(cfg.heads as u64 / 2, super::DECODE_ROWS * 576 * 2)?;
                let pages = context.extent.div_ceil(PAGE_ROWS);
                let half = pages.div_ceil(2);
                mul(half, 8)?;
                let staging = mul(pages, RECORD_PAGE_BYTES + INDEX_PAGE_BYTES)?;
                let workspace = super::context_workspace(
                    cfg,
                    context.extent,
                    shape.rows,
                    shape.waves,
                    context.scratch,
                );
                workspace.iter().try_fold(FLOOR, |sum, &(_, bytes)| {
                    sum.checked_add(bytes.max(FLOOR))
                        .ok_or(Refusal::Overflow("GLM context workspace"))
                })?;
                let buffers = super::context_buffers(
                    cfg,
                    context.extent,
                    shape.rows,
                    shape.waves,
                    context.scratch,
                );
                let exchange = crate::placement::ContextExchangeLayout::new(&buffers)
                    .bytes()
                    .ok_or(Refusal::Overflow("GLM context exchange"))?;
                add("context.exchange", location, exchange, Category::Transport);
                for ((_, bytes), name) in workspace.iter().take(6).zip([
                    "candidates",
                    "own",
                    "peer",
                    "topk_scratch",
                    "partial_scratch",
                    "gather",
                ]) {
                    add(
                        &format!("context.{name}"),
                        location,
                        (*bytes).max(FLOOR),
                        Category::Workspace,
                    );
                }
                for (lane, &(_, bytes)) in workspace.iter().skip(6).enumerate() {
                    let bytes = bytes.max(FLOOR);
                    add(
                        &format!("context.commit.{lane}"),
                        location,
                        bytes,
                        Category::Workspace,
                    );
                    add(
                        &format!("context.upload.commit.{lane}"),
                        Location::PINNED,
                        bytes,
                        Category::Staging,
                    );
                }
                for parity in 0..2 {
                    add(
                        &format!("context.staging.{parity}"),
                        location,
                        staging.max(FLOOR),
                        Category::Workspace,
                    );
                    // Each gather destination is a region, not the floored whole device owner.
                    add(
                        &format!("context.upload.gather.{parity}"),
                        Location::PINNED,
                        mul(half, 4)?,
                        Category::Staging,
                    );
                }
                add("context.sink", location, FLOOR, Category::Workspace);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::planner::{Contract, Facts, Variable};

    fn fixture() -> (GlmDsaConfig, ExchangeBuffers) {
        (
            GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap(),
            ExchangeBuffers {
                physical_devices: [1, 0],
                maximum_pages: 128,
                receive_bytes: vec![16, 512],
                terminal_abort: false,
                diagnostics: false,
                context: None,
            },
        )
    }

    fn allocations(cfg: &GlmDsaConfig, buffers: &ExchangeBuffers) -> Vec<Allocation> {
        let mut contract = Contract::fixed(
            64,
            buffers.maximum_pages,
            2,
            Facts { driver_page_tables: vec![], resident_weights: vec![],
                phase_order: crate::placement::planner::PhaseOrder::DRAFT_THEN_TARGET,
                allocations: vec![],
                deferred: vec![],
                sequential: vec![],
                reservations: vec![], allowances: vec![],
            },
        );
        contract.variables = vec![
            Variable::Prefill(vec![buffers
                .context
                .map_or(PrefillShape { rows: 64, waves: 1 }, |c| c.prefill)]),
            Variable::Lanes(vec![1]),
        ];
        contract.components = buffers.components(cfg).unwrap();
        contract.facts(&[0, 0], 1).unwrap().allocations
    }

    #[test]
    fn peer_owners_keep_rank_identity_floors_and_optional_allocations() {
        let (cfg, mut buffers) = fixture();
        for abort in [false, true] {
            for diagnostics in [false, true] {
                buffers.terminal_abort = abort;
                buffers.diagnostics = diagnostics;
                let owners = allocations(&cfg, &buffers);
                for rank in 0..2 {
                    let prefix = format!("glm.exchange.rank.{rank}.peer.");
                    let rank_owners: Vec<_> = owners
                        .iter()
                        .filter(|a| a.id.starts_with(&prefix))
                        .collect();
                    assert_eq!(
                        rank_owners.len(),
                        3 + usize::from(abort) + usize::from(diagnostics)
                    );
                    assert!(rank_owners
                        .iter()
                        .all(|a| a.location == Location::coordinator(rank as u8)));
                    for (name, bytes) in [("recv.0", 256), ("recv.1", 512), ("control", 256)] {
                        assert_eq!(
                            rank_owners
                                .iter()
                                .find(|a| a.id == format!("{prefix}{name}"))
                                .unwrap()
                                .bytes,
                            bytes
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn context_owners_match_workspace_staging_and_exact_pinned_regions() {
        let (cfg, mut buffers) = fixture();
        for extent in [1, 65, 4096] {
            for prefill in [
                PrefillShape { rows: 64, waves: 1 },
                PrefillShape {
                    rows: 128,
                    waves: 3,
                },
            ] {
                let scratch = GlmContext {
                    topk_scratch: 17,
                    partial_scratch: 1024,
                };
                buffers.context = Some(ContextOwners {
                    extent,
                    prefill,
                    scratch,
                });
                let owners = allocations(&cfg, &buffers);
                let geometry = super::super::context_buffers(
                    &cfg,
                    extent,
                    prefill.rows,
                    prefill.waves,
                    scratch,
                );
                let device_bytes: u64 = geometry
                    .demands()
                    .unwrap()
                    .iter()
                    .filter(|d| d.gpu == 0)
                    .map(|d| d.bytes)
                    .sum();
                for rank in 0..2 {
                    let prefix = format!("glm.exchange.rank.{rank}.context.");
                    let rank_owners: Vec<_> = owners
                        .iter()
                        .filter(|a| a.id.starts_with(&prefix))
                        .collect();
                    assert_eq!(
                        rank_owners
                            .iter()
                            .filter(|a| a.location != Location::PINNED)
                            .map(|a| a.bytes)
                            .sum::<u64>(),
                        device_bytes + FLOOR
                    );
                    let commit =
                        (3 * 3 * super::super::context_commit_pages(prefill.rows) * 8).max(FLOOR);
                    assert_eq!(
                        rank_owners
                            .iter()
                            .filter(|a| a.location == Location::PINNED)
                            .map(|a| a.bytes)
                            .sum::<u64>(),
                        2 * extent.div_ceil(PAGE_ROWS).div_ceil(2) * 4 + prefill.waves * commit
                    );
                    assert_eq!(
                        rank_owners
                            .iter()
                            .find(|a| a.id.ends_with(".sink"))
                            .unwrap()
                            .bytes,
                        FLOOR
                    );
                }
            }
        }
    }

    #[test]
    fn malformed_exchange_geometry_refuses_before_publication() {
        let (cfg, buffers) = fixture();
        for receive in [vec![], vec![0], vec![17]] {
            assert!(ExchangeBuffers {
                receive_bytes: receive,
                ..buffers.clone()
            }
            .components(&cfg)
            .is_err());
        }
        for physical_devices in [[0, 0], [0, -1]] {
            assert!(ExchangeBuffers {
                physical_devices,
                ..buffers.clone()
            }
            .components(&cfg)
            .is_err());
        }
        assert!(ExchangeBuffers {
            maximum_pages: 0,
            ..buffers.clone()
        }
        .components(&cfg)
        .is_err());
        let context = ContextOwners {
            extent: 4096,
            prefill: PrefillShape {
                rows: 128,
                waves: 3,
            },
            scratch: GlmContext::default(),
        };
        let mut context_buffers = ExchangeBuffers {
            context: Some(context),
            ..buffers
        };
        let mut broken = cfg.clone();
        broken.index_topk = usize::MAX - 1;
        assert!(matches!(
            context_buffers.components(&broken),
            Err(Refusal::Overflow(_))
        ));
        context_buffers.context.as_mut().unwrap().extent = 0;
        assert!(context_buffers.components(&cfg).is_err());
    }
}
