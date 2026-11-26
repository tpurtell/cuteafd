//! Single-rank TP1 execution engine for Inkling 2 Mini (0 Sparks, 48 visits, exact-row graphs).
use super::{
    cache::Placement,
    experts::Dispatcher,
    graphs::{DecodeGraphs, Mode},
    head::MiniHead,
    layers::MiniKernels,
    owners::RequestOwners,
    prefill::{HeadProjection, Stateful, TokenIo},
    weights::MiniTargetWeights,
};
use crate::shared::{decode_graph::device::Cycle, prefill_pipeline::Unit, token_io::DeviceLogits};
use anyhow::{ensure, Result};
use cuteafd_loader::families::inkling2_mini::{Inkling2MiniConfig, LoopVisit};

pub(crate) struct Execution<'a> {
    pub(super) graphs: Option<DecodeGraphs<'a>>,
    pub requests: RequestOwners<'a>,
    pub head: MiniHead<'a>,
    pub target: MiniTargetWeights<'a>,
    pub kernels: MiniKernels<'a>,
    pub local: Vec<Box<dyn Dispatcher + 'a>>,
    pub cfg: Inkling2MiniConfig,
    pub visits: Vec<LoopVisit>,
    pub draft: Option<std::rc::Rc<super::draft::Binding<'a>>>,
    terminal: bool,
}

impl<'a> Execution<'a> {
    pub fn new(
        requests: RequestOwners<'a>,
        head: MiniHead<'a>,
        target: MiniTargetWeights<'a>,
        kernels: MiniKernels<'a>,
        local: Vec<Box<dyn Dispatcher + 'a>>,
        cfg: Inkling2MiniConfig,
    ) -> Result<Self> {
        ensure!(
            target.layers.iter().copied().eq(0..cfg.text.layers)
                && !requests.prefill.is_empty()
                && local.len() == cfg.text.layers,
            "Inkling 2 Mini TP1 execution differs from admitted owners"
        );
        let visits = cfg.visits();
        Ok(Self {
            graphs: None,
            requests,
            head,
            target,
            kernels,
            local,
            cfg,
            visits,
            draft: None,
            terminal: false,
        })
    }

    pub fn require_live(&self) -> Result<()> {
        ensure!(!self.terminal, "Inkling 2 Mini execution is terminal");
        Ok(())
    }

    pub fn restore_verify(&self, mark: &super::verify::TargetMark<'_>) -> Result<()> {
        let draft = self
            .draft
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Inkling 2 Mini drafter missing"))?;
        self.requests
            .decode
            .stream
            .device
            .run(|| draft.verify.restore(&mark.conv, self.requests.cache.conv_backing()))?;
        Ok(())
    }

    fn completed(&self) -> Result<()> {
        self.requests.decode.stream.drain()
    }

    pub fn prefill(
        &mut self,
        placements: &mut [Placement],
        tokens: &[&[u32]],
        units: &[Unit],
        head_projection: HeadProjection,
        on_logits: &mut dyn FnMut(usize, usize, DeviceLogits) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !self.terminal && placements.len() == tokens.len(),
            "Inkling 2 Mini execution terminal or request count differs"
        );
        let slots: Vec<usize> = placements.iter().map(|p| p.slot).collect();
        let starts: Vec<usize> = placements.iter().map(|p| p.len).collect();
        let media: Vec<_> = placements.iter().map(|p| p.media.clone()).collect();
        let result = (|| {
            let mut io = TokenIo {
                kernels: &self.kernels,
                head: &self.head,
                cache: &self.requests.cache,
                target: &self.target,
                cfg: &self.cfg,
                tokens,
                starts: &starts,
                slots: &slots,
                retained_rows: false,
                head_projection,
                draft: self.draft.as_deref(),
                publish_draft: true,
                media: &media,
                on_logits,
            };
            Stateful {
                graphs: None,
                kernels: &self.kernels,
                owners: &self.requests.prefill,
                cache: &self.requests.cache,
                target: &self.target,
                cfg: &self.cfg,
                visits: &self.visits,
                placements,
                uploads: &mut self.requests.prefill_uploads,
                local: &self.local,
                io: &mut io,
            }
            .run(units)?;
            self.completed()
        })();
        if result.is_err() {
            self.terminate();
        }
        result
    }

    pub fn decode(
        &mut self,
        placements: &mut [Placement],
        tokens: &[&[u32]],
        unit: &Unit,
    ) -> Result<DeviceLogits> {
        self.decode_inner(placements, tokens, unit, false, true)
    }

    pub fn decode_eager(
        &mut self,
        placements: &mut [Placement],
        tokens: &[&[u32]],
        unit: &Unit,
    ) -> Result<DeviceLogits> {
        self.decode_inner(placements, tokens, unit, false, false)
    }

    pub fn verify(
        &mut self,
        placements: &mut [Placement],
        tokens: &[&[u32]],
        unit: &Unit,
    ) -> Result<DeviceLogits> {
        self.decode_inner(placements, tokens, unit, true, true)
    }

    fn decode_inner(
        &mut self,
        placements: &mut [Placement],
        tokens: &[&[u32]],
        unit: &Unit,
        staged: bool,
        use_graphs: bool,
    ) -> Result<DeviceLogits> {
        self.require_live()?;
        ensure!(
            !placements.is_empty()
                && placements.len() == tokens.len()
                && unit.rows.len() <= self.requests.decode.layout.rows
                && unit.slot == 0,
            "Inkling 2 Mini decode exceeds admitted head/owner"
        );
        let mode = if self.requests.decode.replaying() {
            Mode::Verify
        } else {
            Mode::Decode
        };
        let cycle = if mode == Mode::Verify {
            Cycle::Verify
        } else {
            Cycle::Decode
        };
        if use_graphs {
            if let Some(graphs) = &mut self.graphs {
                graphs.cycle(cycle);
            }
        }
        let slots: Vec<usize> = placements.iter().map(|p| p.slot).collect();
        let starts: Vec<usize> = placements.iter().map(|p| p.len).collect();
        let media: Vec<_> = placements.iter().map(|p| p.media.clone()).collect();
        let result = (|| {
            let mut io = TokenIo {
                kernels: &self.kernels,
                head: &self.head,
                cache: &self.requests.cache,
                target: &self.target,
                cfg: &self.cfg,
                tokens,
                starts: &starts,
                slots: &slots,
                retained_rows: true,
                head_projection: HeadProjection::AllHeadRows,
                draft: self.draft.as_deref(),
                publish_draft: false,
                media: &media,
                on_logits: &mut |_, _, _| Ok(()),
            };
            Stateful {
                graphs: if use_graphs { self.graphs.as_mut() } else { None },
                kernels: &self.kernels,
                owners: std::slice::from_ref(&self.requests.decode),
                cache: &self.requests.cache,
                target: &self.target,
                cfg: &self.cfg,
                visits: &self.visits,
                placements,
                uploads: std::slice::from_mut(&mut self.requests.decode_uploads),
                local: &self.local,
                io: &mut io,
            }
            .run(std::slice::from_ref(unit))?;
            let observe = use_graphs && self.graphs.as_ref().is_some_and(DecodeGraphs::needs_boundary);
            if !staged || observe {
                let wait = std::time::Instant::now();
                self.completed()?;
                if staged && observe { crate::shared::decode_graph::device::record_boundary_drain([self.requests.decode.stream.device.id], wait.elapsed()); }
            }
            if observe {
                self.observe_completed()?;
            } else if use_graphs {
                if let Some(graphs) = &mut self.graphs {
                    graphs.tick();
                }
            }
            Ok(DeviceLogits {
                ptr: self.head.logits.buffer.ptr,
                rows: unit.rows.len(),
                vocab: self.cfg.text.unpadded_vocab_size,
                stride: self.cfg.text.unpadded_vocab_size,
                stream: self.requests.decode.stream.raw,
                greedy: None,
            })
        })();
        if result.is_err() {
            self.terminate();
        }
        result
    }

    fn observe_completed(&mut self) -> Result<()> {
        let Some(graphs) = &mut self.graphs else {
            return Ok(());
        };
        let library = self.requests.decode.stream.device.library;
        // SAFETY: decode stream has drained; captured storage is persistent.
        unsafe {
            crate::shared::decode_graph::device::observe_completed(library, |visit| graphs.visit(visit))?;
        }
        graphs.observed();
        Ok(())
    }

    /// Pre-captures the shipping exact-row decode/verify graph set at startup so
    /// C1..C16 serving runs capture-free and with zero eager fallbacks.
    pub(super) fn prewarm_graphs(&mut self, max_sequences: usize) -> Result<(usize, usize)> {
        if self.graphs.is_none() {
            return Ok((0, 0));
        }
        let Some(_startup) = crate::shared::decode_graph::sites::startup(false) else {
            return Ok((0, 0));
        };
        let has_draft = self.draft.is_some();
        let decode_rows = self.requests.decode.layout.rows;
        let keys = super::graphs::shipping_prewarm_keys(max_sequences, decode_rows, has_draft);
        if keys.is_empty() {
            return Ok((0, 0));
        }
        let max_decode = cuteafd_loader::serving_capacity::inkling2_mini_graphs::KeySpace::for_config(
            max_sequences as u64,
            decode_rows,
            has_draft,
        )
        .decode_keys;
        let warm_tokens = vec![0u32; decode_rows];

        let run_one = |exec: &mut Self, rows: usize, mode: Mode, capture: bool| -> Result<bool> {
            let replay = if mode == Mode::Verify {
                let draft = exec
                    .draft
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Inkling 2 Mini verify prewarm requires drafter"))?;
                Some((draft.verify.replay.buffer, exec.requests.cache.layout.conv.width))
            } else {
                None
            };
            exec.requests.decode.set_replay(replay)?;
            let res = (|| -> Result<bool> {
                if capture {
                    if let Some(graphs) = &mut exec.graphs {
                        graphs.cycle(if mode == Mode::Verify {
                            Cycle::Verify
                        } else {
                            Cycle::Decode
                        });
                    }
                }
                let mut placements = [Placement {
                    units: (0..rows.div_ceil(64) as u32).collect(),
                    slot: 0,
                    len: 0,
                    state_len: 0,
                    media: None,
                }];
                let unit = Unit {
                    index: 0,
                    sequence: 0,
                    rows: 0..rows,
                    layers: 0..exec.cfg.total_visits(),
                    slot: 0,
                    barrier: false,
                    segments: vec![crate::shared::prefill_pipeline::UnitSegment {
                        sequence: 0,
                        rows: 0..rows,
                        tokens: 0..rows,
                        prompt_end: false,
                        snapshot: false,
                        head: false,
                    }],
                };
                let token_slices: [&[u32]; 1] = [&warm_tokens[..rows]];
                let starts = [0usize];
                let slots = [0usize];
                let media = [None];
                let mut io = TokenIo {
                    kernels: &exec.kernels,
                    head: &exec.head,
                    cache: &exec.requests.cache,
                    target: &exec.target,
                    cfg: &exec.cfg,
                    tokens: &token_slices,
                    starts: &starts,
                    slots: &slots,
                    retained_rows: true,
                    head_projection: HeadProjection::AllHeadRows,
                    draft: exec.draft.as_deref(),
                    publish_draft: false,
                    media: &media,
                    on_logits: &mut |_, _, _| Ok(()),
                };
                Stateful {
                    graphs: if capture { exec.graphs.as_mut() } else { None },
                    kernels: &exec.kernels,
                    owners: std::slice::from_ref(&exec.requests.decode),
                    cache: &exec.requests.cache,
                    target: &exec.target,
                    cfg: &exec.cfg,
                    visits: &exec.visits,
                    placements: &mut placements,
                    uploads: std::slice::from_mut(&mut exec.requests.decode_uploads),
                    local: &exec.local,
                    io: &mut io,
                }
                .prewarm_unit(&unit, capture)
            })();
            let _ = exec.requests.decode.set_replay(None);
            res
        };

        // 1. Eager warm pass so one-time kernel/workspace initialization lands before
        //    measuring graph arena growth.
        run_one(self, max_decode, Mode::Decode, false)?;
        if has_draft {
            run_one(self, decode_rows, Mode::Verify, false)?;
        }
        self.completed()?;
        let device = self.requests.decode.stream.device;
        crate::shared::decode_graph::device::observe_device(device.library, device.id)?;

        // 2. Capture-only pass for every shipping key.
        let mut captured = 0usize;
        for &(rows, mode) in &keys {
            if run_one(self, rows, mode, true)? {
                captured += 1;
            }
        }
        self.completed()?;
        self.observe_completed()?;
        self.requests.decode.zero()?;
        self.completed()?;
        Ok((captured, keys.len()))
    }

    pub fn terminate(&mut self) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        let drained = (|| -> Result<()> {
            self.requests.decode.stream.drain()?;
            for owner in &self.requests.prefill {
                owner.stream.drain()?;
            }
            self.requests.cache.drain()
        })();
        crate::shared::decode_graph::fatal_drain(drained, "Inkling 2 Mini executor retirement");
        if let Some(graphs) = &mut self.graphs {
            graphs.clear_drained();
        }
    }
}

impl Drop for Execution<'_> {
    fn drop(&mut self) {
        self.terminate();
    }
}
