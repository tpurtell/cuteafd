//! Physical target buffers, not a placement policy or a complete family admission.
//! Native dense/K6 hook scratch, experts, exchanges, graphs and serving owners
//! join these components separately before the family can authorize Ready.
use super::{GlmScratch, DECODE_ROWS, INDEX_PAGE_BYTES, PAGE_ROWS};
use crate::families::glm5::{GlmDsaConfig, GlmIndexer};
use crate::placement::planner::{
    Allocation, AllocationGranularity, ByteCurve, BytePiece, Component, GranularityBacking,
    GranularityEvidence, Lifetime, Location, Phase, PrefillShape, Refusal, SM120_ALLOCATION_GRANULARITY_BYTES,
    View, allocation_granularity_component, first_independent_pages,
};
use cuteafd_core::memory_layout::Category;

/// Smallest allocation a latent slot or the pools ever request.
pub(crate) const FLOOR: u64 = 256;

/// One reusable synchronous upload backing after load staging is released.
/// Warmup presizes it for decode lengths and the largest admitted prefill arena.
/// Serialized ranks and slots reuse this store; they do not own separate copies.
pub fn sync_h2d_successor(prefill: PrefillShape) -> Result<Allocation, Refusal> {
    if prefill.rows == 0 || prefill.waves == 0 {
        return Err(Refusal::InvalidFact("empty GLM workspace upload shape".into()));
    }
    let rows = mul(prefill.rows, prefill.waves)?.max(DECODE_ROWS);
    let bytes = mul(rows, 4)?;
    usize::try_from(bytes).map_err(|_| Refusal::Overflow("GLM workspace upload"))?;
    Ok(Allocation::buffer(
        "glm.upload.sync-successor", Location::PINNED, bytes,
        Lifetime { first: Phase::Warmup, last: Phase::Growth },
        Category::Staging, "GLM warmup sync-H2D",
    ))
}

/// Prefix-owned storage only. KV pools and the drafter's live windows are
/// aliases owned by their respective target/draft recipes, not new backings.
#[derive(Debug, Clone, Copy)]
pub struct PrefixBuffers {
    pub device: u8,
    /// One slot's window payload from the drafter layout; None for GQA/no draft.
    pub draft_windows: Option<u64>,
    pub mark_slots: u64,
}

impl PrefixBuffers {
    pub fn allocations(self, lifetime: Lifetime) -> Result<Vec<Allocation>, Refusal> {
        if self.device > 1 {
            return Err(Refusal::InvalidFact("invalid GLM prefix device".into()));
        }
        // NativePrefix owns one scratch and optional mark arena on its lead,
        // not one copy per attention rank. Match its receipt identities.
        let mut owners = vec![Allocation::buffer("glm.prefix.scratch",
            Location::coordinator(self.device), FLOOR, lifetime, Category::Prefix, "GLM prefix storage")];
        if let Some(windows) = self.draft_windows.filter(|_| self.mark_slots > 0) {
            let mark = if windows == 0 { 0 } else {
                windows.checked_add(super::DRAFT_MARK_HEADER as u64)
                    .ok_or(Refusal::Overflow("GLM prefix mark header"))?
            };
            let bytes = mul(mark, self.mark_slots)?.max(FLOOR);
            usize::try_from(bytes).map_err(|_| Refusal::Overflow("GLM prefix marks"))?;
            owners.push(Allocation::buffer("glm.prefix.draft_marks",
                Location::coordinator(self.device), bytes, lifetime, Category::Prefix, "GLM prefix storage"));
        }
        Ok(owners)
    }

    pub fn components(self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 {
            return Err(Refusal::InvalidFact("GLM prefix requires a positive pool domain".into()));
        }
        Ok(self.allocations(Lifetime { first: Phase::Warmup, last: Phase::Growth })?
            .into_iter().map(|owner| {
                let bytes = ByteCurve::fixed(owner.bytes, maximum_pages);
                Component::allocation(owner, bytes, false, maximum_pages)
            }).collect())
    }
}

#[derive(Debug, Clone)]
pub struct TargetBuffers {
    pub device: u8,
    pub lead: bool,
    /// Solver page quantum; context partitioning requires an even number of KV pages.
    pub page_tokens: u64,
    pub maximum_pages: u64,
    pub context: u64,
    pub partitioned: bool,
    pub owned_layers: Vec<bool>,
    pub split_heads: bool,
    pub router: bool,
    pub native_head: bool,
    pub decode: Option<GlmScratch>,
    pub prefill: Option<(PrefillShape, GlmScratch)>,
    pub full_prefill_logits: bool,
    /// Optional TP2 payload and K6 row arenas; their hook-local owners are separate.
    pub payload_row_bytes: u64,
    pub core_rows: bool,
    pub attention_partial_bytes: u64,
}

fn mul(a: u64, b: u64) -> Result<u64, Refusal> {
    a.checked_mul(b)
        .ok_or(Refusal::Overflow("GLM target buffers"))
}

/// One pool allocation: its resident per-solver-page stride plus the single
/// scratch page decode rows write. Requests above the driver's shared-block
/// boundary round to whole cudaMalloc blocks independently; smaller ones share
/// blocks, so their curve stays exact and one granularity owner prices the
/// block (`ByteCurve::paged` is the rounding-only form: below the boundary it
/// would charge a block a request cannot take). Refinement splits the rounded
/// extent back into request and tail.
fn pool_curve(stride: u64, page: u64, maximum: u64) -> Result<ByteCurve, Refusal> {
    let first = first_independent_pages(stride, page)?;
    let mut pieces = Vec::new();
    let exact_last = (first - 1).min(maximum);
    if exact_last > 0 {
        pieces.push(BytePiece {
            first: 1,
            last: exact_last,
            bytes_at_first: pool_request(1, stride, page)?,
            bytes_per_page: stride,
            granularity: 1,
            backings: 1,
        });
    }
    if first <= maximum {
        pieces.push(BytePiece {
            first,
            last: maximum,
            bytes_at_first: pool_request(first, stride, page)?,
            bytes_per_page: stride,
            granularity: SM120_ALLOCATION_GRANULARITY_BYTES,
            backings: 1,
        });
    }
    Ok(ByteCurve(pieces))
}

/// Exact cudaMalloc request for one pool allocation at `pages` solver pages.
fn pool_request(pages: u64, stride: u64, page: u64) -> Result<u64, Refusal> {
    mul(pages, stride)?
        .checked_add(page)
        .ok_or(Refusal::Overflow("GLM pool scratch page"))
}

/// max(256, pages * stride), represented by at most two affine pieces.
fn floored(stride: u64, maximum: u64) -> Result<ByteCurve, Refusal> {
    let flat = if stride == 0 {
        maximum
    } else {
        (FLOOR / stride).min(maximum)
    };
    let mut pieces = Vec::new();
    if flat > 0 {
        pieces.push(BytePiece { granularity: 1,
            first: 1,
            last: flat,
            bytes_at_first: FLOOR,
            bytes_per_page: 0, backings: 1,
        });
    }
    if flat < maximum {
        let first = flat
            .checked_add(1)
            .ok_or(Refusal::Overflow("GLM buffer floor"))?;
        pieces.push(BytePiece { granularity: 1,
            first,
            last: maximum,
            bytes_at_first: mul(first, stride)?,
            bytes_per_page: stride, backings: 1,
        });
    }
    Ok(ByteCurve(pieces))
}

struct Builder {
    prefix: String,
    location: Location,
    maximum: u64,
    out: Vec<Component>,
}
impl Builder {
    fn add(
        &mut self,
        name: &str,
        bytes: ByteCurve,
        lifetime: Lifetime,
        category: Category,
        view: Option<(String, u64, Option<u32>)>,
    ) -> Result<(), Refusal> {
        let mut spec = Allocation::buffer(
            format!("{}.{name}", self.prefix),
            self.location,
            bytes.at(1)?,
            lifetime,
            category,
            "GLM target buffers",
        );
        if let Some((backing, offset, slot)) = view {
            spec.view = Some(View {
                backing: format!("{}.{backing}", self.prefix),
                offset,
            });
            spec.slot = slot;
        }
        let mut component = Component::allocation(spec, bytes.clone(), false, self.maximum);
        component.byte_extent(bytes)?;
        self.out.push(component);
        Ok(())
    }
    fn fixed(
        &mut self,
        name: &str,
        bytes: u64,
        lifetime: Lifetime,
        category: Category,
    ) -> Result<(), Refusal> {
        self.add(
            name,
            ByteCurve::fixed(bytes, self.maximum),
            lifetime,
            category,
            None,
        )
    }
    fn workspace(
        &mut self,
        cfg: &GlmDsaConfig,
        facts: &TargetBuffers,
        decode: bool,
        shape: PrefillShape,
        scratch: GlmScratch,
        table_stride: u64,
    ) -> Result<(), Refusal> {
        let name = if decode { "decode" } else { "prefill" };
        // The decode workspace is allocated by the startup warm pass. Prefill
        // lanes are allocated by the first prefill: during startup only for
        // scoring, otherwise by live traffic after Ready.
        let lifetime = Lifetime {
            first: if decode || facts.full_prefill_logits {
                Phase::Warmup
            } else {
                Phase::Growth
            },
            last: Phase::Growth,
        };
        let span = mul(shape.rows, shape.waves)?;
        let lead = facts.lead;
        let mut fields =
            super::step_row_widths(cfg, lead, facts.split_heads, facts.router).to_vec();
        fields.push(("select", 8));
        if facts.payload_row_bytes > 0 {
            fields.push(("payload", facts.payload_row_bytes));
        }
        if facts.core_rows {
            fields.extend([
                ("core.attention", mul(cfg.hidden as u64, facts.attention_partial_bytes)?),
                ("core.down", mul(cfg.hidden as u64, 2)?),
                ("core.reduction", mul(cfg.hidden as u64, 4)?),
            ]);
        }
        for (field, width) in fields {
            let backing = format!("{name}.rows.{field}");
            self.fixed(
                &backing,
                mul(span, width)?.max(FLOOR),
                lifetime,
                Category::Workspace,
            )?;
            if !decode {
                for slot in 0..shape.waves {
                    // Zero-width fields clone the same placeholder, not independent row slots.
                    let bytes = if width == 0 {
                        FLOOR
                    } else {
                        mul(shape.rows, width)?
                    };
                    let offset = mul(mul(slot, shape.rows)?, width)?;
                    self.add(
                        &format!("{name}.slot.{slot}.{field}"),
                        ByteCurve::fixed(bytes, self.maximum),
                        lifetime,
                        Category::Workspace,
                        Some((
                            backing.clone(),
                            offset,
                            if width == 0 || shape.waves == 1 {
                                None
                            } else {
                                Some(slot as u32)
                            },
                        )),
                    )?;
                }
            }
        }
        let logits_rows = super::logits_rows(shape.rows, decode, facts.full_prefill_logits);
        let logits = if lead {
            mul(mul(logits_rows, cfg.vocab_size as u64)?, 4)?.max(FLOOR)
        } else {
            FLOOR
        };
        self.fixed(
            &format!("{name}.logits"),
            logits,
            lifetime,
            Category::Workspace,
        )?;
        self.fixed(
            &format!("{name}.scratch"),
            scratch.programs.max(FLOOR),
            lifetime,
            Category::Workspace,
        )?;
        self.fixed(
            &format!("{name}.topk"),
            scratch.topk.max(FLOOR),
            lifetime,
            Category::Workspace,
        )?;
        if let Some(bytes) = super::head_workspace_bytes(lead, facts.native_head) {
            self.fixed(
                &format!("{name}.head"),
                bytes,
                lifetime,
                Category::Workspace,
            )?;
        }
        let tables = floored(
            mul(table_stride, if decode { shape.rows } else { 1 })?,
            self.maximum,
        )?;
        for slot in 0..shape.waves {
            self.add(
                &format!("{name}.page_table.{slot}"),
                tables.clone(),
                lifetime,
                Category::Workspace,
                None,
            )?;
            if !decode {
                self.add(
                    &format!("{name}.slot.{slot}.logits"),
                    ByteCurve::fixed(logits, self.maximum),
                    lifetime,
                    Category::Workspace,
                    Some((format!("{name}.logits"), 0, None)),
                )?;
            }
            for (field, width) in [
                ("positions", 8),
                ("slots", 8),
                ("cache_lengths", 4),
                ("lengths", 4),
                ("ids", 4),
            ] {
                // Prefill reserves all six destinations eagerly; decode's peer never embeds ids.
                if decode && !lead && field == "ids" {
                    continue;
                }
                let bytes = mul(shape.rows, width)?;
                let bytes = if decode { bytes.max(FLOOR) } else { bytes };
                let location = self.location;
                self.location = Location::PINNED;
                self.fixed(
                    &format!("{name}.upload.{slot}.{field}"),
                    bytes,
                    lifetime,
                    Category::Staging,
                )?;
                self.location = location;
            }
            let location = self.location;
            self.location = Location::PINNED;
            self.add(
                &format!("{name}.upload.{slot}.page_table"),
                tables.clone(),
                lifetime,
                Category::Staging,
                None,
            )?;
            self.location = location;
        }
        Ok(())
    }
}

impl TargetBuffers {
    pub fn components(&self, cfg: &GlmDsaConfig) -> Result<Vec<Component>, Refusal> {
        let divisor = if self.partitioned { 2 } else { 1 };
        if self.device > 1
            || self.maximum_pages == 0
            || self.context == 0
            || self.page_tokens == 0
            || self.page_tokens % (PAGE_ROWS * divisor) != 0
            || self.owned_layers.is_empty()
            || self.owned_layers.len() > cfg.layers
            || cfg.indexers.len() != cfg.layers
            || (self.split_heads && cfg.heads % 2 != 0)
            || cfg.hidden == 0
            || cfg.vocab_size == 0
            || cfg.qk_rope_head_dim == 0
            || self.prefill.as_ref().is_some_and(|(s, _)| {
                !(64..=4096).contains(&s.rows)
                    || s.rows % PAGE_ROWS != 0
                    || !(1..=4).contains(&s.waves)
            })
        {
            return Err(Refusal::InvalidFact(
                "invalid GLM target buffer geometry".into(),
            ));
        }
        // Validate the existing row-geometry helper's arithmetic before calling it.
        for (n, width) in [
            (cfg.hidden as u64, 4),
            (cfg.heads as u64, 1152),
            (cfg.q_lora_rank as u64, 2),
            (cfg.index_heads as u64, 4),
            (cfg.index_topk as u64, 4),
            (cfg.experts as u64, 4),
            (cfg.topk as u64, 4),
        ] {
            mul(n, width)?;
        }
        mul(cfg.index_heads as u64, cfg.index_head_dim as u64)?;
        let local_pages = self.page_tokens / (PAGE_ROWS * divisor);
        let mut b = Builder {
            prefix: format!("glm.target.gpu.{}", self.device),
            location: Location::coordinator(self.device),
            maximum: self.maximum_pages,
            out: Vec::new(),
        };
        let mut small = Vec::new();
        for (layer, &owned) in self.owned_layers.iter().enumerate() {
            let kv = if owned && cfg.owns_kv(layer) {
                let page = mul(PAGE_ROWS, cfg.record_bytes(layer) as u64)?;
                pool_curve(mul(local_pages, page)?, page, self.maximum_pages)?
            } else {
                small.push(format!("glm.target.gpu.{}.kv.{layer}", self.device));
                ByteCurve::fixed(FLOOR, self.maximum_pages)
            };
            b.add(
                &format!("kv.{layer}"),
                kv,
                Lifetime::TARGET,
                Category::Kv,
                None,
            )?;
            if owned && cfg.indexers[layer] == GlmIndexer::Full {
                b.add(
                    &format!("index.{layer}"),
                    pool_curve(mul(local_pages, INDEX_PAGE_BYTES)?, INDEX_PAGE_BYTES,
                        self.maximum_pages)?,
                    Lifetime::TARGET,
                    Category::Kv,
                    None,
                )?;
            }
        }
        if !small.is_empty() {
            // Unowned latent slots still cudaMalloc their 256-byte floor; the
            // driver packs those sub-block requests into one shared block.
            b.out.push(allocation_granularity_component(
                AllocationGranularity {
                    id: format!("glm.target.gpu.{}.kv.small-tail", self.device),
                    location: Location::coordinator(self.device),
                    lifetime: Lifetime::TARGET,
                    backings: small.into_iter()
                        .map(|backing| GranularityBacking { backing, requests: vec![FLOOR] })
                        .collect(),
                    evidence: GranularityEvidence::Forecast,
                },
                self.maximum_pages,
            )?);
        }
        b.fixed(
            "rope",
            mul(mul(self.context, cfg.qk_rope_head_dim as u64)?, 4)?,
            Lifetime::TARGET,
            Category::Weights,
        )?;
        let tables = mul(self.page_tokens / PAGE_ROWS, 4)?;
        if let Some(scratch) = self.decode {
            b.workspace(
                cfg,
                self,
                true,
                PrefillShape {
                    rows: DECODE_ROWS,
                    waves: 1,
                },
                scratch,
                tables,
            )?;
        }
        if let Some((shape, scratch)) = self.prefill {
            b.workspace(cfg, self, false, shape, scratch, tables)?;
        }
        Ok(b.out)
    }

    /// Every KV-pool cudaMalloc request this rank makes at `pages` solver
    /// pages, in the order the executor allocates them. Identities match
    /// `components` so the solved ledger can be split into exact requests.
    /// Returns the real pool requests and the ids of the 256-byte latent slots
    /// whose shared driver block `components` already prices.
    pub fn pool_requests(&self, cfg: &GlmDsaConfig, pages: u64)
        -> Result<(Vec<(String, u64)>, Vec<String>), Refusal> {
        let divisor = if self.partitioned { 2 } else { 1 };
        if self.device > 1
            || pages == 0
            || pages > self.maximum_pages
            || self.page_tokens % (PAGE_ROWS * divisor) != 0
            || self.owned_layers.len() > cfg.layers
            || cfg.indexers.len() != cfg.layers
        {
            return Err(Refusal::InvalidFact("invalid GLM pool request geometry".into()));
        }
        let local_pages = self.page_tokens / (PAGE_ROWS * divisor);
        let prefix = format!("glm.target.gpu.{}", self.device);
        let mut out = Vec::new();
        let mut floors = Vec::new();
        for (layer, &owned) in self.owned_layers.iter().enumerate() {
            if owned && cfg.owns_kv(layer) {
                let page = mul(PAGE_ROWS, cfg.record_bytes(layer) as u64)?;
                out.push((format!("{prefix}.kv.{layer}"),
                    pool_request(pages, mul(local_pages, page)?, page)?));
            } else {
                floors.push(format!("{prefix}.kv.{layer}"));
            }
            if owned && cfg.indexers[layer] == GlmIndexer::Full {
                out.push((
                    format!("{prefix}.index.{layer}"),
                    pool_request(pages, mul(local_pages, INDEX_PAGE_BYTES)?, INDEX_PAGE_BYTES)?,
                ));
            }
        }
        Ok((out, floors))
    }
}

#[cfg(test)]
mod tests;
