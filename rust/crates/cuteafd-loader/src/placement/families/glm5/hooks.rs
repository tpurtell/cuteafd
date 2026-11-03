//! Hook-local native owners, composed beside target rows and caches. These are
//! allocation facts only; experts, exchanges and serving owners join separately.
use crate::families::glm5::{core::CoreRankLayout, native::ScratchLayout};
use crate::formats::trellis_core::ProjectionScratchLayout;
use crate::placement::planner::{
    Allocation, ByteCurve, Component, Lifetime, Location, Phase, Refusal, View,
};
use cuteafd_core::memory_layout::Category;

#[derive(Debug, Clone)]
pub struct HookBuffers {
    pub device: u8,
    pub maximum_pages: u64,
    pub native: Option<[ScratchLayout; 2]>,
    pub core: Option<CoreRankLayout>,
    /// Manifest-derived shared execution owners and bindings, once per hook.
    pub shared_workspace: Option<crate::plan::exlr8::ExecutionWorkspace>,
}

struct Builder {
    prefix: String,
    device: u8,
    maximum: u64,
    out: Vec<Component>,
}
impl Builder {
    fn add(
        &mut self,
        name: &str,
        bytes: u64,
        lifetime: Lifetime,
        category: Category,
        view: Option<(String, u64)>,
    ) -> Result<(), Refusal> {
        if bytes == 0 {
            return Err(Refusal::InvalidFact(format!("empty GLM hook owner {name}")));
        }
        let mut allocation = Allocation::buffer(
            format!("{}.{name}", self.prefix),
            Location::coordinator(self.device),
            bytes,
            lifetime,
            category,
            "GLM target hooks",
        );
        if let Some((backing, offset)) = view {
            allocation.view = Some(View {
                backing: format!("{}.{backing}", self.prefix),
                offset,
            });
        }
        self.out.push(Component::allocation(
            allocation,
            ByteCurve::fixed(bytes, self.maximum),
            false,
            self.maximum,
        ));
        Ok(())
    }
    fn projection(
        &mut self,
        name: &str,
        layout: &ProjectionScratchLayout,
        lifetime: Lifetime,
        category: Category,
    ) -> Result<(), Refusal> {
        if !(1..=4096).contains(&layout.rows) {
            return Err(Refusal::InvalidFact("GLM hook projection rows".into()));
        }
        let storage = layout
            .storage()
            .map_err(|e| Refusal::InvalidFact(format!("GLM hook projection: {e}")))?;
        for (owner, bytes) in storage.owners {
            self.add(
                &format!("{name}.storage.{}", owner.replace(' ', "_")),
                bytes as u64,
                lifetime,
                category,
                None,
            )?;
        }
        for (&field, &bytes) in &layout.extents {
            let (owner, offset) = if let Some(&offset) = storage.route_offsets.get(field) {
                ("route_arena", offset)
            } else {
                (field, 0)
            };
            self.add(
                &format!("{name}.view.{field}"),
                bytes as u64,
                lifetime,
                category,
                Some((format!("{name}.storage.{owner}"), offset as u64)),
            )?;
        }
        Ok(())
    }
}

impl HookBuffers {
    pub fn components(&self) -> Result<Vec<Component>, Refusal> {
        if self.device > 1 || self.maximum_pages == 0 {
            return Err(Refusal::InvalidFact("GLM hook device/capacity".into()));
        }
        let mut b = Builder {
            prefix: format!("glm.target.gpu.{}", self.device),
            device: self.device,
            maximum: self.maximum_pages,
            out: Vec::new(),
        };
        let workspace = Lifetime {
            first: Phase::Warmup,
            last: Phase::Growth,
        };
        for (index, kind) in ["decode", "prefill"].into_iter().enumerate() {
            if let Some(layouts) = &self.native {
                let layout = &layouts[index];
                layout
                    .bytes()
                    .map_err(|e| Refusal::InvalidFact(format!("GLM native hook: {e}")))?;
                let linear = &layout.linear;
                for (field, bytes) in [
                    ("rotated", Some(linear.rotated)),
                    ("table", Some(linear.table)),
                    ("block_scales", Some(linear.block_scales)),
                    ("partial", Some(linear.partial)),
                    ("records", linear.records),
                    ("acc", Some(linear.acc)),
                ] {
                    if let Some(bytes) = bytes {
                        b.add(
                            &format!("{kind}.native.linear.{field}"),
                            bytes as u64,
                            workspace,
                            Category::Workspace,
                            None,
                        )?;
                    }
                }
                for (&field, &bytes) in &layout.stages {
                    b.add(
                        &format!("{kind}.native.stage.{field}"),
                        bytes as u64,
                        workspace,
                        Category::Workspace,
                        None,
                    )?;
                }
            }
            if let Some(core) = &self.core {
                let projection = if index == 0 {
                    &core.decode
                } else {
                    &core.prefill
                };
                if let Some(projection) = projection {
                    b.projection(
                        &format!("{kind}.core.projection"),
                        projection,
                        workspace,
                        Category::Workspace,
                    )?;
                }
                if let Some(stages) = &core.stages {
                    for (&field, &bytes) in &stages[index].extents {
                        b.add(
                            &format!("{kind}.core.stage.{field}"),
                            bytes as u64,
                            workspace,
                            Category::Workspace,
                            None,
                        )?;
                    }
                }
            }
            if let Some(layout) = &self.shared_workspace {
                if layout.compact || layout.unit_tables || layout.top_k != 1
                    || layout.rows != cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS {
                    return Err(Refusal::InvalidFact("GLM shared hook execution geometry".into()));
                }
                let allocations = layout.allocations(&format!("{}.{kind}.native.shared", b.prefix),
                    Location::coordinator(self.device), workspace)
                    .map_err(|e| Refusal::InvalidFact(format!("GLM shared hook: {e}")))?;
                for allocation in allocations {
                    let bytes = allocation.bytes;
                    b.out.push(Component::allocation(allocation, ByteCurve::fixed(bytes, self.maximum_pages),
                        false, self.maximum_pages));
                }
            }
        }
        if let Some(core) = &self.core {
            // bind_core/bind_draft_core_head allocate after target KV, before warmup.
            for (name, layout, category) in [
                ("target_head", &core.target_head, Category::Workspace),
                ("draft_head", &core.draft_head, Category::Drafter),
            ] {
                if let Some(layout) = layout {
                    b.projection(&format!("core.{name}"), layout, Lifetime::TARGET, category)?;
                }
            }
        }
        Ok(b.out)
    }
}

#[cfg(test)]
mod tests;
