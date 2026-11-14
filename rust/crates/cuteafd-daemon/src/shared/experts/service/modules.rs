//! Initialize only installed code; keep owners alive through execution teardown.
use super::*;
use crate::shared::experts::{
    exl3::{execution::{Exl3Epilogue, Exl3InputFormat, Exl3Modules}, worker::Exl3Worker},
    fp8::{self, Fp8Modules},
};

pub(super) enum WarmModules<'a> {
    Native { _kernels: Vec<cuteafd_ffi::V41ExpertKernel<'a>> },
    Exl3(Option<Vec<Exl3Modules<'a>>>),
    Fp8(Option<Vec<Fp8Group>>),
    Exlr8(&'a cuteafd_ffi::programs::Programs<'a>),
}
/// Code and immutable preflight for one homogeneous native group. The module
/// phase validates every package before warming any of them or loading weights.
pub(super) struct Fp8Group {
    pub tensors: cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors,
    pub layers: Vec<usize>,
    pub directory: std::path::PathBuf,
    pub bf16_directory: Option<std::path::PathBuf>,
    pub slicing: cuteafd_loader::formats::fp8_experts::Slicing,
    pub scratch: usize,
    pub io: usize,
    pub modules: Option<Fp8Modules>,
}

impl Fp8Group {
    fn plan(catalog: &OfficialV41Catalog, config: &NativeExpertServiceConfig) -> Result<Vec<Self>> {
        use cuteafd_loader::formats::fp8_experts::ExpertFormat;
        let tensors = catalog.fp8().context("native expert groups require checkpoint tensors")?;
        config.spark_world()?;
        let layers = config.resident_layers(catalog.routed_experts().layers)?;
        let mut groups = Vec::<Self>::new();
        for layer in layers {
            let tensors = tensors.for_layer(layer)?;
            let layers = vec![layer];
            tensors.slice(config.world)?;
            let directory = config.fp8_package.clone().unwrap_or_else(||
                fp8::package_directory(&config.library, config.world, tensors.format()));
            let (directory, slicing) = fp8::exact_layout(&directory, &tensors, config.world, config.rank);
            let family = fp8::package_family(cuteafd_core::expert_geometry().family(), tensors.format());
            let format = match tensors.format() {
                ExpertFormat::Nvfp4 => "nvfp4", ExpertFormat::Bf16 => "bf16", _ => "fp8",
            };
            super::backend::validate_fp8_recipe(&format!("{family}:{format}"), tensors.shape().intermediate,
                config.world, &directory,
                cuteafd_loader::plan::experts::spark_recipe(&format!("{family}:{format}"),
                    tensors.shape().intermediate, config.world)?.rank_for_layer(config.rank, layer), slicing,
                tensors.rank_width(config.world, config.rank, slicing)?)?;
            fp8::validate_reply_exports(&directory, tensors.format())?;
            let metadata = fp8::package_metadata(&directory, &tensors, &layers, config.world, config.rank, slicing)?;
            let capacity = config.capacity as usize;
            let mut scratch = cuteafd_ffi::fp8_moe::fp8_slot_scratch_bytes(metadata.scratch_layout(), capacity)?;
            let bf16_directory = metadata.info.wire_input.then(|| fp8::bf16_sibling(&directory))
                .flatten().filter(|path| path.is_dir());
            if let Some(path) = &bf16_directory {
                fp8::validate_reply_exports(path, tensors.format())?;
                let sibling = fp8::package_metadata(path, &tensors, &layers, config.world, config.rank, slicing)?;
                fp8::validate_bf16_package(&metadata.info, &sibling.info, capacity)?;
                scratch = scratch.max(cuteafd_ffi::fp8_moe::fp8_slot_scratch_bytes(sibling.scratch_layout(), capacity)?);
            }
            let io = super::backend::native_slot_io_bytes(&metadata.info, bf16_directory.is_some(), capacity)?;
            if let Some(group) = groups.iter_mut().find(|group| group.directory == directory
                && group.bf16_directory == bf16_directory && group.slicing == slicing
                && group.tensors.format() == tensors.format()
                && group.tensors.activation(group.layers[0]) == tensors.activation(layer)) {
                group.layers.push(layer);
                group.scratch = group.scratch.max(scratch);
            } else {
                groups.push(Self { tensors, layers, directory, bf16_directory, slicing, scratch, io, modules: None });
            }
        }
        Ok(groups)
    }
}

impl<'a> WarmModules<'a> {
    pub(super) fn load(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
        config: &NativeExpertServiceConfig,
        programs: Option<&'a cuteafd_ffi::programs::Programs<'a>>) -> Result<Self> {
        let partial_replies = Exl3Epilogue::for_catalog(catalog) == Exl3Epilogue::PartialReplies;
        if partial_replies {
            let programs = programs.context("partial Spark replies require a program table")?;
            crate::shared::experts::reply::validate_exports(programs, "partial")?;
        } else if cuteafd_core::spark_reply::requested().can_compress() {
            if let Some(tensors) = catalog.fp8() {
                fp8::validate_reply_formats(tensors.format(), cuteafd_core::spark_reply::requested())?;
            }
            let programs = programs.context("compressed Spark reply admission requires a program table")?;
            let kind = if catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8 {
                if cuteafd_core::expert_geometry().hidden == 4096 { "exlr8_glmf" } else { "exlr8" }
            } else if catalog.fp8().is_some() { "fp8" }
            else if catalog.exl3().is_some() { "exl3" } else { "native" };
            crate::shared::experts::reply::validate_exports(programs, kind)?;
        }
        if catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8 {
            let programs = programs.context("EXLR8 module owner requires validated programs")?;
            let geometry = catalog.routed_experts().geometry()?;
            let family = cuteafd_ffi::exlr8::KernelFamily::from_geometry(geometry.hidden as usize,
                geometry.intermediate as usize / 128, geometry.topk as usize, geometry.experts as usize)?;
            let decode_rows = (config.capacity as usize)
                .min(cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS);
            // Both decode instances are part of every Spark EXLR8 image;
            // validation here precedes any workspace allocation.
            crate::shared::experts::exlr8::Exlr8Execution::decode_compact_workspace_bytes(programs, family,
                decode_rows, 16)?;
            programs.program(family.program(cuteafd_ffi::exlr8::KernelKind::Group), &["ids", "x", "signs_in", "group_expert", "route_slots", "route_group_lane", "group_count", "activation"])?;
            crate::shared::experts::exlr8::Exlr8Execution::decode_grouped_workspace_bytes(programs, family,
                decode_rows, 16)?;
            if config.capacity as usize > cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS {
                crate::shared::experts::exlr8::Exlr8Execution::workspace_bytes_with_limit(programs, family,
                    config.capacity as usize, cuteafd_loader::plan::exlr8::residency::ExpertSet::Routed, 16,
                    config.exlr8_prefill_activations)?;
                programs.program(family.program(cuteafd_ffi::exlr8::KernelKind::GroupedGroup), &["ids", "x", "signs_in", "group_expert", "route_slots", "route_group_lane", "group_count", "activation"])?;
            }
            return Ok(Self::Exlr8(programs));
        }
        if catalog.fp8().is_some() {
            let mut groups = Fp8Group::plan(catalog, config)?;
            for group in &mut groups {
                group.modules = Some(Fp8Modules::load(&group.directory, group.bf16_directory.as_deref(), &cuteafd_loader::plan::fp8_prefill::Policy::from_env().map_err(anyhow::Error::msg)?)?);
            }
            return Ok(Self::Fp8(Some(groups)));
        }
        if catalog.exl3().is_some() {
            let directory = config.exl3_directory(catalog)?;
            Exl3Worker::validate_exports(&directory, config.capacity, config.rank, config.exl3_schedule)?;
            config.validate_exl3_residency(catalog, &directory)?;
            let modules = cuteafd_core::Exl3RowPolicy::active().capacities(config.capacity as usize)?
                .into_iter().map(|capacity| {
                    // SAFETY: exports were validated above; no arenas or weights are allocated.
                    unsafe { Exl3Modules::load(library,
                        &config.exl3_schedule.directory(&directory, capacity), Exl3InputFormat::Fp8K32) }
                }).collect::<Result<Vec<_>>>()?;
            return Ok(Self::Exl3(Some(modules)));
        }
        let started = std::time::Instant::now();
        let nvfp4 = catalog.nvfp4().is_some();
        let compiled = config.compiled_capacity()?;
        let mut roles = Vec::new();
        let mut kernels = Vec::new();
        for index in config.resident_layers(catalog.routed_experts().layers)? {
            let layer = config.selection(index)?;
            if roles.contains(&layer.role()) { continue; }
            roles.push(layer.role());
            ensure!(roles.len() <= 2, "native rank has more than two adjacent width roles");
            let info = layer.select_info(library, compiled, nvfp4)?;
            let mut capacities = vec![compiled];
            if crate::shared::experts::execution::compact_output_role(info.role) {
                if compiled > 1 { capacities.push(1); }
                let grouped = crate::shared::experts::execution::grouped_decode_rows(layer, library, compiled, nvfp4)?;
                if compiled > grouped { capacities.push(grouped); }
            }
            for rows in capacities { kernels.push(layer.select_kernel(library, rows, nvfp4)?); }
        }
        tracing::info!(rank = config.rank, width_states = roles.len(), kernel_states = kernels.len(),
            warm_ms = started.elapsed().as_secs_f64() * 1000., "native width modules warmed");
        Ok(Self::Native { _kernels: kernels })
    }

    pub(super) fn exlr8(&self) -> Result<&'a cuteafd_ffi::programs::Programs<'a>> {
        match self { Self::Exlr8(programs) => Ok(*programs),
            _ => anyhow::bail!("EXLR8 module owner/backend mismatch") }
    }
    pub(super) fn take_fp8(&mut self) -> Result<Vec<Fp8Group>> {
        match self {
            Self::Fp8(modules) => modules.take().context("FP8 modules already consumed"),
            _ => anyhow::bail!("FP8 module owner/backend mismatch"),
        }
    }
    /// The warmed EXL3 modules for the first execution slot; later slots
    /// (`None`) load their own module set from the same exports, so no two
    /// slots share kernel state.
    pub(super) fn take_exl3(&mut self) -> Result<Option<Vec<Exl3Modules<'a>>>> {
        match self {
            Self::Exl3(modules) => Ok(modules.take()),
            _ => anyhow::bail!("EXL3 module owner/backend mismatch"),
        }
    }
}

#[cfg(test)]
mod tests {
    /// Every execution slot builds its own EXL3 execution: the first takes the
    /// warmed modules, later slots load their own (a second take is not an error).
    #[test]
    fn later_execution_slots_load_their_own_exl3_modules() {
        let mut modules = super::WarmModules::Exl3(Some(Vec::new()));
        assert!(modules.take_exl3().unwrap().is_some());
        assert!(modules.take_exl3().unwrap().is_none());
        assert!(modules.take_exl3().unwrap().is_none());
    }

    #[test]
    fn owner_binds_primary_context_without_census_before_all_backends() {
        let worker = include_str!("local.rs");
        let context = worker.split("census.sample(\"before_context\", None)?;").nth(1).unwrap();
        let first_statement = context.lines().map(str::trim)
            .find(|line| !line.is_empty() && !line.starts_with("//")).unwrap();
        assert_eq!(first_statement, "library.cuda_set_device(library.cuda_get_device()?)?;");
        let modules = context.find("let mut modules = match super::modules::WarmModules::load").unwrap();
        let execution = context.find("weights.execution(&library").unwrap();
        // Warm-up and execution stay on the synchronous GPU owner, not a
        // separate census thread with a different thread-local context.
        let owner = &context[..execution];
        assert!(modules < execution);
        for handoff in [".spawn(", "spawn_blocking(", ".await"] {
            assert!(!owner.contains(handoff), "CUDA owner handed off through {handoff}");
        }
    }

}
