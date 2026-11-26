//! Cache/attention placement follows compressed-source consumer groups.
use anyhow::{ensure, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CachePlacement { attention: [usize; 40] }
impl CachePlacement {
    pub fn new(attention: [usize; 40]) -> Result<Self> {
        ensure!(attention.iter().all(|&gpu| gpu < 2), "cache GPU must be zero or one");
        for (source,end) in [(2,8),(8,14),(14,20),(20,40)] {
            ensure!(attention[source..end].iter().all(|&gpu| gpu == attention[source]),
                "compressed source {source} must share its GPU with attention consumers");
        }
        Ok(Self { attention })
    }
    pub fn from_execution_plan(plan: &cuteafd_loader::placement::LayerExecutionPlan) -> Result<Self> {
        use cuteafd_loader::placement::{FfnMode, HopStorage, LayerMode, ResidualHome};
        ensure!(plan.modes().len() == 40, "V4.1 execution requires 40 layers");
        ensure!(plan.spec().storage == HopStorage::DirectInput && plan.spec().row_bytes == 40976,
            "V4.1 residual hops must land in the existing block inputs");
        let mut attention = [0; 40];
        for (layer, owner) in attention.iter_mut().enumerate() {
            let LayerMode::Whole { gpu, ffn: FfnMode::Owner } = plan.mode(layer)? else {
                anyhow::bail!("V4.1 layer {layer} requires whole attention and owner-reduced FFN")
            };
            ensure!(plan.homes()[layer + 1] == ResidualHome::Owned(gpu),
                "V4.1 residual must stay on the attention owner");
            *owner = usize::from(gpu);
        }
        Self::new(attention)
    }
    pub fn encoder_decoder() -> Self {
        Self { attention: std::array::from_fn(|layer| usize::from(layer >= 20)) }
    }
    pub fn attention(&self, layer: usize) -> Result<usize> {
        self.attention.get(layer).copied().ok_or_else(|| anyhow::anyhow!("invalid attention layer"))
    }
    pub fn sources(&self) -> [usize; 4] {
        [self.attention[2],self.attention[8],self.attention[14],self.attention[20]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn execution_plan_binds_single_and_ced_ownership() -> Result<()> {
        use cuteafd_loader::placement::families::deepseek_v41::admission::{
            live_budgets, production_request, ProductionInput,
        };
        use cuteafd_loader::placement::planner::{Operation, PrefillShape};
        use cuteafd_loader::placement::*;
        let config: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
        ))?;
        for ranks in [1, 2] {
            let prefill = PrefillShape { rows: 128, waves: 1 };
            let placement = ProductionInput {
            blas: Vec::new(),
            expert_usage: false,
                config: &config, gpus: ranks, context: 4096, pool: cuteafd_loader::placement::prepare_pool(cuteafd_loader::placement::PoolFacts::new(4096, Some(4096), 512)).unwrap(),
                prefill, concurrency: 1, retained_turns: 0, replicated: false, spark_ranks: 4,
                onboard_minimum: 0, onboard_counts: None,
                experts: vec![ExpertCost { whole: Bytes2::default(), half: [Bytes2::default(); 2],
                    tp2: ranks == 2, whole_supported: true, spark_ok: true }; 40],
                workspace: 0, owners: Vec::new(), weight_arenas: Vec::new(), startup: Vec::new(), allowances: Vec::new(),
                draft: None, encoders: Vec::new(), media_cache_bytes: 0,
                resident_weight_limit: u64::MAX, spark_reply_formats: Default::default(),
                operation: Operation::Serving,
                budgets: live_budgets(&vec![(1 << 40, 1 << 40, 1 << 39); ranks],
                    &cuteafd_loader::serving_capacity::spark_memory::serving_bytes()[..4])?,
            }.prepare()?.admit(production_request(4096, Some(4096), Onboard::Layers(0), prefill,
                Operation::Serving, u64::MAX, Default::default()))?.placement;
            let spec = cuteafd_loader::placement::families::deepseek_v41::execution_hops(ranks);
            let plan = LayerExecutionPlan::new(&placement, spec)?;
            let map = CachePlacement::from_execution_plan(&plan)?;
            for layer in 0..40 {
                assert_eq!(map.attention(layer)?, usize::from(ranks == 2 && layer >= 20));
            }
            assert_eq!(plan.hops().len(), ranks - 1);
            let mut wrong = placement.clone();
            wrong.residual[1] = ResidualHome::Replicated;
            assert!(LayerExecutionPlan::new(&wrong, spec).is_err());
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs"]
    fn distributed_cache_allocations_and_leases_follow_placement() -> Result<()> {
        cache_allocations(false)
    }
    #[test]
    #[ignore = "requires SM peer copy and two CUDA GPUs"]
    fn replicated_cache_allocations_preserve_capacity_and_reset_both_devices() -> Result<()> {
        cache_allocations(true)
    }
    fn cache_allocations(replicated:bool) -> Result<()> {
        use super::super::BackboneCache;
        use cuteafd_ffi::NativeLibrary;
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        for boundary in [20,14] {
            let placement = CachePlacement::new(std::array::from_fn(|layer| usize::from(layer >= boundary)))?;
            let pages = [2,2,2,4];
            let bytes = BackboneCache::distributed_device_bytes(placement,2,pages)?;
            assert_eq!(bytes.iter().sum::<usize>(),BackboneCache::device_bytes(2,pages)?);
            assert!(BackboneCache::new_distributed(&lib,placement,2,pages,[bytes[0]-1,bytes[1]]).is_err());
            let mut bank = if replicated {
                let replicated_bytes=BackboneCache::replicated_device_bytes(placement,2,pages)?;
                assert!(BackboneCache::new_replicated(&lib,placement,2,pages,
                    [replicated_bytes[0]-1,replicated_bytes[1]]).is_err());
                BackboneCache::new_replicated(&lib,placement,2,pages,replicated_bytes)?
            } else { BackboneCache::new_distributed(&lib,placement,2,pages,bytes)? };
            for cycle in 0..2 {
                let lease = bank.begin_request(0,42+cycle)?;
                assert_eq!(bank.committed_end(lease)?,0);
                let request = bank.request(lease)?;
                for layer in 0..40 {
                    let view = bank.windows[layer].view(request.windows[layer])?;
                    let expected = placement.attention(layer)? as i32;
                    assert_eq!(bank.attention_device(layer)?.id,expected);
                    assert_eq!(view.device_end.device_id,expected);
                    let mut end = [255;8];
                    bank.attention_device(layer)?.run(|| lib.copy_d2h_quiescent(&mut end,view.device_end))?;
                    assert_eq!(end,[0;8]);
                    if replicated {
                        let replica=bank.windows[layer].replica().unwrap();
                        let peer=unsafe { replica.view(&bank.windows[layer],request.windows[layer])? };
                        assert_eq!(peer.device_end.device_id,1-expected);
                        replica.device().run(||lib.copy_d2h_quiescent(&mut end,peer.device_end))?;
                        assert_eq!(end,[0;8]);
                    }
                }
                for (index,gpu) in placement.sources().into_iter().enumerate() {
                    let kv = bank.sources[index].kv_cache(request.sources[index])?;
                    let keys = bank.sources[index].index_cache(request.sources[index])?;
                    for buffer in [kv.values,kv.scales,kv.device_pages,kv.device_rows,keys.packed,keys.scales] {
                        assert_eq!(buffer.device_id,gpu as i32);
                    }
                    assert_eq!(kv.rows,0);
                    assert_eq!(bank.sources[index].source_cache().capacity,pages[index]*256);
                    if replicated {
                        let replica=bank.sources[index].replica().unwrap();
                        let peer=unsafe { replica.view(bank.sources[index].source_cache(),0,0)? };
                        assert_eq!(peer.values.device_id,1-gpu as i32);
                        let mut end=[255;8];
                        replica.device().run(||lib.copy_d2h_quiescent(&mut end,peer.device_rows))?;
                        assert_eq!(end,[0;8]);
                    }
                }
                bank.release(&[lease])?;
                assert!(bank.committed_end(lease).is_err());
                assert_eq!(lib.cuda_get_device()?,0);
            }
            drop(bank);
            assert_eq!(lib.cuda_get_device()?,0);
        }
        Ok(())
    }
    #[test]
    fn placement_keeps_sources_with_consumers_and_allows_rebalancing() -> Result<()> {
        let initial = CachePlacement::encoder_decoder();
        assert_eq!(initial.sources(),[0,0,0,1]);
        let mut rebalanced = initial.attention;
        rebalanced[14..20].fill(1);
        assert_eq!(CachePlacement::new(rebalanced)?.sources(),[0,0,1,1]);
        rebalanced[19] = 0;
        assert!(CachePlacement::new(rebalanced).is_err());
        rebalanced[19] = 1;
        rebalanced[0] = 2;
        assert!(CachePlacement::new(rebalanced).is_err());
        assert!(initial.attention(40).is_err());
        Ok(())
    }
}
