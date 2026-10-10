//! Callback-local GPU owners must retire before the enclosing engine closes.
use super::{engine::MimoEngine, prefix::MimoPrefix};
use crate::shared::prefix::CudaCopyEngine;
use crate::shared::token_io::TokenSelector;
use anyhow::Result;
use cuteafd_engine::prefix::{PrefixCache, PrefixFamily};
use cuteafd_hostcache::copy::Stream;

/// Unlike request/grammar metadata, these owners contain CUDA allocations or
/// pinned storage addressed by asynchronous compute and host-copy queues.
struct Buffers<'e, 'a> {
    prefix: Option<(MimoPrefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)>,
    selector: Option<TokenSelector<'a>>,
}

/// Installed before the first serving-owned GPU buffer is created. This also
/// protects partially initialized serving state and non-engine body errors.
pub(super) struct ServingOwners<'e, 'a> {
    engine: &'e MimoEngine<'a>,
    buffers: Option<Buffers<'e, 'a>>,
}

impl<'e, 'a> ServingOwners<'e, 'a> {
    pub(super) fn new(engine: &'e MimoEngine<'a>) -> Self {
        Self { engine, buffers: Some(Buffers { prefix: None, selector: None }) }
    }

    pub(super) fn prefix(&mut self, family: MimoPrefix<'e, 'a>, cache: PrefixCache<CudaCopyEngine<'a>>) {
        self.buffers.as_mut().expect("live serving owners").prefix = Some((family, cache));
    }

    pub(super) fn selector(&mut self, selector: TokenSelector<'a>) {
        self.buffers.as_mut().expect("live serving owners").selector = Some(selector);
    }

    pub(super) fn parts(&mut self) -> (&MimoPrefix<'e, 'a>, &mut PrefixCache<CudaCopyEngine<'a>>, &mut TokenSelector<'a>) {
        let buffers = self.buffers.as_mut().expect("live serving owners");
        let (family, cache) = buffers.prefix.as_mut().expect("initialized serving prefix");
        let selector = buffers.selector.as_mut().expect("initialized serving selector");
        (family, cache, selector)
    }
}

/// No owner drops on a failed retirement proof. The callback escalates the
/// enclosing engine too, preserving its native library, streams and buffers.
fn retire_or_retain<T>(mut owners: T, retire: impl FnOnce(&mut T) -> Result<()>,
    retained: impl FnOnce(anyhow::Error)) {
    match retire(&mut owners) {
        Ok(()) => drop(owners),
        Err(error) => {
            retained(error);
            std::mem::forget(owners);
        }
    }
}

/// Failure of one local queue cannot skip drainage of the remaining queues.
fn retire_copy_queues(prefix: impl FnOnce() -> Result<()>, mut copy: impl FnMut(Stream) -> Result<()>) -> Result<()> {
    let mut failures = Vec::new();
    if let Err(error) = prefix() { failures.push(format!("prefix copy drainage: {error:#}")); }
    for stream in [Stream::Store, Stream::Restore] {
        if let Err(error) = copy(stream) { failures.push(format!("host {stream:?} drainage: {error:#}")); }
    }
    anyhow::ensure!(failures.is_empty(), "{}", failures.join("; "));
    Ok(())
}

impl Drop for ServingOwners<'_, '_> {
    fn drop(&mut self) {
        let Some(buffers) = self.buffers.take() else { return; };
        let engine = self.engine;
        retire_or_retain(buffers, |buffers| {
            // A host restore can wait on a compute event. Do not attempt its
            // queues when publication/compute drainage failed; keep all owners.
            engine.terminal_shutdown()?;
            if let Some((family, cache)) = &mut buffers.prefix {
                retire_copy_queues(|| family.drain().map_err(|error| anyhow::anyhow!("{error:#}")),
                    |stream| cache.host_engine_mut().map_or(Ok(()), |copy| copy.synchronize(stream)))?;
            }
            Ok(())
        }, |error| {
            engine.retain_serving_storage();
            tracing::error!(%error, "serving owners did not drain; retaining complete engine and serving storage");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};

    struct Owner(Rc<RefCell<Vec<&'static str>>>);
    impl Drop for Owner {
        fn drop(&mut self) { self.0.borrow_mut().push("owner-drop"); }
    }

    #[test]
    fn owner_release_follows_successful_retirement() {
        let events = Rc::new(RefCell::new(Vec::new()));
        retire_or_retain(Owner(Rc::clone(&events)), |owner| {
            owner.0.borrow_mut().extend(["engine-drain", "prefix-drain", "host-drain"]);
            Ok(())
        }, |_| panic!("successful retirement was quarantined"));
        assert_eq!(*events.borrow(), ["engine-drain", "prefix-drain", "host-drain", "owner-drop"]);
    }

    #[test]
    fn engine_failure_keeps_every_callback_owner() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let retained = Rc::clone(&events);
        retire_or_retain(Owner(Rc::clone(&events)), |owner| {
            owner.0.borrow_mut().push("engine-failure");
            anyhow::bail!("injected abort publication failure")
        }, |_| retained.borrow_mut().push("engine-retained"));
        assert_eq!(*events.borrow(), ["engine-failure", "engine-retained"]);
    }

    #[test]
    fn local_copy_failure_keeps_owners_after_compute_has_drained() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let retained = Rc::clone(&events);
        retire_or_retain(Owner(Rc::clone(&events)), |owner| {
            owner.0.borrow_mut().push("engine-drain");
            retire_copy_queues(|| {
                owner.0.borrow_mut().push("prefix-failure");
                anyhow::bail!("injected prefix copy failure")
            }, |stream| match stream {
                Stream::Store => {
                    owner.0.borrow_mut().push("host-store-failure");
                    anyhow::bail!("injected host store failure")
                },
                Stream::Restore => {
                    owner.0.borrow_mut().push("host-restore-attempt");
                    Ok(())
                },
            })
        }, |_| retained.borrow_mut().push("engine-retained"));
        assert_eq!(*events.borrow(), ["engine-drain", "prefix-failure", "host-store-failure", "host-restore-attempt", "engine-retained"]);
    }
}
