//! One-shot transfer of freshly loaded, synchronized device owners.
use anyhow::{anyhow, Result};

struct Loaded<T>(T);
// SAFETY: only load_pair constructs this private wrapper, under its caller's
// unaliased, synchronized, device-owned result contract. It is consumed after join.
unsafe impl<T> Send for Loaded<T> {}

/// Load one fresh owner per GPU concurrently and join both before publication.
///
/// # Safety
/// Each successful result must be newly owned and unaliased (no shared Rc or
/// thread-local references), with all CUDA work drained. It must destroy its
/// allocations on their owning device even on the receiving thread. Failures
/// and unwinds must drain queued work before releasing partial allocations.
/// Neither executor state nor an already-published owner may cross this boundary.
pub unsafe fn load_pair<T>(load: impl Fn(usize) -> Result<T> + Sync) -> Result<[T; 2]> {
    std::thread::scope(|scope| {
        let left = scope.spawn(|| load(0).map(Loaded));
        let right = scope.spawn(|| load(1).map(Loaded));
        // Join both even when one fails or panics; never release an owner while
        // its partner is still loading. Scope also joins on a parent unwind.
        let left = left.join().map_err(|_| anyhow!("rank0 weight loader panicked"));
        let right = right.join().map_err(|_| anyhow!("rank1 weight loader panicked"));
        Ok([left??.0, right??.0])
    })
}

/// [`load_pair`], with `beside` running on the calling thread meanwhile (the
/// coordinator's dense weights while its two GPUs receive pushed experts).
/// Both GPU owners are joined before `beside`'s result is returned, on every path.
///
/// # Safety
/// As [`load_pair`] for the two GPU results; `beside` runs on this thread only.
pub unsafe fn load_pair_beside<T, U>(load: impl Fn(usize) -> Result<T> + Sync,
    beside: impl FnOnce() -> U) -> (Result<[T; 2]>, U) {
    // SAFETY: forwarded caller contract; the inputs are empty.
    unsafe { load_pair_with([(), ()], |rank, ()| load(rank), beside) }
}

/// [`load_pair_beside`], handing each GPU's loader its own input (for example
/// descriptors of device memory the caller owns, whose owners outlive this call).
///
/// # Safety
/// As [`load_pair`] for the two GPU results; each input must be unaliased by
/// the other GPU's loader and by `beside`, and anything it refers to must stay
/// alive and unmoved until this call returns.
pub unsafe fn load_pair_with<I, T, U>(inputs: [I; 2], load: impl Fn(usize, I) -> Result<T> + Sync,
    beside: impl FnOnce() -> U) -> (Result<[T; 2]>, U) {
    let [left_input, right_input] = inputs.map(Loaded);
    std::thread::scope(|scope| {
        let load = &load;
        // Capture each whole wrapper (not its field) so only `Loaded` crosses threads.
        let left = scope.spawn(move || { let input = left_input; load(0, input.0).map(Loaded) });
        let right = scope.spawn(move || { let input = right_input; load(1, input.0).map(Loaded) });
        let value = beside();
        let left = left.join().map_err(|_| anyhow!("rank0 weight loader panicked"));
        let right = right.join().map_err(|_| anyhow!("rank1 weight loader panicked"));
        let pair = (|| Ok([left??.0, right??.0]))();
        (pair, value)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{atomic::{AtomicUsize, Ordering}, Arc, Barrier};

    #[test]
    fn both_ranks_load_concurrently() -> Result<()> {
        let rendezvous = Barrier::new(2);
        // SAFETY: plain integers are unaliased and have no queued device work.
        let ranks = unsafe { load_pair(|rank| { rendezvous.wait(); Ok(rank) }) }?;
        assert_eq!(ranks, [0, 1]);
        Ok(())
    }

    #[test]
    fn beside_runs_concurrently_and_both_ranks_join_first() -> Result<()> {
        let rendezvous = Barrier::new(3);
        // SAFETY: plain integers are unaliased and have no queued device work.
        let (pair, beside) = unsafe { load_pair_beside(|rank| { rendezvous.wait(); Ok(rank) }, || {
            rendezvous.wait();
            7
        }) };
        assert_eq!((pair?, beside), ([0, 1], 7));
        let released = Arc::new(AtomicUsize::new(0));
        struct Owner(Arc<AtomicUsize>);
        impl Drop for Owner {
            fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
        }
        // SAFETY: mock owners hold only an Arc.
        let (pair, ()) = unsafe { load_pair_beside(|rank| {
            if rank == 1 { return Err(anyhow!("injected")); }
            Ok(Owner(released.clone()))
        }, || ()) };
        assert!(pair.is_err());
        assert_eq!(released.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn failure_and_panic_join_and_release_successful_partner() {
        struct Owner(Arc<AtomicUsize>);
        impl Drop for Owner {
            fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
        }
        for failing_rank in 0..2 {
            for panic in [false, true] {
                let released = Arc::new(AtomicUsize::new(0));
                let rendezvous = Barrier::new(2);
                // SAFETY: each mock owner has no device work and owns only an Arc.
                let result = unsafe { load_pair(|rank| {
                    rendezvous.wait();
                    if rank == failing_rank {
                        if panic { panic!("injected rank load panic"); }
                        return Err(anyhow!("injected rank load failure"));
                    }
                    Ok(Owner(released.clone()))
                }) };
                assert!(result.is_err());
                assert_eq!(released.load(Ordering::SeqCst), 1);
            }
        }
    }
}
