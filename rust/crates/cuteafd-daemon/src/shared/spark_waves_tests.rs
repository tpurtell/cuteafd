//! The transport worker against a fake GPU and a fake transport: wave order,
//! slot/staging ownership hand-off, and failure drain, without hardware.
use super::*;
use crate::shared::prefill_pipeline::{prefill_schedule, sequence_units, FfnRoute, Step};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Events complete when the test (the "GPU") says so; replies arrive a few
/// polls after a post, in post order, one payload per rank.
#[derive(Default)]
struct World {
    /// Completed event handles.
    complete: Mutex<std::collections::HashSet<usize>>,
    /// Events recorded so far, in order.
    recorded: Mutex<Vec<(usize, usize)>>,
    /// Payload owners alive (a held receive slot each).
    live: AtomicUsize,
    /// Uploads queued and not yet passed by a drain.
    queued: AtomicUsize,
    drained: AtomicBool,
}

struct Payload { world: Arc<World>, slot: usize, sequence: u64 }
impl Drop for Payload {
    fn drop(&mut self) { self.world.live.fetch_sub(1, Ordering::SeqCst); }
}

struct Fake {
    world: Arc<World>,
    ranks: usize,
    waves: usize,
    /// Fail the post (`Some(n)`: the n-th post, 1-based) or the poll.
    fail_post: Option<u64>,
    fail_poll: Option<u64>,
    posts: u64,
    /// Per slot: the wave currently in it.
    occupied: Vec<Option<u64>>,
    posted: Vec<u64>,
}

struct FakeWave { sequence: u64, slot: usize, polls: usize, sent: usize }

const STAGED: usize = 1000;
const LANDED: usize = 2000;

impl Backend for Fake {
    type Wave = FakeWave;
    type Payload = Payload;
    fn complete(&self, event: Handle) -> Result<bool> {
        Ok(self.world.complete.lock().unwrap().contains(&(event.raw() as usize)))
    }
    fn staged(&self, slot: usize) -> Handle { Handle::of((STAGED + slot) as *mut c_void) }
    fn post(&mut self, job: Job) -> Result<FakeWave> {
        self.posts += 1;
        if self.fail_post == Some(self.posts) { anyhow::bail!("injected post failure"); }
        assert!(job.slot < self.waves);
        assert!(self.occupied[job.slot].is_none(), "slot {} reposted before its wave landed", job.slot);
        if let Some(previous) = self.posted.last() { assert!(job.sequence > *previous, "posts out of order"); }
        self.posted.push(job.sequence);
        self.occupied[job.slot] = Some(job.sequence);
        Ok(FakeWave { sequence: job.sequence, slot: job.slot, polls: 0, sent: 0 })
    }
    fn poll(&mut self, wave: &mut FakeWave, arrived: &mut Vec<(usize, u32, Payload)>) -> Result<Option<u8>> {
        wave.polls += 1;
        if self.fail_poll == Some(wave.sequence) && wave.sent == 1 { anyhow::bail!("injected receive failure"); }
        if wave.polls % 3 == 0 && wave.sent < self.ranks {
            self.world.live.fetch_add(1, Ordering::SeqCst);
            arrived.push((wave.sent, 0, Payload { world: self.world.clone(), slot: wave.slot, sequence: wave.sequence }));
            wave.sent += 1;
        }
        if wave.sent == self.ranks {
            self.occupied[wave.slot] = None;
            return Ok(Some(0));
        }
        Ok(None)
    }
    fn upload(&self, slot: usize, _rows: usize, rank: usize, _first: u32, payload: &Payload) -> Result<()> {
        assert!(rank < self.ranks && payload.slot == slot);
        self.world.queued.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn landed(&self, slot: usize) -> Result<Handle> {
        let event = LANDED + slot;
        self.world.recorded.lock().unwrap().push((event, self.posted.len()));
        // The copy stream finishes uploads in order; this fake completes them at once.
        self.world.complete.lock().unwrap().insert(event);
        Ok(Handle::of(event as *mut c_void))
    }
    fn drain(&self) -> Result<()> {
        self.world.queued.store(0, Ordering::SeqCst);
        self.world.drained.store(true, Ordering::SeqCst);
        Ok(())
    }
}

fn fake(world: &Arc<World>, waves: usize) -> Fake {
    Fake { world: world.clone(), ranks: 3, waves, fail_post: None, fail_poll: None, posts: 0,
        occupied: vec![None; waves], posted: Vec::new() }
}

fn job(sequence: u64, slot: usize, consumed: Option<usize>) -> Job {
    Job { sequence, slot, rows: 64, layout: RouteStaging::new(64, 8, 16).unwrap(),
        build: Box::new(|_, _| anyhow::bail!("the fake posts without building")),
        consumed: consumed.map(|event| Handle::of(event as *mut c_void)), arrived: Instant::now() }
}

/// The engine side over the fake: drives the executor's schedule, sending
/// jobs and landing them in dispatch order, completing each staged event a
/// little after its send (the GPU lagging the host).
fn drive(waves: usize, units: usize, layers: usize, fail_post: Option<u64>, fail_poll: Option<u64>)
    -> (Result<Vec<u64>>, Arc<World>, Retired<Payload>) {
    let world = Arc::new(World::default());
    let mut backend = fake(&world, waves);
    backend.fail_post = fail_post;
    backend.fail_poll = fail_poll;
    let routes: Vec<_> = (0..layers).map(|layer| if layer == 0 { FfnRoute::Local } else { FfnRoute::Spark }).collect();
    let units = sequence_units(units * 256, 256, waves, layers).unwrap();
    let steps = prefill_schedule(&routes, &units, waves).unwrap();
    let (jobs, inbox) = mpsc::channel();
    let (report, done) = mpsc::channel();
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| run_worker(&mut backend, inbox, report, Duration::from_secs(5)));
        let engine = (|| -> Result<Vec<u64>> {
            let mut sequence = 0u64;
            let mut outstanding = VecDeque::new();
            let mut landed = Vec::new();
            let mut slot_of_unit = vec![0; units.len()];
            let mut next_slot = 0;
            let mut unread = vec![false; waves];
            for step in steps {
                match step {
                    Step::Send { unit, .. } => {
                        assert!(!unread[next_slot], "slot reused before its reduce");
                        sequence += 1;
                        let slot = next_slot;
                        next_slot = (next_slot + 1) % waves;
                        slot_of_unit[unit] = slot;
                        // A slot read before carries its consumed event (completed below).
                        let consumed = (sequence > waves as u64).then_some(3000 + slot);
                        jobs.send(job(sequence, slot, consumed))
                            .map_err(|_| anyhow::anyhow!("worker gone"))?;
                        world.complete.lock().unwrap().insert(STAGED + slot);
                        outstanding.push_back((sequence, slot));
                    }
                    Step::Land { .. } => {
                        let (expected, slot) = outstanding.pop_front().unwrap();
                        let landed_wave: Landed = done.recv_timeout(Duration::from_secs(5))
                            .map_err(|_| anyhow::anyhow!("no completion"))??;
                        assert_eq!(landed_wave.sequence, expected, "waves landed out of dispatch order");
                        landed.push(landed_wave.sequence);
                        // The reduce reads and releases the slot.
                        unread[slot] = false;
                        world.complete.lock().unwrap().remove(&(STAGED + slot));
                        world.complete.lock().unwrap().insert(3000 + slot);
                    }
                    _ => {}
                }
            }
            Ok(landed)
        })();
        drop(jobs);
        let retired = worker.join().unwrap();
        (engine, world, retired)
    })
}

#[test]
fn waves_post_and_land_in_dispatch_order_across_slots() {
    for waves in 1..=4 {
        let (landed, world, retired) = drive(waves, 6, 7, None, None);
        let landed = landed.unwrap();
        assert_eq!(landed, (1..=landed.len() as u64).collect::<Vec<_>>());
        assert!(retired.error.is_none() && !retired.abandoned && retired.drained);
        assert_eq!(world.queued.load(Ordering::SeqCst), 0, "uploads not drained");
        // Only receive owners of waves whose `landed` events the worker had
        // not yet observed may remain, and they are handed back.
        assert_eq!(world.live.load(Ordering::SeqCst), retired.held.len());
        drop(retired);
        assert_eq!(world.live.load(Ordering::SeqCst), 0, "receive slots leaked");
    }
}

#[test]
fn a_job_waits_for_its_staged_copies_and_its_slots_previous_reduce() {
    let world = Arc::new(World::default());
    let mut backend = fake(&world, 2);
    let (jobs, inbox) = mpsc::channel();
    let (report, done) = mpsc::channel::<Result<Landed>>();
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| run_worker(&mut backend, inbox, report, Duration::from_secs(5)));
        jobs.send(job(1, 0, Some(3000))).unwrap();
        // Neither staged nor consumed yet: nothing may be posted or landed.
        std::thread::sleep(Duration::from_millis(30));
        assert!(done.try_recv().is_err());
        world.complete.lock().unwrap().insert(STAGED);
        std::thread::sleep(Duration::from_millis(30));
        assert!(done.try_recv().is_err(), "posted before the slot's previous reduce completed");
        world.complete.lock().unwrap().insert(3000);
        let first = done.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        assert_eq!(first.sequence, 1);
        drop(jobs);
        let retired = worker.join().unwrap();
        assert!(retired.error.is_none() && !retired.abandoned);
    });
}

#[test]
fn a_failure_mid_wave_drains_uploads_and_returns_every_receive_slot() {
    for (fail_post, fail_poll) in [(Some(3), None), (None, Some(2)), (Some(1), None), (None, Some(5))] {
        let (result, world, retired) = drive(2, 4, 5, fail_post, fail_poll);
        assert!(result.is_err(), "the engine saw no failure ({fail_post:?}, {fail_poll:?})");
        assert!(retired.error.is_some());
        assert!(world.drained.load(Ordering::SeqCst), "uploads not drained before owners returned");
        assert_eq!(world.queued.load(Ordering::SeqCst), 0);
        // Every arrived payload is in `held` (none dropped while its upload
        // could still be queued), and none outlives the hand-back.
        assert_eq!(world.live.load(Ordering::SeqCst), retired.held.len());
        drop(retired);
        assert_eq!(world.live.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn closing_with_waves_in_flight_abandons_them_without_deadlock() {
    let world = Arc::new(World::default());
    let mut backend = fake(&world, 2);
    let (jobs, inbox) = mpsc::channel();
    let (report, _done) = mpsc::channel::<Result<Landed>>();
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| run_worker(&mut backend, inbox, report, Duration::from_secs(5)));
        jobs.send(job(1, 0, None)).unwrap();
        jobs.send(job(2, 1, None)).unwrap();
        // The second wave's copies never complete: the engine errored before.
        world.complete.lock().unwrap().insert(STAGED);
        std::thread::sleep(Duration::from_millis(10));
        drop(jobs);
        let retired = worker.join().unwrap();
        assert!(retired.abandoned && retired.error.is_none() && retired.drained);
        drop(retired);
        assert_eq!(world.live.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn a_stalled_gpu_times_out_instead_of_hanging() {
    let world = Arc::new(World::default());
    let mut backend = fake(&world, 1);
    let (jobs, inbox) = mpsc::channel();
    let (report, done) = mpsc::channel::<Result<Landed>>();
    let retired = std::thread::scope(|scope| {
        let worker = scope.spawn(|| run_worker(&mut backend, inbox, report, Duration::from_millis(50)));
        jobs.send(job(1, 0, None)).unwrap();
        let error = done.recv_timeout(Duration::from_secs(5)).unwrap().err().expect("timeout reported");
        assert!(format!("{error:#}").contains("no progress"), "{error:#}");
        drop(jobs);
        worker.join().unwrap()
    });
    assert!(retired.error.is_some() && retired.abandoned && retired.drained);
}
