//! Sleeping, independent completion waits for a bounded set of copy streams.
use crate::*;
use std::{
    io::{Read, Write},
    os::{fd::{AsRawFd, BorrowedFd}, unix::net::UnixStream},
    sync::mpsc,
    thread::JoinHandle,
};

struct Wait {
    event: *mut c_void,
    tag: u32,
}
// SAFETY: submit's caller retains the recorded event until all workers join;
// CUDA permits event synchronization from a different host thread.
unsafe impl Send for Wait {}

pub struct EventWaiters {
    queues: Vec<mpsc::Sender<Wait>>,
    completed: mpsc::Receiver<(u32, Result<()>)>,
    workers: Vec<JoinHandle<()>>,
    wake: UnixStream,
}

impl EventWaiters {
    pub fn new(library: &NativeLibrary, device: i32, streams: usize) -> Result<Self> {
        anyhow::ensure!(streams > 0, "empty event waiter set");
        let synchronize = library.cuteafd_cuda_event_synchronize()?;
        let set_device = library.cuteafd_cuda_set_device()?;
        let last_error = library.cuteafd_last_error()?;
        let (complete, completed) = mpsc::channel();
        let (wake, notify) = UnixStream::pair()?;
        let mut waiters = Self { queues: Vec::new(), completed, workers: Vec::new(), wake };
        for index in 0..streams {
            let module = Arc::clone(&library.lib);
            let (queue, pending) = mpsc::channel::<Wait>();
            let complete = complete.clone();
            let mut notify = notify.try_clone()?;
            let worker = std::thread::Builder::new()
                .name(format!("push-copy-wait-{device}-{index}"))
                .spawn(move || {
                    let _module = module;
                    while let Ok(wait) = pending.recv() {
                        // SAFETY: the module is retained, output buffer is local,
                        // and submit's caller retains every recorded event.
                        let status = unsafe {
                            let status = set_device(device);
                            if status == 0 { synchronize(wait.event) } else { status }
                        };
                        let result = if status == 0 { Ok(()) } else {
                            let mut text = [0 as c_char; 512];
                            // SAFETY: valid writable buffer and retained function pointer.
                            let message = if unsafe { last_error(text.as_mut_ptr(), text.len()) } == 0 {
                                // Do not rely on an erroneous DSO to terminate its message.
                                text[text.len() - 1] = 0;
                                // SAFETY: text is a live local buffer with its final byte
                                // explicitly NUL-terminated immediately above.
                                unsafe { CStr::from_ptr(text.as_ptr()) }.to_string_lossy().into_owned()
                            } else { "last error unavailable".into() };
                            Err(anyhow::anyhow!("push copy event wait returned status {status}: {message}"))
                        };
                        if complete.send((wait.tag, result)).is_err() { break; }
                        // At most the retained ring's slot count can be pending,
                        // so these single-byte notifications remain bounded.
                        if notify.write_all(&[1]).is_err() { break; }
                    }
                })?;
            waiters.queues.push(queue);
            waiters.workers.push(worker);
        }
        Ok(waiters)
    }

    /// # Safety
    /// The event was recorded on this worker's stream using a blocking event.
    /// Retain it, its copy sources and destinations until completion or join.
    /// Do not record it again before receiving its completion.
    pub unsafe fn submit(&self, stream: usize, event: *mut c_void, tag: u32) -> Result<()> {
        self.queues.get(stream).context("invalid copy waiter index")?
            .send(Wait { event, tag }).map_err(|_| anyhow::anyhow!("copy waiter closed"))
    }

    /// One blocking wake source for socket arrivals and copy completions.
    /// A true result means at least one completed event can be received.
    pub fn wait(&self, socket: Option<BorrowedFd<'_>>) -> Result<bool> {
        let mut fds = [
            libc::pollfd { fd: self.wake.as_raw_fd(), events: libc::POLLIN, revents: 0 },
            libc::pollfd { fd: socket.map_or(-1, |fd| fd.as_raw_fd()), events: libc::POLLIN, revents: 0 },
        ];
        loop {
            // SAFETY: both descriptors remain borrowed/live, the array is writable,
            // and poll retains no pointers. No cadence or timeout is involved.
            if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, -1) } >= 0 {
                return Ok(fds[0].revents != 0);
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted { return Err(error.into()); }
        }
    }

    pub fn complete(&self) -> Result<u32> {
        (&self.wake).read_exact(&mut [0])?;
        let (tag, result) = self.completed.recv().context("copy waiters closed")?;
        result?;
        Ok(tag)
    }

    /// Close submissions and join every waiter, including on cancellation.
    pub fn join(&mut self) -> Result<()> {
        self.queues.clear();
        let mut failed = false;
        for worker in self.workers.drain(..) {
            failed |= worker.join().is_err();
        }
        anyhow::ensure!(!failed, "copy waiter panicked");
        Ok(())
    }
}

impl Drop for EventWaiters {
    fn drop(&mut self) {
        if let Err(error) = self.join() { tracing::error!(%error, "joining copy event waiters"); }
    }
}
