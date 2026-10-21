//! Per-thread coordinator weight-load split: checkpoint reads, synchronous
//! uploads, and the remainder (host transforms, GPU conversions, allocation).
//! Measurement only; it changes no bytes and no ordering.
use std::{cell::Cell, time::Instant};

thread_local! {
    static READ: Cell<(u64, f64)> = const { Cell::new((0, 0.0)) };
    static UPLOAD: Cell<(u64, f64)> = const { Cell::new((0, 0.0)) };
    static QUANTIZE: Cell<(u64, f64)> = const { Cell::new((0, 0.0)) };
}

fn timed<T>(cell: &'static std::thread::LocalKey<Cell<(u64, f64)>>, bytes: usize, body: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let value = body();
    let seconds = started.elapsed().as_secs_f64();
    cell.with(|c| { let (b, s) = c.get(); c.set((b + bytes as u64, s + seconds)); });
    value
}

pub(crate) fn read<T>(bytes: usize, body: impl FnOnce() -> T) -> T { timed(&READ, bytes, body) }
pub(crate) fn upload<T>(bytes: usize, body: impl FnOnce() -> T) -> T { timed(&UPLOAD, bytes, body) }
/// Host BF16 -> FP8 quantization (a part of the transform remainder).
pub(crate) fn quantize<T>(bytes: usize, body: impl FnOnce() -> T) -> T { timed(&QUANTIZE, bytes, body) }

/// Brackets one loader call on this thread and logs its split once.
pub(crate) struct Split { started: Instant, read: (u64, f64), upload: (u64, f64), quantize: (u64, f64) }
impl Split {
    pub fn start() -> Self {
        Self { started: Instant::now(), read: READ.with(Cell::get), upload: UPLOAD.with(Cell::get),
            quantize: QUANTIZE.with(Cell::get) }
    }
    pub fn log(self, label: &str) {
        let elapsed = self.started.elapsed().as_secs_f64();
        let (rb, rs) = READ.with(Cell::get);
        let (ub, us) = UPLOAD.with(Cell::get);
        let (read_bytes, read_seconds) = (rb - self.read.0, rs - self.read.1);
        let (upload_bytes, upload_seconds) = (ub - self.upload.0, us - self.upload.1);
        let (qb, qs) = QUANTIZE.with(Cell::get);
        let (quantize_bytes, quantize_seconds) = (qb - self.quantize.0, qs - self.quantize.1);
        tracing::info!(label, elapsed_seconds = elapsed, read_bytes, read_seconds, upload_bytes, upload_seconds,
            quantize_bytes, quantize_seconds,
            transform_seconds = (elapsed - read_seconds - upload_seconds).max(0.0),
            "coordinator weight load split (transform = host conversion, GPU conversion and allocation)");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn split_counts_only_this_threads_bracketed_work() {
        super::read(7, || ());
        let split = super::Split::start();
        super::read(3, || ());
        super::upload(5, || ());
        std::thread::spawn(|| super::read(100, || ())).join().unwrap();
        let (rb, _) = super::READ.with(std::cell::Cell::get);
        assert_eq!(rb - split.read.0, 3);
        let (ub, _) = super::UPLOAD.with(std::cell::Cell::get);
        assert_eq!(ub - split.upload.0, 5);
        split.log("test");
    }
}
