//! Opt-in drafter spans, read only after an existing stream-completion barrier.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use std::{cell::RefCell, ffi::c_void, io::Write, time::Instant};

struct Retained<'a> {
    group: &'static str,
    sample: Sample,
    events: Events<'a>,
}
pub(crate) struct RetainedSpan(usize);

struct Events<'a> {
    library: &'a NativeLibrary,
    raw: [*mut c_void; 2],
}
impl<'a> Events<'a> {
    fn new(library: &'a NativeLibrary) -> Result<Self> {
        let first = library.cuda_event_create()?;
        match library.cuda_event_create() {
            Ok(last) => Ok(Self {
                library,
                raw: [first, last],
            }),
            Err(error) => {
                // SAFETY: the newly created event has never been recorded.
                unsafe {
                    library.cuda_event_destroy(first)?;
                }
                Err(error.into())
            }
        }
    }
}
impl Drop for Events<'_> {
    fn drop(&mut self) {
        for event in self.raw {
            // SAFETY: the owning drafter drains its stream before destruction.
            if let Err(error) = unsafe { self.library.cuda_event_destroy(event) } {
                tracing::error!(%error, "destroying draft timing event");
            }
        }
    }
}

pub(crate) struct Span {
    index: usize,
    started: Instant,
}
struct Sample {
    stage: &'static str,
    layer: Option<usize>,
    rows: usize,
    host_launch_us: Option<f64>,
}
struct State<'a> {
    library: &'a NativeLibrary,
    events: Vec<Events<'a>>,
    pending: Vec<Sample>,
    retained: Vec<Retained<'a>>,
    host: String,
    device: i32,
    family: &'static str,
    output: Option<std::io::BufWriter<std::fs::File>>,
    directory: Option<std::path::PathBuf>,
    dumped: std::collections::BTreeMap<usize, usize>,
}

fn enabled(value: Option<&str>) -> bool {
    value == Some("1")
}
fn safe_name(value: &str) -> String {
    let name: String = value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if name.is_empty() {
        "unknown".into()
    } else {
        name
    }
}

fn dump_round(dumped: &std::collections::BTreeMap<usize, usize>, rows: usize) -> Option<usize> {
    if !(1..=128).contains(&rows) || (!dumped.contains_key(&rows) && dumped.len() >= 4) {
        return None;
    }
    let round = dumped.get(&rows).copied().unwrap_or(0);
    (round < 3).then_some(round)
}

pub(crate) struct DraftTiming<'a>(Option<RefCell<State<'a>>>);
impl<'a> DraftTiming<'a> {
    pub(crate) fn from_env(library: &'a NativeLibrary, family: &'static str) -> Result<Self> {
        if !enabled(std::env::var("CUTEAFD_DRAFT_TIMING").ok().as_deref()) {
            return Ok(Self(None));
        }
        let host = safe_name(
            &std::env::var("CUTEAFD_PROFILE_HOST")
                .or_else(|_| std::env::var("HOSTNAME"))
                .unwrap_or_else(|_| "unknown".into()),
        );
        let device = library.cuda_get_device()?;
        let directory = std::env::var_os("CUTEAFD_PROFILE_DIR").map(std::path::PathBuf::from);
        let output = directory
            .as_ref()
            .map(|directory| {
                let path =
                    directory.join(format!("draft-timing-{host}-{family}-gpu{device}.jsonl"));
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map(std::io::BufWriter::new)
            })
            .transpose()?;
        Ok(Self(Some(RefCell::new(State {
            library,
            events: Vec::new(),
            pending: Vec::new(),
            retained: Vec::new(),
            host,
            device,
            family,
            output,
            directory,
            dumped: std::collections::BTreeMap::new(),
        }))))
    }

    /// The caller retains this owner and stream until publish_completed.
    pub(crate) fn begin(
        &self,
        stage: &'static str,
        layer: Option<usize>,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<Option<Span>> {
        let Some(state) = &self.0 else {
            return Ok(None);
        };
        let mut state = state.borrow_mut();
        let index = state.pending.len();
        // Bound a diagnostic-only wave even if a caller never reaches completion.
        ensure!(
            index < 4096,
            "draft timing wave exceeds 4096 spans without completion"
        );
        if index == state.events.len() {
            let events = Events::new(state.library)?;
            state.events.push(events);
        }
        let events = &state.events[index];
        // SAFETY: the drafter retains both events and its live stream.
        unsafe {
            events.library.cuda_event_record(events.raw[0], stream)?;
        }
        state.pending.push(Sample {
            stage,
            layer,
            rows,
            host_launch_us: None,
        });
        Ok(Some(Span {
            index,
            started: Instant::now(),
        }))
    }

    pub(crate) fn end(&self, span: Option<Span>, stream: *mut c_void) -> Result<()> {
        let Some(span) = span else { return Ok(()) };
        let host_launch_us = span.started.elapsed().as_secs_f64() * 1e6;
        let mut state = self.0.as_ref().expect("enabled timing span").borrow_mut();
        let events = &state.events[span.index];
        // SAFETY: begin retained these events on this same stream.
        unsafe {
            events.library.cuda_event_record(events.raw[1], stream)?;
        }
        state.pending[span.index].host_launch_us = Some(host_launch_us);
        Ok(())
    }

    /// Retained markers are re-recorded by each graph replay. The owner must
    /// destroy every graph before dropping timing, and serialize their replays.
    pub(crate) fn begin_retained(
        &self,
        group: &'static str,
        stage: &'static str,
        layer: Option<usize>,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<Option<RetainedSpan>> {
        let Some(state) = &self.0 else {
            return Ok(None);
        };
        let mut state = state.borrow_mut();
        let index = if let Some(index) = state.retained.iter().position(|entry| {
            entry.group == group
                && entry.sample.stage == stage
                && entry.sample.layer == layer
                && entry.sample.rows == rows
        }) {
            index
        } else {
            ensure!(
                state.retained.len() < 4096,
                "draft timing exceeds 4096 retained spans"
            );
            let events = Events::new(state.library)?;
            state.retained.push(Retained {
                group,
                events,
                sample: Sample {
                    stage,
                    layer,
                    rows,
                    host_launch_us: None,
                },
            });
            state.retained.len() - 1
        };
        let events = &state.retained[index].events;
        // SAFETY: the graph owner pins timing events through drained teardown.
        unsafe {
            events
                .library
                .cuda_event_record_external(events.raw[0], stream)?;
        }
        Ok(Some(RetainedSpan(index)))
    }

    pub(crate) fn end_retained(
        &self,
        span: Option<RetainedSpan>,
        stream: *mut c_void,
    ) -> Result<()> {
        let Some(RetainedSpan(index)) = span else {
            return Ok(());
        };
        let state = self.0.as_ref().expect("enabled retained timing").borrow();
        let events = &state.retained[index].events;
        // SAFETY: retained markers share the owner's serialized stream and graph lifetime.
        unsafe {
            events
                .library
                .cuda_event_record_external(events.raw[1], stream)?;
        }
        Ok(())
    }

    /// Publish only the group that actually ran, after the owner's existing drain.
    /// Host capture time is not execution time, so these graph-internal samples
    /// intentionally leave host_launch_us null; outer spans time graph submission.
    pub(crate) fn publish_retained(&self, group: &'static str, rows: usize) -> Result<()> {
        let Some(state) = &self.0 else { return Ok(()) };
        let mut state = state.borrow_mut();
        for index in 0..state.retained.len() {
            let entry = &state.retained[index];
            if entry.group != group || entry.sample.rows != rows {
                continue;
            }
            // SAFETY: the caller completed the latest replay of this retained group.
            let device_us = unsafe {
                entry
                    .events
                    .library
                    .cuda_event_elapsed_ms(entry.events.raw[0], entry.events.raw[1])?
            } * 1000.0;
            let record = serde_json::json!({ "event": "draft_compute", "host": state.host,
                "family": state.family, "device": state.device, "stage": entry.sample.stage,
                "layer": entry.sample.layer, "rows": rows, "device_us": device_us,
                "host_launch_us": null, "retained": true });
            if let Some(output) = state.output.as_mut() {
                serde_json::to_writer(&mut *output, &record)?;
                output.write_all(b"\n")?;
            } else {
                tracing::debug!(target: "cuteafd::draft_timing", %record, "drafter graph compute");
            }
        }
        if let Some(output) = &mut state.output {
            output.flush()?;
        }
        Ok(())
    }

    /// No new GPU wait: the caller has already completed this stream, including
    /// every earlier context wave. Event storage is reused only after this drain.
    pub(crate) fn publish_completed(&self) -> Result<()> {
        let Some(state) = &self.0 else { return Ok(()) };
        let mut state = state.borrow_mut();
        for index in 0..state.pending.len() {
            let sample = &state.pending[index];
            let Some(host_launch_us) = sample.host_launch_us else {
                continue;
            };
            let events = &state.events[index];
            // SAFETY: the existing drafter barrier completed both event records.
            let device_us = unsafe {
                events
                    .library
                    .cuda_event_elapsed_ms(events.raw[0], events.raw[1])?
            } * 1000.0;
            let record = serde_json::json!({ "event": "draft_compute", "host": state.host,
                "family": state.family, "device": state.device, "stage": sample.stage,
                "layer": sample.layer, "rows": sample.rows, "device_us": device_us,
                "host_launch_us": host_launch_us });
            if let Some(output) = state.output.as_mut() {
                serde_json::to_writer(&mut *output, &record)?;
                output.write_all(b"\n")?;
            } else {
                tracing::debug!(target: "cuteafd::draft_timing", %record, "drafter compute");
            }
        }
        if let Some(output) = &mut state.output {
            output.flush()?;
        }
        state.pending.clear();
        Ok(())
    }

    /// Diagnostic input only: call after the existing drain while final-norm
    /// storage is still pinned. Limit both shapes and rounds to keep dumps small.
    pub(crate) fn dump_head_input(
        &self,
        buffer: cuteafd_ffi::CuteafdDeviceBuffer,
        rows: usize,
        width: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let Some(state) = &self.0 else { return Ok(()) };
        let mut state = state.borrow_mut();
        let Some(directory) = &state.directory else {
            return Ok(());
        };
        let Some(round) = dump_round(&state.dumped, rows) else {
            return Ok(());
        };
        let bytes = rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(2))
            .context("draft timing head input extent overflow")?;
        ensure!(
            bytes <= buffer.bytes,
            "draft timing head input exceeds admitted storage"
        );
        let path = directory.join(format!(
            "draft-head-{}-{}-gpu{}-rows{rows}-round{round}.bf16",
            state.host, state.family, state.device
        ));
        let mut input = vec![0u8; bytes];
        // SAFETY: `stream` produced `buffer` on this device and remains live.
        unsafe {
            state.library.copy_d2h(
                &mut input,
                cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..buffer },
                stream,
            )?;
        }
        std::fs::write(path, input)?;
        state.dumped.insert(rows, round + 1);
        Ok(())
    }

    pub(crate) fn host_span(&self) -> Option<Instant> {
        self.0.as_ref().map(|_| Instant::now())
    }
    pub(crate) fn publish_host(
        &self,
        stage: &'static str,
        rows: usize,
        started: Option<Instant>,
    ) -> Result<()> {
        let Some(started) = started else {
            return Ok(());
        };
        let host_us = started.elapsed().as_secs_f64() * 1e6;
        let mut state = self.0.as_ref().expect("enabled host timing").borrow_mut();
        let record = serde_json::json!({ "event": "draft_host", "host": state.host,
            "family": state.family, "device": state.device, "stage": stage, "rows": rows,
            "host_us": host_us });
        if let Some(output) = &mut state.output {
            serde_json::to_writer(&mut *output, &record)?;
            output.write_all(b"\n")?;
            output.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn diagnostic_inputs_bound_shapes_rows_and_rounds() {
        let mut dumped = std::collections::BTreeMap::new();
        assert_eq!(super::dump_round(&dumped, 0), None);
        assert_eq!(super::dump_round(&dumped, 129), None);
        for rows in [8, 16, 64, 128] {
            for round in 0..3 {
                assert_eq!(super::dump_round(&dumped, rows), Some(round));
                dumped.insert(rows, round + 1);
            }
            assert_eq!(super::dump_round(&dumped, rows), None);
        }
        assert_eq!(super::dump_round(&dumped, 32), None);
    }

    #[test]
    fn timing_is_explicit_and_file_names_stay_local() {
        assert!(super::enabled(Some("1")));
        for value in [None, Some(""), Some("0"), Some("true")] {
            assert!(!super::enabled(value));
        }
        assert_eq!(super::safe_name("../host/a"), "___host_a");
        assert_eq!(super::safe_name(""), "unknown");
    }
}
