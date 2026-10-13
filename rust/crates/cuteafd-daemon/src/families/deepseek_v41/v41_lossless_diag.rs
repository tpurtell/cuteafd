//! Private, bounded whole-layer snapshots for one exact strict prompt.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use std::{collections::BTreeSet, path::PathBuf, sync::{Mutex, OnceLock}};
use super::v41_requests::RequestBatch;

struct Config {
    directory: PathBuf,
    prompt: Vec<u32>,
    requests: Mutex<BTreeSet<u64>>,
}
static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
fn config() -> Option<&'static Config> {
    CONFIG.get_or_init(|| {
        let directory = std::env::var_os("CUTEAFD_DIAG_V41_GEOMETRY")?;
        let prompt = serde_json::from_str::<Vec<u32>>(&std::env::var("CUTEAFD_DIAG_V41_PROMPT_IDS").ok()?).ok()?;
        if prompt.is_empty() { return None; }
        Some(Config { directory: directory.into(), prompt, requests: Mutex::new(BTreeSet::new()) })
    }).as_ref()
}
pub(super) fn enabled() -> bool { config().is_some() }
fn matches_prompt(prompt: &[u32], position: u64, tokens: &[u32]) -> bool {
    position == 0 && !prompt.is_empty() && tokens == prompt
}
fn selected_position(prompt_len: usize, position: u64) -> bool {
    prompt_len > 0 && (prompt_len as u64 - 1..prompt_len as u64 + 6).contains(&position)
}
pub(super) fn observe_prompt(id: u64, position: u64, tokens: &[u32]) {
    if let Some(config) = config() {
        if matches_prompt(&config.prompt, position, tokens) {
            config.requests.lock().unwrap().insert(id);
        }
    }
}
#[derive(Clone)]
pub(super) struct Snapshot {
    directory: PathBuf,
    rows: usize,
    selected: Vec<usize>,
}
impl Snapshot {
    pub(super) fn begin(batch: &RequestBatch) -> Result<Option<Self>> {
        let Some(config) = config() else { return Ok(None) };
        let rows = batch.cache()?.expert_rows();
        if rows.len() > 80 { return Ok(None); }
        let ids = config.requests.lock().unwrap();
        let selected: Vec<_> = rows.iter().enumerate().filter_map(|(i, row)|
            (ids.contains(&row.request_id) && row.kind != cuteafd_transport::ExpertV2SourceKind::Prefill && selected_position(config.prompt.len(), row.position)).then_some(i)).collect();
        if selected.is_empty() { return Ok(None); }
        let directory = config.directory.join(format!("batch{}-{:?}-{}rows", batch.cache()?.identity(), batch.cache()?.stage(), rows.len()));
        std::fs::create_dir_all(&directory)?;
        let metadata: Vec<_> = rows.iter().zip(batch.diagnostic_tokens()).enumerate().map(|(i, (row, token))|
            serde_json::json!({"row":i,"request":row.request_id,"position":row.position,"token":token,"kind":format!("{:?}",row.kind),"selected":selected.contains(&i)})).collect();
        std::fs::write(directory.join("rows.json"), serde_json::to_vec_pretty(&metadata)?)?;
        Ok(Some(Self { directory, rows: rows.len(), selected }))
    }
    pub(super) fn dump(&self, library: &NativeLibrary, layer: usize, name: &str,
        mut buffer: CuteafdDeviceBuffer, row_bytes: usize) -> Result<()> {
        ensure!(buffer.bytes >= self.rows * row_bytes, "diagnostic buffer extent differs");
        crate::shared::memory::chain::settle(library)?;
        buffer.bytes = self.rows * row_bytes;
        let mut bytes = vec![0; buffer.bytes];
        library.copy_d2h(&mut bytes, buffer)?;
        for &row in &self.selected {
            let path = self.directory.join(format!("layer{layer:02}-{name}-row{row}.bin"));
            use std::io::Write;
            std::fs::OpenOptions::new().write(true).create_new(true).open(path)
                .context("create private V4.1 snapshot")?.write_all(&bytes[row * row_bytes..(row + 1) * row_bytes])?;
        }
        Ok(())
    }
    pub(super) fn logits(&self, library: &NativeLibrary, buffer: CuteafdDeviceBuffer, selected: &[usize]) -> Result<()> {
        ensure!(!selected.is_empty() && buffer.bytes % selected.len() == 0, "diagnostic logits shape differs");
        crate::shared::memory::chain::settle(library)?;
        let mut bytes = vec![0; buffer.bytes];
        library.copy_d2h(&mut bytes, buffer)?;
        let row_bytes = buffer.bytes / selected.len();
        for (index, &row) in selected.iter().enumerate().filter(|(_, row)| self.selected.contains(row)) {
            std::fs::write(self.directory.join(format!("logits-row{row}.bin")), &bytes[index * row_bytes..(index + 1) * row_bytes])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_prompt_excludes_other_requests_and_continuations() {
        assert!(matches_prompt(&[1, 2, 3], 0, &[1, 2, 3]));
        assert!(!matches_prompt(&[1, 2, 3], 0, &[1, 2]));
        assert!(!matches_prompt(&[1, 2, 3], 0, &[1, 2, 4]));
        assert!(!matches_prompt(&[1, 2, 3], 1, &[1, 2, 3]));
        assert!(!matches_prompt(&[], 0, &[]));
    }
    #[test]
    fn snapshots_are_bounded_to_first_matching_verify_rows() {
        for position in 0..80 {
            assert_eq!(selected_position(52, position), (51..58).contains(&position));
        }
        assert!(!selected_position(0, 0));
    }
}
