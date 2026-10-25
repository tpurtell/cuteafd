//! Opt-in golden-only downloads. No wire format or model arithmetic changes.
use super::*;
use std::path::PathBuf;

pub(crate) struct ReplyTap {
    pub partials: Vec<Vec<u8>>,
    pub shared: Vec<Vec<u8>>,
    pub residual: Vec<u8>,
    pub real_rows: usize,
    pub decode: bool,
}

pub(crate) struct ReplyCapture {
    pub dir: PathBuf,
    pub layers: Vec<usize>,
    pub rows: usize,
    pub max_bytes: usize,
    written: usize,
    next: usize,
}

impl ReplyCapture {
    pub fn new(dir: PathBuf, layers: Vec<usize>, rows: usize, max_bytes: usize) -> Result<Self> {
        ensure!(!layers.is_empty() && rows > 0 && max_bytes > 0, "reply tap needs layers, rows and a byte budget");
        std::fs::create_dir_all(&dir)?;
        ensure!(std::fs::read_dir(&dir)?.next().is_none(), "reply tap directory must be empty");
        Ok(Self { dir, layers, rows, max_bytes, written: 0, next: 0 })
    }

    fn selected_rows(&self, layer: usize, real: usize) -> Option<usize> {
        self.layers.contains(&layer).then_some(real.min(self.rows))
    }

    pub fn save(&mut self, tap: &WidthTap, hidden: usize) -> Result<()> {
        let reply = tap.reply.as_ref().context("missing reply planes")?;
        let bytes = tap.x.len() + tap.routes.len() + tap.delta.len() + reply.residual.len()
            + reply.partials.iter().map(Vec::len).sum::<usize>() + reply.shared.iter().map(Vec::len).sum::<usize>();
        ensure!(bytes <= self.max_bytes.saturating_sub(self.written), "reply tap byte budget exceeded");
        let path = self.dir.join(format!("tap-{:05}-layer{:02}", self.next, tap.layer));
        std::fs::create_dir(&path)?;
        for (name, data) in [("x", &tap.x), ("routes", &tap.routes), ("delta", &tap.delta), ("residual", &reply.residual)] {
            std::fs::write(path.join(format!("{name}.bin")), data)?;
        }
        for (kind, planes) in [("rank", &reply.partials), ("shared", &reply.shared)] {
            for (rank, data) in planes.iter().enumerate() {
                std::fs::write(path.join(format!("{kind}{rank}.bin")), data)?;
            }
        }
        let metadata = serde_json::json!({ "layer": tap.layer, "hidden": hidden,
            "rows": tap.x.len() / (hidden * 2), "real_rows": reply.real_rows,
            "row_selection": "first rows of each wave", "decode": reply.decode,
            "ranks": reply.partials.len(), "shared_ranks": reply.shared.len(),
            "dtype": "bf16_le", "routes_layout": "int32 ids then float32 weights",
            "delta_semantics": "bf16(fp32 rank-ordered sum + shared0); shared1 added separately at join",
            "tensor_bytes": bytes });
        std::fs::write(path.join("meta.json"), serde_json::to_vec_pretty(&metadata)?)?;
        self.written += bytes;
        self.next += 1;
        Ok(())
    }
}

impl GlmEngine<'_> {
    pub(super) fn capture_reply(&self, layer: usize, work: [&Workspace<'_>; 2],
        planes: [*const u16; crate::shared::spark_intake::MAX_INTAKE_RANKS], ranks: usize,
        real: usize, decode: bool) -> Result<Option<ReplyTap>> {
        let rows = self.reply_capture.borrow().as_ref().and_then(|c| c.selected_rows(layer, real));
        let Some(rows) = rows else { return Ok(None); };
        ensure!(ranks > 0 && ranks <= planes.len(), "invalid reply tap rank count");
        let bytes = rows * self.cfg.hidden * 2;
        // SAFETY: moe_land ordered every live intake plane on this stream. The
        // synchronous copies finish before reduce/consumed may recycle the slot.
        unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
        let mut partials = Vec::new();
        for &ptr in &planes[..ranks] {
            ensure!(!ptr.is_null(), "missing reply tap plane");
            let mut data = vec![0u8; bytes];
            self.library.copy_d2h(&mut data, cuteafd_ffi::CuteafdDeviceBuffer {
                ptr: ptr.cast_mut().cast(), bytes, device_id: self.device, flags: 0 })?;
            partials.push(data);
        }
        let mut shared = vec![self.download(&work[0].shared, bytes)?];
        self.on(1, || {
            // SAFETY: rank 1's shared expert has been queued on its owned stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream_of(1))?; }
            let mut data = vec![0u8; bytes];
            self.library.copy_d2h(&mut data, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..work[1].shared.buffer })?;
            shared.push(data);
            Ok(())
        })?;
        Ok(Some(ReplyTap { partials, shared, residual: self.download(&work[0].h, bytes)?, real_rows: real, decode }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reply_tap_selection_and_budget_are_bounded() {
        let mut capture = ReplyCapture { dir: PathBuf::new(), layers: vec![4, 8], rows: 3,
            max_bytes: 1, written: 0, next: 0 };
        assert_eq!(capture.selected_rows(4, 8), Some(3));
        assert_eq!(capture.selected_rows(8, 1), Some(1));
        assert_eq!(capture.selected_rows(5, 8), None);
        let tap = WidthTap { layer: 4, x: vec![0; 2], routes: vec![], delta: vec![0; 2],
            reply: Some(ReplyTap { partials: vec![vec![0; 2]; 6], shared: vec![vec![0; 2]; 2],
                residual: vec![0; 2], real_rows: 1, decode: true }) };
        assert!(capture.save(&tap, 1).unwrap_err().to_string().contains("budget exceeded"));
        assert_eq!(capture.written, 0);
        assert_eq!(capture.next, 0);
    }
}
