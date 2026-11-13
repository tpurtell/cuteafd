//! Independent expert backings share one rank's device prefix and host tail.
use super::*;
use std::ops::Range;

#[derive(Debug)]
struct Span {
    device: Range<u64>,
    host: Range<u64>,
}

#[derive(Debug)]
pub(super) struct SetLayout {
    spans: Vec<Span>,
    device_bytes: u64,
    logical_bytes: u64,
}

fn separated(end: u64) -> Result<u64> {
    end.checked_add(256).map(|n| n / 256 * 256).context("expert-set logical extent overflow")
}

impl SetLayout {
    pub(super) fn new(sizes: impl IntoIterator<Item = (u64, u64)>) -> Result<Self> {
        let mut spans = Vec::new();
        let mut device_bytes = 0u64;
        for (device, logical) in sizes {
            ensure!(device <= logical, "expert-set device prefix exceeds logical extent");
            let start = if spans.is_empty() { 0 } else { separated(device_bytes)? };
            device_bytes = start.checked_add(device).context("expert-set device extent overflow")?;
            spans.push(Span { device: start..device_bytes, host: 0..logical - device });
        }
        let mut logical_bytes = device_bytes;
        for span in &mut spans {
            let bytes = span.host.end - span.host.start;
            if bytes == 0 { span.host = logical_bytes..logical_bytes; continue; }
            let start = if logical_bytes == device_bytes { device_bytes } else { separated(logical_bytes)? };
            logical_bytes = start.checked_add(bytes).context("expert-set host extent overflow")?;
            span.host = start..logical_bytes;
        }
        Ok(Self { spans, device_bytes, logical_bytes })
    }

    pub(super) fn plans(plans: &[CoordinatorExpertPlan]) -> Result<Self> {
        Self::new(plans.iter().map(|p| (p.layout.bytes() as u64, p.logical_bytes())))
    }

    pub(super) fn offsets(&self, index: usize) -> Result<(u64, u64)> {
        let span = self.spans.get(index).context("expert-set index outside rank")?;
        Ok((span.device.start, span.host.start))
    }

    fn locate(&self, offset: u64, bytes: u64) -> Result<(usize, u64)> {
        ensure!(bytes > 0, "empty expert-set fragment");
        let end = offset.checked_add(bytes).context("expert-set fragment overflow")?;
        for (index, span) in self.spans.iter().enumerate() {
            if offset >= span.device.start && end <= span.device.end {
                return Ok((index, offset - span.device.start));
            }
            if offset >= span.host.start && end <= span.host.end {
                return Ok((index, offset - span.host.start + span.device.end - span.device.start));
            }
        }
        anyhow::bail!("fragment crosses expert-set boundary or padding")
    }

    pub(super) fn device_bytes(&self) -> u64 { self.device_bytes }
    pub(super) fn logical_bytes(&self) -> u64 { self.logical_bytes }
}

pub(super) struct Sets<T> {
    sets: Vec<T>,
    layout: SetLayout,
}

impl<T: PushDestination> Sets<T> {
    pub(super) fn new(sets: Vec<T>) -> Result<Self> {
        let layout = SetLayout::new(sets.iter().map(|set| (set.device_bytes() as u64, set.logical_bytes() as u64)))?;
        usize::try_from(layout.logical_bytes()).context("expert-set logical extent exceeds address space")?;
        Ok(Self { sets, layout })
    }
}

impl<T: PushDestination> PushDestination for Sets<T> {
    type Delivered = Vec<T::Delivered>;
    fn belongs_to(&self, device: crate::shared::memory::device::Device<'_>) -> bool {
        self.sets.iter().all(|set| set.belongs_to(device))
    }
    fn logical_bytes(&self) -> usize { self.layout.logical_bytes as usize }
    fn device_bytes(&self) -> usize { self.layout.device_bytes as usize }
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        let (index, local) = self.layout.locate(offset, bytes)?;
        self.sets[index].check_fragment(local, bytes)
    }
    fn destination(&self, offset: usize, bytes: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        let (index, local) = self.layout.locate(offset as u64, bytes as u64)?;
        self.sets[index].destination(usize::try_from(local)?, bytes)
    }
    fn copy_validation(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let (index, local) = self.layout.locate(offset, bytes.len() as u64)?;
        self.sets[index].copy_validation(local, bytes)
    }
    fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> {
        let (index, local) = self.layout.locate(offset, bytes)?;
        self.sets[index].copied(local, bytes)
    }
    fn done(self) -> Result<Self::Delivered> {
        self.sets.into_iter().map(PushDestination::done).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_device_spans_precede_independent_host_tails() -> Result<()> {
        let layout = SetLayout::new([(512, 576), (256, 288), (256, 256)])?;
        assert_eq!(layout.device_bytes(), 1536);
        assert_eq!(layout.logical_bytes(), 1824);
        assert_eq!(layout.offsets(0)?, (0, 1536));
        assert_eq!(layout.offsets(1)?, (768, 1792));
        assert_eq!(layout.offsets(2)?.0, 1280);
        assert_eq!(layout.locate(768, 256)?, (1, 0));
        assert_eq!(layout.locate(1536, 64)?, (0, 512));
        assert_eq!(layout.locate(1792, 32)?, (1, 256));
        for (offset, bytes) in [(511, 2), (512, 1), (1599, 2), (1824, 1)] {
            assert!(layout.locate(offset, bytes).is_err());
        }
        assert!(SetLayout::new([(2, 1)]).is_err());
        assert!(SetLayout::new([(u64::MAX, u64::MAX), (1, 1)]).is_err());
        Ok(())
    }

    #[test]
    fn single_set_keeps_its_existing_offsets() -> Result<()> {
        for (device, logical) in [(256, 256), (256, 4096)] {
            let layout = SetLayout::new([(device, logical)])?;
            assert_eq!(layout.offsets(0)?, (0, device));
            assert_eq!((layout.device_bytes(), layout.logical_bytes()), (device, logical));
        }
        Ok(())
    }
}
