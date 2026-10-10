//! Registered snapshot allocations map UVA ranges to their owning copy streams.
//! A host-cache coalescer may join adjacent allocations, even across devices;
//! split those ranges back at registration boundaries before CUDA submission.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_hostcache::{copy::DeviceRange, pool::HostRange};

pub(crate) struct Regions {
    buffers: Vec<CuteafdDeviceBuffer>,
    pub devices: Vec<i32>,
}

impl Regions {
    pub fn new(buffers: &[CuteafdDeviceBuffer]) -> Result<Self> {
        ensure!(!buffers.is_empty(), "snapshot copy registration is empty");
        let mut buffers = buffers.to_vec();
        buffers.sort_unstable_by_key(|buffer| buffer.ptr as u64);
        let mut previous_end = 0;
        let mut devices = Vec::new();
        for buffer in &buffers {
            let start = buffer.ptr as u64;
            ensure!(start != 0 && buffer.bytes > 0 && buffer.device_id >= 0,
                "invalid snapshot allocation registration");
            let end = start.checked_add(buffer.bytes as u64).context("snapshot allocation address overflow")?;
            ensure!(start >= previous_end, "overlapping snapshot allocation registrations");
            previous_end = end;
            if !devices.contains(&buffer.device_id) { devices.push(buffer.device_id); }
        }
        devices.sort_unstable();
        Ok(Self { buffers, devices })
    }

    /// Append a complete copy, partitioned by allocation ownership. No CUDA work
    /// is submitted until every copy in the enclosing batch has validated.
    pub fn route(&self, host: HostRange, device: DeviceRange,
        out: &mut Vec<(usize, HostRange, CuteafdDeviceBuffer)>) -> Result<()> {
        ensure!(host.bytes == device.bytes, "copy length mismatch");
        let end = device.addr.checked_add(device.bytes as u64).context("snapshot copy address overflow")?;
        host.offset.checked_add(host.bytes).context("snapshot host offset overflow")?;
        let mut address = device.addr;
        while address < end {
            let index = self.buffers.partition_point(|buffer| buffer.ptr as u64 <= address);
            let buffer = index.checked_sub(1).and_then(|index| self.buffers.get(index))
                .context("snapshot copy address is not registered")?;
            let limit = buffer.ptr as u64 + buffer.bytes as u64;
            ensure!(address < limit, "snapshot copy crosses an unregistered address gap");
            let bytes = (end.min(limit) - address) as usize;
            let rank = self.devices.binary_search(&buffer.device_id).expect("registered device");
            out.push((rank, HostRange { chunk: host.chunk,
                offset: host.offset + (address - device.addr) as usize, bytes },
                CuteafdDeviceBuffer { ptr: address as *mut std::ffi::c_void, bytes, ..*buffer }));
            address += bytes as u64;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(addr: u64, bytes: usize, device_id: i32) -> CuteafdDeviceBuffer {
        CuteafdDeviceBuffer { ptr: addr as *mut std::ffi::c_void, bytes, device_id, flags: 0 }
    }

    #[test]
    fn coalesced_cross_device_copy_keeps_exact_host_offsets() -> Result<()> {
        let regions = Regions::new(&[buffer(2048, 1024, 1), buffer(1024, 1024, 0)])?;
        let mut routed = Vec::new();
        regions.route(HostRange { chunk: 7, offset: 96, bytes: 1024 },
            DeviceRange { addr: 1536, bytes: 1024 }, &mut routed)?;
        assert_eq!(routed.len(), 2);
        for (part, (rank, host, device)) in routed.iter().enumerate() {
            assert_eq!(*rank, part);
            assert_eq!(host.chunk, 7);
            assert_eq!(host.offset, 96 + 512 * part);
            assert_eq!(host.bytes, 512);
            assert_eq!(device.ptr as u64, 1536 + 512 * part as u64);
            assert_eq!(device.bytes, 512);
            assert_eq!(device.device_id, part as i32);
        }
        Ok(())
    }

    #[test]
    fn gaps_overflow_and_overlapping_registrations_are_rejected() -> Result<()> {
        assert!(Regions::new(&[buffer(1024, 1024, 0), buffer(1536, 1024, 1)]).is_err());
        let regions = Regions::new(&[buffer(1024, 512, 0), buffer(2048, 512, 1)])?;
        for (addr, bytes) in [(0, 1), (1280, 1024), (u64::MAX - 8, 16)] {
            let mut routed = Vec::new();
            assert!(regions.route(HostRange { chunk: 0, offset: 0, bytes },
                DeviceRange { addr, bytes }, &mut routed).is_err());
        }
        Ok(())
    }
}
