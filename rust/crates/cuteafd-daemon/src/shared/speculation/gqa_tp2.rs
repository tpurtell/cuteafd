//! Shared per-rank lifetime and BF16 reduction handshake for GQA draft bodies.
use crate::shared::{memory::{device::{DeviceOwner,Device},DeviceAllocation,Tag},peer_split::{PeerExchange,DIRECT}};
use anyhow::{ensure,Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::rc::Rc;

/// The peer body drops before its stream/exchange and both reduction backings.
/// Body-specific projections, nonlinearities and context caches stay in `D`.
pub(crate) struct GqaHeadSplit<'a,D> {
    pub peer:DeviceOwner<'a,Box<D>>,
    pub exchange:Rc<PeerExchange<'a>>,
    pub lead:usize,
    pub reduced:DeviceAllocation<'a>,
    pub peer_reduced:DeviceOwner<'a,DeviceAllocation<'a>>,
}

impl<'a,D> GqaHeadSplit<'a,D> {
    pub fn new(peer:DeviceOwner<'a,Box<D>>,exchange:Rc<PeerExchange<'a>>,lead:usize,bytes:usize,
        tags:[Tag;2])->Result<Self> {
        ensure!(lead<2 && bytes>0 && peer.device.id==exchange.device(1-lead),"invalid GQA TP2 physical owners");
        let library=peer.device.library;
        let lead_device=Device { library,id:exchange.device(lead) };
        let reduced=lead_device.run(||DeviceAllocation::new(library,bytes,tags[0]))?;
        let peer_reduced=peer.device.own(||DeviceAllocation::new(library,bytes,tags[1]))?;
        Ok(Self { peer,exchange,lead,reduced,peer_reduced })
    }

    /// Reuse admitted per-rank reduction buffers owned by a typed draft recipe (e.g. ML4 TP2).
    pub fn from_buffers(peer:DeviceOwner<'a,Box<D>>,exchange:Rc<PeerExchange<'a>>,lead:usize,
        reduced:DeviceAllocation<'a>,peer_reduced:DeviceOwner<'a,DeviceAllocation<'a>>)->Result<Self> {
        ensure!(lead<2 && peer.device.id==exchange.device(1-lead)
            && reduced.buffer.device_id==exchange.device(lead)
            && peer_reduced.device.id==exchange.device(1-lead)
            && peer_reduced.buffer.device_id==exchange.device(1-lead)
            && reduced.buffer.bytes>0 && reduced.buffer.bytes==peer_reduced.buffer.bytes,
            "invalid TP2 head-split admitted reduction buffers");
        Ok(Self { peer,exchange,lead,reduced,peer_reduced })
    }

    /// Exchange complete F32 partials and deliver them in physical rank order
    /// to a local reducer. Both receivers acknowledge before the slot is reused.
    pub fn reduce_f32(&self, partials: [CuteafdDeviceBuffer; 2], elements: usize,
        mut reduce: impl FnMut(usize, [CuteafdDeviceBuffer; 2]) -> Result<()>) -> Result<()> {
        let exchange = &self.exchange;
        let bytes = elements.checked_mul(4).ok_or_else(|| anyhow::anyhow!("draft F32 reduction extent overflow"))?;
        ensure!(bytes > 0 && bytes <= exchange.slot_bytes() && partials.iter().enumerate().all(|(index, partial)|
            !partial.ptr.is_null() && partial.device_id == exchange.device(index ^ self.lead) && partial.bytes >= bytes),
            "GQA F32 TP2 reduction exceeds retained owners");
        for (index, partial) in partials.iter().enumerate() {
            exchange.push(index ^ self.lead, 0, partial.ptr, bytes)?;
        }
        for (index, &local) in partials.iter().enumerate() {
            let rank = index ^ self.lead;
            exchange.wait(rank, 0)?;
            let remote = CuteafdDeviceBuffer { ptr: exchange.recv(rank, 0)?, bytes, ..local };
            let ordered = if rank == 0 { [local, remote] } else { [remote, local] };
            exchange.on(rank, || reduce(index, ordered))?;
        }
        for rank in 0..2 { exchange.signal(rank, DIRECT)?; }
        for rank in 0..2 { exchange.wait(rank, DIRECT)?; }
        Ok(())
    }

    /// Same push/add/ack ordering as the existing DFlash TP2 path. Each caller
    /// retains these local partials through both waits; no host payload staging.
    pub fn reduce(&self,partials:[CuteafdDeviceBuffer;2],elements:usize)->Result<()> {
        let exchange=&self.exchange;
        let window=exchange.slot_bytes()/2;
        ensure!(window>0,"empty draft peer exchange");
        let bytes=elements.checked_mul(2).ok_or_else(||anyhow::anyhow!("draft reduction extent overflow"))?;
        ensure!(elements>0 && partials.iter().enumerate().all(|(rank,p)|!p.ptr.is_null()
            && p.device_id==exchange.device(rank^self.lead) && p.bytes>=bytes)
            && self.reduced.buffer.bytes>=bytes && self.peer_reduced.buffer.bytes>=bytes,
            "GQA TP2 reduction exceeds retained owners");
        for first in (0..elements).step_by(window) {
            let count=window.min(elements-first);
            for (rank,partial) in partials.iter().enumerate() {
                exchange.push(rank^self.lead,0,partial.ptr.wrapping_byte_add(first*2),count*2)?;
            }
            for (rank,partial) in partials.iter().enumerate() {
                let reduced=if rank==0 { &self.reduced } else { &*self.peer_reduced };
                exchange.wait(rank^self.lead,0)?;
                exchange.add(rank^self.lead,partial.ptr.wrapping_byte_add(first*2),
                    exchange.recv(rank^self.lead,0)?,reduced.buffer.ptr.wrapping_byte_add(first*2),count)?;
            }
            for rank in 0..2 { exchange.signal(rank^self.lead,DIRECT)?; }
            for rank in 0..2 { exchange.wait(rank^self.lead,DIRECT)?; }
        }
        Ok(())
    }
}
