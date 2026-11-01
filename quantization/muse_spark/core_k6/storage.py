"""Audited projection/phase stores and Muse's pre-projected routed-input spool."""
from __future__ import annotations
import os
import hashlib
from pathlib import Path
import numpy as np
import torch
from .codec import covariance_accumulator, ACCUMULATION_CONTRACT
from glm5.core_k6.storage import ProjectionStore, RouteSpool as SharedRouteSpool, load_tensors
from kimi_k3.core_k6.storage import StageStore
from kimi_k3.uniform_k2.storage import atomic_json,sha256_file,file_identity,fsync_directory


class RouteSpool(SharedRouteSpool):
    @classmethod
    def create(cls,root,identity,tokens,batches,progress,layer,minimum=1024):
        root=Path(root)
        if (root/'manifest.json').exists():
            return cls(root,identity,tokens,width=4096,experts=256,minimum=minimum)
        root.mkdir(parents=True,exist_ok=True)
        maps=dict(inputs=np.memmap(root/'inputs.bf16',mode='w+',dtype=np.uint16,shape=(tokens,4096)),
            routes=np.memmap(root/'routes.i16',mode='w+',dtype=np.int16,shape=(tokens,8)),
            candidates=np.memmap(root/'candidates.i16',mode='w+',dtype=np.int16,shape=(8,tokens)))
        offset=0
        try:
            for index,state in batches:
                hidden,selected,near=state['input'],state['selected'],state['near']
                n=len(hidden)
                if (hidden.shape!=(n,4096) or hidden.dtype!=torch.bfloat16 or not torch.isfinite(hidden).all()
                        or selected.shape!=(n,8) or near.shape!=(n,8)
                        or selected.dtype!=torch.int64 or near.dtype!=torch.int64
                        or (selected<0).any() or (selected>=256).any()
                        or (near<0).any() or (near>=256).any()
                        or (selected[:,:,None]==near[:,None,:]).any()
                        or selected.sort(dim=1).values.diff(dim=1).eq(0).any()
                        or near.sort(dim=1).values.diff(dim=1).eq(0).any()):
                    raise ValueError('invalid Muse routed observations')
                maps['inputs'][offset:offset+n]=hidden.cpu().contiguous().view(__import__('torch').uint16).numpy()
                maps['routes'][offset:offset+n]=selected.cpu().short().numpy()
                maps['candidates'][:,offset:offset+n]=near.cpu().short().numpy().T
                offset+=n
                if index%64==0:
                    progress.emit('route_capture_progress',layer=layer,batches=index+1)
            if offset!=tokens:
                raise ValueError('routed spool observation count differs')
            for array in maps.values():
                array.flush();array._mmap.close()
            maps.clear()
            files={}
            for name in ('inputs.bf16','routes.i16','candidates.i16'):
                with (root/name).open('rb') as f:os.fsync(f.fileno())
                files[name]=dict(sha256=sha256_file(root/name),file_identity=file_identity(root/name))
            fsync_directory(root)
            atomic_json(root/'manifest.json',dict(identity=identity,tokens=tokens,width=4096,experts=256,files=files))
        finally:
            for array in maps.values():array._mmap.close()
        return cls(root,identity,tokens,width=4096,experts=256,minimum=minimum)

    @torch.inference_mode()
    def hessian(self, expert, down=None, device='cuda:0', chunk=2048):
        # Keep activation generation at its native 2048-row chunks. Partition
        # only the FP32 covariance products, never the gate/up GEMMs.
        fused=None if down is None else (torch.cat(down,dim=0) if isinstance(down,tuple) else down)
        if fused is not None and (fused.ndim!=2 or fused.shape[0]%2 or fused.shape[1]!=self.width):
            raise ValueError('down capture requires native fused gate/up geometry')
        width=self.width if fused is None else fused.shape[0]//2
        state=covariance_accumulator(width,device)
        natural,recovery=self.route_rows[expert],self.recovery_rows[expert]
        for selection in (natural,recovery):
            for start in range(0,len(selection),chunk):
                x=self.gather(selection[start:start+chunk],device)
                if fused is not None:
                    gate,up=torch.nn.functional.linear(x,fused).chunk(2,dim=-1)
                    x=torch.nn.functional.silu(gate)*up
                state.add(x)
                del x
        if state.count != len(natural)+len(recovery):
            raise ValueError('routed covariance count differs')
        H=state.snapshot()
        residual=max(0,self.minimum-len(natural)-len(recovery))
        H.diagonal().add_(float(residual))
        evidence=dict(natural_rows=len(natural),augmented_rows=len(recovery),identity_rows=residual,
            effective_rows=len(natural)+len(recovery)+residual,ranks=[9,16],
            selection_policy='rank-major-then-original-corpus-row',
            recovery_selection_sha256=hashlib.sha256(recovery.tobytes()).hexdigest(),
            down_inputs=None if down is None else 'serialized-reconstructed-fused-gate-up-native-swiglu-bf16',
            identity_contract='raw-XTX-missing-times-I-equivalent-to-normalized-2I',
            accumulation_contract=ACCUMULATION_CONTRACT)
        return dict(H=H,count=evidence['effective_rows'],finalized=False,identity_rows=residual,
            accumulation_contract=ACCUMULATION_CONTRACT),evidence
