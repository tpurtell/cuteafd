"""Shared durable stores and Step-5 native expert-down Hessians."""
from __future__ import annotations
import hashlib
import torch
from glm5.core_k6.storage import ProjectionStore,RouteSpool as _RouteSpool,StageStore,load_tensors

class RouteSpool(_RouteSpool):
    def hessian(self,expert,down=None,device="cuda:0",chunk=2048):
        if down is not None and not callable(down):
            raise ValueError("Step-5 down capture requires native SwiGLU callback")
        natural,recovery=self.route_rows[expert],self.recovery_rows[expert]
        width=self.width if down is None else self.down_width
        H=torch.zeros((width,width),device=device,dtype=torch.float32)
        for selection in (natural,recovery):
            for start in range(0,len(selection),chunk):
                value=self.gather(selection[start:start+chunk],device)
                if down is not None:
                    value=down(value)
                    if value.shape[1]!=width or value.dtype!=torch.bfloat16:
                        raise ValueError("native Step-5 down-input geometry/dtype differs")
                value=value.float();H.addmm_(value.T,value);del value
        residual=max(0,self.minimum-len(natural)-len(recovery))
        H.diagonal().add_(float(residual))
        evidence=dict(natural_rows=len(natural),augmented_rows=len(recovery),identity_rows=residual,
            effective_rows=len(natural)+len(recovery)+residual,ranks=[9,16],
            selection_policy="rank-major-then-original-corpus-row",
            recovery_selection_sha256=hashlib.sha256(recovery.tobytes()).hexdigest(),
            down_inputs=None if down is None else "serialized-reconstructed-native-fp32-clamped-swiglu-bf16",
            identity_contract="raw-XTX-missing-times-I-equivalent-to-normalized-2I")
        return dict(H=H,count=evidence['effective_rows'],finalized=False,identity_rows=residual),evidence
