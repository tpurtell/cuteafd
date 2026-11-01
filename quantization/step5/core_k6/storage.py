"""Shared durable stores and Step-5 native expert-down Hessians."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from glm5.core_k6.storage import ProjectionStore as _ProjectionStore,RouteSpool as _RouteSpool,StageStore as _StageStore,load_tensors
from kimi_k3.uniform_k2.storage import RollingBoundary as _RollingBoundary,atomic_json,verify_receipt
from gptqmodel.utils.step5_core import accumulate_hessian_fp32,HESSIAN_ACCUMULATION_CONTRACT
from .compat import accepts_identity,identity_digest,load_module_origins,module_producer_digest

class RollingBoundary(_RollingBoundary):
    def current(self,*,checksum=False):
        path=self.root/'current.json'
        if not path.exists():return None
        record=json.loads(path.read_text())
        if not accepts_identity(record.get('identity'),self.identity):
            raise ValueError('boundary producer is outside audited lineage')
        directory=self.root/record['directory']
        if directory!=self.directory(record['layer']) or directory.is_symlink():
            raise ValueError('invalid boundary directory')
        manifest=json.loads((directory/'manifest.json').read_text())
        if (manifest!=record['manifest'] or manifest.get('identity')!=record['identity']
                or manifest.get('layer')!=record['layer']):
            raise ValueError('boundary manifest differs from commitment')
        for entry in manifest['batches']:
            verify_receipt(directory/entry['filename'],entry,checksum=checksum)
        return record

class StageStore(_StageStore):
    def load(self,name,*,checksum=False):
        directory=self.directory(name);path=directory/'manifest.json'
        if not path.exists():return None
        record=json.loads(path.read_text())
        if not accepts_identity(record.get('identity'),self.identity) or record.get('stage')!=name:
            raise ValueError('phase producer is outside audited lineage')
        self._validate_entries(record['entries'])
        for entry in record['entries']:
            target=directory/entry['filename']
            if target.is_symlink():raise ValueError('refuse symlink phase payload')
            verify_receipt(target,entry,checksum=checksum)
        return record

class ProjectionStore(_ProjectionStore):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lineage=load_module_origins(self.evidence.parent.parent,self.identity)

    def load(self,name):
        path,receipt=self.paths(name)
        if name in self.verified:
            verify_receipt(path,self.verified[name]);return self.verified[name]
        if not receipt.exists():return None
        saved=json.loads(receipt.read_text());g=self.selection[name]
        if (not accepts_identity(saved.get('identity'),self.identity) or saved.get('module')!=name
                or saved.get('bits')!=g['bits'] or saved.get('logical_shape')!=[g['out_features'],g['in_features']]):
            raise ValueError('projection identity/geometry differs from audited lineage')
        if identity_digest(saved['identity'])!=module_producer_digest(name,self.identity,self.lineage):
            raise ValueError('projection module producer differs from origin manifest')
        verify_receipt(path,saved,checksum=True);self.verified[name]=saved
        self.record_evidence(name,saved);return saved

    def record_evidence(self,name,saved):
        expected=module_producer_digest(name,self.identity,self.lineage)
        if identity_digest(saved['identity'])!=expected:
            raise ValueError('projection evidence producer differs from origin manifest')
        core={k:saved[k] for k in ('module','bits','logical_shape','metrics','evidence')}
        path=self.evidence/(name+'.json')
        if path.exists():
            prior=json.loads(path.read_text())
            if {k:prior.get(k) for k in core}!=core:
                raise ValueError('existing projection evidence differs')
            old=prior.get('producer_identity_sha256')
            if old is not None and old!=expected:
                raise ValueError('existing evidence origin differs')
            if old is None and (self.lineage is None or name not in self.lineage['module_origins']):
                # A pre-lineage fresh producer has no historical origin map.
                if self.identity.get('predecessor_identities'):
                    raise ValueError('new projection evidence has no producer identity')
            return  # Preserve historical evidence bytes exactly.
        atomic_json(path,dict(core,producer_identity_sha256=expected))

class RouteSpool(_RouteSpool):
    def __init__(self,root,identity,tokens,width=4096,experts=352,minimum=1024):
        path=Path(root)/'manifest.json'
        saved=json.loads(path.read_text())
        if not accepts_identity(saved.get('identity'),identity):
            raise ValueError('route producer is outside audited lineage')
        super().__init__(root,saved['identity'],tokens,width,experts,minimum)
        self.identity=identity

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
                value=value.float();accumulate_hessian_fp32(H,value);del value
        residual=max(0,self.minimum-len(natural)-len(recovery))
        H.diagonal().add_(float(residual))
        evidence=dict(natural_rows=len(natural),augmented_rows=len(recovery),identity_rows=residual,
            effective_rows=len(natural)+len(recovery)+residual,ranks=[9,16],
            hessian_accumulation=HESSIAN_ACCUMULATION_CONTRACT,
            selection_policy="rank-major-then-original-corpus-row",
            recovery_selection_sha256=hashlib.sha256(recovery.tobytes()).hexdigest(),
            down_inputs=None if down is None else "serialized-reconstructed-native-fp32-clamped-swiglu-bf16",
            identity_contract="raw-XTX-missing-times-I-equivalent-to-normalized-2I")
        return dict(H=H,count=evidence['effective_rows'],finalized=False,identity_rows=residual),evidence
