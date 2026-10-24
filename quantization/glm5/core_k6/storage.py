"""Committed GLM projections and bounded disk-backed router inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from kimi_k3.core_k6.storage import StageStore
from kimi_k3.uniform_k2.storage import (atomic_json, commit_safetensors, file_identity,
    fsync_directory, sha256_file, verify_receipt)


def load_tensors(path, device="cuda:0"):
    return {key: value.clone().to(device) for key, value in load_file(str(path)).items()}


class ProjectionStore:
    def __init__(self, root, evidence, identity, selection, progress, numerical, device="cuda:0"):
        self.root, self.evidence = Path(root), Path(evidence)
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.identity, self.selection, self.progress = identity, selection, progress
        self.numerical, self.device, self.verified = numerical, device, {}

    def paths(self, name):
        if name not in self.selection:
            raise ValueError("projection outside reviewed selection")
        return self.root/(name+".safetensors"), self.root/(name+".json")

    def load(self, name):
        path, receipt = self.paths(name)
        if name in self.verified:
            verify_receipt(path, self.verified[name])
            return self.verified[name]
        if not receipt.exists():
            return None
        saved = json.loads(receipt.read_text())
        g = self.selection[name]
        if (saved.get("identity") != self.identity or saved.get("module") != name
                or saved.get("bits") != g["bits"]
                or saved.get("logical_shape") != [g["out_features"], g["in_features"]]):
            raise ValueError("projection receipt identity/geometry differs")
        verify_receipt(path, saved, checksum=True)
        self.verified[name] = saved
        self.record_evidence(name, saved)
        return saved

    def record_evidence(self, name, saved):
        atomic_json(self.evidence/(name+".json"), {key:saved[key] for key in
            ("module", "bits", "logical_shape", "metrics", "evidence")})

    def ensure(self, name, weight, raw_factory=None, extra_evidence=None):
        path, receipt = self.paths(name)
        if self.load(name) is not None:
            return path
        if raw_factory is None:
            raise ValueError("uncommitted projection has no Hessian")
        raw = raw_factory() if callable(raw_factory) else raw_factory
        if path.exists():
            path.unlink()
        packed, metrics, evidence = self.numerical.quantize_projection(weight, raw,
            bits=self.selection[name]["bits"], seed=787, sigma=.025, device=self.device)
        if extra_evidence:
            evidence = dict(evidence, **extra_evidence)
        saved = dict(identity=self.identity, module=name, bits=self.selection[name]["bits"],
            logical_shape=list(weight.shape), metrics=metrics, evidence=evidence,
            **commit_safetensors(path, {name+"."+k:v for k,v in packed.items()}))
        atomic_json(receipt, saved)
        self.verified[name] = saved
        self.record_evidence(name, saved)
        self.progress.emit("projection_committed", module=name, bits=saved["bits"],
            hessian_rows=evidence.get("hessian_rows", raw["count"]),
            hessian_error=metrics["hessian_weighted_relative_error"])
        return path

    def dense(self, name):
        saved = self.load(name)
        if saved is None:
            raise ValueError("projection is not committed")
        tensors = load_file(str(self.paths(name)[0]))
        packed = {k.removeprefix(name+"."):v.clone() for k,v in tensors.items()}
        return self.numerical.reconstruct_projection(packed, saved["logical_shape"],
            device=self.device, dtype=torch.bfloat16)

    def install(self, name, linear):
        if self.load(name) is None:
            return False
        linear.weight = torch.nn.Parameter(self.dense(name), requires_grad=False)
        return True

    def replacements(self, names):
        result = {}
        for name in names:
            if self.load(name) is None:
                raise ValueError("missing projection during assembly")
            result[name] = self.paths(name)[0]
        return result

    def retire(self):
        for path in self.root.glob("*.safetensors"):
            path.unlink()
        fsync_directory(self.root)
        self.verified.clear()


class RouteSpool:
    """One layer's actual top-eight routes and rank-nine-to-sixteen recovery."""
    def __init__(self, root, identity, tokens, width=6144, experts=256, minimum=1024):
        self.root, self.identity, self.tokens = Path(root), identity, tokens
        self.width, self.experts, self.minimum = width, experts, minimum
        manifest = json.loads((self.root/"manifest.json").read_text())
        if (manifest["identity"] != identity or manifest["tokens"] != tokens
                or manifest["width"] != width or manifest["experts"] != experts):
            raise ValueError("route spool geometry/identity differs")
        for name, receipt in manifest["files"].items():
            verify_receipt(self.root/name, receipt, checksum=True)
        self.manifest = manifest
        self.inputs = np.memmap(self.root/"inputs.bf16", mode="r", dtype=np.uint16, shape=(tokens,width))
        self.routes = np.memmap(self.root/"routes.i16", mode="r", dtype=np.int16, shape=(tokens,8))
        self.candidates = np.memmap(self.root/"candidates.i16", mode="r", dtype=np.int16, shape=(8,tokens))
        self.counts, self.route_rows, self.recovery_rows = [], [], []
        if (np.any(self.routes < 0) or np.any(self.routes >= experts)
                or np.any(self.candidates < 0) or np.any(self.candidates >= experts)):
            raise ValueError("route spool expert index outside geometry")
        for expert in range(experts):
            natural = np.flatnonzero(np.any(self.routes == expert, axis=1))
            missing, recovery = max(0, minimum-len(natural)), []
            for rank in range(8):
                if not missing:
                    break
                selected = np.flatnonzero(self.candidates[rank] == expert)[:missing]
                recovery.extend(selected.tolist())
                missing -= len(selected)
            self.counts.append(len(natural))
            self.route_rows.append(natural)
            self.recovery_rows.append(np.asarray(recovery,dtype=np.int64))
        if sum(self.counts) != tokens*8:
            raise ValueError("route spool loses or duplicates natural expert slots")

    @classmethod
    def create(cls, root, identity, tokens, batches, router, progress, layer,
               width=6144, experts=256, minimum=1024):
        root = Path(root)
        if (root/"manifest.json").exists():
            return cls(root,identity,tokens,width,experts,minimum)
        root.mkdir(parents=True,exist_ok=True)
        maps = dict(inputs=np.memmap(root/"inputs.bf16",mode="w+",dtype=np.uint16,shape=(tokens,width)),
            routes=np.memmap(root/"routes.i16",mode="w+",dtype=np.int16,shape=(tokens,8)),
            candidates=np.memmap(root/"candidates.i16",mode="w+",dtype=np.int16,shape=(8,tokens)))
        offset=0
        try:
            for index, row, state in batches:
                hidden=state["mlp_input"].reshape(-1,width)
                selected, weights, near = router(hidden)
                if (selected.shape != (len(hidden),8) or near.shape != selected.shape
                        or torch.any(torch.sort(selected,dim=1).values.diff(dim=1)==0)
                        or torch.any((near[:,:,None] == selected[:,None,:]))
                        or torch.any(torch.sort(near,dim=1).values.diff(dim=1)==0)):
                    raise ValueError("native/recovery route sets differ from the unique top-eight contract")
                end=offset+len(hidden)
                maps["inputs"][offset:end] = hidden.detach().cpu().contiguous().view(torch.uint16).numpy()
                maps["routes"][offset:end] = selected.cpu().short().numpy()
                maps["candidates"][:,offset:end] = near.cpu().short().numpy().T
                offset=end
                if index % 64 == 0:
                    progress.emit("route_capture_progress",layer=layer,batches=index+1)
                del hidden,selected,weights,near,state
            if offset != tokens:
                raise ValueError("route spool token count differs")
            for mapping in maps.values():
                mapping.flush()
                mapping._mmap.close()
            maps.clear()
            files={}
            for name in ("inputs.bf16","routes.i16","candidates.i16"):
                path=root/name
                with path.open("rb") as f:
                    os.fsync(f.fileno())
                files[name]=dict(sha256=sha256_file(path),file_identity=file_identity(path))
            fsync_directory(root)
            atomic_json(root/"manifest.json",dict(identity=identity,tokens=tokens,width=width,experts=experts,files=files))
        finally:
            for mapping in maps.values():
                mapping._mmap.close()
        return cls(root,identity,tokens,width,experts,minimum)

    def gather(self, rows, device="cuda:0"):
        return torch.from_numpy(np.array(self.inputs[rows],copy=True)).view(torch.bfloat16).to(device)

    def hessian(self, expert, down=None, device="cuda:0", chunk=2048):
        fused=None if down is None else (torch.cat(down,dim=0) if isinstance(down,tuple) else down)
        if fused is not None and (fused.ndim!=2 or fused.shape[0]%2 or fused.shape[1]!=self.width):
            raise ValueError("down capture requires the native fused gate/up geometry")
        width=self.width if fused is None else fused.shape[0]//2
        H=torch.zeros((width,width),device=device,dtype=torch.float32)
        natural,recovery=self.route_rows[expert],self.recovery_rows[expert]
        for selection in (natural,recovery):
            for start in range(0,len(selection),chunk):
                x=self.gather(selection[start:start+chunk],device)
                if fused is not None:
                    gate,up=torch.nn.functional.linear(x,fused).chunk(2,dim=-1)
                    x=torch.nn.functional.silu(gate)*up
                x=x.float()
                H.addmm_(x.T,x)
                del x
        residual=max(0,self.minimum-len(natural)-len(recovery))
        H.diagonal().add_(float(residual))
        evidence=dict(natural_rows=len(natural),augmented_rows=len(recovery),identity_rows=residual,
            effective_rows=len(natural)+len(recovery)+residual,ranks=[9,16],
            selection_policy="rank-major-then-original-corpus-row",
            recovery_selection_sha256=hashlib.sha256(recovery.tobytes()).hexdigest(),
            down_inputs=None if down is None else "serialized-reconstructed-fused-gate-up-native-swiglu-bf16",
            identity_contract="raw-XTX-missing-times-I-equivalent-to-normalized-2I")
        return dict(H=H,count=evidence["effective_rows"],finalized=False,identity_rows=residual),evidence

    def close(self):
        for name in ("inputs","routes","candidates"):
            mapping=getattr(self,name,None)
            if mapping is not None:
                mapping._mmap.close()
                setattr(self,name,None)
