#!/usr/bin/env python3
"""Audit and activate the one explicit Step-5 K128 producer resume lineage.

No tensor, prior receipt, frontier, phase, route or numerical evidence is
rewritten. The original plan is archived byte-for-byte; only plan.json is
replaced after the data, code semantics and independent GPU gate pass.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
if __package__ in (None, ""):
    __package__ = "step5.core_k6"

from kimi_k3.uniform_k2.storage import (
    atomic_json, canonical, file_identity, fsync_directory, read_header, sha256_file, verify_receipt,
)

OLD_ROOT_COMMIT = "39df3b71e"
OLD_FORK = "ed84937b20ab3ccbfc7d06c3b001fc24e545b37a"
NEW_FORK = "25dd11c391ce6bf3ceadf8722e8274c3cde7079a"
CONTRACT = "step5-native-fp32-XTX-k128-v1"
COMPLETED = (0, 1, 2)
ACTIVE_LAYER, EXPERT_STOP, EXPERTS = 3, 217, 352
QUANTIZER = "gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py"
ADMIN = frozenset(("schema", "code", "gptqmodel_commit", "predecessor_identities",
                   "resume_lineage_audit_sha256", "resume_lineage_manifest_sha256"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def safe_path(root, relative):
    path = Path(root) / relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("unsafe migration object path: " + str(path))
    return path


def checked_file(path, receipt=None):
    """Real full-file checksums; metadata-only receipt checks are insufficient."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or unsafe migration object: " + str(path))
    if receipt is not None:
        verify_receipt(path, receipt, checksum=True)
        return dict(sha256=receipt["sha256"],file_identity=file_identity(path))
    return dict(sha256=sha256_file(path), file_identity=file_identity(path))


class Objects:
    def __init__(self):
        self.files = {}

    def add(self, path, receipt=None):
        path = Path(path).absolute()
        if str(path) not in self.files:
            self.files[str(path)] = checked_file(path, receipt)
        elif receipt is not None:
            recorded = self.files[str(path)]
            if recorded != {key:receipt[key] for key in ("sha256", "file_identity")}:
                raise ValueError("inconsistent receipts for the same migration object")
        return self.files[str(path)]

    def document(self, path):
        self.add(path)
        return json.loads(Path(path).read_text())

    def unchanged(self):
        for name, record in self.files.items():
            if file_identity(Path(name)) != record["file_identity"]:
                raise ValueError("audited historical object changed during migration: " + name)


def _tree(source):
    return ast.parse(source)


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _symbols(tree):
    return {node.name:node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}


def validate_core_delta(old, new):
    """Only the new bounded accumulator may replace the old Hessian alias."""
    left, right = _tree(old), _tree(new)
    additions = {"accumulate_hessian_fp32", "DenseRawHessian"}
    if set(_symbols(right)) - set(_symbols(left)) != additions:
        raise ValueError("unreviewed Step-5 numerical helper additions")
    def normalized(tree, current):
        result = []
        for node in tree.body:
            if current and isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in additions:
                continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if current and name == "HESSIAN_ACCUMULATION_CONTRACT":
                    if not isinstance(node.value, ast.Constant) or node.value.value != CONTRACT:
                        raise ValueError("unreviewed Hessian accumulation contract")
                    continue
                if not current and name == "DenseRawHessian":
                    if _dump(node.value) != _dump(ast.parse("base.DenseRawHessian", mode="eval").body):
                        raise ValueError("original Hessian alias differs")
                    continue
            result.append(node)
        return ast.Module(body=result, type_ignores=[])
    if _dump(normalized(left,False)) != _dump(normalized(right,True)):
        raise ValueError("old Step-5 native forward or quantization bit math changed")
    return dict(allowed=["new K128 accumulator", "new DenseRawHessian subclass", "accumulation contract"],
                protected_old_symbols=sorted(_symbols(left)))


class _Metadata(ast.NodeTransformer):
    """Remove only lineage admission/annotation additions for AST comparison."""
    def visit_ImportFrom(self,node):
        if node.module == "compat" and node.level == 1:
            return None
        if node.module in ("storage","kimi_k3.uniform_k2.storage"):
            node.names = [n for n in node.names if n.name != "RollingBoundary"]
            if not node.names:
                return None
        return node

    def visit_UnaryOp(self,node):
        if (isinstance(node.op,ast.Not) and isinstance(node.operand,ast.Call)
                and isinstance(node.operand.func,ast.Name) and node.operand.func.id == "accepts_identity"
                and len(node.operand.args) == 2 and not node.operand.keywords):
            return ast.Compare(left=node.operand.args[0],ops=[ast.NotEq()],comparators=[node.operand.args[1]])
        return self.generic_visit(node)

    def visit_If(self,node):
        if any(isinstance(n,ast.Name) and n.id == "lineage" for n in ast.walk(node.test)):
            return None
        return self.generic_visit(node)

    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id == "lineage":
                return None
            if node.targets[0].id == "evidence_root" and _dump(node.value)==_dump(ast.parse("Path(evidence_root)",mode="eval").body):
                return None
        return self.generic_visit(node)

    def visit_Call(self,node):
        node = self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id == "dict":
            node.keywords = [k for k in node.keywords if k.arg != "producer_identity_sha256"]
        return node


def validate_run_delta(old,new,flash):
    left,right = _tree(old),_tree(new)
    old_symbols,new_symbols = _symbols(left),_symbols(right)
    if set(old_symbols) != set(new_symbols):
        raise ValueError("unreviewed producer function additions/removals")
    old_engine,new_engine = old_symbols["Engine"],new_symbols["Engine"]
    before,after = _symbols(old_engine),_symbols(new_engine)
    if set(after)-set(before) != {"checkpoints"} or set(before)-set(after):
        raise ValueError("unreviewed producer Engine methods")
    expected_checkpoints = _symbols(_symbols(_tree(flash))["Engine"])["checkpoints"]
    if _dump(after["checkpoints"]) != _dump(expected_checkpoints):
        raise ValueError("producer projection factory changed beyond its compatible store binding")
    expected_init = copy.deepcopy(before["__init__"])
    expected_init.body.append(ast.parse('self.boundary=RollingBoundary(self.root/"frontier",self.identity)').body[0])
    if _dump(after["__init__"]) != _dump(expected_init):
        raise ValueError("producer initialization changed beyond compatible frontier binding")
    for name,node in before.items():
        if name != "__init__" and _dump(node) != _dump(after[name]):
            raise ValueError("producer numerical method changed: " + name)
    for tree in (left,right):
        tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.ClassDef)) or n.name not in ("Engine","make_plan")]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="RECIPE" for t in node.targets):
                if not isinstance(node.value,ast.Call) or not isinstance(node.value.func,ast.Name) or node.value.func.id!="dict":
                    raise ValueError("unexpected producer recipe expression")
                for k in node.value.keywords:
                    if k.arg=="hessian_accumulation" and (not isinstance(k.value,ast.Constant) or k.value.value!=CONTRACT):
                        raise ValueError("unexpected producer accumulation contract")
                node.value.keywords=[k for k in node.value.keywords if k.arg!="hessian_accumulation"]
    left,right = _Metadata().visit(left),_Metadata().visit(right)
    if _dump(left)!=_dump(right):
        raise ValueError("producer changes extend outside lineage admission and recipe metadata")
    return dict(allowed=["compatible boundary/store factory", "make_plan lineage admission", "recipe contract addition"],
                protected_engine_methods=sorted(set(before)-{"__init__"}))


def validate_storage_delta(old,new):
    left,right = _tree(old),_tree(new)
    before,after = _symbols(left),_symbols(right)
    if set(before)!={"RouteSpool"} or set(after)!={"RouteSpool","ProjectionStore","StageStore","RollingBoundary"}:
        raise ValueError("unreviewed storage wrapper symbols")
    methods=_symbols(after["RouteSpool"])
    if set(methods)!={"__init__","hessian"}:
        raise ValueError("unreviewed route spool methods")
    allowed={"ProjectionStore":{"__init__","load","record_evidence"},"StageStore":{"load"},"RollingBoundary":{"current"}}
    for name,names in allowed.items():
        if set(_symbols(after[name])) != names:
            raise ValueError("unreviewed compatibility wrapper methods: "+name)
    class Accumulation(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=="accumulate_hessian_fp32":
                if _dump(node)!=_dump(ast.parse("accumulate_hessian_fp32(H,value)",mode="eval").body):
                    raise ValueError("unreviewed routed Hessian accumulation call")
                return ast.parse("H.addmm_(value.T,value)",mode="eval").body
            if isinstance(node.func,ast.Name) and node.func.id=="dict":
                for k in node.keywords:
                    if k.arg=="hessian_accumulation" and not (
                        isinstance(k.value,ast.Name) and k.value.id=="HESSIAN_ACCUMULATION_CONTRACT"):
                        raise ValueError("unreviewed routed Hessian contract metadata")
                node.keywords=[k for k in node.keywords if k.arg!="hessian_accumulation"]
            return node
    normalized=Accumulation().visit(copy.deepcopy(methods["hessian"]))
    if _dump(normalized)!=_dump(_symbols(before["RouteSpool"])["hessian"]):
        raise ValueError("routed rows, recovery, damping or Hessian arithmetic changed beyond K128 products")
    for name,node in after.items():
        nodes=_symbols(node)
        for method,body in nodes.items():
            if name=="RouteSpool" and method=="hessian":
                continue
            if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
                   ("quantize_projection","reconstruct_projection","addmm_","matmul","copy_") for n in ast.walk(body)):
                raise ValueError("compatibility wrapper performs numerical tensor work")
    # No executable top-level work may hide beside the reviewed wrappers.
    if any(not isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Expr)) or
           (isinstance(n,ast.Expr) and not isinstance(n.value,ast.Constant)) for n in right.body):
        raise ValueError("unreviewed executable storage module statements")
    return dict(allowed=["explicit compatible receipt readers", "bounded routed XTX call", "routed accumulation metadata"])


def validate_export_delta(old,new):
    left,right=_tree(old),_tree(new)
    for tree in (left,right):
        tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!="_model_card"]
    if _dump(_Metadata().visit(left))!=_dump(_Metadata().visit(right)):
        raise ValueError("export changes extend beyond lineage admission/metadata")
    return dict(allowed=["compatible receipt identity predicate", "per-module producer metadata", "copied lineage evidence", "model-card lineage disclosure"])


def semantic_code_audit(identity,repo=REPO):
    repo=Path(repo);fork=repo/"third_party/gptqmodel"
    current_fork=subprocess.check_output(["git","-C",str(fork),"rev-parse","HEAD"],text=True).strip()
    if identity.get("gptqmodel_commit")!=OLD_FORK or current_fork!=NEW_FORK:
        raise ValueError("migration requires the explicitly GPU-qualified old/new fork pair")
    def git_bytes(directory,revision,path):
        return subprocess.check_output(["git","-C",str(directory),"show",revision+":"+path])
    originals={};current={};checks={}
    for relative,expected in identity["code"].items():
        in_fork=relative.startswith("third_party/gptqmodel/")
        path=relative.removeprefix("third_party/gptqmodel/") if in_fork else relative
        original=git_bytes(fork if in_fork else repo,OLD_FORK if in_fork else OLD_ROOT_COMMIT,path)
        if hashlib.sha256(original).hexdigest()!=expected:
            raise ValueError("old Git source does not match the recorded producer: "+relative)
        originals[relative]=original.decode();current_bytes=(repo/relative).read_bytes()
        if in_fork and current_bytes!=git_bytes(fork,NEW_FORK,path):
            raise ValueError("new numerical helper differs from its GPU-qualified fork commit: "+relative)
        current[relative]=current_bytes.decode()
        checks[relative]=dict(before_sha256=expected,after_sha256=sha256_file(repo/relative))
    changed={p for p,v in checks.items() if v["before_sha256"]!=v["after_sha256"]}
    expected={"quantization/step5/core_k6/"+n for n in ("run.py","storage.py","export.py")}|{
        "third_party/gptqmodel/gptqmodel/utils/step5_core.py"}
    if changed!=expected:
        raise ValueError("unexpected changed/unchanged numerical source set: "+canonical(sorted(changed)))
    results={
        "core":validate_core_delta(originals["third_party/gptqmodel/gptqmodel/utils/step5_core.py"],current["third_party/gptqmodel/gptqmodel/utils/step5_core.py"]),
        "run":validate_run_delta(originals["quantization/step5/core_k6/run.py"],current["quantization/step5/core_k6/run.py"],originals["quantization/glm5_flash/core_k6/run.py"]),
        "storage":validate_storage_delta(originals["quantization/step5/core_k6/storage.py"],current["quantization/step5/core_k6/storage.py"]),
        "export":validate_export_delta(originals["quantization/step5/core_k6/export.py"],current["quantization/step5/core_k6/export.py"])}
    for relative in ("gptqmodel/utils/glm5_core.py",QUANTIZER):
        original=git_bytes(fork,OLD_FORK,relative)
        if original!=(fork/relative).read_bytes():
            raise ValueError("generic LDLQ/trellis bit math changed: "+relative)
        checks["third_party/gptqmodel/"+relative]=dict(before_sha256=hashlib.sha256(original).hexdigest(),after_sha256=sha256_file(fork/relative))
    return dict(status="passed",old_root_commit=subprocess.check_output(["git","-C",str(repo),"rev-parse",OLD_ROOT_COMMIT],text=True).strip(),
        old_fork=OLD_FORK,new_fork=current_fork,files=checks,semantic_checks=results,
        new_admission_files={"compat.py":sha256_file(repo/"quantization/step5/core_k6/compat.py"),
                             "migrate.py":sha256_file(Path(__file__))})


def validate_probe(report,counts=None):
    if (report.get("status")!="passed" or report.get("fork_commit")!=NEW_FORK or report.get("sigma")!=.025
            or report.get("maximum_natural_expert")!=104 or len(report.get("rows",[]))!=2
            or {r.get("expert") for r in report["rows"]}!={217,104}):
        raise ValueError("the two controlled K128 GPU gates are required")
    expected={217:1955,104:253367}
    if counts is not None and (int(counts.argmax())!=104 or any(int(counts[e])!=n for e,n in expected.items())):
        raise ValueError("route spool differs from the GPU-qualified witnesses")
    for row in report["rows"]:
        value=row.get("relative_frobenius")
        if (row.get("actual_rows")!=expected[row["expert"]] or row.get("contract")!=CONTRACT
                or row.get("full_regularization_congruence_ldl_passed") is not True
                or isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<=value<=2e-6):
            raise ValueError("controlled K128 covariance/LDL proof failed")
    return report


def _core_names(layer):
    mlp="mlp" if layer<3 else "share_expert"
    return {f"model.layers.{layer}."+n for n in (
        "self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
        mlp+".gate_proj",mlp+".up_proj",mlp+".down_proj")}


def scope_modules():
    complete=set().union(*(_core_names(n) for n in COMPLETED))
    partial=_core_names(ACTIVE_LAYER)|{f"model.layers.3.moe.experts.{e}.{p}" for e in range(EXPERT_STOP) for p in ("gate_proj","up_proj","down_proj")}
    return complete,partial


def checked_manifest(directory,identity,objects,*,entries_key="entries"):
    directory=Path(directory)
    document=objects.document(directory/"manifest.json")
    if document.get("identity")!=identity:
        raise ValueError("historical manifest identity differs")
    entries=document[entries_key]
    if not entries or len({e["filename"] for e in entries})!=len(entries):
        raise ValueError("historical manifest has empty/duplicate entries")
    expected={"manifest.json"}
    for entry in entries:
        path=safe_path(directory,entry["filename"])
        if path.parent!=directory:
            raise ValueError("manifest entry is not a direct child")
        objects.add(path,entry);expected.add(entry["filename"])
    if {p.name for p in directory.iterdir()}!=expected:
        raise ValueError("unreceipted historical manifest payload")
    return document


def route_counts(directory,manifest):
    import numpy as np
    tokens,width,experts=manifest["tokens"],manifest["width"],manifest["experts"]
    if width!=4096 or experts!=EXPERTS or set(manifest["files"])!={"inputs.bf16","routes.i16","candidates.i16"}:
        raise ValueError("historical route geometry differs")
    sizes={"inputs.bf16":tokens*width*2,"routes.i16":tokens*8*2,"candidates.i16":tokens*8*2}
    if any((directory/name).stat().st_size!=size for name,size in sizes.items()):
        raise ValueError("route spool payload length differs")
    routes=np.memmap(directory/"routes.i16",mode="r",dtype=np.int16,shape=(tokens,8))
    candidates=np.memmap(directory/"candidates.i16",mode="r",dtype=np.int16,shape=(8,tokens))
    counts=np.zeros(experts,dtype=np.int64)
    try:
        for first in range(0,tokens,65536):
            selected=routes[first:first+65536];recovery=candidates[:,first:first+65536].T
            if (np.any((selected<0)|(selected>=experts)) or np.any((recovery<0)|(recovery>=experts))
                    or np.any(np.diff(np.sort(selected,axis=1),axis=1)==0)
                    or np.any(np.diff(np.sort(recovery,axis=1),axis=1)==0)
                    or np.any(selected[:,:,None]==recovery[:,None,:])):
                raise ValueError("historical route/recovery sets are invalid")
            counts+=np.bincount(selected.reshape(-1),minlength=experts)
    finally:
        routes._mmap.close();candidates._mmap.close()
    if int(counts.sum())!=tokens*8:
        raise ValueError("route counts do not conserve actual slots")
    return counts


def _metric(record,name,geometry,tokens,counts):
    bits=geometry["bits"]
    if record.get("module")!=name or record.get("bits")!=bits or record.get("logical_shape")!=[geometry["out_features"],geometry["in_features"]]:
        raise ValueError("historical module geometry differs: "+name)
    metrics,evidence=record["metrics"],record["evidence"]
    if (metrics.get("quantizer_path")!="hessian_ldlq" or metrics.get("hessian_metric_status")!="ok"
            or metrics.get("hessian_regularization_sigma")!=.025
            or metrics.get("hessian_numerical_contract")!="signed-block-hadamard-congruence-fp64-v1"
            or metrics.get("hessian_storage_dtype")!="torch.float32"
            or metrics.get("hessian_transform_compute_dtype")!="torch.float64"):
        raise ValueError("historical module has no accepted unchanged precision/damping proof: "+name)
    for key in ("reported_metric_value","hessian_weighted_error_numerator","hessian_weighted_reference_denominator","hessian_weighted_relative_error"):
        value=metrics.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError("historical module has an invalid numerical metric: "+name)
    if metrics["hessian_weighted_reference_denominator"]<=0:
        raise ValueError("historical module has no reference energy")
    rows=evidence.get("hessian_rows")
    if metrics.get("hessian_sample_count")!=rows or not isinstance(rows,int) or rows<=0:
        raise ValueError("historical module sample counts differ")
    if ".experts." in name:
        expert=int(name.split(".experts.")[1].split(".")[0])
        components=[evidence.get(k) for k in ("natural_rows","augmented_rows","identity_rows")]
        if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in components):
            raise ValueError("historical expert has invalid recovery counts")
        natural,augmented,synthetic=components
        if (evidence.get("natural_rows")!=int(counts[expert]) or evidence.get("effective_rows")!=rows
                or sum(components)!=rows or augmented>max(0,1024-natural)
                or synthetic!=max(0,1024-natural-augmented) or rows<1024):
            raise ValueError("historical expert route evidence differs: "+name)
    elif rows!=tokens or evidence.get("identity_rows")!=0:
        raise ValueError("historical dense calibration is not the full actual stream")


def audit_existing(root,probe_report,repo=REPO):
    from .recipe import reviewed_modules,packed_specs
    from .export import _check_packed
    from step5.prepare_corpus import load_token_records
    root=Path(root).resolve();probe_report=Path(probe_report).resolve();objects=Objects()
    raw=(root/"plan.json").read_bytes();plan=objects.document(root/"plan.json");identity=plan["identity"]
    if (identity.get("schema")!="cuteafd-step5-k4-core-k6-production-v1" or identity.get("gptqmodel_commit")!=OLD_FORK
            or plan.get("root")!=str(root) or identity.get("calibration")!={"samples":1426,"tokens":1105269}):
        raise ValueError("migration is scoped to the stopped v1 Step-5 producer")
    for filename,key in (("config.json","source_config_sha256"),("model.safetensors.index.json","source_index_sha256")):
        # HF snapshots use symlinks into their managed immutable blob store.
        # Resolve these read-only source references; recovery payloads remain
        # subject to the strict no-symlink checks above.
        if objects.add((Path(plan["source"])/filename).resolve(strict=True))["sha256"]!=identity[key]:
            raise ValueError("historical source metadata changed")
    token_manifest=objects.document(Path(plan["tokens"])/"manifest.json")
    if objects.files[str((Path(plan["tokens"])/"manifest.json").absolute())]["sha256"]!=identity["token_manifest_sha256"]:
        raise ValueError("historical calibration manifest changed")
    for split in token_manifest["splits"].values():
        path=Path(plan["tokens"])/split["file"]
        if objects.add(path)["sha256"]!=split["sha256"]:
            raise ValueError("historical token payload changed")
    records=load_token_records(plan["tokens"],"calibration")
    if len(records)!=1426 or sum(len(r["input_ids"]) for r in records)!=1105269:
        raise ValueError("historical calibration rows changed")
    tokens=1105269
    def batches(entries):
        rows=[e for e in entries if e.get("kind","batch")=="batch"]
        if [(e.get("id"),e.get("tokens")) for e in rows]!=[(r["id"],len(r["input_ids"])) for r in records]:
            raise ValueError("historical activation corpus order differs")
    current=objects.document(root/"frontier/current.json")
    if current.get("layer")!=2 or current.get("directory")!="layer-002" or current.get("identity")!=identity:
        raise ValueError("migration requires the verified completed layer2 frontier")
    frontier=checked_manifest(root/"frontier/layer-002",identity,objects,entries_key="batches")
    if frontier!=current["manifest"] or {p.name for p in (root/"frontier").iterdir()}!={"current.json","layer-002"}:
        raise ValueError("frontier commitment or rolling retirement differs")
    batches(frontier["batches"])
    stages=list((root/"layers").glob("*/stages/*/manifest.json"))
    if {str(p.relative_to(root)) for p in stages}!={"layers/003/stages/d/manifest.json","layers/003/stages/e/manifest.json"}:
        raise ValueError("migration requires exactly layer3 phases d/e")
    for phase,width,name in (("d",4096,"mlp"),("e",1536,"down")):
        state=checked_manifest(root/f"layers/003/stages/{phase}",identity,objects)
        if state.get("stage")!=phase or state.get("metadata")!={"batches":1426,"tokens":tokens}:
            raise ValueError("historical phase metadata differs")
        if phase=="d":batches(state["entries"])
        hs=[e for e in state["entries"] if e.get("kind")=="hessian"]
        if len(hs)!=1 or hs[0].get("count")!=tokens or hs[0].get("name")!=name:
            raise ValueError("historical dense Hessian manifest differs")
        header,_=read_header(root/f"layers/003/stages/{phase}"/hs[0]["filename"])
        if set(header)!={"H"} or (header["H"]["dtype"],header["H"]["shape"])!=("F32",[width,width]):
            raise ValueError("historical dense Hessian geometry differs")
    route_manifests=list((root/"layers").glob("*/routes/manifest.json"))
    if len(route_manifests)!=1 or route_manifests[0]!=root/"layers/003/routes/manifest.json":
        raise ValueError("migration requires the one layer3 routing spool")
    routes=objects.document(route_manifests[0]);route_root=route_manifests[0].parent
    if routes.get("identity")!=identity or routes.get("tokens")!=tokens:
        raise ValueError("historical route identity differs")
    for name,receipt in routes["files"].items():objects.add(safe_path(route_root,name),receipt)
    if {p.name for p in route_root.iterdir()}!={"manifest.json",*routes["files"]}:
        raise ValueError("unreceipted route spool object")
    counts=route_counts(route_root,routes)
    probe=objects.document(probe_report);validate_probe(probe,counts)
    complete,partial=scope_modules();expected=complete|partial;selection=reviewed_modules()
    evidence_files=list((root/"evidence").glob("*/*.json"))
    receipt_files=list((root/"layers").glob("*/projections/*.json"))
    packed_files=list((root/"layers").glob("*/projections/*.safetensors"))
    if {p.stem for p in evidence_files}!=expected or len(evidence_files)!=679:
        raise ValueError("historical evidence is not exactly the679 accepted modules before expert217")
    if {p.stem for p in receipt_files}!=expected or len(receipt_files)!=679:
        raise ValueError("historical projection receipt scope differs")
    if {p.stem for p in packed_files}!=partial or len(packed_files)!=658:
        raise ValueError("partial packed scope includes missing/unaccepted payloads")
    if list((root/"layers").rglob("*.partial")) or (root/"layers/003/experts-complete.json").exists():
        raise ValueError("unaccepted partial/expert-complete state cannot be migrated")
    for path in receipt_files:
        saved=objects.document(path);name=path.stem;geometry=selection[name]
        if saved.get("identity")!=identity:
            raise ValueError("historical packed receipt producer differs")
        _metric(saved,name,geometry,tokens,counts)
        observed=objects.document(root/f"evidence/layer-{int(name.split('.')[2]):03d}"/(name+".json"))
        if observed!={key:saved[key] for key in ("module","bits","logical_shape","metrics","evidence")}:
            raise ValueError("historical scalar evidence differs from its packed receipt")
        payload=path.with_suffix(".safetensors")
        if name in partial:
            objects.add(payload,saved);_check_packed(name,geometry,payload)
        elif payload.exists():
            raise ValueError("completed layer retained obsolete packed payload")
    export=Path(plan["export"])
    if {p.name for p in export.glob("layer-*-receipt.json")}!={f"layer-{n:03d}-receipt.json" for n in COMPLETED}:
        raise ValueError("completed layer shard receipt scope differs")
    for layer in COMPLETED:
        receipt=objects.document(export/f"layer-{layer:03d}-receipt.json")
        if receipt.get("identity")!=identity or receipt.get("layer")!=layer:
            raise ValueError("completed shard producer differs")
        shard=export/receipt["filename"];objects.add(shard,receipt);header,_=read_header(shard)
        for name in _core_names(layer):
            for key,spec in packed_specs(name,selection[name]).items():
                if key not in header or (header[key]["dtype"],header[key]["shape"])!=spec:
                    raise ValueError("completed shard packed geometry differs")
    if {p.name for p in export.glob("*.safetensors")}!={f"model-{n+1:05d}-of-00096.safetensors" for n in COMPLETED}:
        raise ValueError("unaccepted completed export shard scope")
    for path in [root/"status.json",root/"events.jsonl",root/"latest-attempt.json"]:
        if path.exists():objects.add(path)
    code=semantic_code_audit(identity,repo)
    for relative in code["files"]:
        objects.add(repo/relative)
    objects.add(repo/"quantization/step5/core_k6/compat.py")
    objects.add(Path(__file__))
    # The active plan is the sole deliberately replaced historical control.
    # Its original bytes are instead bound to the immutable epoch archive.
    objects.files.pop(str((root/"plan.json").absolute()))
    audit=dict(schema="cuteafd-step5-k128-resume-audit-v1",status="passed",old_plan_sha256=hashlib.sha256(raw).hexdigest(),
        old_identity_sha256=digest(identity),old_plan_ref="epochs/000-plan.json",old_identity=identity,
        scope=dict(completed_layers=list(COMPLETED),frontier_layer=2,active_layer=3,next_expert=217,next_part="gate_proj",
            accepted_modules=679,retired_completed_modules=21,verified_partial_packed=658,
            completed_packed_proof="checked complete shard checksums and packed headers; original per-projection files were retired at their commit barriers",
            active_stages=["d","e"],calibration_samples=1426,calibration_tokens=tokens),
        gpu_probe=dict(path=str(probe_report),sha256=objects.files[str(probe_report)]["sha256"],report=probe),
        route_natural_counts=counts.tolist(),semantic_code_audit=code,objects=objects.files)
    lineage=dict(schema="cuteafd-step5-producer-lineage-v1",predecessor_identities=[identity],
        module_origins={name:digest(identity) for name in sorted(expected)},
        old_plan=dict(path="epochs/000-plan.json",sha256=audit["old_plan_sha256"],identity_sha256=digest(identity)))
    return audit,lineage,objects


def validate_active_plan(old,new,lineage,audit_sha,manifest_sha):
    left,right=old["identity"],new["identity"]
    if ({k:v for k,v in old.items() if k!="identity"}!={k:v for k,v in new.items() if k!="identity"}
            or right.get("schema")!="cuteafd-step5-k4-core-k6-production-v2" or right.get("gptqmodel_commit")!=NEW_FORK
            or right.get("predecessor_identities")!=[left] or right.get("resume_lineage_audit_sha256")!=audit_sha
            or right.get("resume_lineage_manifest_sha256")!=manifest_sha):
        raise ValueError("new active plan does not bind the audited predecessor/objects")
    before={k:v for k,v in left.items() if k not in ADMIN|{"recipe"}}
    after={k:v for k,v in right.items() if k not in ADMIN|{"recipe"}}
    if before!=after or right.get("recipe")!={**left["recipe"],"hessian_accumulation":CONTRACT}:
        raise ValueError("source, calibration, runtime or quantization choices changed during migration")
    if lineage.get("predecessor_identities")!=[left] or lineage.get("resume_lineage_audit_sha256")!=audit_sha:
        raise ValueError("producer manifest differs from the activated plan")


def _write_once(path,data):
    path=Path(path)
    if path.is_symlink():raise ValueError("refuse symlink lineage output")
    if path.exists():
        if path.read_bytes()!=data:raise ValueError("refuse to replace existing lineage evidence: "+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=".lineage-",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        try:os.link(temporary,path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes()!=data:
                raise ValueError("lineage evidence changed during publication")
        fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def exclusive_migration(root):
    path=Path(root)/"run.lock"
    if path.is_symlink():raise ValueError("refuse symlink producer lock")
    with path.open("a+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError("producer is running; stop it before migration") from e
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def migrate(root,probe_report,*,repo=REPO,auditor=None,make_plan=None,source_factory=None):
    root=Path(root).resolve();repo=Path(repo).resolve()
    with exclusive_migration(root):
        old_bytes=(root/"plan.json").read_bytes();old=json.loads(old_bytes)
        if old["identity"].get("schema")!="cuteafd-step5-k4-core-k6-production-v1":
            raise ValueError("migration requires the original v1 plan; no second migration is inferred")
        audit,lineage,objects=(auditor or audit_existing)(root,Path(probe_report),repo)
        if audit.get("status")!="passed" or audit["old_plan_sha256"]!=hashlib.sha256(old_bytes).hexdigest():
            raise ValueError("migration audit does not cover the active old plan")
        audit_bytes=(canonical(audit)+"\n").encode();audit_sha=hashlib.sha256(audit_bytes).hexdigest()
        lineage=dict(lineage,resume_lineage_audit_sha256=audit_sha)
        lineage_bytes=(canonical(lineage)+"\n").encode();manifest_sha=hashlib.sha256(lineage_bytes).hexdigest()
        _write_once(root/"epochs/000-plan.json",old_bytes)
        _write_once(root/"resume-lineage-audit.json",audit_bytes)
        _write_once(root/"producer-lineage.json",lineage_bytes)
        if make_plan is None:
            from .run import make_plan
        if source_factory is None:
            sys.path.insert(0,str(repo/"third_party/gptqmodel"))
            from gptqmodel.utils.step5_core import Step5BF16Source
            source_factory=lambda snapshot:Step5BF16Source(snapshot,reference_root=repo/"python/reference")
        source=source_factory(Path(old["source"]))
        try:new=make_plan(source,Path(old["tokens"]),root,Path(old["export"]))
        finally:source.close()
        validate_active_plan(old,new,lineage,audit_sha,manifest_sha)
        semantic=audit.get("semantic_code_audit")
        if semantic is not None:
            expected={p:entry["after_sha256"] for p,entry in semantic["files"].items() if p in old["identity"]["code"]}
            expected.update({"quantization/step5/core_k6/"+name:value for name,value in semantic["new_admission_files"].items()})
            if any(new["identity"]["code"].get(p)!=value for p,value in expected.items()):
                raise ValueError("new active plan source does not match the audited code seal")
        objects.unchanged()
        if (root/"plan.json").read_bytes()!=old_bytes:
            raise ValueError("active plan changed during its migration audit")
        atomic_json(root/"plan.json",new)
        return new


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--probe-report",type=Path,required=True)
    args=parser.parse_args(argv)
    plan=migrate(args.root,args.probe_report)
    print(canonical(dict(status="passed",schema=plan["identity"]["schema"],active_identity_sha256=digest(plan["identity"]),
        predecessor_identity_sha256=digest(plan["identity"]["predecessor_identities"][0]),
        accepted_historical_modules=679,old_objects_rewritten=0)))


if __name__=="__main__":main()
