"""Strict Step-5 producer-lineage admission with real tiny receipt payloads."""
from __future__ import annotations
import copy
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"quantization"))
from step5.core_k6 import migrate as migration
from kimi_k3.uniform_k2.storage import file_identity,sha256_file


def receipt(path):
    return dict(sha256=sha256_file(path),file_identity=file_identity(path))


def probe():
    return dict(status="passed",fork_commit=migration.NEW_FORK,sigma=.025,maximum_natural_expert=104,
        rows=[dict(expert=e,actual_rows=n,relative_frobenius=error,contract=migration.CONTRACT,
              full_regularization_congruence_ldl_passed=True) for e,n,error in ((217,1955,7e-8),(104,253367,6.6e-7))])


def plans(root):
    old=dict(identity=dict(schema="cuteafd-step5-k4-core-k6-production-v1",gptqmodel_commit=migration.OLD_FORK,
        source_repo="native/source",calibration=dict(samples=1426,tokens=1105269),runtime={"native":"pinned"},
        code={"old.py":"a"*64},recipe=dict(sigma_reg=.025,core_bits=6,expert_bits=4)),
        source=str(root/"source"),tokens=str(root/"tokens"),root=str(root),export=str(root/"export"),
        inventory={"reviewed":"unchanged"})
    return old


def new_plan(old,root):
    lineage=json.loads((root/"producer-lineage.json").read_text())
    new=copy.deepcopy(old)
    new["identity"].update(schema="cuteafd-step5-k4-core-k6-production-v2",gptqmodel_commit=migration.NEW_FORK,
        code={"new.py":"b"*64},recipe={**old["identity"]["recipe"],"hessian_accumulation":migration.CONTRACT},
        predecessor_identities=lineage["predecessor_identities"],
        resume_lineage_audit_sha256=lineage["resume_lineage_audit_sha256"],
        resume_lineage_manifest_sha256=sha256_file(root/"producer-lineage.json"))
    return new


def auditor_for(old,objects):
    def audit(root,probe_report,repo):
        raw=(root/"plan.json").read_bytes()
        return (dict(status="passed",old_plan_sha256=migration.hashlib.sha256(raw).hexdigest(),
                    old_identity_sha256=migration.digest(old["identity"]),objects=objects.files),
                dict(predecessor_identities=[old["identity"]],module_origins={"old.module":migration.digest(old["identity"])},
                     old_plan={"path":"epochs/000-plan.json","sha256":migration.hashlib.sha256(raw).hexdigest()}),objects)
    return audit


def test_exact_stopped_scope_contains_only_accepted_experts():
    complete,partial=migration.scope_modules()
    assert len(complete)==21 and len(partial)==658 and len(complete|partial)==679
    assert all(".experts.217." not in name for name in partial)
    assert all(f"model.layers.3.moe.experts.216.{part}" in partial for part in ("gate_proj","up_proj","down_proj"))


def test_real_receipt_payload_checksum_cannot_be_skipped(tmp_path):
    path=tmp_path/"data.bin";path.write_bytes(b"original actual payload")
    saved=receipt(path)
    assert migration.checked_file(path,saved)==saved
    path.write_bytes(b"corrupted actual payload")
    with pytest.raises(ValueError,match="committed file changed"):
        migration.checked_file(path,saved)


def test_hf_source_blob_resolution_does_not_allow_recovery_symlinks(tmp_path):
    payload=tmp_path/"native-blob";payload.write_bytes(b"native source bytes")
    link=tmp_path/"config.json";link.symlink_to(payload)
    with pytest.raises(ValueError,match="unsafe"):
        migration.checked_file(link)
    assert migration.checked_file(link.resolve())["sha256"]==sha256_file(payload)


def test_real_binary_route_spool_counts_and_disjoint_recovery(tmp_path):
    import numpy as np
    tokens=4
    selected=np.tile(np.arange(8,dtype=np.int16),(tokens,1))
    recovery=np.tile(np.arange(8,16,dtype=np.int16)[:,None],(1,tokens))
    selected.tofile(tmp_path/"routes.i16");recovery.tofile(tmp_path/"candidates.i16")
    (tmp_path/"inputs.bf16").write_bytes(bytes(tokens*4096*2))
    manifest=dict(tokens=tokens,width=4096,experts=352,files={name:{} for name in ("inputs.bf16","routes.i16","candidates.i16")})
    counts=migration.route_counts(tmp_path,manifest)
    assert counts[:8].tolist()==[4]*8 and counts[8:].sum()==0
    recovery[0,0]=0;recovery.tofile(tmp_path/"candidates.i16")
    with pytest.raises(ValueError,match="route/recovery sets"):
        migration.route_counts(tmp_path,manifest)


def test_real_manifest_checks_all_files_and_rejects_unreceipted_payload(tmp_path):
    payload=tmp_path/"batch-0.safetensors";payload.write_bytes(b"tiny bounded payload")
    identity={"same":"producer"}
    manifest=dict(identity=identity,entries=[dict(filename=payload.name,**receipt(payload))])
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    assert migration.checked_manifest(tmp_path,identity,migration.Objects())==manifest
    (tmp_path/"unaccepted-217.bin").write_bytes(b"never accepted")
    with pytest.raises(ValueError,match="unreceipted"):
        migration.checked_manifest(tmp_path,identity,migration.Objects())


@pytest.mark.parametrize("change",["sigma","fork","missing","frobenius","negative","not-finite","ldl","rows"])
def test_gpu_gate_requires_both_real_cases_at_the_unchanged_precision_recipe(change):
    report=probe()
    if change=="sigma":report["sigma"]=.05
    elif change=="fork":report["fork_commit"]=migration.OLD_FORK
    elif change=="missing":report["rows"].pop()
    elif change=="frobenius":report["rows"][0]["relative_frobenius"]=2.1e-6
    elif change=="negative":report["rows"][0]["relative_frobenius"]=-1
    elif change=="not-finite":report["rows"][0]["relative_frobenius"]=float("nan")
    elif change=="ldl":report["rows"][0]["full_regularization_congruence_ldl_passed"]=False
    else:report["rows"][0]["actual_rows"]=1956
    with pytest.raises(ValueError):migration.validate_probe(report)


def test_exact_gpu_gate_is_accepted():
    assert migration.validate_probe(probe())["sigma"]==.025


def test_native_functions_and_quantization_bit_math_cannot_hide_inside_hessian_migration():
    old="""import torch
DenseRawHessian=base.DenseRawHessian
def quantize_projection(x):
    return x*2
class Native:
    def forward(self,x):
        return x+1
"""
    new=old.replace("DenseRawHessian=base.DenseRawHessian\n",'HESSIAN_ACCUMULATION_CONTRACT="'+migration.CONTRACT+'"\n')+"""
def accumulate_hessian_fp32(H,x):
    return len(x)
class DenseRawHessian(base.DenseRawHessian):
    pass
"""
    assert migration.validate_core_delta(old,new)["protected_old_symbols"]==["Native","quantize_projection"]
    with pytest.raises(ValueError,match="bit math changed"):
        migration.validate_core_delta(old,new.replace("return x*2","return x*3"))
    with pytest.raises(ValueError,match="bit math changed"):
        migration.validate_core_delta(old,new.replace("return x+1","return x+2"))


@pytest.mark.parametrize("field",["source","calibration","runtime","sigma","unknown"])
def test_semantic_plan_gate_rejects_every_non_accumulation_change(tmp_path,field):
    old=plans(tmp_path)
    lineage=dict(predecessor_identities=[old["identity"]],resume_lineage_audit_sha256="a"*64)
    (tmp_path/"producer-lineage.json").write_text(json.dumps(lineage))
    new=new_plan(old,tmp_path)
    if field=="source":new["identity"]["source_repo"]="different/source"
    elif field=="calibration":new["identity"]["calibration"]["tokens"]+=1
    elif field=="runtime":new["identity"]["runtime"]["native"]="different"
    elif field=="sigma":new["identity"]["recipe"]["sigma_reg"]=.05
    else:new["identity"]["unreviewed_native_option"]=True
    with pytest.raises(ValueError,match="choices changed"):
        migration.validate_active_plan(old,new,lineage,"a"*64,sha256_file(tmp_path/"producer-lineage.json"))


def test_activation_archives_exact_plan_bytes_and_preserves_all_existing_receipt_bytes(tmp_path):
    old=plans(tmp_path);old_bytes=(json.dumps(old,indent=4)+"\n").encode()
    (tmp_path/"plan.json").write_bytes(old_bytes)
    legacy=tmp_path/"legacy-receipt.json";legacy.write_bytes(b'{"unchanged": "receipt bytes"}\n')
    payload=tmp_path/"legacy-packed.bin";payload.write_bytes(b"unchanged packed bits")
    objects=migration.Objects();objects.add(legacy);objects.add(payload)
    before={p:(p.read_bytes(),file_identity(p)) for p in (legacy,payload)}
    source=SimpleNamespace(close=lambda:None)
    activated=migration.migrate(tmp_path,tmp_path/"probe.json",auditor=auditor_for(old,objects),
        source_factory=lambda path:source,make_plan=lambda source,tokens,root,export:new_plan(old,root))
    assert (tmp_path/"epochs/000-plan.json").read_bytes()==old_bytes
    assert json.loads((tmp_path/"plan.json").read_text())==activated
    lineage=json.loads((tmp_path/"producer-lineage.json").read_text())
    assert lineage["module_origins"]=={"old.module":migration.digest(old["identity"])}
    assert "active_identity_sha256" not in lineage
    assert all((p.read_bytes(),file_identity(p))==state for p,state in before.items())


def test_failed_semantics_leave_old_active_plan_and_old_data_untouched_then_retry_succeeds(tmp_path):
    old=plans(tmp_path);raw=(json.dumps(old,indent=2)+"\n").encode();(tmp_path/"plan.json").write_bytes(raw)
    payload=tmp_path/"packed.bin";payload.write_bytes(b"original")
    objects=migration.Objects();objects.add(payload)
    source=SimpleNamespace(close=lambda:None)
    def bad(source,tokens,root,export):
        result=new_plan(old,root);result["identity"]["recipe"]["sigma_reg"]=.1;return result
    kwargs=dict(auditor=auditor_for(old,objects),source_factory=lambda path:source)
    with pytest.raises(ValueError,match="choices changed"):
        migration.migrate(tmp_path,tmp_path/"probe.json",make_plan=bad,**kwargs)
    assert (tmp_path/"plan.json").read_bytes()==raw and payload.read_bytes()==b"original"
    # Reuse the same immutable lineage files, replacing only the active plan.
    migration.migrate(tmp_path,tmp_path/"probe.json",make_plan=lambda s,t,r,e:new_plan(old,r),**kwargs)
    assert json.loads((tmp_path/"plan.json").read_text())["identity"]["schema"].endswith("v2")


def test_nonblocking_migration_lock_never_mutates_a_live_producer(tmp_path):
    with (tmp_path/"run.lock").open("a+") as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match="producer is running"):
            with migration.exclusive_migration(tmp_path):
                raise AssertionError("must not enter")


def test_immutable_lineage_outputs_reject_symlinks_or_different_existing_bytes(tmp_path):
    path=tmp_path/"evidence.json"
    migration._write_once(path,b"byte exact original")
    before=file_identity(path)
    migration._write_once(path,b"byte exact original")
    assert file_identity(path)==before
    with pytest.raises(ValueError,match="refuse to replace"):
        migration._write_once(path,b"changed")
    link=tmp_path/"unsafe";link.symlink_to(path)
    with pytest.raises(ValueError,match="symlink"):
        migration._write_once(link,b"wrong")
