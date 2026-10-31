"""Only explicitly audited, same-input Step-5 K128-XTX resume identities cross."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kimi_k3.uniform_k2.storage import canonical
from step5.core_k6.compat import (ACCUMULATION,accepts_identity,identity_digest,
                                load_module_origins,module_producer_digest)


def identities():
    old=dict(schema='cuteafd-step5-k4-core-k6-production-v1',
        source_repo='stepfun-ai/Step-5-Preview-BF16',source_revision='a'*40,
        source_config_sha256='b'*64,source_index_sha256='c'*64,token_manifest_sha256='d'*64,
        calibration=dict(samples=1426,tokens=12345),packages=dict(torch='2.14.1+cu130',numpy='2.2.6'),
        recipe=dict(core_bits=6,expert_bits=4,late_expert_bits=6,late_expert_layers=[88,89,90],
                    mtp_bits=4,seed=787,sigma_reg=.025,hessian='signed-block-hadamard-congruence-fp64-v1'),
        code={'quantization/step5/core_k6/run.py':'e'*64},gptqmodel_commit='f'*40)
    current=deepcopy(old);current['schema']='cuteafd-step5-k4-core-k6-production-v2'
    current['code']['quantization/step5/core_k6/run.py']='0'*64;current['gptqmodel_commit']='1'*40
    current['recipe']['hessian_accumulation']=ACCUMULATION
    current.update(predecessor_identities=[deepcopy(old)],resume_lineage_audit_sha256='2'*64,resume_lineage_manifest_sha256='3'*64)
    return old,current


def manifest(root,old,current,module='model.layers.0.self_attn.q_proj'):
    audit=(canonical(dict(status="passed",prior_identity_sha256=identity_digest(old)))+'\n').encode()
    (root/'resume-lineage-audit.json').write_bytes(audit)
    current['resume_lineage_audit_sha256']=hashlib.sha256(audit).hexdigest()
    value=dict(predecessor_identities=[old],resume_lineage_audit_sha256=current['resume_lineage_audit_sha256'],
               module_origins={module:identity_digest(old)},original_plan='old-plan.json')
    raw=(canonical(value)+'\n').encode();(root/'producer-lineage.json').write_bytes(raw)
    current['resume_lineage_manifest_sha256']=hashlib.sha256(raw).hexdigest()
    return value


def test_exact_identity_unchanged_and_explicit_compatible_predecessor_only():
    old,current=identities()
    assert accepts_identity(old,old) and accepts_identity('legacy-toy','legacy-toy')
    assert accepts_identity(old,current)
    unknown=deepcopy(old);unknown['code']['unknown']='4'*64
    assert not accepts_identity(unknown,current)
    del current['predecessor_identities'];assert not accepts_identity(old,current)


@pytest.mark.parametrize('field',['source_repo','source_revision','source_config_sha256','source_index_sha256',
                                  'token_manifest_sha256','calibration','packages','extra_runtime'])
def test_explicit_listing_cannot_override_changed_inputs_or_runtime(field):
    old,current=identities()
    current[field]={'changed':True}
    assert not accepts_identity(old,current)


@pytest.mark.parametrize('field',['core_bits','expert_bits','late_expert_bits','late_expert_layers','mtp_bits','seed','sigma_reg','hessian','new_semantics'])
def test_explicit_listing_cannot_override_recipe_or_damping_change(field):
    old,current=identities();current['recipe'][field]='changed'
    assert not accepts_identity(old,current)


@pytest.mark.parametrize('damage',['wrong_accumulation','missing_accumulation','missing_audit','missing_manifest','malformed_audit','bad_predecessor'])
def test_incomplete_or_other_numerical_lineage_rejected(damage):
    old,current=identities()
    if damage=='wrong_accumulation':current['recipe']['hessian_accumulation']='k2048'
    elif damage=='missing_accumulation':del current['recipe']['hessian_accumulation']
    elif damage=='missing_audit':del current['resume_lineage_audit_sha256']
    elif damage=='missing_manifest':del current['resume_lineage_manifest_sha256']
    elif damage=='malformed_audit':current['resume_lineage_audit_sha256']='not-a-digest'
    elif damage=='bad_predecessor':current['predecessor_identities']='all'
    assert not accepts_identity(old,current)


def test_module_origin_manifest_is_bound_and_unknown_modules_default_active(tmp_path):
    old,current=identities();m='model.layers.0.self_attn.q_proj';value=manifest(tmp_path,old,current,m)
    loaded=load_module_origins(tmp_path,current)
    assert loaded==value
    assert module_producer_digest(m,current,loaded)==identity_digest(old)
    assert module_producer_digest('model.layers.4.self_attn.q_proj',current,loaded)==identity_digest(current)
    with pytest.raises(ValueError,match='manifest'):module_producer_digest(m,current)
    assert load_module_origins(tmp_path,old) is None


@pytest.mark.parametrize('damage',['bytes','audit','audit_bytes','predecessor','origin','symlink'])
def test_module_origin_manifest_rejects_changed_bytes_or_unreviewed_origins(tmp_path,damage):
    old,current=identities();value=manifest(tmp_path,old,current)
    path=tmp_path/'producer-lineage.json'
    if damage=='bytes':path.write_text(path.read_text()+' ')
    elif damage=='audit_bytes':(tmp_path/'resume-lineage-audit.json').write_text('{}')
    elif damage=='symlink':
        other=tmp_path/'other.json';path.rename(other);path.symlink_to(other)
    else:
        if damage=='audit':value['resume_lineage_audit_sha256']='f'*64
        elif damage=='predecessor':value['predecessor_identities']=[]
        elif damage=='origin':value['module_origins']['model.layers.0.self_attn.q_proj']='f'*64
        raw=(canonical(value)+'\n').encode();path.write_bytes(raw)
        current['resume_lineage_manifest_sha256']=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):load_module_origins(tmp_path,current)


def test_other_identity_schema_is_not_admitted_by_this_specific_migration():
    old,current=identities();old['schema']='another-model-v1'
    current['predecessor_identities']=[old]
    assert not accepts_identity(old,current)
