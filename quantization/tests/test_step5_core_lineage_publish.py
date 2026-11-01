"""Legacy shard receipts stay immutable and fully hashed during publication."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step5.core_k6 import publish
from step5.core_k6.compat import identity_digest
from kimi_k3.uniform_k2.storage import atomic_json,file_identity,sha256_file
from test_step5_core_compat import identities,manifest
from test_step5_core_export import save


@pytest.fixture
def completed(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir();out=tmp_path/'export';out.mkdir()
    old,current=identities();module='lm_head';manifest(root,old,current,module)
    for n in ('producer-lineage.json','resume-lineage-audit.json'):(out/n).write_bytes((root/n).read_bytes())
    monkeypatch.setattr(publish,'SHARD_COUNT',1)
    monkeypatch.setattr(publish,'TARGET_TENSOR_COUNT',1)
    monkeypatch.setattr(publish,'TARGET_PAYLOAD_BYTES',2)
    monkeypatch.setattr(publish,'MODULE_COUNT',1)
    for name in publish.REQUIRED_ASSETS:(out/name).write_text('native '+name)
    name='model-00001-of-00001.safetensors';path=save(out/name,{'native':('BF16',[1],1)})
    receipt=dict(identity=old,filename=name,sha256=sha256_file(path),file_identity=file_identity(path),payload_bytes=2,tensors=1)
    atomic_json(out/'extras-receipt.json',receipt)
    atomic_json(out/'model.safetensors.index.json',dict(metadata=dict(total_size=2),weight_map={'native':name}))
    atomic_json(out/'quantization-provenance.json',current)
    atomic_json(out/'quantize_config.json',dict(tensor_storage={module:dict(producer_identity_sha256=identity_digest(old))}))
    atomic_json(root/'plan.json',dict(identity=current,root=str(root),export=str(out),publication_repo=publish.TARGET_REPO))
    atomic_json(root/'quantization-complete.json',dict(identity=current,summary=dict(shards=1,tensors=1,payload_bytes=2,quantized_projections=1)))
    return root,out,old,current


def test_old_receipt_is_accepted_only_under_bound_explicit_lineage(completed):
    root,out,old,current=completed;p=out/'extras-receipt.json';before=p.read_bytes()
    assert publish._complete_plan(out,root,publish.TARGET_REPO)==current
    files=publish._collect_local(out,current)
    assert 'model-00001-of-00001.safetensors' in files
    assert p.read_bytes()==before
    unrelated=deepcopy(current);unrelated['recipe']['core_bits']=5
    with pytest.raises(ValueError,match='receipt'):publish._collect_local(out,unrelated)


@pytest.mark.parametrize('damage',['manifest','audit','origin_metadata','missing_origin','provenance','completion','shard_bytes'])
def test_publication_rejects_origin_or_byte_damage(completed,damage):
    root,out,old,current=completed
    if damage=='manifest':(out/'producer-lineage.json').write_text('{}')
    elif damage=='audit':(out/'resume-lineage-audit.json').write_text('{}')
    elif damage in ('origin_metadata','missing_origin'):
        value={} if damage=='missing_origin' else dict(tensor_storage={'lm_head':dict(producer_identity_sha256=identity_digest(current))})
        atomic_json(out/'quantize_config.json',value)
    elif damage=='provenance':atomic_json(out/'quantization-provenance.json',old)
    elif damage=='completion':
        v=json.loads((root/'quantization-complete.json').read_text());v['identity']=old;atomic_json(root/'quantization-complete.json',v)
    elif damage=='shard_bytes':
        p=out/'model-00001-of-00001.safetensors'
        with p.open('r+b') as f:f.seek(-1,2);f.write(b'\xff')
    with pytest.raises(ValueError):
        publish._complete_plan(out,root,publish.TARGET_REPO)
        publish._collect_local(out,current)
