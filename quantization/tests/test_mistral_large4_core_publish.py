"""Completed-run Hub publication checks with fake APIs and tiny local shards."""
import hashlib
import fnmatch
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import struct

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mistral_large4.core_k6 import publish as publication
from kimi_k3.uniform_k2.storage import atomic_json, file_identity

REVISION = "1a" * 20
ATTRIBUTES = b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"


def git_blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


@pytest.fixture
def finished(tmp_path, monkeypatch):
    root, export, cache = tmp_path/"run", tmp_path/"export", tmp_path/"local-hub"
    root.mkdir();export.mkdir()
    monkeypatch.setattr(publication,"SHARD_COUNT",3)
    monkeypatch.setattr(publication,"TARGET_TENSOR_COUNT",3)
    monkeypatch.setattr(publication,"TARGET_PAYLOAD_BYTES",6)
    monkeypatch.setattr(publication,"MODULE_COUNT",1)
    monkeypatch.setattr(publication,"LOCAL_HUB_CACHE",cache)
    identity=dict(schema="toy-complete-producer",source="pinned-fp8",recipe=dict(bits=4,core_bits=6))
    weight_map={}
    for ordinal in range(1,4):
        filename=f"model-{ordinal:05d}-of-00003.safetensors"
        name=f"layer-{ordinal}.native"
        header={name:dict(dtype="BF16",shape=[1],data_offsets=[0,2])}
        encoded=json.dumps(header).encode();encoded+=b" "*(-len(encoded)%8)
        path=export/filename;path.write_bytes(struct.pack("<Q",len(encoded))+encoded+b"\x80\x3f")
        receipt=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),file_identity=file_identity(path))
        receipt.update(identity=identity,filename=filename,payload_bytes=2,tensors=1)
        if ordinal!=3:receipt["layer"]=ordinal-1
        rp=export/(f"layer-{ordinal-1:03d}-receipt.json" if ordinal!=3 else "extras-receipt.json")
        atomic_json(rp,receipt)
        weight_map[name]=filename
    for name in publication.REQUIRED_ASSETS:
        (export/name).write_text("native asset "+name)
    atomic_json(export/"model.safetensors.index.json",dict(metadata=dict(total_size=6),weight_map=weight_map))
    atomic_json(export/"consolidated.safetensors.index.json",dict(metadata=dict(total_size=6),weight_map=weight_map))
    atomic_json(export/"quantization-provenance.json",identity)
    atomic_json(root/"plan.json",dict(identity=identity,root=str(root),export=str(export),publication_repo=publication.TARGET_REPO))
    atomic_json(root/"quantization-complete.json",dict(identity=identity,
        summary=dict(shards=3,tensors=3,payload_bytes=6,quantized_projections=1)))
    atomic_json(root/"completed-export-audit.json",dict(identity=identity,findings=[],
        summary=dict(shards=3,tensors=3,payload_bytes=6,quantized_projections=1)))
    monkeypatch.setattr(publication,"reviewed_modules",lambda:{"layers.71.attention.wq":dict(bits=6,out_features=128,in_features=128)})
    errors=export/"quantization-errors/layer-071";errors.mkdir(parents=True)
    (errors/"projection.json").write_text('{"module":"layers.71.attention.wq","bits":6,"logical_shape":[128,128],"finite":true}')
    ignored=export/".cache/huggingface";ignored.mkdir(parents=True)
    (ignored/"upload-state.json").write_text("private local upload state")
    return SimpleNamespace(root=root,export=export,cache=cache,identity=identity)


@pytest.fixture
def fake_hub(monkeypatch):
    import huggingface_hub
    state=dict(instances=[],uploads=[],creates=[],downloads=[],offline=[],damage=None,
               modern=False,private=False,metadata_lfs=False,mutate_after_info=False)
    def upload(self,**kwargs):state["uploads"].append(kwargs)
    def create(self,*args,**kwargs):state["creates"].append((args,kwargs))
    def model_info(self,repo_id,files_metadata=False,revision=None):
        assert repo_id == publication.TARGET_REPO
        if revision is not None:assert revision==REVISION
        export=Path(state["uploads"][-1]["folder_path"])
        remote=[]
        for path in export.rglob("*"):
            relative=path.relative_to(export).as_posix()
            if (not path.is_file() or path.name.endswith("-receipt.json")
                    or ".cache" in path.relative_to(export).parts):continue
            if any(fnmatch.fnmatchcase(relative,pattern) for pattern in state["uploads"][-1]["ignore_patterns"]):continue
            data=path.read_bytes()
            is_lfs=path.suffix==".safetensors" or (state["metadata_lfs"] and relative=="quantize_config.json")
            remote.append(SimpleNamespace(rfilename=relative,size=len(data),
                lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if is_lfs else None,
                blob_id=None if is_lfs else git_blob(data)))
        if not any(item.rfilename==".gitattributes" for item in remote):
            remote.append(SimpleNamespace(rfilename=".gitattributes",size=len(ATTRIBUTES),lfs=None,blob_id=git_blob(ATTRIBUTES)))
        damage=state["damage"]
        if damage=="lfs":next(item for item in remote if item.lfs is not None).lfs.sha256="f"*64
        elif damage=="metadata":next(item for item in remote if item.rfilename=="config.json").blob_id="f"*40
        elif damage=="size":next(item for item in remote if item.rfilename=="tokenizer.json").size+=1
        elif damage=="attributes":next(item for item in remote if item.rfilename==".gitattributes").blob_id="f"*40
        elif damage=="missing":remote=[item for item in remote if item.rfilename!="config.json"]
        elif damage=="extra":remote.append(SimpleNamespace(rfilename="old-model.safetensors",size=1,lfs=None,blob_id="f"*40))
        elif damage=="not-lfs":next(item for item in remote if item.rfilename.endswith(".safetensors")).lfs=None
        if state["mutate_after_info"]:
            (export/"README.md").write_text("changed during publication after upload")
        return SimpleNamespace(sha=REVISION,siblings=remote,private=state["private"])
    def init(self,**kwargs):state["instances"].append(self);self.kwargs=kwargs
    def download(repo_id,filename,**kwargs):
        state["downloads"].append((repo_id,filename,kwargs))
        path=Path(kwargs["local_dir"])/filename;path.write_bytes(ATTRIBUTES);return str(path)
    def offline(repo_id,**kwargs):
        state["offline"].append((repo_id,kwargs))
        assert kwargs["local_files_only"]
        return str(Path(kwargs["cache_dir"])/("models--"+repo_id.replace("/","--"))/"snapshots"/kwargs["revision"])
    def head(url,token=None):
        assert token is False
        name = url.rsplit("/", 1)[-1]
        path = Path(state["uploads"][-1]["folder_path"]) / name
        return SimpleNamespace(size=path.stat().st_size,etag=hashlib.sha256(path.read_bytes()).hexdigest(),commit_hash=REVISION)
    def install(*,modern=False):
        state["modern"]=modern
        attrs=dict(__init__=init,create_repo=create,model_info=model_info)
        attrs["upload_folder" if modern else "upload_large_folder"]=upload
        api=type("FakeApi",(),attrs)
        monkeypatch.setattr(huggingface_hub,"HfApi",api)
        monkeypatch.setattr(huggingface_hub,"hf_hub_download",download)
        monkeypatch.setattr(huggingface_hub,"snapshot_download",offline)
        monkeypatch.setattr(huggingface_hub,"get_hf_file_metadata",head)
        monkeypatch.setattr("huggingface_hub.utils._runtime.is_xet_available",lambda:True)
        return state
    state["install"]=install
    install()
    return state


@pytest.mark.parametrize("modern",[False,True])
def test_publish_audits_hardlinks_offline_cache_and_idempotent_completion(finished,fake_hub,modern):
    fake_hub["install"](modern=modern)
    fake_hub["metadata_lfs"]=True
    revision=publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    assert revision==REVISION
    marker=json.loads((finished.root/"publication-complete.json").read_text())
    snapshot=Path(marker["local_snapshot"])
    assert snapshot.is_relative_to(finished.cache)
    assert (snapshot.parents[1]/"refs/main").read_text()==REVISION
    assert marker["identity"]==finished.identity
    assert not any(path.name.endswith("-receipt.json") for path in snapshot.rglob("*"))
    assert not (snapshot/".cache").exists()
    for path in snapshot.rglob("*"):
        if path.is_file():assert os.path.samefile(path,finished.export/path.relative_to(snapshot))
    assert fake_hub["uploads"][0]["ignore_patterns"]==publication.IGNORE_PATTERNS
    assert fake_hub["offline"][0][1]["cache_dir"]==finished.cache
    assert fake_hub["downloads"][0][2]["revision"]==REVISION
    assert fake_hub["creates"][0][1]["private"] is False
    assert fake_hub["instances"][-1].kwargs == dict(token=False)
    assert publication.publish(finished.export,finished.root,publication.TARGET_REPO)==REVISION
    assert len(fake_hub["uploads"])==1


def test_complete_diagnostics_are_bundled_and_raw_records_stay_local(finished,fake_hub):
    source=finished.export/"quantization-errors/layer-071/projection.json"
    original=source.read_bytes()
    publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    errors=finished.export/"quantization-errors";bundle=errors/"layer-071.jsonl"
    assert [json.loads(line) for line in bundle.read_text().splitlines()]==[json.loads(original)]
    assert source.read_bytes()==original
    index=json.loads((errors/"index.json").read_text())
    assert index["records"]==1 and index["bundles"][bundle.name]["records"]==1
    assert index["bundles"][bundle.name]["sha256"]==hashlib.sha256(bundle.read_bytes()).hexdigest()
    marker=json.loads((finished.root/"publication-complete.json").read_text());snapshot=Path(marker["local_snapshot"])
    assert os.path.samefile(snapshot/"quantization-errors/layer-071.jsonl",bundle)
    assert not (snapshot/"quantization-errors/layer-071/projection.json").exists()
    inode=bundle.stat().st_ino
    assert publication.publish(finished.export,finished.root,publication.TARGET_REPO)==REVISION
    assert bundle.stat().st_ino==inode and len(fake_hub["uploads"])==1


def test_changed_diagnostics_cannot_rewrite_an_already_published_bundle(finished,fake_hub):
    publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    bundle=finished.export/"quantization-errors/layer-071.jsonl";before=bundle.read_bytes()
    source=finished.export/"quantization-errors/layer-071/projection.json"
    changed=json.loads(source.read_text());changed["finite"]=False;source.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="completed diagnostic bundle differs"):
        publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    assert bundle.read_bytes()==before and len(fake_hub["uploads"])==1


@pytest.mark.parametrize("damage",["missing","symlink","invalid-json"])
def test_unsafe_or_incomplete_diagnostic_sources_fail_before_hub(finished,fake_hub,damage,tmp_path):
    path=finished.export/"quantization-errors/layer-071/projection.json"
    if damage=="missing":path.unlink()
    elif damage=="symlink":
        outside=tmp_path/"outside.json";outside.write_text('{}');path.unlink();path.symlink_to(outside)
    else:path.write_text('invalid JSON')
    with pytest.raises(ValueError):publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    assert not fake_hub["instances"]


@pytest.mark.parametrize("damage",["identity","summary","paths","target","index","provenance","receipt","missing","checksum","audit","native_index"])
def test_uncompleted_or_changed_runs_fail_before_any_hub_call(finished,fake_hub,damage):
    root,export=finished.root,finished.export
    if damage=="missing":(root/"quantization-complete.json").unlink()
    elif damage in ("identity","summary"):
        p=root/"quantization-complete.json";data=json.loads(p.read_text())
        if damage=="identity":data["identity"]["source"]="different"
        else:data["summary"]["shards"]=94
        atomic_json(p,data)
    elif damage in ("paths","target"):
        p=root/"plan.json";data=json.loads(p.read_text())
        if damage=="paths":data["export"]=str(export/"wrong")
        else:data["publication_repo"]="wrldsuksgo2mars/Kimi-K3-EXL3-K2"
        atomic_json(p,data)
    elif damage=="index":
        p=export/"model.safetensors.index.json";data=json.loads(p.read_text());data["metadata"]["total_size"]=0;atomic_json(p,data)
    elif damage=="provenance":atomic_json(export/"quantization-provenance.json",dict(source="different"))
    elif damage=="audit":
        p=root/"completed-export-audit.json";data=json.loads(p.read_text());data["findings"]=["damaged native bytes"];atomic_json(p,data)
    elif damage=="native_index":
        atomic_json(export/"consolidated.safetensors.index.json",dict(metadata=dict(total_size=0),weight_map={}))
    elif damage=="receipt":
        p=export/"layer-000-receipt.json";data=json.loads(p.read_text());data["identity"]["source"]="different";atomic_json(p,data)
    else:
        path=export/"model-00001-of-00003.safetensors"
        receipt=json.loads((export/"layer-000-receipt.json").read_text())
        with path.open("r+b") as f:f.seek(-1,2);f.write(b"\xff")
        os.utime(path,ns=(path.stat().st_atime_ns,receipt["file_identity"]["mtime_ns"]))
    with pytest.raises((ValueError,FileNotFoundError)):
        publication.publish(export,root,publication.TARGET_REPO)
    assert not fake_hub["instances"] and not fake_hub["uploads"]


def test_wrong_external_repository_is_rejected(finished,fake_hub):
    with pytest.raises(ValueError,match="target differs"):
        publication.publish(finished.export,finished.root,"wrldsuksgo2mars/Kimi-K3-EXL3-K2")
    assert not fake_hub["instances"]


@pytest.mark.parametrize("damage",["lfs","metadata","size","attributes","missing","extra","not-lfs"])
def test_remote_failures_never_publish_completion(finished,fake_hub,damage):
    fake_hub["damage"]=damage
    with pytest.raises(ValueError):publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    assert not (finished.root/"publication-complete.json").exists()


def test_private_repository_cannot_be_marked_publication_complete(finished,fake_hub):
    fake_hub["private"]=True
    with pytest.raises(ValueError,match="not public"):
        publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    assert not (finished.root/"publication-complete.json").exists()


def test_cross_filesystem_cache_fails_before_upload(finished,fake_hub,monkeypatch):
    original=Path.stat
    def stat(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        if path==finished.cache:
            fields=list(value);fields[2]+=1;return os.stat_result(fields)
        return value
    monkeypatch.setattr(Path,"stat",stat)
    with pytest.raises(ValueError,match="export filesystem"):
        publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    assert not fake_hub["uploads"]


def test_conflicting_existing_snapshot_is_not_overwritten(finished,fake_hub):
    snapshot=finished.cache/("models--"+publication.TARGET_REPO.replace("/","--"))/"snapshots"/REVISION
    snapshot.mkdir(parents=True)
    (snapshot/"LICENSE").write_text("different existing cache object")
    with pytest.raises(ValueError,match="existing snapshot file differs"):
        publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    assert (snapshot/"LICENSE").read_text()=="different existing cache object"
    assert not (finished.root/"publication-complete.json").exists()


def test_changed_local_metadata_invalidates_publication_marker(finished,fake_hub):
    publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    (finished.export/"README.md").write_text("changed after publication")
    with pytest.raises(ValueError,match="completion identity or payload"):
        publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    assert len(fake_hub["uploads"])==1


def test_export_mutation_during_publication_cannot_produce_a_cache_marker(finished,fake_hub):
    fake_hub["mutate_after_info"]=True
    with pytest.raises(ValueError,match="committed file changed"):
        publication.publish(finished.export,finished.root,publication.TARGET_REPO)
    assert not (finished.root/"publication-complete.json").exists()


def test_all_73_receipt_names_are_unambiguous(tmp_path,monkeypatch):
    monkeypatch.setattr(publication,"SHARD_COUNT",73)
    for ordinal in range(1,74):
        path=publication._receipt_path(tmp_path,f"model-{ordinal:05d}-of-00073.safetensors")
        assert path.name==("extras-receipt.json" if ordinal==73 else f"layer-{ordinal-1:03d}-receipt.json")
    with pytest.raises(ValueError):publication._receipt_path(tmp_path,"model-00094-of-00094.safetensors")
