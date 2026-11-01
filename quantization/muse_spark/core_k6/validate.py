"""Independent CPU reload, native-byte and mixed EXL3 checkpoint audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'quantization'))
if __package__ in (None,''):
    __package__='muse_spark.core_k6'
from kimi_k3.uniform_k2.storage import atomic_json,canonical,read_header,verify_receipt,sha256_file
from .source import MuseSource
from .recipe import (BASE_REPO,BASE_REVISION,TARGET_PAYLOAD_BYTES,TARGET_TENSOR_COUNT,
    MODULE_COUNT,MCG_MULTIPLIER,reviewed_modules,inventory,removed_source_names,packed_specs)
from .export import SHARD_COUNT,_target_shard,_spec,_preserved_audit,_check_packed


def audit_completed(source,root,export,*,require_complete=True):
    root,export=Path(root).expanduser().resolve(),Path(export).expanduser().resolve()
    plan=json.loads((root/'plan.json').read_text());identity=plan['identity']
    if (plan.get('publication_repo')!='wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-Core-K6'
            or Path(plan['root']).resolve()!=root or Path(plan['export']).resolve()!=export
            or identity.get('source_repo')!=BASE_REPO or identity.get('source_revision')!=BASE_REVISION):
        raise ValueError('completed export plan identity/paths differ')
    if require_complete:
        complete=json.loads((root/'quantization-complete.json').read_text())
        if complete.get('identity')!=identity:
            raise ValueError('completed export marker producer differs')
    original=inventory(source);selection=reviewed_modules();removed=removed_source_names(selection)
    original_headers={}
    native={}
    for name,shard in source.weight_map.items():
        if name not in removed:
            if shard not in original_headers:original_headers[shard]=read_header(source.snapshot/shard)[0]
            native[name]=_spec(original_headers[shard],name)
    expected_map={name:_target_shard(name) for name in native}
    expected_specs=dict(native)
    for module,g in selection.items():
        for name,spec in packed_specs(module,g).items():
            expected_map[name]=_target_shard(name);expected_specs[name]=spec
    index=json.loads((export/'model.safetensors.index.json').read_text())
    if index!={'metadata':{'total_size':TARGET_PAYLOAD_BYTES},'weight_map':expected_map}:
        raise ValueError('completed index differs from independent expected map')
    expected_shards=set(expected_map.values())
    if {p.name for p in export.glob('*.safetensors')}!=expected_shards or len(expected_shards)!=SHARD_COUNT:
        raise ValueError('completed shard inventory differs')
    storage=json.loads((export/'quantize_config.json').read_text())
    if (set(storage.get('tensor_storage',{}))!=set(selection)
            or storage.get('core_bits')!=6 or storage.get('head_bits')!=6
            or storage.get('bits')!=4 or storage.get('codebook')!='mcg'
            or storage.get('group_size')!=-1 or storage.get('out_scales')!='auto'
            or storage.get('per_module_metadata')!='quantize_config.json:tensor_storage'):
        raise ValueError('completed quantization metadata differs')
    config=json.loads((export/'config.json').read_text());declaration=config.pop('quantization_config',None)
    if config!=source.document or not declaration or declaration.get('core_bits')!=6 or declaration.get('head_bits')!=6:
        raise ValueError('completed source config/declaration differs')
    if json.loads((export/'quantization-provenance.json').read_text())!=identity:
        raise ValueError('completed provenance differs')
    expected_native={name:dict(dtype=dtype,shape=shape) for name,(dtype,shape) in native.items()}
    if storage.get('retained_native_tensors')!=expected_native:
        raise ValueError('retained-native metadata differs')
    total,count,preserved,shards=0,0,0,{}
    modules_by_shard={}
    for module,g in selection.items():modules_by_shard.setdefault(_target_shard(module),{})[module]=g
    for shard in sorted(expected_shards):
        path=export/shard;header,base=read_header(path)
        expected={n:spec for n,spec in expected_specs.items() if expected_map[n]==shard}
        if {n:_spec(header,n) for n in header if n!='__metadata__'}!=expected:
            raise ValueError(f'completed header differs: {shard}')
        ordinal=int(shard.split('-')[1])
        receipt_path=export/('extras-receipt.json' if ordinal==SHARD_COUNT else f'layer-{ordinal-1:03d}-receipt.json')
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('identity')!=identity or receipt.get('filename')!=shard:
            raise ValueError('completed shard receipt identity differs')
        verify_receipt(path,receipt,checksum=True)
        originals={n:(source.snapshot/source.weight_map[n],n) for n in native if expected_map[n]==shard}
        native_audit=_preserved_audit(originals,path)
        if native_audit!=receipt.get('preserved_audit'):
            raise ValueError('independent retained-native audit differs')
        preserved+=native_audit['payload_bytes']
        for module,g in modules_by_shard.get(shard,{}).items():
            _check_packed(module,g,path,header_and_base=(header,base))
            entry=storage['tensor_storage'][module]
            expected_tensors={n:dict(shape=shape,torch_dtype={'I16':'int16','F16':'float16','I32':'int32'}[dtype])
                for n,(dtype,shape) in packed_specs(module,g).items()}
            if entry!=dict(quant_format='exl3',bits_per_weight=g['bits'],mcg_multiplier=MCG_MULTIPLIER,
                in_features=g['in_features'],out_features=g['out_features'],stored_out_features=g['stored_out_features'],
                sharded_side=1,stored_tensors=expected_tensors):
                raise ValueError('module storage declaration differs')
        size=sum(v['data_offsets'][1]-v['data_offsets'][0] for n,v in header.items() if n!='__metadata__')
        total+=size;count+=len(expected)
        shards[shard]=dict(sha256=receipt['sha256'],payload_bytes=size,retained=native_audit)
    if total!=TARGET_PAYLOAD_BYTES or count!=TARGET_TENSOR_COUNT or preserved!=original['preserved_bytes']:
        raise ValueError('independent completed payload totals differ')
    # Verify every copied asset against its original bytes (never infer from file names).
    excluded={'config.json','quantize_config.json','model.safetensors.index.json','README.md','.gitattributes','quantization-provenance.json'}
    assets={}
    for path in source.snapshot.iterdir():
        if not path.is_file() or path.name in excluded or path.name.endswith('.safetensors'):continue
        target=export/path.name
        if target.is_symlink() or not target.is_file() or sha256_file(path)!=sha256_file(target):
            raise ValueError(f'completed source asset differs: {path.name}')
        assets[path.name]=dict(bytes=target.stat().st_size,sha256=sha256_file(target))
    evidence={}
    for path in (export/'quantization-errors').rglob('*.json'):
        # Bundle indexes are publication metadata; original module records live in subdirs.
        if path.parent==export/'quantization-errors':continue
        if path.is_symlink():raise ValueError('unsafe numerical evidence')
        record=json.loads(path.read_text());module=record.get('module')
        if (module not in selection or module in evidence or record.get('bits')!=selection[module]['bits']
                or record.get('logical_shape')!=[selection[module]['out_features'],selection[module]['in_features']]
                or not record.get('metrics') or not record.get('evidence')):
            raise ValueError('completed numerical record differs')
        canonical(record);evidence[module]=record
    if set(evidence)!=set(selection):raise ValueError('completed numerical record inventory differs')
    return dict(status='passed',identity=identity,shards=shards,assets=assets,tensors=count,
        payload_bytes=total,preserved_tensors=len(native),preserved_bytes=preserved,
        numerical_records=len(evidence),finished=datetime.now(timezone.utc).isoformat())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-snapshot','root','export','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=audit_completed(MuseSource(args.source_snapshot),args.root,args.export)
    atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('identity','shards','assets')}),flush=True)


if __name__=='__main__':main()
