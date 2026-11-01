"""Bounded direct BF16 span reads from the pinned Muse checkpoint."""
from __future__ import annotations

import json
import math
from pathlib import Path

from kimi_k3.uniform_k2.storage import read_header
from .recipe import BASE_REVISION, EXPERT_COUNT, LAYER_COUNT


class MuseSource:
    def __init__(self,snapshot):
        self.snapshot=Path(snapshot).expanduser().absolute()
        self.document=json.loads((self.snapshot/'config.json').read_text())
        self.config=self.document['text_config']
        self.weight_map=json.loads((self.snapshot/'model.safetensors.index.json').read_text())['weight_map']
        self._headers={}

    def header(self,name):
        shard=self.weight_map[name]
        if not isinstance(shard,str) or '/' in shard or '\\' in shard or not shard.endswith('.safetensors'):
            raise ValueError('invalid source shard basename')
        if shard not in self._headers:
            self._headers[shard]=read_header(self.snapshot/shard)
        header,base=self._headers[shard]
        return self.snapshot/shard,header[name],base

    def span(self,name,*,offset=0,length=None):
        path,spec,base=self.header(name)
        start,end=spec['data_offsets'];extent=end-start
        length=extent-offset if length is None else length
        if type(offset)is not int or type(length)is not int or offset<0 or length<0 or offset+length>extent:
            raise ValueError('source tensor slice outside declared span')
        return path,base+start+offset,length

    def _read(self,name,shape,*,offset=0,device='cpu'):
        import torch
        _,spec,_=self.header(name)
        dtype={'BF16':torch.bfloat16,'F32':torch.float32}.get(spec['dtype'])
        if dtype is None:
            raise ValueError('unsupported Muse source tensor dtype')
        length=math.prod(shape)*{'BF16':2,'F32':4}[spec['dtype']]
        path,position,length=self.span(name,offset=offset,length=length)
        tensor=torch.empty(shape,dtype=dtype,device='cpu')
        buffer=memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast('B')
        with path.open('rb',buffering=0) as file:
            file.seek(position)
            cursor=0
            while cursor<length:
                count=file.readinto(buffer[cursor:min(cursor+16*1024*1024,length)])
                if not count:
                    raise ValueError('truncated Muse source tensor slice')
                cursor+=count
        return tensor.to(device=device)

    def tensor(self,name,device='cpu'):
        _,spec,_=self.header(name)
        if name.endswith(('.experts.gate_up_proj','.experts.down_proj')):
            raise ValueError('read fused Muse expert operands through expert()')
        return self._read(name,spec['shape'],device=device)

    def expert(self,layer,expert,part,device='cpu'):
        if type(layer)is not int or not 0<=layer<LAYER_COUNT or type(expert)is not int or not 0<=expert<EXPERT_COUNT:
            raise ValueError('Muse expert/layer outside geometry')
        if part not in ('gate_proj','up_proj','down_proj'):
            raise ValueError('unknown Muse expert operand')
        fused='down_proj' if part=='down_proj' else 'gate_up_proj'
        name=f'model.language_model.layers.{layer}.mlp.experts.{fused}'
        _,spec,_=self.header(name)
        expected=[256,4096 if fused=='down_proj' else 8192,4096]
        if spec['dtype']!='BF16' or spec['shape']!=expected:
            raise ValueError('Muse fused expert source geometry differs')
        matrix=4096*4096*2
        offset=(expert if fused=='down_proj' else expert*2+(part=='up_proj'))*matrix
        return self._read(name,[4096,4096],offset=offset,device=device)

    def load_block(self, layer, device='cuda:0', **kwargs):
        from .numerical import load_block
        return load_block(self, layer, device=device, **kwargs)

    def rotary(self, device='cuda:0'):
        from .numerical import rotary
        return rotary(self, device=device)

    def close(self):
        pass  # Header metadata only: no long-lived mappings or descriptors.


MuseBF16Source = MuseSource
