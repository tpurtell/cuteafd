"""Run the full causal ERNIE MC quantization from verified inputs and sealed frontiers."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch

from kimi_k3.uniform_k2.storage import atomic_json, fsync_directory
from . import identity
from .bootstrap import SOURCE, TASK, load_bootstrap
from .export import DirectMCWriter, PUBLISHED, projection_inventory
from .first_encodes import add_packed, packed_identity
from .forward import ErnieForward, reference
from .forward_preflight import embed
from .prepare_corpus import load_token_batches
from .production_core import process_layer
from .source import HeaderSource, BASE_REVISION
from .storage import Stages


def now():
    return datetime.now(timezone.utc).isoformat()


def write_errors(root, source):
    _, core, experts = projection_inventory(source)
    expected = {(module, b) for module in core for b in (6, 8)} | {(m, 4) for m in experts}
    files = sorted((root / "metrics").glob("*.json"))
    keys = set()
    seen = set()
    for path in files:
        for row in identity.read(path):
            ident = row["module"], row["bits"]
            if ident in seen:
                raise ValueError(f"duplicate projection metric: {ident}")
            seen.add(ident)
            keys.update(row)
    if seen != expected:
        raise ValueError(f"projection error table differs: {len(expected-seen)} missing, {len(seen-expected)} extra")
    target = root/"per-module-errors.csv"
    temp = target.with_suffix(".partial")
    with temp.open("w",newline="") as stream:
        writer = csv.DictWriter(stream, ["module","bits",*sorted(keys-{"module","bits"})])
        writer.writeheader()
        for path in files:
            for row in identity.read(path):
                writer.writerow({k:json.dumps(v,separators=(",",":")) if isinstance(v,(dict,list)) else v
                                 for k,v in row.items()})
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(target)
    fsync_directory(root)
    return target


class Production:
    def __init__(self, run, snapshot, frozen):
        self.run = run
        self.root = run/"production"
        self.frozen = frozen
        self.identity = identity.digest(frozen)
        self.stages = Stages(self.root/"stages", self.identity)
        self.model = ErnieForward.from_snapshot(snapshot,"cuda:0")
        self.source = HeaderSource(snapshot)
        self.writer = DirectMCWriter(run/PUBLISHED.split("/")[-1], self.source, frozen)
        self.counts, self.tokens, self.ids = {}, {}, []
        for split in ("calibration","heldout"):
            rows = [b["input_ids"][0] for b in load_token_batches(run/"corpus", split)]
            self.counts[split] = len(rows)
            self.tokens[split] = sum(len(ids) for ids in rows)
            self.ids.extend(rows)
        self.count = len(self.ids)
        self.family = self.index = None
        self.embedding = None

    def event(self,phase,**fields):
        record = dict(phase=phase, family=self.family, layer=self.index, time=now(), **fields)
        atomic_json(self.root/"status.json",record)
        print(json.dumps(record),flush=True)

    def embedded(self,start):
        if self.embedding is None:
            self.embedding = self.model.checkpoint.load(reference.OUTER_NAMES[0])
        for i in range(start,self.count):
            yield i, {"state":embed(self.ids[i],self.embedding,self.model.config)}

    def records(self,name,context=None):
        def read(start):
            for i, row in self.stages.records(name, start=start, device="cuda:0"):
                if context and context != name:
                    row.update(self.stages.fields(context,i,("context","context_positions","positions"),
                                                 device="cuda:0"))
                yield i,row
        return read

    def block(self,family,index,prior,*,context=None):
        from .qualified_run import bootstrap_key
        self.family, self.index = family, index
        label = f"{family}-{index:03d}"
        frontier = "frontier-"+label
        receipt_path = self.root/"layers"/(label+".json")
        if receipt_path.is_file():
            receipt = identity.read(receipt_path)
            if receipt["identity"] != self.identity or not self.writer.group_complete(label):
                raise ValueError(f"{label}: committed block no longer matches export or producer")
            self.stages.retire(*receipt["retired"])
            return frontier
        self.event("layer_start")
        start = time.monotonic()
        mark = self.root/"started"/(label+".json")
        interrupted = mark.exists()
        if not interrupted:
            atomic_json(mark,dict(started_at=now(),identity=self.identity))
        bootstrap = None
        records = self.records(prior,context) if prior else self.embedded
        if not prior:
            inputs = self.frozen["inputs"]
            key = bootstrap_key(inputs)
            packs = Stages(self.run/"bootstrap-packed", packed_identity(inputs,key))
            bootstrap = add_packed(packs,load_bootstrap(self.run,key))
        tokens = self.tokens if family=="layers" else {s:n*self.model.config.dspark_block_size for s,n in self.counts.items()}
        result = process_layer(self.model,self.stages,family,index,records,self.counts,tokens,self.event,bootstrap)
        self.event("write_frontier",stage=frontier)
        self.stages.write_activations(frontier,result["outputs"],self.count)
        if family=="layers" and index in self.model.config.dspark_target_layer_ids:
            def taps(start_at):
                for i,row in self.stages.records(frontier,start=start_at,device="cuda:0"):
                    yield i, {"tap":row["state"].mean(dim=1)}
            self.stages.write_activations(f"tap-{index}",taps,self.count)
        self.event("write_mc_block",stage=label)
        block = self.writer.write_block(family,index,k4=result["k4"],k8=result["k8"],k6=result["k6"])
        if not prior:
            self.verify_grouped_output(block)
        metrics = result["metrics"]
        atomic_json(self.root/"metrics"/(label+".json"),metrics)
        retired = list(result["stages"])
        if prior and prior != context:
            retired.append(prior)
        self.stages.load(frontier)
        atomic_json(receipt_path,dict(status="passed",identity=self.identity,family=family,index=index,
                    frontier=frontier,seconds=time.monotonic()-start,interrupted=interrupted,
                    projections=len(metrics),completed_at=now(),retired=retired))
        self.stages.retire(*retired)
        self.embedding = None
        self.model.checkpoint.close()
        return frontier

    def verify_grouped_output(self, group):
        from hy4_flash.run import reconstruction
        from .hca_format import load_grouped_weight
        parent = "model.layers.0.self_attn.linear_o_group_proj"
        config = self.source.document
        rank = config["o_lora_rank"]
        for bits in (8,6):
            mapping = {n:s["filename"] for s in group["shards"] if s["kind"] == f"k{bits}-compact"
                       for n in s["names"]}
            loaded = load_grouped_weight(self.writer.output,mapping,parent,config,bits,device="cuda:0")
            for i,part in enumerate(loaded.split(rank)):
                values = self.stages.read_batch("layers-000-attn-grouped",i)
                expected = reconstruction(values,f"{parent}.groups.{i}",bits,part.shape,"cuda:0")
                if not torch.equal(part,expected):
                    raise ValueError(f"published HCA group {i} K{bits} differs from its sealed reconstruction")
        atomic_json(self.root/"grouped-hca-roundtrip.json",
                    dict(status="passed",identity=self.identity,groups=config["o_groups"],bits=[8,6]))

    def execute(self):
        from .production_draft import prepare_draft
        c = self.model.config
        prior = None
        for i in range(c.num_hidden_layers):
            prior = self.block("layers",i,prior)
        self.family,self.index = "mtp",None
        taps = [f"tap-{i}" for i in c.dspark_target_layer_ids]
        self.event("draft_feature")
        k8,k6,metrics,context = prepare_draft(self.model,self.stages,self.counts,list(map(len,self.ids)),
                                              taps,prior,self.event)
        self.writer.write_block("outer",k4={},k8=k8,k6=k6)
        atomic_json(self.root/"metrics/outer.json",metrics)
        prior = context
        for i in range(c.num_mtp_stages):
            prior = self.block("mtp",i,prior,context=context)
        self.family = self.index = None
        self.event("package_audit")
        errors = write_errors(self.root,self.source)
        receipt = self.writer.finalize(error_csv=errors,card_path=self.root/"README.md")
        atomic_json(self.root/"numerical-complete.json",dict(status="passed",identity=self.writer.identity,
                    output=str(self.writer.output),repository=PUBLISHED,revision=BASE_REVISION,
                    completed_at=now(),package_validation=str(self.writer.output/"package-validation.json"),
                    audit=receipt))
        self.event("numerical_complete")


@torch.inference_mode()
def main():
    from .qualified_run import session
    root = TASK/"production"
    with session(root,SOURCE,TASK/"corpus",identity.read(TASK/"preflight/vendor-merged.json"),
                 identity.read(TASK/"preflight/forward-parity.json")) as frozen:
        job = Production(TASK,SOURCE,frozen)
        try:
            job.execute()
        except Exception as exc:
            job.event("failed",reason=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
