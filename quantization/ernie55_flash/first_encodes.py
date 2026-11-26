"""Encode the four full-corpus layer-zero input matrices from their shared raw Hessians."""
import torch
from hy4_flash import gates
from hy4_flash.run import encode, reconstruction
from kimi_k3.uniform_k2.storage import atomic_json

from . import identity
from .bootstrap import INPUTS, PREFIX, SOURCE, TASK, inputs_key, load_bootstrap
from .forward import ErnieForward, reference
from .forward_preflight import embed
from .prepare_corpus import load_token_batches
from .storage import Stages


def packed_identity(inputs, key):
    paths = ("quantization/ernie55_flash/first_encodes.py", "quantization/hy4_flash/run.py",
             "quantization/hy4_flash/gates.py", "quantization/hy4_flash/forward.py")
    return identity.digest(dict(hessians=key, files=identity.committed_files(identity.REPO,*paths)))


def add_packed(stages, raw):
    for n, name in enumerate(INPUTS):
        label = f"input-{n}"
        manifest = stages.load(label, checksum=True)
        if manifest is None:
            raise ValueError(f"missing sealed {label}")
        full = PREFIX + name + ".weight"
        values = stages.read_batch(label,0)
        raw[full].update(packed={b:{k:v for k,v in values.items() if k.startswith(f"k{b}.")} for b in (8,6)},
                         metrics=manifest["metadata"]["metrics"])
    return raw


@torch.inference_mode()
def main():
    inputs = identity.candidate(SOURCE, TASK/"corpus", identity.REPO)
    key = inputs_key(inputs)
    raw = load_bootstrap(TASK,key)
    stages = Stages(TASK/"bootstrap-packed",packed_identity(inputs,key))
    model = ErnieForward.from_snapshot(SOURCE,"cuda")
    c, weights = model.config, model.load_layer(0)
    embedding = model.checkpoint.load(reference.OUTER_NAMES[0])
    held = {"all":[], "compressor":[]}
    for batch in load_token_batches(TASK/"corpus","heldout"):
        h = embed(batch["input_ids"][0],embedding,c)
        _,_,x = reference.mhc_pre(h,weights,"self_attention_hyper_connection",weights["input_layernorm.weight"],c)
        held["all"].append(x[-16:].clone())
        stop = len(x)//reference.COMPRESS_RATIO_HCA*reference.COMPRESS_RATIO_HCA
        if stop:
            held["compressor"].append(x[max(0,stop-16):stop].clone())
    for n,(name,kind) in enumerate(INPUTS.items()):
        label = f"input-{n}"
        if stages.load(label) is not None:
            continue
        module = PREFIX+name
        w = weights["self_attn."+name+".weight"]
        metrics = []
        values = encode(module,w,raw[module+".weight"],(8,6),"cuda",metrics)
        x = torch.cat(held[kind])
        for row in metrics:
            matrix = reconstruction(values,module,row["bits"],w.shape,"cuda")
            gates.evaluated(row,x,w,matrix,"last-16-actual-inputs-per-heldout-prompt",
                            projection=lambda _,rows,weight: reference.linear(rows,weight))
        stages.write_batches(label,[(0,values)],dict(metrics=metrics))
        print(f"Sealed production K8/K6 {module}",flush=True)
    atomic_json(TASK/"bootstrap-packed/complete.json",dict(status="passed",identity=stages.identity,projections=4))


if __name__ == "__main__":
    main()
