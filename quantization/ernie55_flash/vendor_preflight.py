"""Import the measured vLLM-ERNIE receipt for the reference merged in PR #2808."""
import argparse
import math
from pathlib import Path

from . import identity


def normalize(path):
    report = identity.read(path)
    merged = "913fb6eb5d"
    files = identity.committed_files(identity.REPO, *identity.REFERENCE_FILES)
    for name in files:
        if identity.git(identity.REPO, "rev-parse", f"{merged}:{name}") != identity.git(
                identity.REPO, "rev-parse", "HEAD:" + name):
            raise ValueError("this receipt applies to the reference from #2808")
    if (report["passed"] is not True or report["checkpoint"] != identity.BASE_REPO
            or report["revision"] != identity.BASE_REVISION or report["vllm_version"] != "0.32.0"
            or report["vllm_ernie_revision"] != "ba322a5634f91d3e6f96ae711df72ddaf3c38b6e"
            or report["f32_bound"] != 1e-4):
        raise ValueError("unexpected upstream preflight identity or bound")
    checks = report["f32_checks"]
    if (any(not math.isfinite(x) or not 0 <= x <= report["f32_bound"] for x in checks.values())
            or max(checks.values()) != report["max_f32_rel_l2"]):
        raise ValueError("upstream F32 comparisons exceed the published bound")
    if (not report["op_results"]["dspark_drafter"]["tokens_exact_match"]
            or not report["op_results"]["moe_routing"]["ids_exact_match"]
            or any(report["mla_dsa_module_check"][f"layer_{i}"]["exact_topk2048_match_frac"] != 1
                   for i in (7, 42))):
        raise ValueError("upstream discrete selections differ")
    groups = {
        "hca": ["hca_t260_prefix", "rope_hca_yarn_cos", "rope_hca_yarn_sin"],
        "mla_dsa": ["mla_dsa_layer_7_exact_tail", "mla_dsa_layer_42_exact_tail",
                    "rope_plain_8m_cos", "rope_plain_8m_sin", "rope_indexer_neox_q", "rope_indexer_neox_k"],
        "mhc": ["mhc_layer_in", "mhc_post", "hc_head"], "dense": ["trunc_l0"],
        "latent_moe": ["moe_routing_weights", "trunc_l1"], "situ": ["dense_situ_layer0"],
        "lm_head": ["lm_head_multimax_triton"], "dspark": ["dspark_logits", "dspark_confidence"],
    }
    return dict(schema="ernie55-vllm-ernie-parity-v1", status="passed", source_revision=report["revision"],
                reference_files_sha256=identity.digest(files), raw_receipt_sha256=identity.sha256_file(path),
                reference_merge=identity.git(identity.REPO, "rev-parse", merged),
                vllm_ernie_revision=report["vllm_ernie_revision"], vllm_version=report["vllm_version"],
                measurements={k: dict(error=max(checks[n] for n in names), bound=report["f32_bound"],
                                      upstream_checks=names) for k, names in groups.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    identity.atomic_json(args.output, normalize(args.raw))
