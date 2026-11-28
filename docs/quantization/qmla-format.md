# QMLA Sidecar Format Specification (`cuteafd-qmla-v3`)

**Schema:** `cuteafd-qmla-v3`  
**Status:** Normative contract for QMLA sidecar artifacts and third-party loaders (2026-11-28, [#2909](https://github.com/tpurtell/cuteafd/issues/2909)).  
**Design rationale & engine notes:** [`docs/quantization/qmla-design.md`](qmla-design.md).  
**Primary method reference:** *QuantMLA: Function-Aligned Dual-Path Quantization for Low-Bit MLA KV Caching*, Su et al., 2026-09 ([arXiv:2609.36760](https://arxiv.org/abs/2609.36760)).

---

## 1. What QMLA Is

Quantized Multi-Head Latent Attention (QMLA) compresses the per-token KV cache of full-history MLA layers to low-bit integers (2, 4, or 8 bits on the content latent; 4 or 8 bits on the decoupled RoPE key) without mutating any checkpoint weight shard. Before writing a token's latent to the integer cache, the inference engine applies a calibrated, invertible coordinate transform that concentrates quantization error away from attention outputs; at load time, the exact inverse transform is folded into the attention projections so the quantized cache can be read directly with no extra per-step de-rotation.

QMLA v3 stores one selected transform per **layer × path × serving width**, each tagged with an explicit execution kind (`identity`, `scale_only`, or `full`). On the content path, the transform acts on the post-RMSNorm cached latent $c$ during cache write, and its inverse $R_C S_C^{-1}$ is absorbed once at model load into every consumer's `kv_b_proj` (including any drafter that reads the target MLA cache). On the RoPE path, per-frequency-pair rotations and pair-tied scales commute with native RoPE (including 3D M-RoPE and YaRN/LeRoPE), so the extra phase angle and scale fold directly into the RoPE step on $K_P$ and the reciprocal scale on $Q_P$ with zero weight mutation and zero extra matrix multiplications.

---

## 2. Files and Discovery

A QMLA calibration adds **exactly one binary file** at the checkpoint root:

| File | Location | Indexed in `model*.safetensors.index.json`? | Required on `0:0`? |
|---|---|---|---|
| `mla_calibration.safetensors` | Checkpoint root | **Never** | No |

There is **no** separate `mla_calibration.json`.

- **Self-describing header (all checkpoints):** `mla_calibration.safetensors` always embeds the canonical JSON manifest string under `__metadata__["cuteafd_qmla"]`. Every tensor inside the shard is `F32`.
- **MC checkpoints (`cuteafd-mc.json` present):** The full manifest is mirrored under the optional, advisory top-level key `cuteafd-mc.json["qmla"]` (never listed in `required_features`), with two additional whole-file integrity fields:
  - `tensors_bytes`: exact byte length of `mla_calibration.safetensors`;
  - `tensors_sha256`: lowercase 64-hex SHA-256 of the entire `mla_calibration.safetensors` file.
  Stripping `tensors_bytes` and `tensors_sha256` from `cuteafd-mc.json["qmla"]` MUST equal the JSON parsed from `__metadata__["cuteafd_qmla"]` exactly.
- **Plain / non-MC checkpoints (`cuteafd-mc.json` absent):** Readers parse `__metadata__["cuteafd_qmla"]` directly from `mla_calibration.safetensors`. Never create a stub `cuteafd-mc.json` for a non-MC checkpoint.

---

## 3. Per-Width Selections, Explicit Kinds, Tensor Naming, and Aliasing

A serving mode is written `C:R`, where `C ∈ {0, 2, 4, 8}` is the content bit width and `R ∈ {0, 4, 8}` is the RoPE bit width (`0` retains the engine's native cache format for that path; `*:2` INT2 RoPE is forbidden). Mode `C:R` combines:
- the content selection at width `C` (`profiles.content.widths["C"].per_layer["L"]`) when `C > 0`, and
- the RoPE selection at width `R` (`profiles.rope.widths["R"].per_layer["L"]`) when `R > 0`.

On pure-NoPE models (`qk_rope_dim == 0`, e.g. GLM 5.3 Flash), `profiles.rope` is `null`, every layer geometry has `rope_kind: "none"` and `rope_enabled: false`, and `unsupported_modes_reason` explicitly records `*:4` and `*:8` as structurally unsupported.

### Explicit Transform Kinds and Classification Rules

Every `(layer, path, serving_width)` selection carries an explicit `kind` in `{"identity", "scale_only", "full"}`:

| Kind | Content cache write & `kv_b_proj` | RoPE cache write & query | Stored tensors |
|---|---|---|---|
| `identity` | Quantize native $c$ directly; `kv_b_proj` unmodified | Quantize with native RoPE; $Q_P$ unmodified | **None** (`{}`) |
| `scale_only` | Elementwise channel scaling $c \odot s_C$; column-scale `kv_b_proj` by $s_C^{-1}$ | Pair-tied scale $s_P$ on $K_P$, $s_P^{-1}$ on $Q_P$ | `s` only |
| `full` | Dense rotation + scale $(c R_C) \odot s_C$; absorb $R_C S_C^{-1}$ into `kv_b_proj` | Per-pair angle $\theta$ + scale $s_P$ on $K_P$, $\theta$ + $s_P^{-1}$ on $Q_P$ | `r` + `s` (content) or `theta` + `s` (RoPE) |

**Canonical classification tolerances (`kind_tolerances`):**
- `rotation_identity_tol = 1e-6`
  - Content: rotation is identity iff $\|R_C - I\|_\infty = \max_{i,j}|R_{ij} - \delta_{ij}| \le 10^{-6}$.
  - RoPE: rotation is identity iff $\max_i |\operatorname{atan2}(\sin\theta_i, \cos\theta_i)| \le 10^{-6}\text{ rad}$ (wraps $2\pi$ multiples to $0$).
- `scale_identity_tol = 1e-6`
  - Both paths: scale is unit iff $\|s - \mathbf{1}\|_\infty = \max_j |s_j - 1| \le 10^{-6}$.

**Classification cascade (regardless of which optimizer produced the candidate):**
1. If rotation is identity **and** scale is unit $\implies$ `identity`.
2. Else if rotation is identity **and** scale is non-unit $\implies$ `scale_only`.
3. Else $\implies$ `full`.

**Per-width kind freedom:** `full` is a normal per-layer outcome at **any** supported width (including 8 bits), just as `identity` and `scale_only` are valid outcomes at any width. Loaders must dispatch every `(layer, path, width)` by its recorded `kind` without assuming a fixed kind per bit width.

**Why `1e-6`:** It is $>1000\times$ smaller than BF16 half-ULP ($\approx 3.9\times 10^{-3}$) and INT8 step size ($\ge 1/255 \approx 3.9\times 10^{-3}$), so collapsing a near-identity transform changes no quantized representation; simultaneously it absorbs FP32 Cayley/$\exp(0)$ roundoff ($\text{few}\times 10^{-7}$) while remaining $10\times$ tighter than the orthogonality gate ($10^{-5}$). Fixed Sylvester Hadamard is a calibration measurement baseline only and is never shipped.

### Exact Tensor Names, Dtypes, Shapes, and Aliasing Policy

All tensors in `mla_calibration.safetensors` are stored in little-endian IEEE 754 `F32`:

| Path | Kind | `tensors` map keys | Canonical safetensors name | Dtype | Shape |
|---|---|---|---|---|---|
| `content` | `identity` | `{}` | *(none)* | — | — |
| `content` | `scale_only` | `{"s": ...}` | `content.layer_{l}.t{k}.s` | `F32` | `[d_c]` |
| `content` | `full` | `{"r": ..., "s": ...}` | `content.layer_{l}.t{k}.r`, `content.layer_{l}.t{k}.s` | `F32` | `[d_c, d_c]`, `[d_c]` |
| `rope` | `identity` | `{}` | *(none)* | — | — |
| `rope` | `scale_only` | `{"s": ...}` | `rope.layer_{l}.t{k}.s` | `F32` | `[d_r / 2]` |
| `rope` | `full` | `{"theta": ..., "s": ...}` | `rope.layer_{l}.t{k}.theta`, `rope.layer_{l}.t{k}.s` | `F32` | `[d_r / 2]`, `[d_r / 2]` |

**Aliasing / deduplication (`tensor_aliasing: "exact_per_layer_f32_dedup"`):**  
Within each `(path, layer_id)`, the writer walks serving widths in ascending order (`2, 4, 8` for content; `4, 8` for RoPE). Any `identity` selection emits zero tensors. Each distinct non-identity `(kind, raw_f32_bytes)` tuple is assigned the next zero-based slot index `t0`, `t1`, ... and stored **once**. If two serving widths at the same layer select byte-identical F32 tensors of the same kind (for example, 2-bit and 4-bit content both selecting the 4-bit-trained full fit), both width entries reference the exact same `t{k}` tensor names.

**Loader rule:** Always look up tensor names through `profiles.<path>.widths["<bits>"].per_layer["<layer_id>"].tensors`. Never infer a width or kind by parsing `t{k}` slot indices.

---

## 4. Exact Mathematical Contract

All activation vectors are **row vectors** (`vector_layout: "row"`). Linear projection weights $W$ use standard Hugging Face `[out_features, in_features]` layout so `F.linear(x, W) = x @ W^T`.

### 4.1 Content Path (`content_basis_space: "post_norm"`)

Let $x \in \mathbb{R}^{T \times d_{\text{model}}}$ be the layer attention input, `kv_a_proj_with_mqa` produce the raw compressed latent $h_c \in \mathbb{R}^{T \times d_c}$, $u = \mathrm{RMSNorm}_1(h_c)$ be the weightless RMSNorm result (including any native cast before the learned gain), and $\Gamma = \mathrm{Diag}(\gamma)$ be the checkpoint's `kv_a_layernorm.weight`.

The transform acts on the **exact weighted latent $c$ that the checkpoint's native attention path passes to `kv_b_proj`**:
- On checkpoints with a BF16 cache boundary (e.g. GLM 5.3 Flash, Hy4-flash MC), $c = \mathrm{BF16}(u \cdot \Gamma)$ (then promoted to FP32 for the transform math).
- On checkpoints whose native reference consumes an FP32 product directly, $c = u \cdot \Gamma$ in FP32.

`kv_a_proj_with_mqa` and weighted `kv_a_layernorm` always execute unmodified in their native order; $\Gamma$ is **never** folded into `kv_b_proj` compensation.

For a layer's `kv_b_proj` weight $W \in \mathbb{R}^{H(d_{\text{nope}} + d_v) \times d_c}$ (for quantized `kv_b_proj` such as EXL3 in compact views, dequantize to FP32 first, apply compensation in FP32, then split/pack into head-absorbed $W_{UK} \in \mathbb{R}^{H \times d_c \times d_{\text{nope}}}$ and $W_{UV} \in \mathbb{R}^{H \times d_v \times d_c}$):

- **`identity`:**
  $$z_C = c, \qquad \widetilde{W} = W.$$
- **`scale_only`** (vector $s_C \in \mathbb{R}_{>0}^{d_c}$):
  $$z_C = c \odot s_C, \qquad \widetilde{W} = W \cdot \mathrm{Diag}(s_C^{-1}) \quad (\widetilde{W}_{i,j} = W_{i,j} / s_{C,j}).$$
- **`full`** (orthogonal $R_C \in O(d_c)$, vector $s_C \in \mathbb{R}_{>0}^{d_c}$):
  $$z_C = (c R_C) \odot s_C = c R_C \mathrm{Diag}(s_C), \qquad \widetilde{W} = W R_C \mathrm{Diag}(s_C^{-1}).$$

In exact real arithmetic for all three kinds:
$$z_C \widetilde{W}^\top = (c R_C S_C)(S_C^{-1} R_C^\top W^\top) = c W^\top.$$
Any target-KV-sharing speculative drafter (e.g. Hy4 / Hy4-flash DFly2) must compensate its own `kv_b_proj` using the active selection of the target layer it reads.

### 4.2 RoPE Path (`rope_angle_sign: "right_row_phi_plus_theta"`)

Let $d_r = 2P$ ($P = d_r/2$ frequency pairs). For pair $i \in \{0, \dots, P-1\}$, define coordinates $(x_{i,0}, x_{i,1})$ according to the layer geometry's `rope_kind`:
- `interleaved` or `unrotated_pe`: `(2i, 2i + 1)`;
- `split_half`: `(i, i + P)`.

For any angle $\psi$ in row-vector convention:
$$R_i(\psi) = \begin{pmatrix} \cos\psi & \sin\psi \\ -\sin\psi & \cos\psi \end{pmatrix}, \qquad (x_{i,0}, x_{i,1}) R_i(\psi) = \big(x_{i,0}\cos\psi - x_{i,1}\sin\psi,\; x_{i,0}\sin\psi + x_{i,1}\cos\psi\big).$$

Because $SO(2)$ is abelian and $s_i I_2$ commutes with all $2\times 2$ matrices, native RoPE at position $m$ with angle $\phi_{i,m}$ composes directly by angle addition:
$$R_i(\phi_{i,m})\, R_i(\theta_i)\, (s_i I_2) = R_i(\phi_{i,m} + \theta_i)\,(s_i I_2).$$
This holds identically for 1D RoPE, 3D interleaved M-RoPE, YaRN, and LeRoPE.

- **`identity`:** apply native RoPE $\phi_{i,m}$ to $Q_P$ and $K_P$ unchanged.
- **`scale_only`** (vector $s_P \in \mathbb{R}_{>0}^{P}$): multiply both elements of pair $i$ by $s_{P,i}$ on $K_P$ and by $s_{P,i}^{-1}$ on $Q_P$:
  $$(k'_{i,0}, k'_{i,1}) = s_{P,i}\,(k_{i,0}, k_{i,1}), \qquad (q'_{i,0}, q'_{i,1}) = s_{P,i}^{-1}\,(q_{i,0}, q_{i,1}).$$
- **`full`** (angles $\theta \in \mathbb{R}^{P}$, vector $s_P \in \mathbb{R}_{>0}^{P}$):
  $$(k'_{i,0}, k'_{i,1}) = s_{P,i}\,(k_{i,0}, k_{i,1}) R_i(\theta_i), \qquad (q'_{i,0}, q'_{i,1}) = s_{P,i}^{-1}\,(q_{i,0}, q_{i,1}) R_i(\theta_i).$$

Equivalently, a fused RoPE producer kernel adds $\theta_i$ to $\phi_{i,m}$ and scales $K_P$ by $s_{P,i}$ and $Q_P$ by $s_{P,i}^{-1}$. For all three kinds, $Q'_P (K'_P)^\top = Q_P K_P^\top$ in exact arithmetic.

### 4.3 Quantizer and Packed Cache Record Format

Every quantized path uses **per-token asymmetric affine quantization** with group size $G = 64$, round-to-nearest ties-to-even (`rne`), and $\epsilon = 10^{-8}$. For a single token vector $x \in \mathbb{R}^D$ split into $D / G$ contiguous groups $g$ of 64 elements and integer width $b \in \{2, 4, 8\}$ with $M = 2^b - 1$:

1. $x_{\min} = \min_{j \in g} x_j$, $\; x_{\max} = \max_{j \in g} x_j$.
2. $s_{\text{raw}} = \max(x_{\max} - x_{\min},\, 10^{-8}) / M$.
3. Stored scale dtype:
   - **Content:** $s = \max(\mathrm{BF16}(s_{\text{raw}}),\, 10^{-8}/M)$, stored as `BF16` (2 B/group).
   - **RoPE:** $s = \max(s_{\text{raw}},\, 10^{-8}/M)$, stored as `F32` (4 B/group).
4. Zero point: $z = \mathrm{clip}(\mathrm{RNE}(-x_{\min} / s),\, 0,\, M)$, stored as `U8` (1 B/group) for content and `F32` (4 B/group) for RoPE.
5. Integer code: $q = \mathrm{clip}(\mathrm{RNE}(x / s + z),\, 0,\, M) \in \{0, \dots, M\}$.
6. Dequantization: $\hat{x} = s \cdot (q - z)$.

**Bit packing (`pack_order: "lsb_first"`):**
- INT8: 1 code per byte (`uint8`).
- INT4: 2 codes per byte: `q[2k] | (q[2k+1] << 4)`.
- INT2: 4 codes per byte: `q[4k] | (q[4k+1] << 2) | (q[4k+2] << 4) | (q[4k+3] << 6)`.

**16-byte-aligned record layout (for $d_c = 512 \Rightarrow 8$ groups, $d_r = 64 \Rightarrow 1$ group):**
- `C_meta`: 8 × `BF16` scales (16 B) + 8 × `U8` zero points (8 B) = 24 B.
- `R_meta`: 1 × `F32` scale (4 B) + 1 × `F32` zero point (4 B) = 8 B.
- Joint (`C > 0, R > 0`): `C_codes || C_meta || R_meta || zero_pad_to_16B || R_codes` (`4:4` = 256 + 24 + 8 + 32 = **320 B/tok**, matching QuantMLA Eq. 40; `2:4` = **192 B**; `4:8` = **352 B**; `8:8` = **608 B**).
- Content-only (`C > 0, R = 0`): `C_codes || C_meta || 8B_pad || native_BF16_R` (NoPE omits `native_BF16_R`: `4:0` = 288 B, `2:0` = 160 B, `8:0` = 544 B).
- RoPE-only (`C = 0, R > 0`): `native_C || R_codes || R_meta || 8B_pad`.

---

## 5. Protection Policy (`sink4_local128`)

`protection_policy: "sink4_local128"` with `protection_scope: "full_history_mla_only"` is the default for every nonzero `C:R` mode (calibration and serving alike):
- **Which tokens stay unquantized:** For a query at position $t$, keys at indices $j \in \{0, 1, 2, 3\}$ (the 4 attention sinks) and $j \in \{\max(0, t - 127), \dots, t\}$ (the 128 most recent tokens, inclusive of $t$) are attended in exact unquantized precision (BF16 in serving; exact transformed or native values per the layer's kind). All older keys $4 \le j \le t - 128$ are read from the quantized integer cache.
- **Why:** Attention sinks carry disproportionate magnitude and softmax mass, and recent tokens are already held unquantized in the extending staging block before block finalization. Calibrating with `sink4_local128` active avoids spending transform capacity on sink/recent outliers that are never read quantized at runtime.
- **Scope:** Full-history MLA layers only. Bounded sliding-window attention (SWA) rings, linear/KDA recurrence, and DSA indexer keys are **never** touched by QMLA or `sink4_local128`.
- **Storage:** Every token still writes its packed integer record at insertion so window eviction requires zero re-quantization; each active sequence and prefix-cache mark additionally keeps the 4 sink + 128 recent unquantized rows (132 rows per full-history MLA layer).
- `"none"` is permitted only when explicitly calibrated and recorded as `protection_policy: "none"`.

---

## 6. Manifest JSON Schema (`cuteafd-qmla-v3`)

### Annotated Example

```json
{
  "schema": "cuteafd-qmla-v3",
  "artifact_status": "calibrated_unqualified",
  "protection_policy": "sink4_local128",
  "protection_scope": "full_history_mla_only",
  "source": {
    "repo": "wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC",
    "revision": "<40-hex-commit-sha>",
    "base_repo": "tencent/Hy4-flash",
    "base_revision": "<40-hex-commit-sha>",
    "family": "hy4_flash",
    "model_type": "hy_v4_flash",
    "config_sha256": "<64-hex-sha256>",
    "checkpoint_kind": "mc",
    "calibrated_view": "original"
  },
  "applicable_views": ["original", "k8-compact"],
  "view_transfer_validation": {
    "k8-compact": {
      "view": "k8-compact",
      "split": "test",
      "reference": "k8-compact 0:0",
      "source_revision": "<40-hex-commit-sha>",
      "transform_fingerprint": "<64-hex-sha256>",
      "test_digest": "<64-hex-sha256>",
      "combinations": {
        "4:4": {"attn_rel_err": 0.0012, "rtn_attn_rel_err": 0.0020, "hadamard_attn_rel_err": 0.0019, "ratio_to_rtn": 0.60, "layer_tokens": 163625}
      }
    }
  },
  "conventions": {
    "vector_layout": "row",
    "content_basis_space": "post_norm",
    "content_forward_formula": "c @ R_C @ diag(s_C)",
    "content_inverse_formula": "diag(1/s_C) @ R_C^T",
    "rope_angle_sign": "right_row_phi_plus_theta",
    "rope_forward_key_formula": "k_rope @ R_P(theta) @ diag(s_P_pair)",
    "rope_query_compensation_formula": "q_rope @ R_P(theta) @ diag(1/s_P_pair)",
    "storage_dtype": "F32"
  },
  "quantizer_recipe": {
    "scheme": "affine_asymmetric_per_token",
    "group_size": 64,
    "rounding": "rne",
    "content_scale_dtype": "BF16",
    "content_zp_dtype": "U8",
    "rope_scale_dtype": "F32",
    "rope_zp_dtype": "F32",
    "eps_scale": 1e-8,
    "pack_order": "lsb_first"
  },
  "geometry": [
    {
      "layer_id": 0,
      "kv_source_layer": 0,
      "kind": "dsa",
      "heads": 32,
      "kv_lora_rank": 512,
      "qk_nope_dim": 192,
      "qk_rope_dim": 64,
      "v_head_dim": 256,
      "rope_kind": "interleaved",
      "mrope_section": [12, 10, 10],
      "rms_norm_eps": 1e-6,
      "attention_scale": null,
      "sliding_window": null,
      "shared_consumers": [0],
      "content_enabled": true,
      "rope_enabled": true
    }
  ],
  "profiles": {
    "content": {
      "bits": [2, 4, 8],
      "layers": [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41],
      "protection_calibrated": "sink4_local128",
      "selection_objective": "per_width_mean_val_ratio_to_rtn",
      "kind_tolerances": {
        "rotation_identity_tol": 1e-6,
        "scale_identity_tol": 1e-6
      },
      "tensor_aliasing": "exact_per_layer_f32_dedup",
      "widths": {
        "2": {
          "width_fingerprint": "<64-hex-sha256>",
          "kind_counts": {"identity": 0, "scale_only": 0, "full": 12},
          "per_layer": {
            "0": {
              "kind": "full",
              "tensors": {"r": "content.layer_0.t0.r", "s": "content.layer_0.t0.s"},
              "source_candidate": "fitted:c4",
              "source_bits": 4,
              "val_score": {"2:0": 0.0085, "2:4": 0.0091},
              "equiv_max_rel_err": 4.4e-7,
              "orth_max_err": 4.1e-7
            }
          }
        },
        "4": {
          "width_fingerprint": "<64-hex-sha256>",
          "kind_counts": {"identity": 0, "scale_only": 0, "full": 12},
          "per_layer": {
            "0": {
              "kind": "full",
              "tensors": {"r": "content.layer_0.t0.r", "s": "content.layer_0.t0.s"},
              "source_candidate": "fitted:c4",
              "source_bits": 4,
              "val_score": {"4:0": 0.00030, "4:4": 0.00035, "4:8": 0.00031},
              "equiv_max_rel_err": 4.4e-7,
              "orth_max_err": 4.1e-7
            }
          }
        },
        "8": {
          "width_fingerprint": "<64-hex-sha256>",
          "kind_counts": {"identity": 12, "scale_only": 0, "full": 0},
          "per_layer": {
            "0": {
              "kind": "identity",
              "tensors": {},
              "source_candidate": "identity",
              "source_bits": null,
              "val_score": {"8:0": 1.2e-5, "8:8": 1.4e-5},
              "equiv_max_rel_err": 0.0,
              "orth_max_err": 0.0
            }
          }
        }
      },
      "validation": {
        "equivalence_metric": "expanded_kv_rel_l2",
        "equiv_max_rel_err": 4.4e-7,
        "orth_max_err": 4.1e-7
      },
      "profile_fingerprint": "<64-hex-sha256>"
    },
    "rope": {
      "bits": [4, 8],
      "layers": [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41],
      "protection_calibrated": "sink4_local128",
      "selection_objective": "per_width_mean_val_ratio_to_rtn",
      "kind_tolerances": {
        "rotation_identity_tol": 1e-6,
        "scale_identity_tol": 1e-6
      },
      "tensor_aliasing": "exact_per_layer_f32_dedup",
      "widths": {
        "4": {
          "width_fingerprint": "<64-hex-sha256>",
          "kind_counts": {"identity": 0, "scale_only": 0, "full": 12},
          "per_layer": {
            "0": {
              "kind": "full",
              "tensors": {"theta": "rope.layer_0.t0.theta", "s": "rope.layer_0.t0.s"},
              "source_candidate": "fitted:r4",
              "source_bits": 4,
              "val_score": {"0:4": 0.00010, "2:4": 0.0091, "4:4": 0.00035},
              "equiv_max_rel_err": 1.2e-7
            }
          }
        },
        "8": {
          "width_fingerprint": "<64-hex-sha256>",
          "kind_counts": {"identity": 0, "scale_only": 12, "full": 0},
          "per_layer": {
            "0": {
              "kind": "scale_only",
              "tensors": {"s": "rope.layer_0.t1.s"},
              "source_candidate": "scale_fitted:r8",
              "source_bits": 8,
              "val_score": {"0:8": 1.5e-6, "4:8": 0.00031, "8:8": 1.4e-5},
              "equiv_max_rel_err": 8.0e-8
            }
          }
        }
      },
      "validation": {
        "equiv_max_rel_err": 1.2e-7
      },
      "profile_fingerprint": "<64-hex-sha256>"
    }
  },
  "supported_modes": ["2:0", "4:0", "8:0", "0:4", "0:8", "2:4", "2:8", "4:4", "4:8", "8:4", "8:8"],
  "unsupported_modes_reason": {
    "*:2": "INT2 RoPE is not supported"
  },
  "calibration_provenance": {
    "dataset_repo": "wrldsuksgo2mars/agentic-quant-panel-v3",
    "dataset_revision": "e3ac6ee6126f469e37ddf879a6cb2dccc10c6c99",
    "components": {
      "agentic": {"sequences": 1426, "tokens": 1097388, "input_ids_sha256": "<64-hex-sha256>"},
      "exl3_standard": {"sequences": 250, "tokens": 512000, "input_ids_sha256": "<64-hex-sha256>"},
      "long_context": {"sequences": 118, "tokens": 231947, "input_ids_sha256": "<64-hex-sha256>"}
    },
    "train_sequences": 128,
    "val_sequences": 16,
    "test_sequences": 16,
    "splits": {"salt": "cuteafd-qmla-split-v2", "seed": 0, "train": {"count": 128, "seq_ids": ["..."], "sha256": ["..."], "digest": "<64-hex-sha256>"}, "val": {"count": 16, "seq_ids": ["..."], "sha256": ["..."], "digest": "<64-hex-sha256>"}, "test": {"count": 16, "seq_ids": ["..."], "sha256": ["..."], "digest": "<64-hex-sha256>"}},
    "measurement_split": "test",
    "measurement_reference": "original 0:0",
    "fit_budget": {"content": {"2": ["full", "scale_only"], "4": ["full", "scale_only"], "8": ["full", "scale_only"]}, "rope": {"4": ["full", "scale_only"], "8": ["full", "scale_only"]}},
    "combination_measurements": {
      "4:4": {
        "layer_tokens": 163625,
        "attn_rel_err": 0.0011,
        "rtn_attn_rel_err": 0.0020,
        "v2_shared_attn_rel_err": 0.0012,
        "hadamard_attn_rel_err": 0.0019,
        "ratio_to_rtn": 0.55,
        "v2_shared_ratio_to_rtn": 0.60,
        "ratio_to_v2_shared": 0.917
      }
    },
    "input_ids_sha256": "<64-hex-sha256>",
    "renderer_template_sha256": "<64-hex-sha256>",
    "thinking_mode": "high",
    "optimizer": "adamw",
    "steps_per_path": 300,
    "seed": 0,
    "code_commit": "<40-hex-commit-sha>"
  },
  "tensors_file": "mla_calibration.safetensors",
  "tensors_payload_bytes": 12611584,
  "tensors_payload_sha256": "<64-hex-sha256>",
  "transform_fingerprint": "<64-hex-sha256>",
  "tensors_bytes": 12630144,
  "tensors_sha256": "<64-hex-sha256-in-cuteafd-mc-json-only>"
}
```

### Field Reference & Digests
- `artifact_status`: `"calibrated_unqualified"` after offline calibration; promoted to `"runtime_qualified"` only after end-to-end runtime fidelity, long-context, and speed gates pass.
- `source_candidate` / `source_bits`: training-width provenance for the selected transform (`"identity"` with `source_bits: null`, `"fitted:c4"` with `source_bits: 4`, `"scale_fitted:r8"` with `source_bits: 8`, `"smoothquant:c4"` with `source_bits: 4`, etc.). Note that a `fitted:c8` whose tensors fall within `1e-6` of identity still has `kind: "identity"` and `tensors: {}`.
- `tensors_payload_sha256`: SHA-256 of the raw tensor payload section of `mla_calibration.safetensors` (bytes starting immediately after the 8-byte header length + safetensors header).
- `width_fingerprint`: SHA-256 of canonical JSON (`sort_keys=True, separators=(",", ":")`) over `{schema, path, bits, protection_calibrated, layers, per_layer: {lid: {kind, tensors, source_candidate, source_bits}}, tensors: {name: sha256(raw_f32_bytes)}}`.
- `profile_fingerprint`: SHA-256 of canonical JSON over `{schema, path, bits, protection_calibrated, layers, kind_tolerances, tensor_aliasing, widths: {bits: width_fingerprint}}`.
- `transform_fingerprint`: SHA-256 of canonical JSON over `{schema, protection_policy, protection_scope, source_revision, base_revision, geometry, conventions, quantizer_recipe, profiles: {content: fp|null, rope: fp|null}, tensors_payload_sha256}`.
- **Active mode cache identity (`mode_selection_fingerprint`):** Computed at load time for active mode `C:R` as SHA-256 of canonical JSON over `{schema, mode, protection_policy, protection_scope, source_revision, base_revision, geometry, conventions, quantizer_recipe, content_width_fingerprint, rope_width_fingerprint}`. Prefix caches and compiled execution graphs must key on this digest.

---

## 7. Consumer Validation Checklist (Fail Closed)

A loader or validator consuming `mla_calibration.safetensors` MUST verify all of the following and fail closed on any violation:

1. **Schema & conventions:** `schema == "cuteafd-qmla-v3"`, `conventions` matches the `post_norm` row-vector contract, `quantizer_recipe` matches group-64 RNE asymmetric affine, `protection_policy ∈ {"sink4_local128", "none"}`, `protection_scope == "full_history_mla_only"`.
2. **Integrity & MC agreement:**
   - Recompute `tensors_payload_sha256` and `tensors_payload_bytes` over the payload bytes after the safetensors header.
   - On MC checkpoints, recompute whole-file `tensors_sha256` and `tensors_bytes`, verify `cuteafd-mc.json["qmla"]` minus those two keys equals `__metadata__["cuteafd_qmla"]` exactly, and verify `mla_calibration.safetensors` does not appear in any view's `weight_map`.
3. **Fingerprints:** Recompute every `width_fingerprint`, `profile_fingerprint`, and `transform_fingerprint` from the manifest and per-tensor SHA-256 digests.
4. **Tensor inventory & shapes:** Every tensor in the shard is `F32`, finite, and referenced by at least one selection (zero orphan tensors). Every referenced tensor exists with exact shape (`[d_c, d_c]` for `r`, `[d_c]` for content `s`, `[d_r/2]` for RoPE `theta` and `s`) and canonical slot name (`{path}.layer_{l}.t{k}.{role}`).
5. **Scale positivity:** Every stored `s` tensor satisfies $s_j > 0$ elementwise and is finite.
6. **Orthogonality:** Every stored content `r` matrix satisfies $\|R_C^\top R_C - I\|_\infty \le 10^{-5}$.
7. **Canonical kind consistency:**
   - `kind == "identity"` $\implies$ `tensors == {}`.
   - `kind == "scale_only"` $\implies$ `tensors` contains only `"s"`, and $\|s - \mathbf{1}\|_\infty > 10^{-6}$.
   - `kind == "full"` $\implies$ `tensors` contains `"r"+"s"` (content) or `"theta"+"s"` (RoPE), and $\|R_C - I\|_\infty > 10^{-6}$ (content) or $\max_i|\operatorname{atan2}(\sin\theta_i, \cos\theta_i)| > 10^{-6}$ (RoPE).
   - `kind_counts` equals the actual count of each kind across `per_layer`.
8. **Source & geometry match:** `source.config_sha256` and every `geometry` entry match the loaded checkpoint's `config.json` and MLA layer schedule. NoPE layers (`qk_rope_dim == 0`) never appear in `profiles.rope`.
9. **Mode support:** Active serving mode `C:R` (when nonzero) is present in `supported_modes`; otherwise fail closed citing `unsupported_modes_reason`.

---

## 8. Versioning

`cuteafd-qmla-v3` is the sole supported QMLA sidecar schema. Loaders fail closed on any other `schema` string and do not read earlier schemas. Calibration-time candidate checkpoints (`fit-checkpoints/`) and full candidate score tables (`candidate-table.json`) exist only locally during a run so selection policies can be revisited without refitting; they are never published to Hugging Face and are deleted after verified publication and import, retaining only the small JSON receipts.

---

## 9. HF Model Card Note Template

When publishing a checkpoint revision that includes `mla_calibration.safetensors`, include the following section in the repository's `README.md`, replacing `<PINNED_COMMIT_SHA>` with the commit on `tpurtell/cuteafd` carrying this specification and populating the headline table directly from `qmla-validation-report.json["headline_table_markdown"]` (on pure-NoPE models such as GLM Flash, keep `2:4`, `8:8`, and `4:8` explicitly listed as `n/a` with the structural `qk_rope_dim=0` reason, with `4:0` and `8:0` as the measured headline rows):

```markdown
### Calibrated MLA KV Cache (`mla_calibration.safetensors`)

This checkpoint includes an optional `cuteafd-qmla-v3` calibration sidecar (`mla_calibration.safetensors`, never indexed in `model*.safetensors.index.json`; mirrored in `cuteafd-mc.json["qmla"]` on MC checkpoints) providing per-width, per-layer MLA cache-compression transforms (`identity`, `scale_only`, or `full`) for low-bit KV serving without modifying any model weight shard. Loaders that do not enable QMLA ignore the file and run standard `0:0` attention unchanged. See the self-contained tensor and math specification at [`docs/quantization/qmla-format.md`](https://github.com/tpurtell/cuteafd/blob/<PINNED_COMMIT_SHA>/docs/quantization/qmla-format.md).

| Mode (C:R) | Identity (RTN) | Shipped (v3) | Ratio | Reduction | Per-layer spread (median / best / worst) |
|---|---:|---:|---:|---:|---|
| `4:0` | `<rtn>` | `<v3>` | `<ratio>x` | `<pct>%` | `<med> / L<best> / L<worst>` |
| `2:4` | `<rtn or n/a>` | `<v3 or n/a>` | `<ratio or n/a>` | `<pct or n/a>` | `<spread or n/a (reason)>` |
| `8:0` | `<rtn>` | `<v3>` | `<ratio>x` | `<pct>%` | `<med> / L<best> / L<worst>` |
| `8:8` | `<rtn or n/a>` | `<v3 or n/a>` | `<ratio or n/a>` | `<pct or n/a>` | `<spread or n/a (reason)>` |
| `4:8` | `<rtn or n/a>` | `<v3 or n/a>` | `<ratio or n/a>` | `<pct or n/a>` | `<spread or n/a (reason)>` |

*Joint end-to-end post-`o_proj` attention error on held-out test, combining each layer's selected content transform at the content width with its selected RoPE transform at the RoPE width (never per-path numbers added together). `:0` means RoPE is kept unquantized (native format).*
```
