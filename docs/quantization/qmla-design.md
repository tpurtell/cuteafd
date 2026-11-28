# CuteAFD QMLA Design: Shared Calibrated MLA Cache Compression

**Status:** Phase 0 design **confirmed by TJ** (2026-11-28, tracking [#2909](https://github.com/tpurtell/cuteafd/issues/2909), target **v8**, not for v7).  
**Primary reference:** *QuantMLA: Function-Aligned Dual-Path Quantization for Low-Bit MLA KV Caching*, Su et al., 2026-09 ([arXiv:2609.36760](https://arxiv.org/abs/2609.36760)).  
**Comparison baseline:** `gr8` fixed-Hadamard INT4 MLA (`KV4.md`, Sylvester $H_{512}$, 416 B/tok).

---

## 1. Faithful vs. Adapted

Every deviation from the QuantMLA paper is explicitly marked **`[Adaptation]`**.

### Faithful to QuantMLA (arXiv:2609.36760)
- **Dual-path error model (§2):** Content latent $C \in \mathbb{R}^{n \times d_c}$ enters both QK matching and value aggregation (plus their interaction); decoupled RoPE key $K_P \in \mathbb{R}^{n \times d_r}$ enters positional matching only and amplifies matched cache NMSE into $1.94\times$–$6.86\times$ larger attention-output distortion.
- **Content transformation space (§3.1, Eq. 6):** Orthogonal rotation $R_C \in O(d_c)$ (Cayley-parameterized during learning) and positive channel scale $S_C = \mathrm{Diag}(s_C)$ (log-scale clamped to $[-2, 2]$) acting in **weightless-RMSNorm coordinates** $u = \mathrm{RMSNorm}_1(x)$:
  $$z_C = u R_C S_C, \qquad \widetilde{W}_b = S_C^{-1} R_C^\top \Gamma W_b \implies z_C \widetilde{W}_b = c W_b \quad (c = u\Gamma).$$
- **RoPE transformation space (§3.1, Prop. 1, Eq. 7):** Per-frequency-pair planar rotations $R_P = \mathrm{blockdiag}(R_i)$, $R_i \in SO(2)$, and pair-tied positive scales $S_P = \mathrm{blockdiag}(s_i I_2)$, $s_i > 0$, with reciprocal query compensation:
  $$Q'_P = Q_P R_P S_P^{-1}, \qquad K'_P = K_P R_P S_P \implies Q'_P (K'_P)^\top = Q_P K_P^\top.$$
- **Function-aligned objectives (§3.2, Eqs. 8 & 10):**
  - Content path $\mathcal{L}_C$: normalized post-`o_proj` attention-output MSE against frozen full-precision reference, with QDQ in the loop on content and exact RoPE ($\epsilon_C = N_Y \cdot 10^{-8}$).
  - RoPE path $\mathcal{L}_P$: normalized causal positional-QK MSE over all heads and valid causal pairs $\Omega$, with QDQ in the loop on $K'_P$, unquantized compensated $Q'_P$, and exact content ($\epsilon_P = 10^{-8}$, bounding $\|\Delta Y_P\|_F^2$ by Prop. 2).
- **Quantizer & optimizer (Appendix C.4, Table 7):** Per-token asymmetric affine quantization, group size $G = 64$, $s = \max(x_{\max}-x_{\min}, 10^{-8})/(2^b - 1)$, $z = \mathrm{clip}(\mathrm{round}(-x_{\min}/s), 0, 2^b - 1)$, round-to-nearest-even (STE in calibration), BF16 content scales and FP32 RoPE scales. AdamW (weight decay 0), 300 steps/path, 10% warmup + cosine decay, grad clip 1.0, LRs $3\times 10^{-3}$ (content $R_C$) / $10^{-3}$ (content $S_C$, RoPE $\theta, s$), anchor $10^{-4}$, 99.9th-percentile scale-init grid $\alpha \in \{0, 0.125, 0.25, 0.5, 0.75, 1\}$, validation every 20 steps (incl. step 0). Calibration runs *without* BF16 token protection.
- **Equivalence gate (Appendix C.5):** Quantization-disabled forward equivalence is verified before any quantized evaluation.

### CuteAFD Adaptations
1. **`[Adaptation A1 — Single sidecar file + load-time UK/UV absorption + write-side forward transform]`:** The paper fuses $(R_C, S_C, R_P, S_P)$ offline by mutating checkpoint weights (`kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `q_b_proj`). CuteAFD keeps all checkpoint shards and MC views (`original`, `k8-compact`, `k6-compact`, `fp8-compact`, native FP8/NVFP4) **byte-for-byte immutable** and stores transforms in one compact root sidecar file (`mla_calibration.safetensors`), with its metadata in `cuteafd-mc.json["qmla"]` for MC checkpoints (or in `mla_calibration.safetensors`'s `__metadata__` header for non-MC checkpoints; §5). At load time we absorb $\Gamma R_C S_C^{-1}$ into `W_UK`/`W_UV` (zero history-row inverse cost on decode); at write time we apply one forward transform per newly produced token (§2) and fuse the tiny per-pair RoPE phase/scale directly into the RoPE step (§3). On uncompressed BF16 `original` views, load-time producer folding into `kv_a_proj_with_mqa` remains an optional zero-overhead optimization using the exact same `unit_rms` sidecar.
2. **`[Adaptation A2 — Orthogonal --quant-mla C:R CLI contract]`:** Exposes independent integer widths per path (`0:0`, `8:8`, `4:8`, `4:4`, `2:4`, `4:0`, `0:4`, `8:0`, `0:8`), where `0` retains the family's existing native cache representation (FP8 or BF16 per `--kv-cache`) rather than forcing BF16.
3. **`[Adaptation A3 — Pinned CuteAFD quant panel via panel_corpus (agentic + exl3_standard + long_context)]`:** Calibrates on the panel pinned by `quantization/panel_corpus.py`, `wrldsuksgo2mars/agentic-quant-panel-v3@e3ac6ee6126f469e37ddf879a6cb2dccc10c6c99`, with all three components in order: **`agentic`** (chat-rendered with the checkpoint's own template), **`exl3_standard`** (packed with this tokenizer) and **`long_context`** (16K/32K multi-turn traces, rendered once and cut into 2,048-token windows) — never WikiText-2 alone and never the held-out Standard Fidelity panel. QMLA keeps no pin of its own: it resolves through `panel_corpus.resolve_panel_policy` (full = not pilot; a full run with fewer than three components fails closed) and loads through `panel_corpus.load_component_records`, recording the exact `dataset_repo`, `dataset_revision` and components in `calibration_provenance`. `agentic-quant-panel-v2@54683731eb` (two components) stays pinned for past runs. Captures match each family's exact masks, sparse DSA top-k selection, SWA windows, learnable sinks, output gates, post-proj norm, VHA postmix, and M-RoPE / YaRN / LeRoPE positions.
4. **`[Adaptation A4 — Default sink4/local128 protection in both calibration and serving; SWA untouched]`:** The paper calibrates unprotected and enables `sink4/local128` BF16 protection only at evaluation (and leaves shared-prefix ownership outside scope, Appendix C.5). CuteAFD enables **`sink4_local128` by default from the start in both calibration and runtime** for every nonzero `QUANT_MLA` mode on **full-history MLA layers only (never SWA)**: recent rows are already kept unquantized to form/quantize extending blocks properly, and calibrating with `sink4_local128` active avoids spending transform capacity on sink/recent tokens that are never quantized at runtime. `none` is retained only as a measured opt-in alternative if protection later proves unnecessary. Admission, exact prefix marks, and CUDA graph identity account for `sink4_local128` from Phase 3 onward. SWA layers (e.g., Hy4-flash's 30 bounded SWA-512 rings) are completely untouched by QMLA and retain their `--kv-cache` format (§4).
5. **`[Adaptation A5 — Multi-family MLA extensions]`:** Explicitly handles pure-NoPE MLA (`glm5_flash`), unrotated 64-dim PE slots (`kimi_k3`/`kimi_k31`), alternating RoPE/NoPE + cross-layer KV sharing + scalable softmax (`mistral_large4`), 3D interleaved M-RoPE (`hy4_flash`/`hy4`), EXL3-quantized `kv_b_proj` in Hy4 compact views (already dequantized to absorbed BF16 at load), and target-KV-sharing drafters (Hy4/Hy4-flash DFly2).

---

## 2. The Content Path

### Algebra and Coordinate Convention
In HF `F.linear` layout, `kv_b_proj.weight` has shape $[H(d_{\text{nope}} + d_v), d_c]$ (row $r$ is $w_r \in \mathbb{R}^{1 \times d_c}$). For raw latent $x \in \mathbb{R}^{1 \times d_c}$, weightless normalized latent $u = \mathrm{RMSNorm}_1(x)$, and `kv_a_layernorm.weight` $\Gamma = \mathrm{Diag}(\gamma)$, the native latent is $c = u\Gamma$.

We calibrate and store content transforms in **weightless `unit_rms` coordinates** (matching paper Eq. 6):
- **Forward write transform:** $z_C = u M_C$, where $M_C = R_C S_C \in \mathbb{R}^{d_c \times d_c}$ (pre-multiplied at load time into one BF16 matrix).
- **Consumer compensation (in HF `[out, in]` layout):**
  $$\widetilde{W}_{\text{kv\_b}} = W_{\text{kv\_b}} \, \Gamma \, R_C \, S_C^{-1} \quad\Longrightarrow\quad z_C \widetilde{W}_{\text{kv\_b}}^\top = (u R_C S_C)(S_C^{-1} R_C^\top \Gamma W_{\text{kv\_b}}^\top) = c W_{\text{kv\_b}}^\top.$$
  Splitting $\widetilde{W}_{\text{kv\_b}}$ per head into absorbed $W_{UK}$ (`[H, d_c, d_nope]`) and $W_{UV}$ (`[H, d_v, d_c]`) preserves both content QK scores $(q_{C,h} \widetilde{W}_{UK,h}) z_C^\top$ and value aggregation $(A_h Z_C) \widetilde{W}_{UV,h}^\top$ identically.

**Why `unit_rms` instead of `post_norm` ($c' = c R_C S_C$):**
1. Working on $u$ removes `kv_a_layernorm`'s channel gain variance *before* rotation, matching the paper's exact transformation space and optimization behavior.
2. At runtime, weightless `RMSNorm_1(x)` avoids multiplying by $\gamma$ before $M_C$ (one fewer vector load), while $\Gamma$ is absorbed once for free into $\widetilde{W}_{\text{kv\_b}}$ at load time.
3. Because $\mathrm{RMSNorm}_1(x R_C) = \mathrm{RMSNorm}_1(x) R_C$, the **exact same sidecar** supports both runtime write-side transform (universal) and optional load-time producer folding into BF16 `kv_a_proj_with_mqa`.

### Runtime Write-Side Transform + Load-Time UK/UV Absorption vs. Full Offline/Load-Time Folding

| Dimension | Runtime write-side $z_C = u M_C$ + load-time UK/UV absorption (**Default**) | Full folding into `kv_a_proj_with_mqa` + `kv_a_layernorm` (Paper offline method) |
|---|---|---|
| **Weighted RMSNorm (`kv_a_layernorm`)** | Handled cleanly: producer runs weightless `RMSNorm_1`, $\Gamma$ absorbed into $\widetilde{W}_{\text{kv\_b}}$. | Only possible in `unit_rms` space ($W'_a = W_a R_C$, $\Gamma' = S_C$); impossible for arbitrary post-norm transforms. |
| **Fused `w_qkv_a` (`formats/mla.rs`)** | `w_qkv_a` untouched; transform applies only to the 512-dim content slice after projection. | Must slice and rewrite rows `[q_rank .. q_rank + kv_rank]` of `w_qkv_a` per bit-mode. |
| **EXL3 compact cores (`k8`, `k6`)** | **Works directly** — `kv_a_proj_with_mqa` stays EXL3 trellis; no core re-encoding. | **Incompatible** without dequantizing `kv_a_proj_with_mqa` to BF16 or re-quantizing EXL3 per bit-mode. |
| **Source FP8 / `fp8-compact`** | **Works directly** — `kv_a_proj_with_mqa` keeps its exact 128×128 block FP8 bytes/scales. | Destroys calibrated 128×128 FP8 block scales; requires dequant→rotate→requant drift. |
| **Incremental sidecar updates (Phase 2)** | **Zero weight shard changes** — add sidecar only. | Requires rewriting checkpoint weight shards (breaks Phase 2). |
| **Expanded prefill & target-sharing drafters** | Prefill & DFly2 consume the same compensated $\widetilde{W}_{\text{kv\_b}}$ (DFly2's own `kv_b_proj` absorbs its tapped target layer's $\Gamma_\ell R_{C,\ell} S_{C,\ell}^{-1}$ at load). | Same requirement on `kv_b_proj` and drafters. |
| **History-row decode cost** | **Zero** — history stays in $z_C$ coordinates; $W_{UK}, W_{UV}$ are already compensated. | **Zero**. |
| **New-token write cost** | One $B \times 512 \times 512$ BF16 GEMV/GEMM per MLA layer (estimated below). | **Zero** (optional load-time fold fast path on BF16 `original` views only, using the same `unit_rms` sidecar). |

*Note on `kv_b_proj` formats:* Across GLM 5.3, GLM 5.3 Flash, Kimi K3/K3.1, Mistral Large 4, and ERNIE 5.5 Flash, `kv_b_proj` is native BF16 or FP8 and absorbed at load time. In Hy4 / Hy4-flash compact views (`k8`/`k6`), `kv_b_proj` is EXL3-encoded but CuteAFD *already* dequantizes it to absorbed BF16 `W_UK`/`W_UV` at load (`hy4_common/native.rs:401`). Applying $\Gamma R_C S_C^{-1}$ in FP32 immediately after dequantization and before BF16 (or FP8-absorbed) packing adds zero permanent memory and preserves the single-arena contract.

### Write-Side Transform Cost: Dense GEMV vs. Structured Alternatives
For $d_c = 512$, $M_C = R_C S_C$ in BF16 is **512 KiB per MLA layer** ($512 \times 512 \times 2$ B):
- **Coordinator VRAM for $M_C$ arena:** 6.0 MiB (Hy4-flash, 12 DSA layers), 5.5 MiB (GLM 5.3 Flash, 11 MLA), 12.0 MiB (Kimi K3/K3.1, 24 MLA), 31.5 MiB (Mistral Large 4, 63 unique KV sources), 39.0 MiB (GLM 5.3 / Hy4, 78 MLA), 3.0 MiB (ERNIE 5.5 Flash, 6 MLA-DSA).
- **Decode (C1–C16, $B \le 16$ new tokens/step):** Memory-bandwidth bound on RTX PRO 6000 (~1.79 TB/s GDDR7): $512\text{ KiB} / 1.79\text{ TB/s} \approx 0.29\,\mu\text{s}$ raw data movement (~1–2 µs/layer fused with `RMSNorm_1` + QMLA pack/write). Total per step: **~12–24 µs** on Hy4-flash (12 DSA) and **~80–150 µs** on 78-layer GLM 5.3 / Hy4 (<0.5% of step time).
- **Prefill (8K chunk = 8,192 rows):** Compute-bound BF16 Tensor Core GEMM ($8192 \times 512 \times 512 \times 2 = 4.29\text{ GFLOPs/layer} \approx 8\text{–}12\,\mu\text{s/layer}$, <1 ms total across 78 layers).
- **Structured alternatives (`[Adaptation — NOT QuantMLA]`):**
  - *Fixed Sylvester Hadamard $H_{512}$ (`gr8` KV4 / `QuaRot†`):* 0 weight bytes, $O(d_c \log d_c) = 4{,}608$ ops/token (~0.3 µs in registers). Retained strictly as an **uncalibrated baseline**, not a replacement: paper Tables 1–2 show fixed Hadamard trails QuantMLA by 2.7–5.8 pts at C4R4 and collapses at C2R4 (e.g., GLM-4.7-Flash 31.23% vs 51.43%).
  - *Kronecker ($Q_{16} \otimes Q_{32}$) or block-64 orthogonal + permutation:* 2–64 KiB/layer (fits in registers/SMEM), 8–16× fewer FLOPs, lower expressiveness than full $O(512)$. Only investigate if Phase 3 profiling shows dense $512\times 512$ exceeds budget; always labeled distinctly in schema (`content_transform_kind: "dense_orth"` vs `"kronecker_orth"` / `"fixed_hadamard"`), never mislabeled as QuantMLA.

---

## 3. The RoPE Path

### Per-Frequency-Pair Algebra & Zero-Extra-Matmul Execution
For $d_r = 64$ ($P = 32$ frequency pairs), each pair $i \in \{0, \dots, P-1\}$ has one trainable angle $\theta_i \in \mathbb{R}$ and one positive pair-tied scale $s_i > 0$ ($\log s_i \in [-2, 2]$). In row-vector notation, define the canonical $2\times 2$ block:
$$R_i(\theta_i) = \begin{pmatrix} \cos\theta_i & \sin\theta_i \\ -\sin\theta_i & \cos\theta_i \end{pmatrix} \in SO(2), \qquad S_i = s_i I_2.$$
Standard RoPE at position $m$ applies $R_i(\phi_{i,m})$ to pair $i$ of both $Q_P$ and $K_P$:
$$(y_0, y_1) = (x_0 \cos\phi - x_1 \sin\phi,\; x_0 \sin\phi + x_1 \cos\phi) = (x_0, x_1) R_i(\phi).$$
Because $SO(2)$ is abelian and scalar $s_i I_2$ commutes with all $2\times 2$ matrices:
$$R_i(\phi_{i,m}) R_i(\theta_i) (s_i I_2) = R_i(\phi_{i,m} + \theta_i) (s_i I_2).$$
**Key consequence for CuteAFD runtime:** Applying QMLA's RoPE transform requires **neither weight folding nor an extra matmul**. The existing RoPE producer kernel simply adds the learned per-pair phase offset $\theta_i$ to $\phi_{i,m}$ and multiplies by $s_i$ on $K_P$ and $s_i^{-1}$ on $Q_P$ (exact across all query heads and all core views: BF16, FP8, EXL3 K8/K6). Cost: 32 F32 adds + 32 F32 muls per token/head (~0 µs hidden in existing RoPE memory bandwidth). **Cross-frequency rotations are forbidden** for RoPE layers (they break commutation with position angles and mix M-RoPE axes).

### Pairing Conventions, M-RoPE, YaRN/LeRoPE, and NoPE Exceptions (Verified on `dev`)
- **Interleaved `(2i, 2i+1)`:** `hy4_flash`, `hy4` (settled PR #50589, `hy4/golden.py:20`), `glm5` (`rope_interleave=true`), `mistral_large4` (interleaved YaRN + LeRoPE), and `ernie55_flash` MLA-DSA layers (`ernie55_flash/model.py:835-843`; its *indexer* uses split-half, untouched by QMLA).
- **Split-half / NeoX `(i, i + P)`:** Supported in the schema (`rope_pairing: "split_half"`) using the exact same $R_i(\theta_i), s_i$ on coordinate pair `(i, i + P)` for any future split-half MLA family.
- **3D Interleaved M-RoPE (`hy4_flash`, `hy4`):** `mrope_section = [12, 10, 10]` assigns each of the 32 pairs to one axis (T, H, or W). Because $R_i(\theta_i)$ and $s_i I_2$ act strictly *within* single pair $i$, **they commute identically with 3D M-RoPE** for text, image, and video tokens. Validation explicitly tests non-degenerate 3D positions (`pos_T != pos_H != pos_W`).
- **YaRN, LeRoPE, and Scalable Softmax (`mistral_large4`):** YaRN and LeRoPE (`inv_freq * exp(lerope.alpha)`) only modify per-pair angle $\phi_{i,m}$; scalable softmax (`s_h * log(pos+1) + b_h`) is a per-head scalar. Both commute identically with $(R_P, S_P)$.
- **NoPE exceptions:**
  - **Pure-NoPE MLA (`glm5_flash`, all 11 MLA layers; `mistral_large4` 18 NoPE layers `{3, 7, ..., 71}`):** `qk_rope_dim == 0`. Manifest sets `rope_kind: "none"`, `rope_enabled: false`. On `glm5_flash`, `C:0` modes are valid and any explicit `R > 0` fails closed with the exact reason. On `mistral_large4`, `C:R` applies RoPE compression to the 54 RoPE layers and leaves the 18 NoPE layers content-only.
  - **Unrotated 64-dim PE slot (`kimi_k3`, `kimi_k31`):** `mla_use_nope = true`, yet `qk_rope_head_dim = 64` is concatenated *without* RoPE (`kimi_k3/config.rs:75`). Manifest marks `rope_kind: "unrotated_pe"`. Default calibration uses `pair_so2` for schema uniformity; if quality measurements warrant, an unconstrained `dense_orth` $O(64)$ + diagonal $S_P$ (`[Adaptation]`) is algebraically valid because no positional rotation is applied.

---

## 4. Baselines, Protection Policy, and Scale/Packing Scheme

### Baselines (Measure; Never Assume INT8 Beats FP8/BF16)
1. **`0:0` native FP8 (`mla_fp8_ds_v1`, 656 B/tok RoPE / 528 B/tok NoPE) and BF16 (1,152 B / 1,024 B):** Matched baseline for every fidelity and latency report.
2. **Uncalibrated fixed Sylvester Hadamard (`H_512`):** `gr8` KV4 baseline (`416 B/tok`: 512 INT4 sym `[-7..7]` g32 + 16×FP16 scales + 64 BF16 RoPE, measured +0.010–0.0145 KL vs FP8 on GLM-5.3) and `QuaRot†` (post-RoPE Hadamard on `4:4`).
3. **RTN** (untransformed asymmetric affine INT) and **SmoothQuant†** (scale-only $\alpha$-search, produced automatically at step 0 of our scale-init search).

### Recent-Token and Sink Protection (`[Adaptation A4]` — Default ON from the Start)
- **Paper policy:** Keeps $P_L = \{0,\dots,\min(4,L)-1\} \cup \{\max(0,L-128),\dots,L-1\}$ in BF16 (132 rows × 1,152 B = 152,064 B/layer/request), while still writing packed integer records for every token at insertion so window eviction requires zero re-quantization. On DeepSeek-V2-Lite MMLU C4R4 (Table 3), `sink4/local128` adds +1.64 pts on top of the +4.41 pts from content+RoPE transforms. The paper calibrates *unprotected* and enables `sink4/local128` only at evaluation.
- **CuteAFD design (`[Adaptation A4]`):**
  - **`sink4_local128` is ON by default for every nonzero `QUANT_MLA` mode** on **full-history MLA layers only, never SWA** — in **both calibration and runtime**.
  - **Why:** The extending block already keeps recent rows unquantized before finalizing/quantizing blocks, so protecting the 4 sink + 128 recent tokens aligns runtime execution with natural block staging, avoids wasting learned transform capacity on sink/recent outliers that are never read quantized at runtime, and delivers the paper's full +1.64-pt protection gain from day one.
  - **`none` is only a measured alternative:** If later measurements show a mode/family meets its fidelity bar without protection, `protection_policy: "none"` can be tested and recorded explicitly as a measured alternative.
  - **Prefix-cache & admission accounting (from Phase 3 onward):** Every token still writes its packed integer record into `RefPagePool` at insertion (keeping shared prefix pages self-contained and compatible with `ContextSplit` parity pages), while each active sequence and each exact **prefix mark** holds the 4 sink + 128 recent BF16 rows (`152,064 B` per full-history MLA layer: **1.74 MiB/seq** for Hy4-flash's 12 DSA layers, **1.60 MiB/seq** for GLM 5.3 Flash's 11 NoPE MLA layers, **11.31 MiB/seq** for 78-layer GLM 5.3 / Hy4 — identical mechanism to Hy4-flash's existing 512-row SWA prefix mark). Attention merges protected BF16 + quantized history under a single global softmax (Eq. 39). Admission, prefix identity, and CUDA graph identity account for `sink4_local128` from the first runtime.
  - **SWA layers (e.g., Hy4-flash's 30 SWA-512 rings):** SWA rings are bounded per-sequence circular rings (640 rows/seq/layer), not context-scaling full-history pools. **SWA is untouched by QMLA** — SWA rings keep their `--kv-cache` format and never use `sink4_local128` protection. QMLA applies to full-history MLA layers (Hy4-flash's 12 DSA layers).

### Scale and Packing Scheme (Paper-Faithful, 16-Byte Aligned)
Every quantized path uses per-token asymmetric affine quantization with group size $G = 64$, low-bits-first packing (`q0 | (q1 << 4)` for INT4; `q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)` for INT2; raw `u8` for INT8):
- **Content metadata ($d_c = 512 \Rightarrow 8$ groups):** `8 × BF16 scale` (16 B) + `8 × U8 zero_point` (8 B, exact for INT2/INT4/INT8) = **24 B**.
- **RoPE metadata ($d_r = 64 \Rightarrow 1$ group):** `1 × F32 scale` (4 B) + `1 × F32 zero_point` (4 B) = **8 B**.
- Together, `C_meta + R_meta = 32 B` (16-byte aligned!), reproducing **QuantMLA Eq. 40's exact 320 B/token (`20,480 B` per 64-token page) for `4:4`**! Single-path modes pad metadata by 8 B to keep every sub-section and row stride a multiple of 16 bytes for SM120/SM121 vectorized/TMA loads.

| Mode (`C:R`) | RoPE MLA (`d_c=512, d_r=64`) layout | Row bytes | vs BF16 / FP8 | NoPE MLA (`d_c=512, d_r=0`) row bytes |
|---|---|---:|---:|---:|
| `0:0` (BF16) | 512 BF16 + 64 BF16 | 1,152 B | 1.00× / 0.57× | 1,024 B |
| `0:0` (FP8) | 512 E4M3 + 16 B scales + 64 BF16 | 656 B | 1.76× / 1.00× | 528 B |
| `8:8` | 512 C8 + 24 B C_meta + 8 B R_meta + 64 R8 | **608 B** | 1.89× / 1.08× | — (invalid, $R>0$) |
| `8:0` | 512 C8 + 32 B (C_meta+8B pad) + 128 B BF16 R | **672 B** | 1.71× / 0.98× | **544 B** |
| `0:8` (FP8 C) | 512 E4M3 + 16 B scales + 64 R8 + 16 B (R_meta+8B pad) | **608 B** | 1.89× / 1.08× | — (invalid, $R>0$) |
| `4:8` | 256 C4 + 24 B C_meta + 8 B R_meta + 64 R8 | **352 B** | 3.27× / 1.86× | — (invalid, $R>0$) |
| `4:0` | 256 C4 + 32 B (C_meta+8B pad) + 128 B BF16 R | **416 B** | 2.77× / 1.58× | **288 B** |
| `0:4` (FP8 C) | 512 E4M3 + 16 B scales + 32 R4 + 16 B (R_meta+8B pad) | **576 B** | 2.00× / 1.14× | — (invalid, $R>0$) |
| `4:4` | 256 C4 + 24 B C_meta + 8 B R_meta + 32 R4 | **320 B** | **3.60× / 2.05×** | — (invalid, $R>0$) |
| `2:4` | 128 C2 + 24 B C_meta + 8 B R_meta + 32 R4 | **192 B** | **6.00× / 3.42×** | — (invalid, $R>0$) |
| `2:0` | 128 C2 + 32 B (C_meta+8B pad) + 128 B BF16 R | **288 B** | 4.00× / 2.28× | **160 B** |

*(For `0:8`/`0:4` with `--kv-cache bf16`, content stays 1,024 B BF16 → 1,104 B / 1,072 B. In Phase 3 kernels we will benchmark decoding `s * (q - z)` vs storing pre-multiplied BF16 `-s*z` bias if register pressure warrants; both fit within the same 32 B metadata budget for single-path/NoPE modes.)*

---

## 5. Sidecar Schema (`mla_calibration.safetensors` + `cuteafd-mc.json["qmla"]`)

There is **no separate `mla_calibration.json` file**. The **only** sidecar file added at the checkpoint root is `mla_calibration.safetensors`, **never indexed** in any `model*.safetensors.index.json` (so plain loaders and `0:0` ignore it completely).

### Where the QMLA Manifest Lives (MC vs. Non-MC Checkpoints)
- **MC checkpoints (have `cuteafd-mc.json`):** The entire QMLA manifest lives as an optional, advisory **`"qmla"` section inside `cuteafd-mc.json`** (same pattern as `engram_fp8` and `exl3_codebook` in `docs/mc-checkpoints.md`; **never** in `required_features`). All existing `cuteafd-mc.json` fields (`schema`, `source`, `original_core_dtype`, `shared_inventory_sha256`, `variants`, `exl3_codebook`, `engram_fp8`) stay byte-for-byte unchanged, and every existing MC validator continues to pass. A checkpoint without `"qmla"` or without `mla_calibration.safetensors` serves `0:0` unchanged.
- **Non-MC checkpoints (official / plain checkpoints without `cuteafd-mc.json`):** The exact same canonical QMLA manifest JSON is embedded in `mla_calibration.safetensors`'s **`__metadata__["cuteafd_qmla"]`** header (zero extra JSON files!).
  - **Why `__metadata__` instead of a minimal `cuteafd-mc.json`:** `docs/mc-checkpoints.md` defines a repository as MC *iff* `cuteafd-mc.json` exists, and MC discovery fails closed if `cuteafd-mc.json` lacks `variants` / `shared_inventory_sha256`. Putting the metadata in `mla_calibration.safetensors`'s own safetensors `__metadata__` header preserves the MC discovery invariant cleanly, requires adding **only one file** for a calibration-only non-MC revision, and avoids a circular file-hash dependency (the embedded manifest carries `tensors_payload_sha256` over the raw tensor payload bytes after the safetensors header, while `cuteafd-mc.json["qmla"]` on MC checkpoints additionally pins the whole-file `tensors_sha256` and `tensors_bytes` of `mla_calibration.safetensors`).
  - `mla_calibration.safetensors` always writes `__metadata__["cuteafd_qmla"]` for both MC and non-MC checkpoints, so the file is self-describing and `validate_sidecar` verifies exact agreement between `cuteafd-mc.json["qmla"]` and the safetensors header on MC checkpoints.

### One Shipped Transform per Path, Selected Across Every Width (TJ, 2026-11-28)
Calibration fits every width independently (content at 2/4/8, RoPE at 4/8), because the QDQ inside each objective is width-specific. Those per-width fits are for analysis. **The sidecar ships one content transform $(R_C, S_C)$ and one RoPE transform $(\theta, s)$ per layer, and serves them at every calibrated width.**

- **Candidates per layer and path:** every width's fit (`fitted:c2`, `fitted:c4`, `fitted:c8`; `fitted:r4`, `fitted:r8`) and each width's SmoothQuant init (`smoothquant:c4` is identity $R_C$ with the 4-bit alpha-grid scales; `smoothquant:r4` is zero angles with its pair scales).
- **Selection (on `val`):** score every candidate on every combination, scoring content × RoPE pairs jointly on joint combinations. Ship the pair with the lowest **mean val ratio to RTN** over all measured combinations (`mean_val_ratio_to_rtn`). Each combination counts equally, so a 2-bit combination's larger absolute error does not outvote the 8-bit ones. This also acts as the joint guard. Per-path fits are chosen on their own objective ($\mathcal{L}_C$ with exact RoPE, $\mathcal{L}_P$ with exact content), but the reported number is the joint post-`o_proj` error. On pilot layer 21, SmoothQuant content + SmoothQuant RoPE beat fitted + fitted at 4:4. Selecting on the joint combinations catches that.
- **Why one transform rather than per-combination references:** the runtime writes one cache coordinate system per layer. A separate transform per width or per combination would multiply sidecar bytes, load-time absorptions and cache identities for a difference the cross-table can measure first. If the measured cross-table shows one transform losing materially at some width, a per-width profile can come back as a schema change, backed by that measurement.
- **Recorded per layer:** `profiles.<path>.selection.per_layer[l] = {candidate, chosen: fitted|smoothquant, source_bits, objective, val_score: {combination: val error}}`. A SmoothQuant choice is still a fingerprinted profile. The validator requires its rotation to be the identity (content) or its angles to be zero (RoPE).
- **Tensors:** `content.layer_{l}.r` `F32[d_c, d_c]`, `content.layer_{l}.s` `F32[d_c]`, `rope.layer_{l}.theta` and `rope.layer_{l}.s` `F32[d_r/2]`. `profiles.content.bits` / `profiles.rope.bits` list the served widths; `supported_modes` is derived from them. A path with no profile (pure NoPE RoPE) is `null`.

Storing `F32` keeps orthogonality validation ($\|R^\top R - I\|_\infty < 10^{-5}$) and FP32 `kv_b_proj` absorption at load time exact. On disk that is 1.0 MiB/layer for content: ~12 MiB on Hy4-flash's 12 DSA layers, ~78 MiB for 78-layer GLM 5.3 / Hy4.

### Embedded `qmla` Section Schema (`schema: "cuteafd-qmla-v2"`)
```json
{
  "schema": "cuteafd-qmla-v2",
  "artifact_status": "calibrated_unqualified",
  "protection_policy": "sink4_local128",
  "protection_scope": "full_history_mla_only",
  "source": {
    "repo": "wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC",
    "revision": "<40-char-sha>",
    "base_repo": "tencent/Hy4-flash",
    "base_revision": "<40-char-sha>",
    "family": "hy4_flash",
    "model_type": "hy_v4_flash",
    "config_sha256": "<sha256>",
    "checkpoint_kind": "mc",
    "calibrated_view": "original"
  },
  "applicable_views": ["original", "k8-compact", "k6-compact", "fp8-compact"],
  "view_transfer_validation": {
    "k8-compact": {"split": "test", "reference": "k8-compact 0:0", "combinations": {"4:4": {"attn_rel_err": 0.0, "rtn_attn_rel_err": 0.0, "ratio_to_rtn": 0.0}}}
  },
  "conventions": {
    "vector_layout": "row",
    "content_basis_space": "unit_rms",
    "content_forward_formula": "u @ R_C @ diag(s_C)",
    "content_inverse_formula": "diag(1/s_C) @ R_C^T @ diag(gamma)",
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
      "content_enabled": true,
      "rope_enabled": true,
      "shared_consumers": [0]
    }
  ],
  "profiles": {
    "content": {
      "bits": [2, 4, 8],
      "transform_kind": "dense_orth",
      "layers": [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41],
      "tensors_prefix": "content",
      "protection_calibrated": "sink4_local128",
      "selection": {
        "objective": "mean_val_ratio_to_rtn",
        "per_layer": {
          "21": {"candidate": "smoothquant:c4", "chosen": "smoothquant", "source_bits": 4,
                 "objective": 0.64, "val_score": {"4:4": 0.0, "4:0": 0.0}}
        }
      },
      "validation": {"equiv_max_rel_err": 0.0, "orth_max_err": 0.0, "per_layer": {}},
      "profile_fingerprint": "<sha256>"
    },
    "rope": {
      "bits": [4, 8],
      "transform_kind": "pair_so2",
      "layers": [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41],
      "tensors_prefix": "rope",
      "protection_calibrated": "sink4_local128",
      "selection": {"objective": "mean_val_ratio_to_rtn", "per_layer": {}},
      "validation": {"equiv_max_rel_err": 0.0, "per_layer": {}},
      "profile_fingerprint": "<sha256>"
    }
  },
  "supported_modes": ["8:8", "4:8", "4:4", "2:4", "4:0", "0:4", "8:0", "0:8", "2:0"],
  "unsupported_modes_reason": {"*:2": "INT2 RoPE is not supported"},
  "calibration_provenance": {
    "dataset_repo": "wrldsuksgo2mars/agentic-quant-panel-v3",
    "dataset_revision": "e3ac6ee6126f469e37ddf879a6cb2dccc10c6c99",
    "components": {
      "agentic": {"sequences": 1426, "tokens": 1097388, "input_ids_sha256": "<sha256>"},
      "exl3_standard": {"sequences": 250, "tokens": 512000, "input_ids_sha256": "<sha256>"},
      "long_context": {"sequences": 118, "tokens": 231947, "input_ids_sha256": "<sha256>", "mode": "windowed", "window_tokens": 2048}
    },
    "train_sequences": 128,
    "val_sequences": 16,
    "test_sequences": 16,
    "splits": {"salt": "cuteafd-qmla-split-v2", "seed": 0,
               "train": {"count": 128, "seq_ids": ["..."], "sha256": ["..."], "digest": "<sha256>"},
               "val": {"count": 16, "seq_ids": ["..."], "sha256": ["..."], "digest": "<sha256>"},
               "test": {"count": 16, "seq_ids": ["..."], "sha256": ["..."], "digest": "<sha256>"}},
    "measurement_split": "test",
    "combination_measurements": {"4:4": {"attn_rel_err": 0.0, "rtn_attn_rel_err": 0.0, "ratio_to_rtn": 0.0}},
    "input_ids_sha256": "<sha256>",
    "renderer_template_sha256": "<sha256>",
    "thinking_mode": "high",
    "optimizer": "adamw",
    "steps_per_path": 300,
    "seed": 0,
    "code_commit": "<sha>"
  },
  "tensors_file": "mla_calibration.safetensors",
  "tensors_payload_sha256": "<sha256>",
  "tensors_bytes": 37748736,
  "tensors_sha256": "<sha256-in-cuteafd-mc-json-only>",
  "transform_fingerprint": "<sha256>"
}
```
- **Shared across views:** One sidecar is shared across `original`, `k8-compact`, `k6-compact`, and `fp8-compact` whenever cross-view transfer validation passes (recorded in `applicable_views` and `view_transfer_validation`). If any view requires separate transforms, it gets an explicit view-scoped manifest entry rather than silent reuse.
- **`transform_fingerprint`:** Canonical SHA256 over `(schema, protection_policy, source.base_revision, geometry, conventions, quantizer_recipe, profiles, tensors_payload_sha256)`. Binds runtime graph & prefix cache identity to exact learned coordinates and protection policy, independent of bit width alone.

---

## 6. Calibration Procedure

1. **Corpus (panel_corpus pin, never Standard Fidelity):** Draw deterministically from the panel pinned by `quantization/panel_corpus.py`, `wrldsuksgo2mars/agentic-quant-panel-v3@e3ac6ee6126f469e37ddf879a6cb2dccc10c6c99`, over its three components in canonical order:
   - **Full calibration (default):** `agentic` (chat-rendered with this checkpoint's own template), `exl3_standard` (packed with this tokenizer under `exl3-standard-pack-v1`) and `long_context` (2,048-token windows of the rendered 16K/32K traces; heldout records never fit). `--corpus-dir` is a `panel_corpus.prepare_panel_corpus` output loaded through `panel_corpus.load_component_records`; fewer than three components fails closed.
   - **Pilot:** `--pilot --components agentic,exl3_standard` (or another subset keeping both base components) is allowed.
   - **Past runs:** `agentic-quant-panel-v2@54683731eb` (two components) via `--panel-revision 54683731eb01c7c2fa5b801a0888e029a0e1b1c7`.
   - Record the exact `dataset_repo`, `dataset_revision`, and components used in `calibration_provenance`. Never touch the held-out Standard Fidelity panel.
   - **Three disjoint splits** by a salt-versioned SHA256 hash of each row ID (`cuteafd-qmla-split-v2`): `train` fits transforms; `val` selects each fit's best step and the shipped transform, and nothing else; `test` produces every reported number. Defaults: full run 128 / 16 / 16, pilot 16 / 4 / 4. Each split's sequence IDs and token-ID SHA256s go into capture provenance, `calibration_provenance.splits` and the fit-checkpoint identity. A reused capture whose salt, seed, counts or membership differ refuses to load. Two-way `cuteafd-qmla-capture-v1` captures refuse outright.
2. **Per-layer capture (Hessians alone are insufficient):** At each MLA layer $\ell$, capture:
   - Weightless normalized content latent $u_\ell = \mathrm{RMSNorm}_1(x_{c,\ell})$ (`[T, d_c]`) and native $\gamma_\ell$.
   - Absorbed content query $\bar{Q}_{C,\ell}$ (or $Q_{C,\ell}$ + `kv_b_proj`, `o_proj`, output gate, learnable sink, post-proj norm, VHA postmix) and post-RoPE positional queries $Q_{P,\ell}$ and keys $K_{P,\ell}$ (since $R_P, S_P$ commute with RoPE, post-RoPE capture avoids re-running RoPE inside the optimization loop).
   - Exact causal / SWA / DSA sparse selection mask $\Omega_\ell$ per sequence.
   - Frozen FP32/BF16 reference post-`o_proj` attention output $Y_{\text{ref},\ell}$ and positional score matrix $Z_{P,\text{ref},\ell}$.
3. **Reuse & streaming:**
   - **Full MC pass (Phase 1):** Tap the 128+32 calibration sequences directly during the existing single causal K8 pass as each MLA layer executes — **zero extra full-model forward pass**.
   - **Incremental update (Phase 2):** Run a single frozen-weight layer-major forward over the 128+32 sequences only (no expert re-quantization, no core re-encoding).
   - **Per-layer streaming & recovery:** Optimize all requested bit-width profiles (`c8, c4, c2` and `r8, r4`) from the same captured layer tensors, write a per-layer sequential checkpoint segment with streamed XXH3-128 receipt (`brief-template.md` §7), and free the layer's activation buffers before moving to layer $\ell+1$.
   - **Shared-KV layers (Mistral Large 4):** Accumulate $\mathcal{L}_C$ across all consumer layers sharing `kv_source_layer` so one $(R_C, S_C)$ optimizes every consumer jointly.
4. **Optimizer & baseline comparison:** For each layer and bit width, run:
   - Step 0 evaluation of RTN and fixed Sylvester Hadamard (`H_512`) baselines + 99.9th-percentile scale-init grid $\alpha \in \{0, 0.125, 0.25, 0.5, 0.75, 1\}$ (gives SmoothQuant† for free).
   - 300 steps of AdamW (Cayley orthogonal $R_C$, log-scale $S_C \in [-2,2]$, angles $\theta_i$, pair-tied log-scales $s_i \in [-2,2]$), selecting the best checkpoint on the `val` split every 20 steps (step 0 is the SmoothQuant init).
   - Then build the candidate cross-table (every width's fit and SmoothQuant init against every combination) on `val`, and select one content and one RoPE transform per layer (§5). Measure the same table on `test` for the report.
   - Mandatory validators: orthogonality ($\|R^\top R - I\|_\infty < 10^{-5}$), scale finiteness & positivity, quantization-off algebraic equivalence against reference output (and 3D M-RoPE commutation where applicable), and cross-view transfer check.
5. **Long-context evaluation (16K and 32K, `--long-eval`):** Fitting stays at the current calibration sequence lengths (`agentic` + `exl3_standard`); long-context evaluation is a separate, optional **evaluation-only** pass (`qmla_run.py --long-eval`, either after fitting or standalone against an existing `--sidecar`).
   - **Why:** Past ~2K tokens, three effects appear that 2K-only captures cannot test: DSA top-k pruning (past `candidate_topk = 2048`), low-frequency RoPE rotation at 16K–32K+ positions, and vanishing protection coverage (`sink4_local128` protects 6.5% at 2K vs 0.4% at 32K).
   - **Chunked attention & compact DSA masks:** Evaluation runs forward-only, streaming layer by layer and freeing each capture immediately after scoring, reusing the same 128-row tiled `eval_attention_output` (`tiled_sink_attention` with per-query `sink4_local128` protection) and `measurement._Scorer` hooks across teacher, shipped transforms, RTN, and fixed Hadamard. Long captures store compact DSA `SparseSelection` group IDs (`int32[T, group_topk]`, 64 MiB at 32K) instead of dense `[T, T]` boolean masks (1 GiB + 8 GiB index temp).
   - **DSA selection drift:** On Hy4-flash, `compressed_indexer` (`index_projections(x, q_norm, ...)`) reads only layer input `x` and `q_a_proj(x)` — quantized MLA latents `u` and `k_rope` never feed it, and indexer keys stay native. Within a layer, DSA selection drift is therefore **structurally zero** (overlap `1.0` by construction); cross-layer residual propagation is out of scope for per-layer evaluation.
   - **Position curve & report:** Full-length 16K/32K `long_context` records are rendered from the pinned v3 snapshot (local HF cache) through `quantization/panel_corpus.py` in `full` mode, `heldout` then `calibration` splits, with the corpus's chat kwargs. Results land in `qmla-validation-report.json["long_context_eval"]` for the shipped single transform per path beside RTN and Hadamard, both overall and bucketed by query position (`0-2k`, `2k-4k`, `4k-8k`, `8k-16k`, `16k-32k`, `32k+`).

---

## 7. Runtime Contract

### CLI, Launcher, and Composition with `--kv-cache`
- **Flags:** `--quant-mla <CONTENT>:<ROPE>` (CLI) and `QUANT_MLA=<CONTENT>:<ROPE>` (launcher / `.config`), default `0:0`. Independent of `CORE_VIEW` and `ATTENTION_PRECISION`.
- **Valid widths:** `CONTENT ∈ {0, 2, 4, 8}`, `ROPE ∈ {0, 4, 8}`. Reject `*:2` (INT2 RoPE) and all unknown widths at parse time with an explicit error.
- **Composition with `--kv-cache` / `KV_CACHE` (compose, never conflict):**
  - `--kv-cache` sets the default cache format for every cache plane (SWA rings, indexer keys, and MLA paths).
  - `--quant-mla` / `QUANT_MLA` overrides **only** the full-history MLA content and/or RoPE path where it specifies a nonzero integer bit width; any `0` path retains the `--kv-cache` format, and SWA rings / indexer keys / KDA recurrence always keep `--kv-cache` untouched.
  - `0:0`: 100% unchanged per-family behavior (`--kv-cache` governs everything).
  - `C:0` (`C > 0`): full-history MLA content uses calibrated INT`C`; RoPE, SWA rings, and indexer keys retain `--kv-cache`.
  - `0:R` (`R > 0`): full-history MLA content, SWA rings, and indexer keys retain `--kv-cache` (`fp8` or `bf16`); RoPE uses calibrated INT`R`.
  - `C:R` (`C > 0, R > 0`): full-history MLA content uses calibrated INT`C` and RoPE uses calibrated INT`R`; SWA rings and indexer keys retain `--kv-cache`. Passing `--kv-cache` alongside `--quant-mla C:R` is valid and composes cleanly (no conflict error).
- **Fail-closed rules (never silent fallback):**
  - Non-MLA family with `--quant-mla != 0:0`: fail closed naming the family.
  - Pure-NoPE MLA family (`glm5_flash`) with `R > 0`: fail closed (`qk_rope_dim == 0; use R=0`).
  - Missing `cuteafd-mc.json["qmla"]` (or `__metadata__["cuteafd_qmla"]` on non-MC) / missing `mla_calibration.safetensors`, mismatched `source` / `config_sha256`, or missing requested `c{C}` / `r{R}` profile: fail closed with the exact missing file/profile (never fall back to `0:0` or RTN).
  - Unqualified runtime family/mode combination: fail closed naming the unimplemented kernel/family limitation.

### Arena Ownership, Graph Identity, Prefix Identity, and Speculators
- **Packed arenas (AGENTS.md):** All QMLA transform tensors (`M_C = R_C S_C`, RoPE `theta / s / s_inv`) live in **one named packed device arena per coordinator GPU** (`qmla.transforms.<gpu>`), inventoried and charged once by admission. Load-time `kv_b_proj` compensation writes directly into the existing absorbed `W_UK`/`W_UV` arena slots via bounded transient scratch (zero permanent BF16 duplicate). Active `sink4_local128` BF16 protection buffers live in one named per-rank arena and inside the existing prefix mark arena (full-history MLA layers only, never SWA), charged once by admission from the first runtime.
- **Coordinate identity vs. bit width:** Cache identity is `(resolved_content_format, resolved_rope_format, base_kv_cache, protection_policy, transform_fingerprint)` — **never** bit width alone. `protection_policy` defaults to `"sink4_local128"` for every nonzero mode.
- **Prefix cache & host cache (`cuteafd-hostcache`, `RefPagePool`):** Include the full cache identity tuple in page-pool / host-cache namespace hashing so incompatible calibrations, modes, or protection policies can never collide. Exact prefix restore copies packed pages plus the `sink4_local128` BF16 mark bytes byte-for-byte.
- **CUDA graph identity:** Graph keys include `(qmla_mode, protection_policy, record_bytes, transform_fingerprint)`. Because transform tensors and protection buffers live in fixed packed arenas, graph pointers remain invariant across requests.
- **Speculative consumers:**
  - *Target-KV-sharing drafters* (`hy4_flash` / `hy4` DFly2): read the same QMLA target KV pages (zero BF16 duplicate); load-time absorption compensates DFly2's `kv_b_proj` with its tapped target layer's $(\Gamma_\ell, R_{C,\ell}, S_{C,\ell})$ and applies $(R_{P,\ell}, S_{P,\ell}^{-1})$ to DFly2's RoPE queries.
  - *Independent-KV drafters* (external DFlash2 on GLM 5.3 / GLM 5.3 Flash / Kimi; ML4 `mtp_dspark`): keep their own KV cache governed by `--draft-kv-cache`, completely independent of target `--quant-mla`.

---

## 8. Sharing and Consolidation Across Families

### 8.1 Quantization & Calibration Sharing (`quantization/qmla/` vs. `quantization/<family>/`)
New CuteAFD quant families are built by copying the previous family's working pipeline (`quantization/<family>/`), not from prose alone. We split the calibration stack cleanly so **Hy4-flash** serves as the working end-to-end reference pipeline (`capture → fit → validate → sidecar → manifest → same-repo HF revision publish`) and the next family (**GLM 5.3 Flash**) only needs a thin family adapter:

- **Shared family-agnostic library (`quantization/qmla/`):**
  - `capture.py` *(mode-agnostic)*: `MlaLayerGeometry`, `CapturedSequence`, `CapturedLayerParams`, `CapturedLayerBank`, `QmlaCaptureAdapter` protocol, exact full-precision attention-output / positional-QK evaluators, deterministic train/val splitting, and sequential segment capture storage with streamed XXH3-128 receipts.
  - `quantizer.py`: paper-faithful per-token asymmetric affine group-64 quantizer (RNE ties-to-even, STE, INT2/INT4/INT8 pack/unpack + CPU numeric oracle).
  - `fit.py`: content-path Cayley orthogonal $R_C$ + log-scale $S_C$ optimizer ($\mathcal{L}_C$), RoPE-path per-pair $SO(2)$ angles + pair-tied scales optimizer ($\mathcal{L}_P$), step-0 RTN / fixed-Hadamard / SmoothQuant† $\alpha$-search baselines, shared-KV multi-consumer loss aggregation, per-layer streaming & checkpoint/resume.
  - `sidecar.py`: `cuteafd-qmla-v2` single-file sidecar (`mla_calibration.safetensors` with embedded `__metadata__["cuteafd_qmla"]` + full `"qmla"` section inside `cuteafd-mc.json` for MC checkpoints; no separate `mla_calibration.json`), `transform_fingerprint`, and fail-closed validators (geometry, source identity, SHA256 digests, orthogonality/invertibility, quantization-off algebraic equivalence, cross-view transfer).
  - `publish.py`: same-repo incremental HF revision publisher over `quantization/hub_lfs.py` (`HF_HUB_DISABLE_XET=1`), rewriting `cuteafd-mc.json` with `"qmla"` (MC) and adding `mla_calibration.safetensors` with zero weight shard re-upload.
- **Thin family adapter (`quantization/hy4_flash/`, reference pipeline to copy):**
  - `qmla_capture.py` *(mode-agnostic)*: `Hy4FlashCaptureAdapter` wiring Hy4-flash's geometry (12 DSA + 30 SWA layers, interleaved 3D M-RoPE `[12,10,10]`, compressed-indexer group top-k masks, gated MLA, learnable sinks, `post_proj_norm`, and DFly2 target taps) into `quantization/qmla/capture.py` — reusable in both full MC causal passes and standalone incremental passes.
  - `qmla_run.py` / `qmla_publish.py`: CLI driver exposing `--pilot` / `--layers` / `--train-sequences` / `--val-sequences` / `--modes` for short pilots and full runs, invoking `quantization/qmla/` for fit/validate/sidecar/publish.
- **What the next family (`quantization/glm5_flash/`) needs to add:** only its geometry descriptor (11 NoPE MLA layers, `qk_rope_dim=0`, 4-token pooled DSA indexer), its capture hook wrapping GLM-5.3-Flash's forward, and its thin CLI entrypoint — zero duplicated fitting, validation, sidecar, or publishing code.
- **Docs templates (`docs/quantization/brief-template.md` and `docs/quantization/qmla-incremental-template.md`):** describe this exact working pipeline and point directly at `quantization/qmla/` + `quantization/hy4_flash/` as the reference to copy.

### 8.2 Runtime Engine Consolidation (`rust/` and `b12x`)
Current engine audit reveals concrete duplication across MLA families that runtime QMLA (Phases 3–5) will touch anyway:

| Area | Current Duplication | Consolidation Proposal | Regression Risk & Boundary |
|---|---|---|---|
| **MLA cache format & sizing** | `cuteafd_core::MlaCacheFormat` (Kimi/GLM/draft) vs `hy4_common::Hy4KvCache` (identical 1152/656 B!) vs `KvLayout::Glm52Compressed*` (`kv_cache/config.rs`). | Introduce one `MlaCacheSpec` + `QmlaMode` + `QmlaProtection` in `cuteafd-core/src/kv_cache/mla.rs` providing single-source record bytes, offsets, alignment, and identity keys for all RoPE & NoPE MLA modes. | **Low.** `0:0` maps directly to existing record bytes; unit-tested bit-for-bit against all existing constants. |
| **QMLA sidecar loader & validator** | Does not exist yet. | Single `cuteafd-loader/src/formats/qmla.rs` module reading `cuteafd-mc.json["qmla"]` (MC) or `mla_calibration.safetensors` `__metadata__["cuteafd_qmla"]` (non-MC), validating schema/digests/orthogonality/geometry and building the packed `qmla.transforms.<gpu>` arena plan. | **Zero** on `0:0` (not invoked unless `--quant-mla != 0:0`). |
| **`kv_b_proj` UK/UV absorption** | Implemented three times: `cuteafd-loader/src/formats/mla.rs`, `native/shared/cuda/fp8_load_quant.h` (`absorb_uk/uv`), and `hy4_common/weights.rs` (`pack_kv_b_absorbed`). | One shared `absorb_kv_b_compensated` (CPU F32 oracle + GPU load kernel) applying optional $\Gamma R_C S_C^{-1}$ before splitting into `W_UK`/`W_UV` arena slots. | **Low.** `0:0` bypasses compensation identically; tested bit-exact against existing `pack_kv_b_absorbed`. |
| **Pack/unpack & b12x MLA kernels** | `native/shared/cuda/kv_cache.cu` + `third_party/sparkinfer/b12x/attention/_shared/mla/{traits,kv_cache,decode_math,prefill_mg}.py`. | Add `QmlaContentCodec`/`QmlaRopeCodec` trait specializations and fused write/dequant primitives in `b12x.attention._shared.mla` + `kv_cache.cu` (upstreamed to `sparkinfer-glmrt` master). | **Low** when guarded as distinct `const_expr` traits so existing `0:0` PTX/SASS is unchanged. |
| **Per-family execution loops & scheduling** | Each family has real structural differences (Hy4-flash SWA-512 + Engram + DFly2; GLM Flash KDA + NoPE; Kimi AttnRes + unrotated PE; ML4 KV sharing + ssmax + no q-LoRA; ERNIE HCA + VHA). | **Do NOT rewrite or unify family attention blocks.** Keep small per-family `MlaGeometryDescriptor` adapters calling the shared sidecar loader, UK/UV compensator, and `b12x` kernels. | **Avoids fake uniformity** and keeps regression surface strictly bounded. |

**Migration order:** Build shared core/loader/kernel primitives in **Phase 3 (`hy4_flash`)**, validate them on a maximally different second family in **Phase 4 (`glm5_flash`: NoPE + KDA + external DFlash2 + 1/2 RTX)** (consolidating any newly visible duplication), then add thin adapters in **Phase 5**.

---

## 9. Expected Costs, Family Matrix, and Verification Plan

### Expected Memory & Capacity Savings (at 128K Context, Including Default `sink4_local128` Protection)
Physical cache accounting **includes** scales, padding, unchanged DSA indexer caches (`index_k`: 132 B/tok full or 33 B/tok 4-token-pooled), **and** the default per-sequence `sink4_local128` BF16 protection buffer on full-history MLA layers (`132 × (d_c + d_r) × 2 B` per MLA layer: `152,064 B/L` RoPE, `135,168 B/L` NoPE), and excludes fixed workspaces/SWA rings:

| Family (full-history MLA layers) | `0:0` BF16 (KV+idx) | `0:0` FP8 (KV+idx) | `4:4` (or `4:0` NoPE) | `2:4` (or `2:0` NoPE) | Default `sink4_local128` / seq | 128K total (1 seq): FP8 → `4:4`/`4:0` → `2:4`/`2:0` |
|---|---:|---:|---:|---:|---:|---|
| **Hy4-flash** (12 DSA; 30 SWA native) | 1,185 B/L | 689 B/L | **353 B/L** | **225 B/L** | +1.74 MiB | 1.01 GiB → **0.52 GiB (1.94×)** → **0.33 GiB (3.04×)** |
| **GLM 5.3 Flash** (11 NoPE MLA; 34 KDA) | 1,057 B/L | 561 B/L | **321 B/L** (`4:0`) | **193 B/L** (`2:0`) | +1.42 MiB | 0.75 GiB → **0.43 GiB (1.74×)** → **0.26 GiB (2.89×)** |
| **GLM 5.3** (78 MLA; 21 full idx) | 1,188 B/L avg | 692 B/L avg | **356 B/L avg** | **228 B/L avg** | +11.31 MiB | 6.58 GiB → **3.40 GiB (1.94×)** → **2.18 GiB (3.02×)** |
| **Hy4** (78 MLA; 21 full idx) | 1,188 B/L avg | 692 B/L avg | **356 B/L avg** | **228 B/L avg** | +11.31 MiB | 6.58 GiB → **3.40 GiB (1.94×)** → **2.18 GiB (3.02×)** |
| **Kimi K3 / K3.1 Pro** (24 MLA; K3.1 24 idx) | 1,284 B/L | 788 B/L | **452 B/L** | **324 B/L** | +3.48 MiB | 2.31 GiB → **1.33 GiB (1.74×)** → **0.95 GiB (2.42×)** |
| **Mistral Large 4** (63 owned: 54 RoPE + 9 NoPE; 44 idx) | 1,226 B/L avg | 730 B/L avg | **408 B/L avg** | **280 B/L avg** | +8.99 MiB | 5.61 GiB → **3.14 GiB (1.79×)** → **2.16 GiB (2.60×)** |
| **ERNIE 5.5 Flash** (6 MLA-DSA; 37 HCA native) | 1,284 B/L | 788 B/L | **452 B/L** | **324 B/L** | +0.87 MiB | 0.58 GiB → **0.33 GiB (1.74×)** → **0.24 GiB (2.42×)** |

*(At short contexts e.g. 4K tokens, `sink4_local128` is ~3.2% of BF16 per active sequence; each prefix mark adds the same per-sequence protection footprint. `none` saves that per-sequence/mark footprint if later measured unnecessary.)*

### Expected Latency & Load-Time Costs
- **Load time:** Sidecar read (36–234 MiB from NVMe/SparkNest: <0.2 s) + GPU FP32 `kv_b_proj` absorption ($H(d_{\text{nope}}+d_v) \times d_c \times d_c$ per layer: ~0.05–0.3 s across 78 layers on RTX). Negligible vs. 30–60 s model readiness.
- **Short-context decode:** +12–150 µs/step for write-side $u M_C$ (0 µs when load-time folded on BF16 `original`); slight ALU cost for INT4/INT2 unpack inside attention.
- **Long-context / high-concurrency decode:** Attention reads **~2.0× fewer KV bytes (`4:4`) / ~3.4× fewer (`2:4`)** vs FP8 (~3.6× / ~6.0× vs BF16), directly speeding up memory-bound long-context attention and ~1.8–3.0× expanding KV pool concurrency before preemption.

### Family Matrix

| Family | MLA Layers | Geometry (`H / d_c / d_nope / d_rope / d_v`) | RoPE Style & Special Features | Target Modes | Drafter Interaction | Phase |
|---|---|---|---|---|---|---|
| **Hy4-flash** | 12 DSA (+30 SWA native) | 32 (SWA 64) / 512 / 192 / 64 / 256 | Interleaved 3D M-RoPE `[12,10,10]`, gated, sink, post-norm, Engram, EXL3 `kv_b` in K8/K6 | `8:8, 4:8, 4:4, 2:4, 4:0, 0:4, 8:0, 0:8` | Bundled DFly2 taps DSA KV (compensate DFly2 `kv_b` & $Q_P$) | **Phase 3** |
| **GLM 5.3 Flash** | 11 DSA (+34 KDA native) | 64 / 512 / 256 / **0 (NoPE)** / 256 | Pure NoPE (`qk_rope_dim=0`), 4-tok pooled DSA indexer | `8:0, 4:0, 2:0` (`R>0` refused) | External DFlash2 (independent KV) | **Phase 4** |
| **GLM 5.3** | 78 DSA | 64 / 512 / 192 / 64 / 256 | Interleaved RoPE, 21 full / 57 shared indexers | All `C:R` | External DFlash2 (independent KV) | **Phase 5** |
| **Hy4** | 78 DSA | 64 / 512 / 192 / 64 / 256 | Interleaved 3D M-RoPE `[12,10,10]`, gated, sink, 21 full / 57 shared indexers | All `C:R` | Bundled DFly2 taps layers 61..77 (compensate DFly2 `kv_b` & $Q_P$) | **Phase 5** |
| **Kimi K3 / K3.1 Pro** | 24 Full / DSA (+69 KDA) | 96 / 512 / 128 / 64 (unrotated) / 128 | `mla_use_nope` with unrotated 64-dim PE slot, output gate, AttnRes | All `C:R` (evaluate `pair_so2` vs `dense_orth` on PE) | External DFlash2 (independent KV) | **Phase 5** |
| **Mistral Large 4** | 72 DSA (63 owned KV) | 64 / 512 / 64 (NoPE 128) / 64 (or 0) / 128 | No q-LoRA, interleaved YaRN+LeRoPE, ssmax, 18 NoPE layers, 9 KV-sharing consumers (src=35) | All `C:R` (RoPE on 54 layers; joint loss on shared KV src) | Bundled `mtp_dspark` (independent KV) | **Phase 5** |
| **ERNIE 5.5 Flash** | 6 MLA-DSA (+37 HCA native) | 64 / 512 / 192 / 64 / 256 | Interleaved plain RoPE (8e6) on MLA, VHA postmix, SWA-128 union, gate | All `C:R` (6 MLA-DSA layers only; HCA untouched) | Bundled DSpark (HCA/independent) | **Phase 5** |
| **Out of scope** | — | — | DeepSeek V4/V4.1 (CSA/HCA), MiniMax (MSA), MiMo/Qwen/Muse/Inkling/Step-5 (GQA/GDN) | Refused (`!= 0:0`) | Unchanged | Excluded |

### Verification & Test Plan
- **Phases 1–2 (CPU unit & numeric oracle tests — no GPU required):**
  - Quantizer exactness (RNE ties-to-even, edge/constant groups, nibble/crumb pack/unpack bit-exactness across INT2/INT4/INT8).
  - Content & RoPE algebraic equivalence with quantization off (both `interleaved` and `split_half`, plus 3D M-RoPE `pos_T != pos_H != pos_W`, plus shared-KV multi-consumer layers, plus NoPE layers).
  - Calibration loop convergence oracle on synthetic low-rank + outlier MLA layers (verifying $\mathcal{L}_C$ and $\mathcal{L}_P$ strictly beat RTN & fixed Hadamard, orthogonality $\|R^\top R - I\|_\infty < 10^{-5}$, and deterministic checkpoint/resume with XXH3-128 receipts).
  - Sidecar writer/validator fail-closed tests (schema, digests, tensor shapes/dtypes, view transfer records, `cuteafd-mc.json` advisory pointer preserving `0:0` compatibility, and incremental revision preservation of existing shard bytes/inventories).
- **Phases 3–5 (Runtime & hardware qualification):**
  - `0:0` byte/launch/memory parity; `--kv-cache` + `--quant-mla` composition and fail-closed non-MLA/NoPE rejection tests.
  - Device kernel parity against CPU decoded-cache oracle across prefill, decode, sparse DSA top-k, 2-GPU `ContextSplit` partial merge, CUDA graph replay, and target-sharing DFly2 speculation.
  - Exact prefix-cache restore byte-identity (including `sink4_local128` mark capture/restore when enabled) and cache-identity isolation across distinct `transform_fingerprint`s.
  - Matched Standard fidelity (mean KL, top-1, NLL, context scores vs `0:0` FP8 and BF16), held-out long-context/needle, agentic multi-turn sessions, and matched throughput / bytes-per-token / load-time profiling.

---

## Confirmed Decisions (TJ, 2026-11-28)

1. **Content path default:** Calibrate in `unit_rms` space ($z_C = u R_C S_C$, $\widetilde{W}_{\text{kv\_b}} = W_{\text{kv\_b}} \Gamma R_C S_C^{-1}$) with runtime write-side $u M_C$ + load-time UK/UV absorption on every core view (`original`, `k8`, `k6`, `fp8`); producer folding into BF16 `kv_a_proj_with_mqa` is only an optional BF16-original fast path using the exact same sidecar.
2. **RoPE execution:** Fuse $(\theta_i, s_i, s_i^{-1})$ directly into the existing RoPE phase/scale step (zero extra matmuls, zero weight mutations, commutes with 3D M-RoPE and YaRN/LeRoPE).
3. **CLI composition (`--kv-cache` + `QUANT_MLA`):** Compose rather than conflict. `--kv-cache` sets the default for SWA rings, indexer keys, and MLA; `QUANT_MLA` overrides only the full-history MLA path(s) set nonzero (`0` retains `--kv-cache`). Cache identity includes the resolved per-path format.
4. **Protection ON from the start (`[Adaptation A4]`):** `sink4_local128` is the **default** policy for every nonzero mode on **full-history MLA layers only (SWA untouched)**, in both calibration and serving (`none` is only a measured alternative). Admission, prefix marks, and CUDA graph identity account for `sink4_local128` from Phase 3 onward.
5. **Single sidecar file (`mla_calibration.safetensors`) + `cuteafd-mc.json["qmla"]` (no `mla_calibration.json`):** The full `cuteafd-qmla-v2` manifest lives in `cuteafd-mc.json["qmla"]` for MC checkpoints (advisory, never in `required_features`, `0:0` unchanged) and in `mla_calibration.safetensors`'s `__metadata__["cuteafd_qmla"]` header for non-MC checkpoints (chosen over a minimal `cuteafd-mc.json` to preserve `docs/mc-checkpoints.md`'s MC discovery invariant and add only one file).
6. **Pinned panel corpus (`[Adaptation A3]`):** `quantization/panel_corpus.py` owns the pin; QMLA has none of its own. Full calibration uses all three components (`agentic`, `exl3_standard`, `long_context`) from `wrldsuksgo2mars/agentic-quant-panel-v3@e3ac6ee6126f469e37ddf879a6cb2dccc10c6c99` and records the exact revision and components in `calibration_provenance`; a pilot may use a subset. `agentic-quant-panel-v2@54683731eb` stays pinned for past runs.
7. **Library placement & reference pipeline:** Shared family-agnostic QMLA code lives in `quantization/qmla/` inside `cuteafd` (**no `GPTQModel` fork changes / no fork PR**); Hy4-flash in `quantization/hy4_flash/` is the end-to-end reference pipeline with pilot/full CLI flags that subsequent families copy.
8. **Three splits, test-only reporting, one shipped transform per path:** `train` fits, `val` selects fit steps and the shipped transform, `test` produces every reported number (128 / 16 / 16 full, 16 / 4 / 4 pilot). Every width is fitted and measured for analysis. The sidecar ships one content and one RoPE transform per layer, selected jointly across all combinations on `val` (§5). The pilot fits every width and measures all nine combinations.
