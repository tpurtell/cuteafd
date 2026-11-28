"""Paper-faithful QuantMLA asymmetric affine group-64 quantizer and bit-exact packing oracle.

Reproduces QuantMLA (Su et al., 2026-09, arXiv:2609.36760) Appendix C.4 Eqs. 35-36
and Appendix C.5 Eq. 40:
- Per-token asymmetric affine quantization with group size 64.
- Round-to-nearest ties-to-even (STE during calibration).
- Content scales stored in BF16 (2 B/group) + U8 zero-points (1 B/group).
- RoPE scales stored in F32 (4 B/group) + F32 zero-points (4 B/group).
- Low-bits-first integer packing (INT8: 1 B, INT4: 2/byte, INT2: 4/byte).
"""
from __future__ import annotations

import math
import struct
from typing import Literal

import torch

GROUP_SIZE = 64
EPS_SCALE = 1e-8
VALID_CONTENT_BITS = frozenset({0, 2, 4, 8})
VALID_ROPE_BITS = frozenset({0, 4, 8})
CONTENT_NONZERO_BITS = frozenset({2, 4, 8})
ROPE_NONZERO_BITS = frozenset({4, 8})


def parse_qmla_mode(spec: str) -> tuple[int, int]:
    """Parse ``CONTENT:ROPE`` mode string, failing closed on unsupported widths."""
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"invalid QUANT_MLA mode {spec!r}: expected CONTENT:ROPE (e.g. 4:4, 4:0, 0:0)")
    parts = spec.strip().split(":")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"invalid QUANT_MLA mode {spec!r}: expected two non-negative integers CONTENT:ROPE")
    c_bits, r_bits = int(parts[0]), int(parts[1])
    if c_bits not in VALID_CONTENT_BITS:
        raise ValueError(
            f"unsupported QMLA content bit width {c_bits} in {spec!r}; supported: {sorted(VALID_CONTENT_BITS)}"
        )
    if r_bits == 2:
        raise ValueError(f"INT2 RoPE ({spec!r}) is not supported; supported RoPE widths: {sorted(VALID_ROPE_BITS)}")
    if r_bits not in VALID_ROPE_BITS:
        raise ValueError(
            f"unsupported QMLA RoPE bit width {r_bits} in {spec!r}; supported: {sorted(VALID_ROPE_BITS)}"
        )
    return c_bits, r_bits


def round_rne_ste(x: torch.Tensor, *, ste: bool = True) -> torch.Tensor:
    """Round-to-nearest ties-to-even (PyTorch ``torch.round``), with optional STE."""
    rounded = torch.round(x)
    return x + (rounded - x).detach() if ste else rounded


def bf16_round_ste(x: torch.Tensor, *, ste: bool = True) -> torch.Tensor:
    """Round FP32 tensor through BF16 (matching stored BF16 content scales)."""
    rounded = x.to(torch.bfloat16).to(torch.float32)
    return x + (rounded - x).detach() if ste else rounded


def quantize_params_asymmetric(
    x: torch.Tensor,
    bits: int,
    *,
    group_size: int = GROUP_SIZE,
    scale_dtype: Literal["BF16", "F32"] = "F32",
    eps: float = EPS_SCALE,
    ste: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-group (q_int_float, scale_f32, zp_float) for the last dimension.

    Input shape: ``[..., D]`` where ``D % group_size == 0``.
    Returns ``(q, scale, zp)`` with ``q`` shaped ``[..., G, group_size]`` in ``[0, 2^b - 1]``,
    and ``scale``, ``zp`` shaped ``[..., G, 1]``.
    """
    if bits not in (2, 4, 8):
        raise ValueError(f"unsupported integer bit width: {bits}")
    if x.ndim < 1 or x.shape[-1] == 0 or x.shape[-1] % group_size != 0:
        raise ValueError(f"last dim {x.shape[-1] if x.ndim else 0} must be a positive multiple of {group_size}")
    if scale_dtype not in ("BF16", "F32"):
        raise ValueError(f"unsupported scale_dtype: {scale_dtype!r}")

    xf = x.float()
    groups = xf.shape[-1] // group_size
    grouped = xf.reshape(*xf.shape[:-1], groups, group_size)
    m = float((1 << bits) - 1)

    xmin = grouped.amin(dim=-1, keepdim=True)
    xmax = grouped.amax(dim=-1, keepdim=True)
    raw_scale = (xmax - xmin).clamp_min(float(eps)) / m
    scale = bf16_round_ste(raw_scale, ste=ste) if scale_dtype == "BF16" else raw_scale
    scale = scale.clamp_min(float(eps) / m)
    zp = round_rne_ste(-xmin / scale, ste=ste).clamp(0.0, m)
    q = round_rne_ste(grouped / scale + zp, ste=ste).clamp(0.0, m)
    return q, scale, zp


def qdq_asymmetric_group(
    x: torch.Tensor,
    bits: int,
    *,
    group_size: int = GROUP_SIZE,
    scale_dtype: Literal["BF16", "F32"] = "F32",
    eps: float = EPS_SCALE,
    ste: bool = True,
) -> torch.Tensor:
    """Quantize-dequantize ``x`` using QuantMLA's asymmetric affine group quantizer."""
    q, scale, zp = quantize_params_asymmetric(
        x, bits, group_size=group_size, scale_dtype=scale_dtype, eps=eps, ste=ste
    )
    deq = scale * (q - zp)
    return deq.reshape(x.shape)


def pack_unsigned_codes(q_u8: torch.Tensor, bits: int) -> bytes:
    """Pack integer codes [0..2^b-1] (1D uint8 tensor) low-bits-first."""
    flat = q_u8.detach().cpu().to(torch.uint8).contiguous().view(-1)
    n = int(flat.numel())
    if bits == 8:
        return flat.numpy().tobytes()
    if bits == 4:
        if n % 2 != 0:
            raise ValueError("INT4 packing requires even element count")
        pairs = flat.reshape(-1, 2)
        packed = (pairs[:, 0] & 0x0F) | ((pairs[:, 1] & 0x0F) << 4)
        return packed.to(torch.uint8).numpy().tobytes()
    if bits == 2:
        if n % 4 != 0:
            raise ValueError("INT2 packing requires element count divisible by 4")
        quads = flat.reshape(-1, 4)
        packed = (
            (quads[:, 0] & 0x03)
            | ((quads[:, 1] & 0x03) << 2)
            | ((quads[:, 2] & 0x03) << 4)
            | ((quads[:, 3] & 0x03) << 6)
        )
        return packed.to(torch.uint8).numpy().tobytes()
    raise ValueError(f"unsupported bit width for packing: {bits}")


def unpack_unsigned_codes(data: bytes, count: int, bits: int) -> torch.Tensor:
    """Unpack low-bits-first packed codes into a 1D uint8 tensor of length ``count``."""
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    if bits == 8:
        if len(raw) != count:
            raise ValueError(f"INT8 byte count {len(raw)} != {count}")
        return raw.clone()
    if bits == 4:
        if count % 2 != 0 or len(raw) != count // 2:
            raise ValueError(f"INT4 byte count {len(raw)} disagrees with count={count}")
        lo = raw & 0x0F
        hi = (raw >> 4) & 0x0F
        return torch.stack((lo, hi), dim=-1).reshape(count).clone()
    if bits == 2:
        if count % 4 != 0 or len(raw) != count // 4:
            raise ValueError(f"INT2 byte count {len(raw)} disagrees with count={count}")
        q0 = raw & 0x03
        q1 = (raw >> 2) & 0x03
        q2 = (raw >> 4) & 0x03
        q3 = (raw >> 6) & 0x03
        return torch.stack((q0, q1, q2, q3), dim=-1).reshape(count).clone()
    raise ValueError(f"unsupported bit width for unpacking: {bits}")


def record_bytes_for_mode(
    c_bits: int,
    r_bits: int,
    *,
    d_c: int = 512,
    d_r: int = 64,
    base_kv_cache: Literal["fp8", "bf16"] = "fp8",
    group_size: int = GROUP_SIZE,
) -> int:
    """Exact 16-byte-aligned physical MLA record stride (bytes/token/layer).

    Matches ``docs/quantization/qmla-design.md`` §4 and QuantMLA Eq. 40 (320 B for 4:4).
    """
    if c_bits not in VALID_CONTENT_BITS or r_bits not in VALID_ROPE_BITS:
        raise ValueError(f"unsupported mode ({c_bits}, {r_bits})")
    if d_c <= 0 or d_c % group_size != 0:
        raise ValueError(f"d_c={d_c} must be a positive multiple of {group_size}")
    if d_r < 0 or (d_r > 0 and d_r % group_size != 0):
        raise ValueError(f"d_r={d_r} must be 0 or a positive multiple of {group_size}")
    if d_r == 0 and r_bits != 0:
        raise ValueError("NoPE layer (d_r=0) cannot enable RoPE compression (r_bits > 0)")
    if base_kv_cache not in ("fp8", "bf16"):
        raise ValueError(f"unsupported base_kv_cache: {base_kv_cache!r}")

    if c_bits == 0 and r_bits == 0:
        if base_kv_cache == "bf16":
            return (d_c + d_r) * 2
        return d_c + (d_c // 128) * 4 + d_r * 2

    c_groups = d_c // group_size
    r_groups = d_r // group_size
    c_meta = c_groups * 3 if c_bits > 0 else 0  # BF16 scale (2 B) + U8 zp (1 B)
    r_meta = r_groups * 8 if r_bits > 0 else 0  # F32 scale (4 B) + F32 zp (4 B)

    if c_bits > 0 and r_bits > 0:
        c_code = (d_c * c_bits) // 8
        r_code = (d_r * r_bits) // 8
        meta = math.ceil((c_meta + r_meta) / 16) * 16
        return c_code + meta + r_code
    if c_bits > 0:
        c_code = (d_c * c_bits) // 8
        meta = math.ceil(c_meta / 16) * 16
        r_native = d_r * 2  # BF16 RoPE in both fp8 and bf16 native layouts
        return c_code + meta + r_native
    # c_bits == 0 and r_bits > 0
    c_native = d_c * 2 if base_kv_cache == "bf16" else (d_c + (d_c // 128) * 4)
    r_code = (d_r * r_bits) // 8
    meta = math.ceil(r_meta / 16) * 16
    return c_native + r_code + meta


def pack_content_row(row: torch.Tensor, bits: int, *, group_size: int = GROUP_SIZE, eps: float = EPS_SCALE) -> bytes:
    """Pack one content latent row [d_c] into (codes || BF16 scales || U8 zero-points)."""
    if row.ndim != 1:
        raise ValueError("pack_content_row expects a 1D vector")
    q, scale, zp = quantize_params_asymmetric(
        row, bits, group_size=group_size, scale_dtype="BF16", eps=eps, ste=False
    )
    codes = pack_unsigned_codes(q.reshape(-1).to(torch.uint8), bits)
    scales_bytes = scale.reshape(-1).to(torch.bfloat16).contiguous().view(torch.uint8).cpu().numpy().tobytes()
    zp_bytes = zp.reshape(-1).to(torch.uint8).contiguous().cpu().numpy().tobytes()
    return codes + scales_bytes + zp_bytes


def unpack_content_row(data: bytes, d_c: int, bits: int, *, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Unpack one content latent row produced by ``pack_content_row`` back to FP32."""
    groups = d_c // group_size
    code_bytes = (d_c * bits) // 8
    scale_bytes = groups * 2
    zp_bytes = groups
    expected = code_bytes + scale_bytes + zp_bytes
    if len(data) != expected:
        raise ValueError(f"packed content row length {len(data)} != expected {expected}")
    codes = unpack_unsigned_codes(data[:code_bytes], d_c, bits).reshape(groups, group_size).float()
    scales = (
        torch.frombuffer(bytearray(data[code_bytes : code_bytes + scale_bytes]), dtype=torch.bfloat16)
        .float()
        .reshape(groups, 1)
    )
    zps = (
        torch.frombuffer(bytearray(data[code_bytes + scale_bytes :]), dtype=torch.uint8)
        .float()
        .reshape(groups, 1)
    )
    return (scales * (codes - zps)).reshape(d_c)


def pack_rope_row(row: torch.Tensor, bits: int, *, group_size: int = GROUP_SIZE, eps: float = EPS_SCALE) -> bytes:
    """Pack one RoPE key row [d_r] into (codes || F32 scales || F32 zero-points)."""
    if row.ndim != 1:
        raise ValueError("pack_rope_row expects a 1D vector")
    q, scale, zp = quantize_params_asymmetric(
        row, bits, group_size=group_size, scale_dtype="F32", eps=eps, ste=False
    )
    codes = pack_unsigned_codes(q.reshape(-1).to(torch.uint8), bits)
    scales_bytes = b"".join(struct.pack("<f", float(v)) for v in scale.reshape(-1).tolist())
    zp_bytes = b"".join(struct.pack("<f", float(v)) for v in zp.reshape(-1).tolist())
    return codes + scales_bytes + zp_bytes


def unpack_rope_row(data: bytes, d_r: int, bits: int, *, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Unpack one RoPE key row produced by ``pack_rope_row`` back to FP32."""
    groups = d_r // group_size
    code_bytes = (d_r * bits) // 8
    scale_bytes = groups * 4
    zp_bytes = groups * 4
    expected = code_bytes + scale_bytes + zp_bytes
    if len(data) != expected:
        raise ValueError(f"packed RoPE row length {len(data)} != expected {expected}")
    codes = unpack_unsigned_codes(data[:code_bytes], d_r, bits).reshape(groups, group_size).float()
    scales = torch.tensor(
        struct.unpack(f"<{groups}f", data[code_bytes : code_bytes + scale_bytes]),
        dtype=torch.float32,
    ).reshape(groups, 1)
    zps = torch.tensor(
        struct.unpack(f"<{groups}f", data[code_bytes + scale_bytes :]),
        dtype=torch.float32,
    ).reshape(groups, 1)
    return (scales * (codes - zps)).reshape(d_r)


def pack_joint_record(
    content_row: torch.Tensor,
    rope_row: torch.Tensor,
    c_bits: int,
    r_bits: int,
    *,
    group_size: int = GROUP_SIZE,
    eps: float = EPS_SCALE,
) -> bytes:
    """Pack one joint quantized (C > 0, R > 0) record matching the 16B-aligned runtime layout."""
    if c_bits not in CONTENT_NONZERO_BITS or r_bits not in ROPE_NONZERO_BITS:
        raise ValueError(f"pack_joint_record requires nonzero (c_bits, r_bits), got ({c_bits}, {r_bits})")
    d_c, d_r = int(content_row.shape[0]), int(rope_row.shape[0])
    qc, sc, zc = quantize_params_asymmetric(
        content_row, c_bits, group_size=group_size, scale_dtype="BF16", eps=eps, ste=False
    )
    qr, sr, zr = quantize_params_asymmetric(
        rope_row, r_bits, group_size=group_size, scale_dtype="F32", eps=eps, ste=False
    )
    c_codes = pack_unsigned_codes(qc.reshape(-1).to(torch.uint8), c_bits)
    c_meta = (
        sc.reshape(-1).to(torch.bfloat16).contiguous().view(torch.uint8).cpu().numpy().tobytes()
        + zc.reshape(-1).to(torch.uint8).contiguous().cpu().numpy().tobytes()
    )
    r_meta = b"".join(struct.pack("<f", float(v)) for v in sr.reshape(-1).tolist()) + b"".join(
        struct.pack("<f", float(v)) for v in zr.reshape(-1).tolist()
    )
    meta_pad = (-len(c_meta + r_meta)) % 16
    r_codes = pack_unsigned_codes(qr.reshape(-1).to(torch.uint8), r_bits)
    out = c_codes + c_meta + r_meta + (b"\x00" * meta_pad) + r_codes
    expected = record_bytes_for_mode(c_bits, r_bits, d_c=d_c, d_r=d_r, group_size=group_size)
    if len(out) != expected:
        raise ValueError(f"joint record bytes {len(out)} != expected {expected}")
    return out


def unpack_joint_record(
    data: bytes,
    d_c: int,
    d_r: int,
    c_bits: int,
    r_bits: int,
    *,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack one joint quantized record back into FP32 (content_row, rope_row)."""
    expected = record_bytes_for_mode(c_bits, r_bits, d_c=d_c, d_r=d_r, group_size=group_size)
    if len(data) != expected:
        raise ValueError(f"joint record bytes {len(data)} != expected {expected}")
    c_groups = d_c // group_size
    r_groups = d_r // group_size
    c_code_len = (d_c * c_bits) // 8
    c_meta_len = c_groups * 3
    r_meta_len = r_groups * 8
    meta_pad = (-(c_meta_len + r_meta_len)) % 16
    r_code_len = (d_r * r_bits) // 8

    off = 0
    c_codes = data[off : off + c_code_len]
    off += c_code_len
    c_meta = data[off : off + c_meta_len]
    off += c_meta_len
    r_meta = data[off : off + r_meta_len]
    off += r_meta_len + meta_pad
    r_codes = data[off : off + r_code_len]
    return (
        unpack_content_row(c_codes + c_meta, d_c, c_bits, group_size=group_size),
        unpack_rope_row(r_codes + r_meta, d_r, r_bits, group_size=group_size),
    )


def sylvester_hadamard(dim: int, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized Sylvester Hadamard matrix H_dim (H @ H^T = I) for power-of-two dim."""
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Sylvester Hadamard requires a power-of-two dimension, got {dim}")
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < dim:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h * (dim ** -0.5)
