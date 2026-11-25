#!/usr/bin/env python3
"""Compare two F32 logits dumps, as written by a golden's --save-logits.

  scripts/bench/compare-logits.py A B [VOCAB] [--tokens tokens.bin --offset K] [--json OUT]

Two layouts are read from their own bytes:

- a flat `.bin` dump: rows x VOCAB little-endian F32, VOCAB required;
- a per-row safetensors dump: a directory holding `row-NNNNNNNN.safetensors`,
  one F32 tensor per row with its width in the tensor header (Kimi and the
  shared probe write this). VOCAB is optional and, when given, must match
  every row's width.

Reports byte equality first, then over the common rows: max |a - b|, mean and
max KL(a||b) (float64), top-1 agreement and, with the token file and the row
offset of A's first row in it, each side's mean NLL of the next token. A row
tensor named `log_probs` is already log-softmaxed and is not re-normalized;
`logits` and `.bin` rows are. A row dump also reports the row holding the
largest KL and the largest |diff|, so a perturbed row is named. Rows are
streamed in blocks, so full-prompt dumps (~1 GB) stay within a few hundred MB
of memory. Exit 0 when both sides were compared (the verdict is in the
output), 2 on unusable input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

BLOCK_ROWS = 64
ROW_SUFFIX = '.safetensors'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def log_softmax(x: np.ndarray) -> np.ndarray:
    top = x.max(1, keepdims=True)
    return x - top - np.log(np.exp(x - top).sum(1, keepdims=True))


def compare(a_path: Path, b_path: Path, vocab: int, tokens: Path | None = None, offset: int = 0) -> dict:
    if vocab <= 0:
        raise ValueError('vocab must be positive')
    sizes = {name: path.stat().st_size for name, path in (('a', a_path), ('b', b_path))}
    for name, size in sizes.items():
        if size % (4 * vocab):
            raise ValueError(f'{name}: {size} bytes is not a whole number of {vocab}-wide F32 rows')
    result = dict(a=str(a_path), b=str(b_path), vocab=vocab, sha256_a=sha256(a_path), sha256_b=sha256(b_path),
                  rows_a=sizes['a'] // (4 * vocab), rows_b=sizes['b'] // (4 * vocab))
    result['byte_identical'] = result['sha256_a'] == result['sha256_b'] and sizes['a'] == sizes['b']
    rows = min(result['rows_a'], result['rows_b'])
    result['rows'] = rows
    if rows == 0:
        return result
    a = np.memmap(a_path, dtype='<f4', mode='r', shape=(result['rows_a'], vocab))
    b = np.memmap(b_path, dtype='<f4', mode='r', shape=(result['rows_b'], vocab))
    ids = None
    if tokens is not None:
        ids = np.fromfile(tokens, dtype='<u4')
    max_diff, kl_sum, kl_max, agree, nll_a, nll_b, scored = 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0
    nonfinite = 0
    for start in range(0, rows, BLOCK_ROWS):
        stop = min(rows, start + BLOCK_ROWS)
        xa, xb = np.asarray(a[start:stop], dtype=np.float64), np.asarray(b[start:stop], dtype=np.float64)
        finite = np.isfinite(xa).all(1) & np.isfinite(xb).all(1)
        nonfinite += int((~finite).sum())
        xa, xb = xa[finite], xb[finite]
        if not len(xa):
            continue
        max_diff = max(max_diff, float(np.abs(xa - xb).max()))
        la, lb = log_softmax(xa), log_softmax(xb)
        kl = (np.exp(la) * (la - lb)).sum(1)
        kl_sum += float(kl.sum())
        kl_max = max(kl_max, float(kl.max()))
        agree += int((xa.argmax(1) == xb.argmax(1)).sum())
        if ids is not None:
            index = np.arange(start, stop)[finite] + offset + 1
            keep = index < len(ids)
            nxt = ids[index[keep]]
            nll_a -= float(la[keep][np.arange(len(nxt)), nxt].sum())
            nll_b -= float(lb[keep][np.arange(len(nxt)), nxt].sum())
            scored += len(nxt)
    compared = rows - nonfinite
    result['last_row_argmax_match'] = bool(np.argmax(a[rows - 1]) == np.argmax(b[rows - 1]))
    result.update(nonfinite_rows=nonfinite, max_abs_diff=max_diff, mean_kl=kl_sum / max(compared, 1),
                  max_kl=kl_max, top1_agreement=agree / max(compared, 1))
    if ids is not None:
        result.update(nll_a=nll_a / max(scored, 1), nll_b=nll_b / max(scored, 1), nll_rows=scored)
    return result


def row_files(directory: Path) -> dict[str, Path]:
    """name -> path of a per-row dump's row-NNNNNNNN.safetensors files."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'{directory}: not a directory of row-*{ROW_SUFFIX} dumps')
    rows = {path.name: path for path in directory.glob('row-*' + ROW_SUFFIX) if path.is_file()}
    if not rows:
        raise ValueError(f'{directory}: no row-*{ROW_SUFFIX} files')
    return rows


def parse_row(raw: bytes, label: str) -> tuple[str, np.ndarray]:
    """The single F32 row tensor of a safetensors file, from its own header."""
    if len(raw) < 8:
        raise ValueError(f'{label}: shorter than a safetensors header length')
    (extent,) = struct.unpack('<Q', raw[:8])
    if extent > len(raw) - 8:
        raise ValueError(f'{label}: header extent {extent} exceeds the file')
    try:
        header = json.loads(raw[8:8 + extent])
    except ValueError as error:
        raise ValueError(f'{label}: unreadable safetensors header ({error})') from error
    if not isinstance(header, dict) or len(header) != 1:
        raise ValueError(f'{label}: expected exactly one tensor, found {sorted(header) if isinstance(header, dict) else header}')
    name, spec = next(iter(header.items()))
    if spec.get('dtype') != 'F32':
        raise ValueError(f'{label}: {name} is {spec.get("dtype")}, not F32')
    shape, offsets = spec.get('shape'), spec.get('data_offsets')
    if not (isinstance(shape, list) and len(shape) == 1 and isinstance(shape[0], int)):
        raise ValueError(f'{label}: {name} is not one-dimensional')
    if not (isinstance(offsets, list) and len(offsets) == 2):
        raise ValueError(f'{label}: {name} has no data_offsets')
    begin, end = offsets
    if end - begin != 4 * shape[0] or begin < 0 or 8 + extent + end > len(raw):
        raise ValueError(f'{label}: {name} offsets do not hold {shape[0]} F32 values')
    values = np.frombuffer(raw[8 + extent + begin:8 + extent + end], dtype='<f4')
    if len(values) != shape[0]:
        raise ValueError(f'{label}: {name} holds {len(values)} values, header says {shape[0]}')
    return name, values


def probabilities(name: str, values: np.ndarray) -> np.ndarray:
    """Log-probabilities of one row: `log_probs` is already normalized."""
    if name == 'log_probs':
        return values
    top = values.max()
    return values - top - np.log(np.exp(values - top).sum())


def compare_rows(a_path: Path, b_path: Path, vocab: int | None = None) -> dict:
    """Compare two per-row safetensors dumps, row file by row file."""
    a_rows, b_rows = row_files(a_path), row_files(b_path)
    only_a, only_b = sorted(set(a_rows) - set(b_rows)), sorted(set(b_rows) - set(a_rows))
    names = sorted(set(a_rows) & set(b_rows))
    result = dict(a=str(a_path), b=str(b_path), layout='rows', vocab=vocab,
                  rows_a=len(a_rows), rows_b=len(b_rows), rows=len(names),
                  only_a=len(only_a), only_b=len(only_b), only_a_rows=only_a[:8], only_b_rows=only_b[:8])
    identical = not only_a and not only_b
    max_diff = kl_sum = kl_max = 0.0
    agree = compared = nonfinite = 0
    worst_kl = worst_diff = None
    for label in names:
        raw_a, raw_b = a_rows[label].read_bytes(), b_rows[label].read_bytes()
        identical = identical and raw_a == raw_b
        name_a, values_a = parse_row(raw_a, f'{a_path}/{label}')
        name_b, values_b = parse_row(raw_b, f'{b_path}/{label}')
        if name_a != name_b:
            raise ValueError(f'{label}: tensor names differ ({name_a} vs {name_b})')
        if values_a.shape != values_b.shape:
            raise ValueError(f'{label}: widths differ ({values_a.shape[0]} vs {values_b.shape[0]})')
        if vocab is None:
            vocab = len(values_a)
        if len(values_a) != vocab:
            raise ValueError(f'{label}: width {len(values_a)} does not match {vocab}')
        finite = bool(np.isfinite(values_a).all() and np.isfinite(values_b).all())
        if not finite:
            nonfinite += 1
            continue
        xa, xb = values_a.astype(np.float64), values_b.astype(np.float64)
        diff = float(np.abs(xa - xb).max())
        if worst_diff is None or diff > max_diff:
            worst_diff = label
        max_diff = max(max_diff, diff)
        la, lb = probabilities(name_a, xa), probabilities(name_b, xb)
        kl = float((np.exp(la) * (la - lb)).sum())
        if worst_kl is None or kl > kl_max:
            worst_kl = label
        kl_sum += kl
        kl_max = max(kl_max, kl)
        agree += int(xa.argmax() == xb.argmax())
        compared += 1
    result.update(vocab=vocab, byte_identical=identical, nonfinite_rows=nonfinite,
                  max_abs_diff=max_diff, mean_kl=kl_sum / max(compared, 1), max_kl=kl_max,
                  top1_agreement=agree / max(compared, 1),
                  worst_row=worst_kl, worst_abs_diff_row=worst_diff)
    return result


def compare_paths(a_path: Path, b_path: Path, vocab: int | None = None,
                  tokens: Path | None = None, offset: int = 0) -> dict:
    """Dispatch on the dumps' own layout: a flat .bin file or a per-row directory."""
    a_dir, b_dir = Path(a_path).is_dir(), Path(b_path).is_dir()
    if a_dir or b_dir:
        if not (a_dir and b_dir):
            raise ValueError(f'{a_path} and {b_path}: a per-row dump compares against another per-row dump')
        return compare_rows(a_path, b_path)
    if vocab is None:
        raise ValueError('a flat .bin logits dump needs VOCAB')
    return compare(a_path, b_path, vocab, tokens, offset)


def describe(result: dict) -> str:
    if result.get('layout') == 'rows':
        lines = [f"byte-identical: {result['byte_identical']}  rows {result['rows_a']} vs {result['rows_b']}"
                 f" ({result['rows']} compared)",
                 f"a {result['a']}", f"b {result['b']}"]
        if result['only_a'] or result['only_b']:
            lines.append(f"rows only in a ({result['only_a']}): {', '.join(result['only_a_rows'])}")
            lines.append(f"rows only in b ({result['only_b']}): {', '.join(result['only_b_rows'])}")
        if result['rows']:
            lines.append(f"rows {result['rows']}: max |diff| {result['max_abs_diff']:.4e} | "
                         f"mean KL(a||b) {result['mean_kl']:.6f} max {result['max_kl']:.6f} | "
                         f"top-1 agreement {100 * result['top1_agreement']:.2f}%"
                         + (f" | {result['nonfinite_rows']} nonfinite rows skipped" if result['nonfinite_rows'] else ''))
            if not result['byte_identical'] and result.get('worst_row'):
                lines.append(f"worst row (max KL) {result['worst_row']} | "
                             f"max |diff| row {result['worst_abs_diff_row']}")
        return '\n'.join(lines)
    lines = [f"byte-identical: {result['byte_identical']}  rows {result['rows_a']} vs {result['rows_b']}",
             f"sha256 a {result['sha256_a']}", f"sha256 b {result['sha256_b']}"]
    if result['rows']:
        lines.append(f"rows {result['rows']}: max |diff| {result['max_abs_diff']:.4e} | mean KL(a||b) {result['mean_kl']:.6f} "
                     f"max {result['max_kl']:.6f} | top-1 agreement {100 * result['top1_agreement']:.2f}%"
                     + (f" | {result['nonfinite_rows']} nonfinite rows skipped" if result['nonfinite_rows'] else ''))
    if 'last_row_argmax_match' in result:
        lines.append(f"last-row argmax match: {result['last_row_argmax_match']}")
    if 'nll_a' in result:
        lines.append(f"mean NLL a {result['nll_a']:.4f} b {result['nll_b']:.4f} over {result['nll_rows']} rows")
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('a', type=Path)
    parser.add_argument('b', type=Path)
    parser.add_argument('vocab', type=int, nargs='?', help='row width (.bin dumps; checked against row headers)')
    parser.add_argument('--tokens', type=Path, help="token file (u32) the rows score against")
    parser.add_argument('--offset', type=int, default=0, help="index in --tokens of A's first row")
    parser.add_argument('--json', type=Path, help='also write the result as JSON')
    args = parser.parse_args()
    try:
        result = compare_paths(args.a, args.b, args.vocab, args.tokens, args.offset)
    except (OSError, ValueError) as error:
        print(f'compare-logits: {error}', file=sys.stderr)
        return 2
    print(describe(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
