"""Shared family-agnostic QuantMLA (QMLA) calibration package.

Phase 0 (mode-agnostic) provides the capture interface, layer geometry
descriptors, exact full-precision attention/positional-QK evaluators,
deterministic train/val splitting, and sequential XXH3-128 capture storage.
Fit routines, quantizers, sidecar writers/validators, and incremental
publication land after Phase 0 design confirmation (#2909).
"""
from __future__ import annotations

from .capture import (
    CAPTURE_SCHEMA,
    CapturedLayerBank,
    CapturedLayerParams,
    CapturedSequence,
    MlaLayerGeometry,
    QmlaCaptureAdapter,
    causal_mask,
    eval_attention_output,
    eval_positional_qk,
    load_layer_capture,
    rms_norm,
    save_layer_capture,
    split_train_val,
    unit_rms_norm,
)

__all__ = [
    "CAPTURE_SCHEMA",
    "CapturedLayerBank",
    "CapturedLayerParams",
    "CapturedSequence",
    "MlaLayerGeometry",
    "QmlaCaptureAdapter",
    "causal_mask",
    "eval_attention_output",
    "eval_positional_qk",
    "load_layer_capture",
    "rms_norm",
    "save_layer_capture",
    "split_train_val",
    "unit_rms_norm",
]
