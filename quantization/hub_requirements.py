"""One qualified Hub SDK for receipt-backed, resumable Xet publication."""
import importlib.metadata

HUB_VERSION = "1.33.0"


def require_qualified_hub(*, allow_http_lfs=False):
    from huggingface_hub import constants
    from huggingface_hub.utils._runtime import is_xet_available

    http_lfs = allow_http_lfs and constants.HF_HUB_DISABLE_XET
    if importlib.metadata.version("huggingface_hub") != HUB_VERSION or not (is_xet_available() or http_lfs):
        raise ValueError(f"publication requires qualified huggingface_hub {HUB_VERSION} with Xet enabled")
