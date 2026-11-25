"""One qualified Hub SDK for receipt-backed, resumable HTTP LFS publication."""
import importlib.metadata
import os

HUB_VERSION = "1.33.0"


def use_http_lfs():
    """Pin this publisher process (and its children) to HTTP LFS; Xet is not used.

    The SDK reads ``constants.HF_HUB_DISABLE_XET`` at call time, so setting it
    here takes effect even after ``huggingface_hub`` was imported.
    """
    from huggingface_hub import constants

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    constants.HF_HUB_DISABLE_XET = True


def require_qualified_hub():
    """Publication runs only on the pinned SDK with Xet disabled (HTTP LFS)."""
    from huggingface_hub import constants

    if importlib.metadata.version("huggingface_hub") != HUB_VERSION or not constants.HF_HUB_DISABLE_XET:
        raise ValueError(f"publication requires qualified huggingface_hub {HUB_VERSION} "
                         "over HTTP LFS (HF_HUB_DISABLE_XET=1)")
