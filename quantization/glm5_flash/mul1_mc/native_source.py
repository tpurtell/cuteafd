"""Materialize source-native tensors once into final MC shards and calibrate from those bytes."""
from pathlib import Path

from safetensors import safe_open

from inkling2.numerical import fork_module

NativeFlash = fork_module("utils.glm5_flash_core").GlmFlashBF16Source
GROUP = "source-native"


class PublishedNativeSource(NativeFlash):
    """Use the fork's owned tensor reads for final native shards and original experts."""

    def __init__(self, snapshot, local_files):
        super().__init__(snapshot)
        self.local_files = local_files

    def _active_reader(self, name):
        path = self.local_files.get(name)
        if path is None:
            return super()._active_reader(name)
        if self._reader_shard != path:
            self.close()
            self._reader_context = safe_open(path, framework="pt", device="cpu")
            self._reader = self._reader_context.__enter__()
            self._reader_shard = path
        return self._reader


def prepare_native_source(snapshot, writer, *, core_modules, expert_modules):
    """Commit native copy proofs before installing any reconstructed matrices."""
    snapshot = Path(snapshot).resolve()
    if snapshot != writer.source or not writer.prove_shared_source:
        raise ValueError("native calibration requires the matching source and byte-copy proof writer")
    locations = writer.source_inventory["locations"]
    originals = {f"{name}.weight" for name in core_modules}
    routed = {f"{name}.weight" for name in expert_modules}
    if originals & routed or not (originals | routed) <= locations.keys():
        raise ValueError("core and routed inventory disagrees with pinned source")
    shared = locations.keys() - originals - routed

    def references(names):
        return {name: (snapshot / locations[name]["shard"], locations[name]["source_name"]) for name in names}

    sealed = writer.write_group(GROUP, original=references(originals), shared=references(shared))
    local_files = {name: writer.output / row["filename"] for row in sealed["shards"] for name in row["names"]}
    if not sealed["complete"] or local_files.keys() != shared | originals:
        raise ValueError("sealed native payload inventory differs")
    return PublishedNativeSource(snapshot, local_files)
