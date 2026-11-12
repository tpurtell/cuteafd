#!/usr/bin/env python3
"""Content identities and measured inputs for CPU-only AOT publications."""
from __future__ import annotations

import fnmatch
import functools
import hashlib
import io
import inspect
import json
import shutil
import subprocess
import os
from pathlib import Path
import struct
import sys
import types
import weakref


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def job_global_inputs(context):
    """The exporter and arguments are inputs even when policy ran before tracing."""
    return {key: context[key] for key in (
        "exporter", "exporter_sha256", "args", "comparison_script_sha256",
        "export_sources_sha256", "sms", "capability", "toolchain", "environment", "build_kind")}


def is_bytecode(path):
    return "__pycache__" in path.parts or path.suffix == ".pyc"


def source_name(path, root):
    try:
        resolved = Path(path).resolve()
        relative = resolved.relative_to(root.resolve())
    except (ValueError, TypeError, OSError):
        return None
    # Import caches embed worktree paths and timestamps; the .py source is recorded.
    return relative.as_posix() if not is_bytecode(relative) and resolved.is_file() else None


def elf_resources(blob):
    """Read CUDA ELF section metadata without initializing a CUDA context."""
    if blob[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected a little-endian ELF64 CUDA image")
    offset = struct.unpack_from("<Q", blob, 40)[0]
    size, count, strings_index = struct.unpack_from("<HHH", blob, 58)
    if size < 64 or strings_index >= count or offset + size * count > len(blob):
        raise ValueError("invalid CUDA ELF section table")
    strings_offset, strings_size = struct.unpack_from("<QQ", blob, offset + strings_index * size + 24)
    strings = blob[strings_offset:strings_offset + strings_size]
    kernels = {}
    symbols = {}
    for index in range(count):
        entry = offset + index * size
        kind = struct.unpack_from("<I", blob, entry + 4)[0]
        if kind == 2:  # SHT_SYMTAB; EIATTR_REGCOUNT references symbol indices.
            start, length = struct.unpack_from("<QQ", blob, entry + 24)
            link = struct.unpack_from("<I", blob, entry + 40)[0]
            names_start, names_size = struct.unpack_from("<QQ", blob, offset + link * size + 24)
            names = blob[names_start:names_start + names_size]
            for symbol_index, symbol_offset in enumerate(range(start, start + length, 24)):
                name_offset = struct.unpack_from("<I", blob, symbol_offset)[0]
                symbols[symbol_index] = names[name_offset:names.find(b"\0", name_offset)].decode()
    register_counts = {}
    for index in range(count):
        entry = offset + index * size
        name_offset = struct.unpack_from("<I", blob, entry)[0]
        end = strings.find(b"\0", name_offset)
        name = strings[name_offset:end if end >= 0 else len(strings)].decode()
        data_offset, data_size = struct.unpack_from("<QQ", blob, entry + 24)
        if name.startswith(".text."):
            kernel = kernels.setdefault(name[6:], {})
            count_from_header = struct.unpack_from("<I", blob, entry + 44)[0] >> 24
            if count_from_header:
                kernel["registers"] = count_from_header
        elif name.startswith(".nv.shared.") and name != ".nv.shared.reserved.0":
            kernels.setdefault(name[11:], {})["static_smem_bytes"] = data_size
        elif name == ".nv.info" or name.startswith(".nv.info."):
            kernel = kernels.setdefault(name[9:], {}) if name != ".nv.info" else None
            data = blob[data_offset:data_offset + data_size]
            position = 0
            while position + 4 <= len(data):
                form, attribute, value = struct.unpack_from("<BBH", data, position)
                position += 4
                if form == 4:  # EIFMT_SVAL; MAX_THREADS is a three-dimensional extent.
                    payload = data[position:position + value]
                    position += value
                    if kernel is not None and attribute in (5, 16) and len(payload) == 12:
                        x, y, z = struct.unpack("<III", payload)
                        kernel["max_threads"] = x * y * z
                    elif name == ".nv.info" and attribute == 47 and len(payload) == 8:
                        symbol, registers = struct.unpack("<II", payload)
                        register_counts[symbols[symbol]] = registers
    for name, registers in register_counts.items():
        if name in kernels:
            kernels[name]["registers"] = registers
    for kernel in kernels.values():
        kernel.setdefault("static_smem_bytes", 0)
        kernel.setdefault("registers", None)
        kernel.setdefault("max_threads", None)
    return {name: kernels[name] for name in sorted(kernels)}


# Keep explicit absent values too; the source-scan regression guards new reads.
EXPORT_ENVIRONMENT_NAMES = frozenset("""
B12X_COMPILE_DISK_CACHE B12X_COMPILE_MEMORY_CACHE CUDA_VISIBLE_DEVICES TRITON_CACHE_DIR
CUTEAFD_BUILD_KIND CUTEAFD_CUTE_HOST_TARGET CUTEAFD_EXL3_ROUTE_BLOCKS
CUTEAFD_EXPORT_NARROW_AOT CUTEAFD_EXPORT_HC_LAGGED CUTEAFD_EXPORT_IMAGE_ID
CUTEAFD_SOURCE_ROOT CUTEAFD_SPARKINFER_SOURCE_DIR CUTEAFD_SPARKINFER_LOCK_FILE
CUTEAFD_EXPORT_HOST_CLASS CUTEAFD_EXPORT_HOST_CONTEXTS CUTEAFD_EXPORT_HOST_WORKERS
CUTEAFD_EXPORT_JOBS CUTEAFD_EXPORT_SLOT_DIR CUTEAFD_EXPORT_HOST_SLOT_DIR
CUTEAFD_EXPORT_GPU_SLOT_DIR CUTEAFD_EXPORT_GPU_BUDGET_MIB
CUTEAFD_EXPORT_WORKER_GPU_MIB CUTEAFD_EXPORT_WORKER_RAM_MIB
CUTEAFD_EXPORT_CONTEXT_SPACING_S HF_HUB_CACHE HF_HOME XDG_CACHE_HOME TRITON_DISABLE_LINE_INFO CUTEAFD_EXPORT_CACHE
""".split())


PROVENANCE_ENVIRONMENT_NAMES = frozenset({
        "CUTEAFD_EXPORT_HOST_CLASS", "CUTEAFD_EXPORT_HOST_CONTEXTS", "CUTEAFD_EXPORT_HOST_WORKERS",
        "CUTEAFD_EXPORT_JOBS", "CUTEAFD_EXPORT_SLOT_DIR", "CUTEAFD_EXPORT_HOST_SLOT_DIR",
        "CUTEAFD_EXPORT_GPU_SLOT_DIR", "CUTEAFD_EXPORT_GPU_BUDGET_MIB",
        "CUTEAFD_EXPORT_WORKER_GPU_MIB", "CUTEAFD_EXPORT_WORKER_RAM_MIB",
        "CUTEAFD_EXPORT_CONTEXT_SPACING_S", "HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME",
})


def export_environment():
    # Only settings read by exporters belong here, not arbitrary CUTEAFD_ launch
    # settings. New reads require classification in the source-scan regression.
    values = {name: os.environ.get(name) for name in sorted(EXPORT_ENVIRONMENT_NAMES)}
    values.update({name: "<provenance>" for name in PROVENANCE_ENVIRONMENT_NAMES})
    # These are forced by the offline worker, not inherited policy. Paths name
    # verified sources or disposable compiler storage rather than semantic inputs.
    values.update(CUTEAFD_BUILD_KIND=os.environ.get("CUTEAFD_BUILD_KIND", "development"),
                  CUTEAFD_EXPORT_NARROW_AOT="1", CUTEAFD_EXPORT_HC_LAGGED="1",
                  B12X_COMPILE_DISK_CACHE="0", B12X_COMPILE_MEMORY_CACHE="0",
                  CUDA_VISIBLE_DEVICES="", TRITON_CACHE_DIR="<worker-cache>", TRITON_DISABLE_LINE_INFO="1",
                  CUTEAFD_EXPORT_CACHE="<export-cache>",
                  CUTEAFD_SOURCE_ROOT="<verified-source>",
                  CUTEAFD_SPARKINFER_SOURCE_DIR="<verified-fork>",
                  CUTEAFD_SPARKINFER_LOCK_FILE="<verified-lock>")
    return values


def compiler_identity(root, payload, capability):
    """An immutable compiler image is mandatory; unknown identity disables reuse."""
    from b12x._lib import compiler
    image = os.environ.get("CUTEAFD_EXPORT_IMAGE_ID", "")
    ptxas = shutil.which("ptxas")
    if not image.startswith("sha256:") or not ptxas:
        return None
    version = subprocess.check_output([ptxas, "--version"], text=True, timeout=30)
    exporter = Path(root) / "python/tools/aot" / payload["exporter"]
    return {"schema": 1, "image_id": image, "runtime": compiler._runtime_toolchain_key(),
            "environment": compiler._compile_environment_key(),
            "export_environment": export_environment(),
            "ptxas_version": version,
            "ptxas_sha256": digest(Path(ptxas).read_bytes()), "sms": payload["sms"],
            "capability": list(capability), "exporter": payload["exporter"], "args": payload["args"],
            "exporter_sha256": digest(exporter.read_bytes())}


def stage0_inputs(root, payload, capability):
    identity = compiler_identity(root, payload, capability)
    if identity is None:
        return None
    root = Path(root)
    rules_path = root / "scripts/build/kernel-input-rules.json"
    rules = json.loads(rules_path.read_text())
    # Orchestration exclusions affect only globals; executed helpers still enter recorded reads.
    hashes = {}
    for directory in ("third_party/sparkinfer/b12x", "python/tools", "scripts/build", "docker"):
        for path in sorted((root / directory).rglob("*")):
            relative = path.relative_to(root).as_posix()
            if not path.is_file() or any(fnmatch.fnmatch(relative, pattern) for pattern in rules["ignore"]):
                continue
            if (any(fnmatch.fnmatch(relative, pattern) for pattern in rules["global"])
                    and not any(fnmatch.fnmatch(relative, pattern) for pattern in rules.get("global_ignore", []))):
                hashes[relative] = digest(path.read_bytes())
    return json.loads(json.dumps({"identity": identity, "global_sources": hashes,
                                  "rules": digest(rules_path.read_bytes())}, sort_keys=True))


def job_candidates(root, previous, current_inputs):
    if current_inputs is None or previous.get("stage0_inputs") != current_inputs:
        return ["job-global-inputs"]
    reads = previous.get("source_reads_sha256")
    if not reads:
        return ["missing-recorded-reads"]
    changed = []
    for relative, checksum in reads.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            return ["invalid-recorded-path"]
        if is_bytecode(path):
            continue
        source = Path(root) / path
        if not source.is_file() or digest(source.read_bytes()) != checksum:
            changed.append(relative)
    return changed


def backend_identity():
    """Lowering inputs only; target, ABI and recipes are in IR/options, not placement."""
    import platform
    from b12x._lib import compiler
    image = os.environ.get("CUTEAFD_EXPORT_IMAGE_ID", "")
    ptxas = shutil.which("ptxas")
    if not image.startswith("sha256:") or not ptxas:
        return None
    from _export_cache import static_environment
    return {"schema": 2, "image": image, "machine": platform.machine(),
            "runtime": compiler._runtime_toolchain_key(),
            "environment": compiler._compile_environment_key(),
            "static_environment": static_environment(),
            "host_target": os.environ.get("CUTEAFD_CUTE_HOST_TARGET", "<portable-default>"),
            "ptxas_sha256": digest(Path(ptxas).read_bytes())}


def install_backend_cache(directory=None):
    """Shared hook for CMake exporters and offline placement workers, including children."""
    store = os.environ.get("CUTEAFD_EXPORT_CACHE", "")
    if directory is None:
        if not store or store == "off":
            return
        directory = Path(store) / "backend"
    try:
        from cutlass.base_dsl.dsl import BaseDSL
    except ModuleNotFoundError:
        return  # source verification and host-only tools do not have CuTe
    if not os.environ.get("CUTEAFD_EXPORT_IMAGE_ID", "").startswith("sha256:"):
        return
    previous = getattr(BaseDSL, "_cuteafd_backend_cache", None)
    if previous is not None and previous.directory == Path(directory):
        return previous
    cache = BackendCache(directory, None)
    BaseDSL._cuteafd_backend_cache = cache
    import atexit
    def report():
        from _export_cache import stats
        stats("backend", layer="backend", hits=cache.hits, misses=cache.misses)
    atexit.register(report)
    if getattr(BaseDSL.compile_and_cache, "_cuteafd_backend_cache", False):
        return cache
    original = BaseDSL.compile_and_cache

    @functools.wraps(original)
    def compile_cached(dsl, module, *args, **kwargs):
        cache = BaseDSL._cuteafd_backend_cache
        cache.identity = backend_identity()
        return cache.compile(dsl, original, module, args, kwargs, digest(str(module).encode()))

    compile_cached._cuteafd_backend_cache = True
    BaseDSL.compile_and_cache = compile_cached
    return cache


class BackendCache:
    """Retain lowered MLIR, never Python executors or a guessed launch ABI."""

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.hits = 0
        self.misses = 0

    def key(self, dsl, original, module, args, kwargs, ir_hash):
        bound = inspect.signature(original).bind(dsl, module, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if not self.identity or not values.get("no_jit_engine") or not values.get("no_cache"):
            return None
        options = {kind.__module__ + "." + kind.__qualname__: option.value
                   for kind, option in dsl.compile_options.options.items()}
        if any(value for name, value in options.items() if name.endswith(".LinkLibraries")):
            return None  # External bitcode contents are not yet fingerprinted.
        inputs = {"schema": 1, "ir_sha256": ir_hash, "identity": self.identity,
                  "options": options, "pipeline": values.get("pipeline"),
                  "function_name": values["function_name"], "signature": str(values["sig"]),
                  "executor_type": str(values["func_type"])}
        try:
            return stable_hash(inputs)
        except (TypeError, ValueError):
            return None  # Unserializable compiler inputs are never a hit.

    def compile(self, dsl, original, module, args, kwargs, ir_hash):
        from cutlass._mlir import ir
        key = self.key(dsl, original, module, args, kwargs, ir_hash)
        metadata = self.directory / (str(key) + ".json")
        bytecode = self.directory / (str(key) + ".mlirbc")
        if key and metadata.is_file() and bytecode.is_file():
            try:
                saved = json.loads(metadata.read_text())
                blob = bytecode.read_bytes()
                restored = (ir.Module.parse(blob, context=module.context)
                            if saved == {"key": key, "sha256": digest(blob)} else None)
            except (OSError, ValueError, RuntimeError):
                restored = None
            if restored is not None:
                provider = dsl.compiler_provider
                compile_method = provider.compile
                def already_compiled(current, *unused_args, **unused_kwargs):
                    if current is not restored:
                        raise ValueError("backend replay received an unexpected module")
                provider.compile = already_compiled
                try:
                    result = original(dsl, restored, *args, **kwargs)
                finally:
                    provider.compile = compile_method
                self.hits += 1
                return result
        self.misses += 1
        result = original(dsl, module, *args, **kwargs)
        if key:
            stream = io.BytesIO()
            result.ir_module.operation.write_bytecode(stream)
            blob = stream.getvalue()
            temporary = self.directory / (key + f".{os.getpid()}.tmp")
            temporary.write_bytes(blob)
            temporary.replace(bytecode)
            temporary.write_text(json.dumps({"key": key, "sha256": digest(blob)}, sort_keys=True))
            temporary.replace(metadata)
        return result


class Recorder:
    """Observe the existing compiler pass; never invoke a second compilation.

    Read sets contain executed frontend functions, referenced module globals,
    newly imported modules and opened inputs. Whole files are fingerprinted,
    retaining module-level constants without making eager unused imports global.
    """

    def __init__(self, root, output, backend_cache=None):
        self.backend_cache = backend_cache
        self.root = Path(root).resolve()
        self.output = Path(output)
        self.executors = {}
        self.modules = {}
        self.exports = {}
        self.opened = set()
        self.frames = set()
        self.active = False
        self.restores = []
        self.frame_paths = {}
        self.functions = set()
        self.imported = set()
        self.construction = False
        self.latest = []
        self.monitoring_id = None

    def _start(self, code, offset):
        frame = sys._getframe(1)
        self._profile(frame, "call", None)
        return sys.monitoring.DISABLE

    def start_monitoring(self):
        for tool_id in (3, 4):
            if sys.monitoring.get_tool(tool_id) is None:
                sys.monitoring.use_tool_id(tool_id, "cuteafd-kernel-records")
                self.monitoring_id = tool_id
                break
        if self.monitoring_id is None:
            raise RuntimeError("no sys.monitoring tool ID available for source recording")
        sys.monitoring.register_callback(self.monitoring_id, sys.monitoring.events.PY_START, self._start)
        sys.monitoring.restart_events()
        sys.monitoring.set_events(self.monitoring_id, sys.monitoring.events.PY_START)

    def stop_monitoring(self):
        if self.monitoring_id is not None:
            sys.monitoring.set_events(self.monitoring_id, 0)
            sys.monitoring.register_callback(self.monitoring_id, sys.monitoring.events.PY_START, None)
            sys.monitoring.free_tool_id(self.monitoring_id)
            self.monitoring_id = None

    def construct(self, function, *args, **kwargs):
        """Trace the entire recipe, including launch construction and scratch."""
        self.latest = []
        self.construction = True
        try:
            return function(*args, **kwargs)
        finally:
            reads = self.reads()
            for record in self.latest:
                record["source_reads_sha256"].update(reads)
            self.construction = False

    def _audit(self, event, args):
        if self.active and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = source_name(args[0], self.root)
            if path and not path.startswith(str(self.output)):
                self.opened.add(path)

    def _profile(self, frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename
            if filename not in self.frame_paths:
                self.frame_paths[filename] = source_name(filename, self.root)
            path = self.frame_paths[filename]
            if path:
                self.frames.add(path)
                if not (path == "scripts/build/kernel-records.py"
                        and frame.f_code.co_qualname.startswith("BackendCache.")):
                    self.functions.add((path, frame.f_code.co_qualname))
                # Active callers can hold function-local package imports that
                # do not execute again inside the measured constructor.
                names = (*frame.f_code.co_names, *frame.f_code.co_varnames,
                         *frame.f_code.co_freevars, *frame.f_code.co_cellvars)
                self._record_dependencies(frame.f_locals.get(name) for name in names)
                self._record_dependencies(frame.f_globals.get(name) for name in frame.f_code.co_names)

    def _record_dependencies(self, values):
        for value in values:
            if not isinstance(value, (types.ModuleType, types.FunctionType, type)):
                continue
            module = sys.modules.get(getattr(value, "__module__", ""))
            for dependency in (module, value):
                filename = getattr(dependency, "__file__", None)
                if filename is not None:
                    source = source_name(filename, self.root)
                    if source:
                        self.frames.add(source)

    def reads(self):
        paths = self.opened | self.frames
        for name in set(sys.modules) - self.imported:
            module = sys.modules.get(name)
            path = source_name(getattr(module, "__file__", None), self.root)
            if path:
                paths.add(path)
        active = self.active
        self.active = False
        try:
            return {path: digest((self.root / path).read_bytes()) for path in sorted(paths)}
        finally:
            self.active = active

    def patch(self, cls, method, replacement):
        original = getattr(cls, method)
        self.restores.append((cls, method, original))
        setattr(cls, method, replacement(original))

    def install(self):
        from cutlass.base_dsl.dsl import BaseDSL
        from cutlass.base_dsl.jit_executor import JitCompiledFunction
        sys.addaudithook(self._audit)
        self.imported = set(sys.modules)
        self.active = True
        self.start_monitoring()

        def frontend(original):
            @functools.wraps(original)
            def wrapped(dsl, *args, **kwargs):
                frame = sys._getframe().f_back
                while frame is not None:
                    # Recipe locals/closures can change between capacities even
                    # when PY_START has already disabled this code object's event.
                    self._profile(frame, "call", None)
                    frame = frame.f_back
                result = original(dsl, *args, **kwargs)
                module = result[0]
                # Do not normalize symbols or locations: both can carry semantics.
                self.modules[id(module)] = {
                    "ir_kind": "original_mlir", "ir_sha256": digest(str(module).encode()),
                    "source_reads_sha256": self.reads(),
                }
                return result
            return wrapped

        def compiled(original):
            @functools.wraps(original)
            def wrapped(dsl, module, *args, **kwargs):
                record = self.modules.pop(id(module), None)
                if record is None:
                    record = {"ir_kind": "mlir", "ir_sha256": digest(str(module).encode()),
                              "source_reads_sha256": self.reads()}
                result = (self.backend_cache.compile(dsl, original, module, args, kwargs, record["ir_sha256"])
                          if self.backend_cache is not None else original(dsl, module, *args, **kwargs))
                self.executors[id(result)] = (weakref.ref(result), record)
                if self.construction:
                    self.latest.append(record)
                return result
            return wrapped

        def exported(original):
            @functools.wraps(original)
            def wrapped(executor, file_path, file_name, function_prefix=""):
                result = original(executor, file_path, file_name, function_prefix)
                found = self.executors.get(id(executor))
                if found is None or found[0]() is not executor:
                    raise ValueError(f"{file_name}: export has no recorded compiler IR")
                path = (Path(file_path) / (file_name + ".o")).relative_to(self.output).as_posix()
                self.exports[path] = found[1]
                self.exports[path]["export_prefix"] = function_prefix or file_name
                return result
            return wrapped

        self.patch(BaseDSL, "generate_original_ir", frontend)
        self.patch(BaseDSL, "compile_and_cache", compiled)
        self.patch(JitCompiledFunction, "export_to_c", exported)
        return self

    def finish(self, inspector, context, plans):
        self.stop_monitoring()
        self.active = False
        for cls, method, original in reversed(self.restores):
            setattr(cls, method, original)
        records = {}
        for path in sorted(self.output.rglob("*")):
            if path.suffix not in (".o", ".cubin"):
                continue
            relative = path.relative_to(self.output).as_posix()
            record = self.exports.get(relative)
            if record is None and path.suffix == ".cubin" and path.with_suffix(".ptx").exists():
                record = {"ir_kind": "ptx", "ir_sha256": digest(path.with_suffix(".ptx").read_bytes()),
                          "source_reads_sha256": self.reads()}
            if record is None:
                raise ValueError(f"{relative}: missing IR identity; refusing incomplete publication")
            blob = path.read_bytes()
            images = [image for _, image in inspector.cuda_images(blob)]
            if not images:
                raise ValueError(f"{relative}: missing CUDA image")
            header = path.with_suffix(".h")
            abi = digest(header.read_bytes()) if header.exists() else None
            resources = [{"cubin_sha256": digest(image), "kernels": elf_resources(image)} for image in images]
            key = {"ir_sha256": record["ir_sha256"], "target": context["capability"],
                   "toolchain": context["toolchain"], "environment": context["environment"],
                   "abi_sha256": abi, "build_kind": context["build_kind"],
                   "job_global_inputs": job_global_inputs(context)}
            records[relative] = {**record, "object_sha256": digest(blob),
                                 "cubin_sha256": [digest(image) for image in images],
                                 "resources": resources, "abi_sha256": abi,
                                 "compile_key": stable_hash(key), "compile_inputs": key,
                                 "plan": plans.get(path.stem)}
        (self.output / "program-records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        return records


def change_report(previous, current, expected=None):
    changes = {"schema": 1, "added": [], "removed": [], "changed": [], "leaks": [],
               "sm170_mismatches": sorted(name for name, record in current.items()
                                         if "sm170_object_sha256" in record
                                         and record["sm170_object_sha256"] != record.get("object_sha256"))}
    for name in sorted(previous.keys() | current.keys()):
        old, new = previous.get(name), current.get(name)
        group = (new or old).get("group", name.split("/", 1)[0])
        if old is None:
            category, entry = "added", {"program": name, "group": group}
        elif new is None:
            category, entry = "removed", {"program": name, "group": group}
        else:
            metadata = [record.get("planner_metadata") for record in (old, new)]
            routes = [value.get("route_preparation") if isinstance(value, dict) else None for value in metadata]
            if all(isinstance(route, dict) and route.get("content_sha256") for route in routes):
                metadata = [{**value, "route_preparation": {key: item for key, item in route.items()
                             if key != "sha256"}} for value, route in zip(metadata, routes)]
            metadata_changed = metadata[0] != metadata[1]
            fields = ("object_sha256", "cubin_sha256", "ir_sha256", "resources", "scratch_bytes_at_capacity",
                      "abi_sha256", "plan")
            if not metadata_changed and all(old.get(field) == new.get(field) for field in fields):
                continue
            reads = old.get("source_reads_sha256", {})
            new_reads = new.get("source_reads_sha256", {})
            entry = {"program": name, "group": group,
                     "object_changed": old.get("object_sha256") != new.get("object_sha256"),
                     "ir_changed": old.get("ir_sha256") != new.get("ir_sha256"),
                     "abi_changed": old.get("abi_sha256") != new.get("abi_sha256"),
                     "plan_changed": old.get("plan") != new.get("plan"),
                     "planner_metadata_changed": metadata_changed,
                     "triggering_sources": sorted(path for path in reads.keys() | new_reads.keys()
                                                  if reads.get(path) != new_reads.get(path)),
                     "resource_deltas": numeric_deltas(old.get("resources"), new.get("resources")),
                     "scratch_deltas": numeric_deltas(old.get("scratch_bytes_at_capacity"),
                                                       new.get("scratch_bytes_at_capacity"))}
            category = "changed"
        changes[category].append(entry)
        if expected is not None and group not in expected:
            changes["leaks"].append(name)
    return changes


def numeric_deltas(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        return {key: value for field in sorted(old.keys() | new.keys())
                for key, value in numeric_deltas(old.get(field), new.get(field),
                                                 f"{prefix}.{field}" if prefix else field).items()}
    if isinstance(old, list) and isinstance(new, list):
        return {key: value for index in range(max(len(old), len(new)))
                for key, value in numeric_deltas(old[index] if index < len(old) else None,
                                                 new[index] if index < len(new) else None,
                                                 f"{prefix}[{index}]").items()}
    if type(old) in (int, float) and type(new) in (int, float) and old != new:
        return {prefix: {"before": old, "after": new, "delta": new - old}}
    if old != new and (old is None or new is None):
        return {prefix: {"before": old, "after": new, "delta": None}}
    return {}


def write_report(destination, report):
    (destination / "CHANGES.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Kernel Changes", "", "Program groups are the publication's export job names.",
             "Source observations are cumulative within each worker; Sources lists show only changed source hashes.", ""]
    if report.get("sm170_mismatches"):
        lines += ["## HARD FAILURE: SM 170 / 188 OBJECT MISMATCH", "",
                  *[f"- `{name}`" for name in report["sm170_mismatches"]], ""]
    if report["leaks"]:
        lines += ["## UNEXPECTED CHANGES OUTSIDE DECLARED TARGETS", "", *[f"- `{name}`" for name in report["leaks"]], ""]
    for category in ("added", "removed", "changed"):
        lines += [f"## {category.title()} ({len(report[category])})", ""]
        for entry in report[category]:
            lines += [f"- `{entry['program']}`"]
            if category == "changed":
                lines += [f"  Object changed: {entry['object_changed']}; IR changed: {entry['ir_changed']}; "
                          f"ABI changed: {entry['abi_changed']}; plan changed: {entry['plan_changed']}; "
                          f"planner metadata changed: {entry['planner_metadata_changed']}.",
                          f"  Sources: {', '.join(entry['triggering_sources']) or '(no recorded source delta)' }.",
                          f"  Resources: {json.dumps(entry['resource_deltas'], sort_keys=True)}.",
                          f"  Scratch: {json.dumps(entry['scratch_deltas'], sort_keys=True)}."]
        lines.append("")
    (destination / "CHANGES.md").write_text("\n".join(lines))
