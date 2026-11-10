"""Content-addressed reuse of AOT export products, keyed by what the export read.

Two layers share one store (``CUTEAFD_EXPORT_CACHE``):

* invocation: one CMake export command (scripts/build/export-cache.py run). A
  hit restores every file the command wrote, without starting the exporter.
* program: one exported object inside an exporter (export_b12x_dsv4_aot.py).
  A hit restores its object, header and launch metadata without running the
  program's frontend or backend.

A key is a static identity (the command line or program description with build
and source paths replaced by placeholders, the compiler image and device) plus
checks: the content of every file the export read, the names in every source
directory it listed or probed for imports, every stat it made of a source path,
and every identity environment variable it read. The recorder is an audit hook
installed in the export process and in every Python child (sitecustomize), so
lookups follow ccache direct mode: an identity holds candidate check sets and
an entry hits only when every check evaluates now as it did when recorded.

Recording fails closed. A read of a file outside the source, build, toolchain
and scratch roots, a child program that is not a known compiler, a Python child
that left no record, or a write into the source tree makes the export
uncacheable, so it is exported again next time rather than ever restored wrong.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import threading

SCHEMA = 2
READS_ENV = "CUTEAFD_EXPORT_CACHE_READS"
ROOT_ENV = "CUTEAFD_EXPORT_CACHE_SOURCE_ROOT"
BUILD_ENV = "CUTEAFD_EXPORT_CACHE_BUILD_DIR"
# Compilers and linkers whose inputs and outputs are named on their command
# line (scanned below); every other program makes the read set unknown.
COMPILERS = frozenset({"c++", "g++", "gcc", "cc", "clang", "clang++", "nvcc", "ptxas",
                       "fatbinary", "nvlink", "ld", "ar"})
# Programs that read no export input and write nothing the build consumes.
QUERIES = frozenset({"nvidia-smi", "git", "uname", "ldconfig", "nproc"})
# Where and how many exports run, never what they compile; and paths that name
# the verified source or disposable compiler storage rather than its content.
PLACEMENT_ENV = frozenset("""
CUTEAFD_EXPORT_HOST_CLASS CUTEAFD_EXPORT_HOST_CONTEXTS CUTEAFD_EXPORT_HOST_WORKERS
CUTEAFD_EXPORT_JOBS CUTEAFD_EXPORT_SLOT_DIR CUTEAFD_EXPORT_HOST_SLOT_DIR
CUTEAFD_EXPORT_GPU_SLOT_DIR CUTEAFD_EXPORT_GPU_BUDGET_MIB CUTEAFD_EXPORT_WORKER_GPU_MIB
CUTEAFD_EXPORT_WORKER_RAM_MIB CUTEAFD_EXPORT_CONTEXT_SPACING_S CUTEAFD_EXPORT_GPU
CUTEAFD_EXPORT_CACHE CUTEAFD_EXPORT_CACHE_READS CUTEAFD_EXPORT_CACHE_SOURCE_ROOT
CUTEAFD_EXPORT_CACHE_BUILD_DIR CUTEAFD_EXPORT_CACHE_STATS CUTEAFD_EXPORT_CACHE_REQUIRE_HIT CUTEAFD_EXPORT_CACHE_OUTPUTS CUTEAFD_EXPORT_CACHE_TRACE
CUTEAFD_EXPORT_IMAGE_ID CUTEAFD_SOURCE_ROOT CUTEAFD_SPARKINFER_SOURCE_DIR CUTEAFD_SPARKINFER_LOCK_FILE
B12X_COMPILE_DISK_CACHE B12X_COMPILE_MEMORY_CACHE B12X_COMPILE_CACHE_DIR B12X_LOG_CUTE_COMPILES
B12X_PRINT_COMPILE_PROGRESS TRITON_CACHE_DIR CUTE_DSL_CACHE_DIR
""".split())
RECORDED_ENV_PREFIXES = ("CUTEAFD_", "B12X_", "CUTE_", "CUTLASS_", "TRITON_", "NVCC_", "PTXAS_")
# Read in C by the toolchain, never through os.environ: part of every identity.
STATIC_ENV_PREFIXES = ("CUTE_DSL_", "CUTLASS_", "NVCC_", "PTXAS_", "TRITON_", "CUDA_")
STATIC_ENV_NEUTRAL = frozenset({"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_HOME", "CUDA_PATH",
                                "TRITON_CACHE_DIR", "CUTE_DSL_CACHE_DIR", "CUDA_MODULE_LOADING"})


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable(value) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def file_digest(path) -> str | None:
    try:
        with open(path, "rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError):
        return None


def stat_kind(path) -> str:
    try:
        mode = os.stat(path).st_mode
    except (FileNotFoundError, NotADirectoryError):
        return "absent"
    return "file" if stat.S_ISREG(mode) else "dir" if stat.S_ISDIR(mode) else "other"


def listing(path) -> str | None:
    try:
        # Metadata discovery lists sys.path entries, including interpreter debris
        # that differs between otherwise identical source checkouts.
        return stable(sorted(name for name in os.listdir(path)
                             if name != "__pycache__" and not name.endswith(".pyc")))
    except OSError:
        return None


def import_probe(directory, tail) -> str | None:
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    return stable(sorted(name for name in names if name == tail or name.startswith(tail + ".")))


def static_environment(environ=os.environ) -> dict:
    return {name: value for name, value in sorted(environ.items())
            if name.startswith(STATIC_ENV_PREFIXES) and name not in STATIC_ENV_NEUTRAL
            and name not in PLACEMENT_ENV}


def toolchain_identity() -> dict | None:
    """The compiler image fixes CUDA, CuTe DSL, Triton and torch; no image, no cache."""
    image = os.environ.get("CUTEAFD_EXPORT_IMAGE_ID", "")
    if not image.startswith("sha256:"):
        return None
    import platform
    return {"image": image, "machine": platform.machine(), "python": list(sys.version_info[:3]),
            "static_environment": static_environment()}


class Roots:
    """Canonical names for every location an export touches."""

    def __init__(self, source_root, build_dir=None):
        self.source = Path(source_root).resolve()
        sparkinfer = os.environ.get("CUTEAFD_SPARKINFER_SOURCE_DIR")
        self.sparkinfer = Path(sparkinfer).resolve() if sparkinfer else self.source / "third_party/sparkinfer"
        self.build = Path(build_dir).resolve() if build_dir else None
        self.outputs = tuple(os.path.realpath(path) for path in
                             json.loads(os.environ.get("CUTEAFD_EXPORT_CACHE_OUTPUTS", "[]")))
        prefixes = {sys.prefix, sys.base_prefix, sys.exec_prefix, "/usr", "/lib", "/lib64", "/opt",
                    "/etc", "/proc", "/sys", "/dev", "/run", "/bin", "/sbin"}
        self.toolchain = tuple(sorted({str(Path(prefix).resolve()) for prefix in prefixes}))
        # Tool-keyed caches and coordination files: their contents are keyed by
        # the tools that own them (Triton's cache by its own source and knobs).
        caches = {os.environ.get("TRITON_CACHE_DIR", ""), os.environ.get(READS_ENV, ""),
                  os.environ.get("CUTEAFD_EXPORT_CACHE", ""), os.environ.get("CUTEAFD_EXPORT_HOST_SLOT_DIR", ""),
                  os.environ.get("CUTEAFD_EXPORT_SLOT_DIR", ""), os.environ.get("CUTEAFD_EXPORT_GPU_SLOT_DIR", "")}
        self.scratch = tuple(sorted({str(Path(path).resolve()) for path in caches if path}))
        # Temporary directories: a file there is an input unless this export made it.
        temporary = {tempfile.gettempdir(), os.environ.get("TMPDIR", ""), "/tmp", "/var/tmp", "/dev/shm"}
        self.temporary = tuple(sorted({str(Path(path).resolve()) for path in temporary if path}))

    @staticmethod
    def _under(path: str, roots) -> bool:
        return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)

    def tool_cache(self, resolved: str) -> bool:
        """A tool-keyed cache, including one an exporter points Triton at after start (#1531)."""
        triton = os.environ.get("TRITON_CACHE_DIR")
        live = (os.path.realpath(triton),) if triton else ()
        return self._under(resolved, self.scratch) or self._under(resolved, live)

    def aliases(self):
        """Other mount points of the source tree (the dev shell's /workspace/cuteafd)."""
        if not hasattr(self, "_aliases"):
            self._aliases = []
            for candidate in ("/workspace/cuteafd",):
                try:
                    if os.path.samefile(candidate, self.source) and os.path.realpath(candidate) != str(self.source):
                        self._aliases.append(os.path.realpath(candidate))
                except OSError:
                    pass
        return self._aliases

    def name(self, path) -> str | None:
        """'src:<rel>' or 'build:<rel>'; None outside both (and for tool caches inside them)."""
        try:
            resolved = os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return None
        if self.tool_cache(resolved):
            return None  # a tool-keyed cache, even one placed in the build tree
        for alias in self.aliases():
            if resolved == alias or resolved.startswith(alias + "/"):
                resolved = str(self.source) + resolved[len(alias):]
                break
        for prefix, root in (("src:third_party/sparkinfer/", self.sparkinfer), ("src:", self.source),
                             ("build:", self.build)):
            if root is None:
                continue
            root = str(root)
            if resolved == root:
                return prefix.rstrip("/") if prefix.endswith("/") else prefix + "."
            if resolved.startswith(root + "/"):
                return prefix + resolved[len(root) + 1:]
        return None

    def classify(self, path) -> tuple[str, str | None]:
        """('src'|'build', name), ('toolchain'|'scratch', None) or ('foreign', resolved)."""
        name = self.name(path)
        if name is not None:
            return name.partition(":")[0], name
        resolved = os.path.realpath(os.fsdecode(path))
        if "/__pycache__/" in resolved:
            return "scratch", None
        if self.tool_cache(resolved):
            return "scratch", None
        if self._under(resolved, self.temporary):
            return "temp", resolved
        if self._under(resolved, self.toolchain) or "/site-packages/" in resolved or "/dist-packages/" in resolved:
            return "toolchain", None
        return "foreign", resolved

    def path(self, name: str) -> Path:
        kind, _, relative = name.partition(":")
        if relative == ".":
            relative = ""
        if kind == "src":
            if relative == "third_party/sparkinfer" or relative.startswith("third_party/sparkinfer/"):
                return self.sparkinfer / relative.removeprefix("third_party/sparkinfer").lstrip("/")
            return self.source / relative
        if kind == "build" and self.build is not None:
            return self.build / relative
        raise ValueError(f"unknown recorded location {name}")


def is_path(value) -> bool:
    return isinstance(value, (str, bytes, os.PathLike))


def user_magic_present():
    return any(os.path.lexists(os.path.expanduser(name)) for name in ("~/.magic", "~/.magic.mgc"))


def evaluate(roots: Roots, check: str):
    """The current value of one recorded check."""
    kind, _, rest = check.partition("|")
    if kind == "file":
        return file_digest(roots.path(rest))
    if kind == "stat":
        return stat_kind(roots.path(rest))
    if kind == "list":
        return listing(roots.path(rest))
    if kind == "import":
        directory, _, tail = rest.rpartition("|")
        return import_probe(roots.path(directory), tail)
    if kind == "env":
        return os.environ.get(rest)
    if kind == "file-user-magic":
        return user_magic_present()
    raise ValueError(f"unknown check {check}")


class Scope:
    """The observations made while one program compiled (see Recorder.scope)."""

    def __init__(self):
        self.checks: set[str] = set()
        self.unknown: set[str] = set()
        self.written: set[str] = set()


class _Unknown(set):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def add(self, item):
        super().add(item)
        for scope in self.recorder.scopes:
            scope.unknown.add(item)


class Recorder:
    """Every input observation of this process, plus what it wrote and spawned.

    Values are taken at first observation, so a file the export overwrites
    later is keyed by the content it found. Reads of files this process tree
    wrote earlier are intermediates, not inputs.
    """

    def __init__(self, roots: Roots):
        self.roots = roots
        self.checks: dict[str, object] = {}
        self.written: set[str] = set()
        self.deleted: set[str] = set()
        self.scopes: list[Scope] = []
        self.unknown: set[str] = _Unknown(self)
        self.env_written: set[str] = set()
        self.made_dirs: set[str] = set()
        self.temp_read: set[str] = set()
        self.temp_written: set[str] = set()
        self.lock = threading.RLock()
        self.local = threading.local()
        self.active = False
        self.paused = 0

    # Observations -----------------------------------------------------------------
    def _observe(self, check, value_fn):
        with self.lock:
            if check not in self.checks:
                self.checks[check] = value_fn()
            for scope in self.scopes:
                scope.checks.add(check)

    def scope(self, closing=None):
        """Context collecting every observation (repeated or first) until it closes.

        `closing(scope)` runs before it closes (ProgramCache adds the sources of
        the code the program executed); by default every loaded module counts.
        """
        recorder = self

        class _Context:
            def __enter__(self):
                self.value = Scope()
                with recorder.lock:
                    recorder.scopes.append(self.value)
                return self.value

            def __exit__(self, *exc):
                recorder.guarded(closing or (lambda scope: recorder.modules()), self.value)
                with recorder.lock:
                    recorder.scopes.remove(self.value)
        return _Context()

    def recording(self) -> bool:
        return self.active and not self.paused and not getattr(self.local, "busy", False)

    def guarded(self, method, *args) -> None:
        """Run one observation with recording off for its own filesystem calls."""
        self.local.busy = True
        try:
            method(*args)
        except Exception as error:
            self.unknown.add(f"recorder-error:{method.__name__}:{type(error).__name__}")
        finally:
            self.local.busy = False

    def note_read(self, path) -> None:
        kind, name = self.roots.classify(path)
        if kind == "foreign":
            self.unknown.add("read:" + name)
            return
        if kind == "temp":
            if not self._temp_made_here(name) and os.path.isfile(name):
                with self.lock:
                    self.temp_read.add(name)
                    for scope in self.scopes:  # a program cannot wait for the merge
                        scope.unknown.add("read-temporary:" + name)
            return
        if name is None:
            return
        resolved = os.path.realpath(os.fsdecode(path))
        if resolved in self.written or not os.path.isfile(resolved):
            return
        self._observe("file|" + name, lambda: file_digest(resolved))

    def _temp_made_here(self, path: str) -> bool:
        """A temporary file this process wrote, or one inside a directory it created
        (a compiler writing its object into our mkdtemp directory from C)."""
        if path in self.temp_written:
            return True
        parent = os.path.dirname(path)
        while parent and parent not in self.roots.temporary and parent != "/":
            if parent in self.temp_written:
                return True
            parent = os.path.dirname(parent)
        return False

    def note_write(self, path) -> None:
        kind, name = self.roots.classify(path)
        if kind == "src":
            self.unknown.add("write-source:" + name)
            return
        if kind == "temp":
            with self.lock:
                self.temp_written.add(name)
            return
        if kind == "build":
            resolved = os.path.realpath(os.fsdecode(path))
            with self.lock:
                self.written.add(resolved)
                for scope in self.scopes:
                    scope.written.add(resolved)

    def note_move(self, source, destination) -> None:
        """A staged file or directory renamed into place: its writes now live there."""
        old = os.path.realpath(os.fsdecode(source))
        new = os.path.realpath(os.fsdecode(destination)) if not os.path.isdir(os.fsdecode(destination)) \
            or os.path.isdir(old) else os.path.join(os.path.realpath(os.fsdecode(destination)), os.path.basename(old))
        with self.lock:
            moved = {path for path in self.written if path == old or path.startswith(old + "/")}
            for path in moved:
                self.written.discard(path)
                target = new + path[len(old):]
                self.written.add(target)
                for scope in self.scopes:
                    scope.written.discard(path)
                    scope.written.add(target)
        self.note_write(destination)

    def note_delete(self, path) -> None:
        kind, name = self.roots.classify(path)
        if kind == "src":
            self.unknown.add("delete-source:" + name)
        elif kind == "build":
            resolved = os.path.realpath(os.fsdecode(path))
            with self.lock:
                self.written.discard(resolved)
                self.deleted.add(resolved)

    def owned(self, resolved: str) -> bool:
        """Under an output directory the command names: it owns those trees.

        Exporters remove and recreate what they produce there, so which names
        exist beforehand is not an input (a file read there still is).
        """
        if any(resolved == root or resolved.startswith(root + "/") for root in self.roots.outputs):
            return True
        # ...or under a directory this process created (a staging directory).
        parent = resolved
        while parent and parent != "/":
            if parent in self.made_dirs:
                return True
            parent = os.path.dirname(parent)
        return False

    def note_listing(self, path) -> None:
        kind, name = self.roots.classify(path)
        if kind == "foreign":
            self.unknown.add("list:" + name)
        elif name is not None:
            resolved = os.path.realpath(os.fsdecode(path))
            if not self.owned(resolved):
                self._observe("list|" + name, lambda: listing(resolved))

    def note_stat(self, path, kind_value) -> None:
        kind, name = self.roots.classify(path)
        if name is None or kind not in ("src", "build"):
            return
        resolved = os.path.realpath(os.fsdecode(path))
        if resolved in self.written or self.owned(resolved):
            return
        self._observe("stat|" + name, lambda: kind_value)

    def note_import(self, directory, tail) -> None:
        kind, name = self.roots.classify(directory)
        if kind == "foreign":
            self.unknown.add("import-dir:" + name)
        elif name is not None:
            self._observe("import|" + name + "|" + tail, lambda: import_probe(directory, tail))

    def note_env(self, name, value) -> None:
        if name in self.env_written:
            return  # this process set it: an effect of the export, not an input
        if name.startswith(RECORDED_ENV_PREFIXES) and name not in PLACEMENT_ENV and not name.startswith("CUTEAFD_WIP_"):
            self._observe("env|" + name, lambda: value)

    def note_child(self, executable, argv, environment=None) -> None:
        argv = [os.fsdecode(item) for item in (argv or [])]
        program = os.path.basename(os.fsdecode(executable)) if executable else (os.path.basename(argv[0]) if argv else "")
        if program.startswith("python"):
            if any(flag in ("-I", "-E", "-s", "-S") for flag in argv[1:3]):
                self.unknown.add("python-child-without-site")
            if environment is not None and environment.get(READS_ENV) != os.environ.get(READS_ENV):
                self.unknown.add("python-child-without-recorder")
            return
        if program == "file":
            # Only plain read-only probes; alternate magic databases and indirect
            # file lists need their own input scanner.
            self.note_read(shutil.which(os.fsdecode(executable)) or os.fsdecode(executable))
            for name in ("MAGIC", "HOME", "POSIXLY_CORRECT", "LC_ALL", "LC_CTYPE", "LANG"):
                value = os.environ.get(name)
                if environment is not None and environment.get(name) != value:
                    self.unknown.add("file-environment:" + name)
                    return
                self._observe("env|" + name, lambda value=value: value)
            # Only the image's default database is known; user databases are
            # refused, including one created between this run and a lookup.
            self._observe("file-user-magic|", user_magic_present)
            if os.environ.get("MAGIC") or user_magic_present():
                self.unknown.add("file-user-magic")
                return
            positional = False
            for path in argv[1:]:
                if path == "--" and not positional:
                    positional = True
                elif path.startswith("-") and not positional:
                    if path not in ("-b", "--brief", "-L", "--dereference"):
                        self.unknown.add("file-option:" + path)
                        return
                else:
                    self.note_read(path)
                    self.note_stat(path, stat_kind(path))
            return
        if program in QUERIES:
            return
        if program not in COMPILERS:
            self.unknown.add("spawn:" + program)
            return
        self._scan_compiler(program, argv[1:])

    def _scan_compiler(self, program, argv) -> None:
        index = 0
        positional = []
        while index < len(argv):
            token = argv[index]
            for flag in ("-o", "-I", "-L", "-isystem", "-include", "-MF", "-iquote"):
                if token == flag and index + 1 < len(argv):
                    value = argv[index + 1]
                    index += 1
                    break
                if token.startswith(flag) and len(token) > len(flag) and flag in ("-o", "-I", "-L"):
                    value = token[len(flag):]
                    break
            else:
                flag, value = None, None
            if flag == "-o" or flag == "-MF":
                self.note_write(value)
            elif flag is not None:
                kind, name = self.roots.classify(value)
                resolved = os.path.realpath(value)
                generated = any(item.startswith(resolved + "/") for item in self.written)
                if kind == "src" or kind == "foreign" or (kind == "build" and not generated):
                    self.unknown.add(f"{program}-search-path:{name or value}")
            elif not token.startswith("-"):
                positional.append(token)
            index += 1
        if program == "ar" and positional:
            # ar OPERATIONS ARCHIVE MEMBER...
            positional = positional[1:]
            if positional:
                self.note_write(positional[0])
                positional = positional[1:]
        for token in positional:
            if os.path.isfile(token):
                self.note_read(token)

    # Hooks ------------------------------------------------------------------------
    def hook(self, event, args):
        if not self.recording():
            return
        self.local.busy = True
        try:
            self._dispatch(event, args)
        except Exception as error:  # never disturb the export; an unexplained event is unknown
            self.unknown.add(f"recorder-error:{event}:{type(error).__name__}")
        finally:
            self.local.busy = False

    def _dispatch(self, event, args):
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            if not isinstance(path, (str, bytes, os.PathLike)):
                return
            writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            reads = not isinstance(mode, str) or "r" in mode or "+" in mode
            if reads and not (isinstance(mode, str) and any(c in mode for c in "wax")):
                self.note_read(path)
            if writes:
                self.note_write(path)
        elif event in ("os.listdir", "os.scandir"):
            target = args[0] if args and args[0] is not None else "."
            if is_path(target) and not getattr(self.local, "importing", 0):
                self.note_listing(target)
        elif event in ("os.rename", "os.replace"):
            if len(args) >= 2 and is_path(args[0]) and is_path(args[1]) and not self._fd_relative(event, args):
                self.note_move(args[0], args[1])
        elif event in ("shutil.copyfile", "shutil.copytree", "os.link", "os.symlink"):
            if len(args) >= 2 and is_path(args[1]) and not self._fd_relative(event, args):
                self.note_write(args[1])
        elif event in ("os.remove", "os.rmdir", "shutil.rmtree"):
            if args and is_path(args[0]) and not self._fd_relative(event, args):
                self.note_delete(args[0])
        elif event == "os.mkdir":
            if args and is_path(args[0]) and not self._fd_relative(event, args):
                target = os.path.realpath(os.fsdecode(args[0]))
                if not os.path.exists(target):  # the audit event precedes the call
                    with self.lock:
                        self.made_dirs.add(target)
                self.note_write(args[0])
        elif event == "subprocess.Popen":
            executable, argv, _, environment = (list(args) + [None] * 4)[:4]
            self.note_child(executable, argv, environment)
        elif event == "os.posix_spawn":
            self.note_child(args[0], args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else None)
        elif event in ("os.exec", "os.system", "os.fork_exec", "ctypes.dlopen_source"):
            self.unknown.add(event)

    @staticmethod
    def _fd_relative(event, args) -> bool:
        """A name relative to a directory descriptor (shutil.rmtree's fd walk)."""
        fds = [item for item in args[1:] if isinstance(item, int) and not isinstance(item, bool) and item != -1]
        return bool(fds) and not os.path.isabs(os.fsdecode(args[0]))

    def install(self):
        global ACTIVE
        ACTIVE = self
        sys.addaudithook(self.hook)
        self._patch_environment()
        self._patch_stat()
        self._patch_imports()
        self.active = True
        return self

    def _patch_environment(self):
        recorder = self
        environ_type = type(os.environ)
        original_getitem = environ_type.__getitem__
        original_contains = environ_type.__contains__

        def getitem(self, key):
            try:
                value = original_getitem(self, key)
            except KeyError:
                if recorder.recording() and isinstance(key, str):
                    recorder.guarded(recorder.note_env, key, None)
                raise
            if recorder.recording() and isinstance(key, str):
                recorder.guarded(recorder.note_env, key, value)
            return value

        def value_of(key):
            try:
                return original_getitem(os.environ, key)
            except KeyError:
                return None

        def contains(self, key):
            result = original_contains(self, key)
            if recorder.recording() and isinstance(key, str):
                recorder.guarded(recorder.note_env, key, value_of(key) if result else None)
            return result
        original_setitem = environ_type.__setitem__
        original_delitem = environ_type.__delitem__

        def setitem(self, key, value):
            if isinstance(key, str):
                recorder.env_written.add(key)
            return original_setitem(self, key, value)

        def delitem(self, key):
            if isinstance(key, str):
                recorder.env_written.add(key)
            return original_delitem(self, key)
        environ_type.__getitem__ = getitem
        environ_type.__contains__ = contains
        environ_type.__setitem__ = setitem
        environ_type.__delitem__ = delitem

    def _patch_stat(self):
        recorder = self
        for function in ("stat", "lstat"):
            original = getattr(os, function)

            def patched(path, *args, _original=original, **kwargs):
                try:
                    result = _original(path, *args, **kwargs)
                except (FileNotFoundError, NotADirectoryError):
                    if recorder.recording() and isinstance(path, (str, bytes, os.PathLike)):
                        recorder.guarded(recorder.note_stat, path, "absent")
                    raise
                if recorder.recording() and isinstance(path, (str, bytes, os.PathLike)):
                    mode = result.st_mode
                    recorder.guarded(recorder.note_stat, path, "file" if stat.S_ISREG(mode) else "dir" if stat.S_ISDIR(mode) else "other")
                return result
            setattr(os, function, patched)

    def _patch_imports(self):
        from importlib.machinery import FileFinder
        recorder = self
        original = FileFinder.find_spec

        def find_spec(self, fullname, target=None):
            local = recorder.local
            local.importing = getattr(local, "importing", 0) + 1
            try:
                spec = original(self, fullname, target)
            finally:
                local.importing -= 1
            if recorder.recording():
                recorder.guarded(recorder.note_import, self.path, fullname.rpartition(".")[2])
            return spec
        FileFinder.find_spec = find_spec

    # Results ----------------------------------------------------------------------
    def modules(self):
        for module in list(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if isinstance(filename, str):
                self.note_read(filename)

    def snapshot(self) -> dict:
        was = self.active
        self.active = False
        try:
            self.paused += 0
            self.active = True
            self.local.busy = False
            self.modules()
        finally:
            self.active = was
        with self.lock:
            return {"checks": dict(self.checks), "written": sorted(self.written),
                    "deleted": sorted(self.deleted), "unknown": sorted(set(self.unknown)),
                    "temp_read": sorted(self.temp_read), "temp_written": sorted(self.temp_written)}


ACTIVE: Recorder | None = None


class paused:
    """Suspend recording around work that verifies, rather than consumes, its inputs."""

    def __enter__(self):
        if ACTIVE is not None:
            ACTIVE.paused += 1

    def __exit__(self, *exc):
        if ACTIVE is not None:
            ACTIVE.paused -= 1


def note_read(path) -> None:
    """Record an input explicitly (one read while recording was paused)."""
    if ACTIVE is not None and ACTIVE.active:
        ACTIVE.note_read(path)


# Functions that read sources only to key b12x's own (disabled) compile caches:
# their reads say nothing about what an export compiles.
BOOKKEEPING = {"b12x._lib.compiler": ("_compute_b12x_package_fingerprint",)}


def install_bookkeeping_pauses() -> None:
    import importlib.abc
    import importlib.util

    def wrap(module):
        for name in BOOKKEEPING.get(module.__name__, ()):
            original = getattr(module, name, None)
            if original is None or getattr(original, "_export_cache_paused", False):
                continue

            def paused_call(*args, _original=original, **kwargs):
                with paused():
                    return _original(*args, **kwargs)
            paused_call._export_cache_paused = True
            setattr(module, name, paused_call)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in BOOKKEEPING:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            loader = spec.loader
            original_exec = loader.exec_module

            def exec_module(module, _exec=original_exec):
                _exec(module)
                wrap(module)
            loader.exec_module = exec_module
            return spec
    for name in BOOKKEEPING:
        if name in sys.modules:
            wrap(sys.modules[name])
    sys.meta_path.insert(0, Finder())


# caller file -> files of modules it imported through importlib.import_module
DYNAMIC_IMPORTS: dict[str, set[str]] = {}


def install_dynamic_import_record() -> None:
    import importlib
    original = importlib.import_module
    if getattr(original, "_export_cache_recorded", False):
        return

    def import_module(name, package=None):
        module = original(name, package)
        caller = sys._getframe(1).f_code.co_filename
        filename = getattr(module, "__file__", None)
        if isinstance(filename, str):
            DYNAMIC_IMPORTS.setdefault(caller, set()).add(filename)
        return module
    import_module._export_cache_recorded = True
    importlib.import_module = import_module


def child_bootstrap() -> None:
    """sitecustomize: record this process's observations into READS_ENV/<pid>.json."""
    directory = os.environ.get(READS_ENV)
    root = os.environ.get(ROOT_ENV)
    if not directory or not root:
        return
    if "resource_tracker" in " ".join(getattr(sys, "orig_argv", [])):
        return  # multiprocessing's tracker only unlinks semaphores
    import atexit
    recorder = Recorder(Roots(Path(root), os.environ.get(BUILD_ENV) or None)).install()
    install_dynamic_import_record()
    pid = [os.getpid()]
    Path(directory, f"{pid[0]}.started").write_text("")

    def flush():
        recorder.active = False
        record = recorder.snapshot()
        temporary = Path(directory) / f".{os.getpid()}.tmp"
        temporary.write_text(json.dumps(record))
        os.replace(temporary, Path(directory) / f"{os.getpid()}.json")
    atexit.register(flush)

    def forked():
        # A forked child inherits the recorder; multiprocessing ends it with
        # os._exit, which skips atexit, so flush first.
        Path(directory, f"{os.getpid()}.started").write_text("")
        real_exit = os._exit

        def exit_after_flush(code):
            try:
                flush()
            finally:
                real_exit(code)
        os._exit = exit_after_flush
    os.register_at_fork(after_in_child=forked)


def merge_records(directory: Path) -> dict | None:
    """Union of every process's record; None when a started process left none or two disagree."""
    merged = {"checks": {}, "written": set(), "deleted": set(), "unknown": set()}
    temp_read, temp_written = set(), set()
    for started in directory.glob("*.started"):
        record = started.with_suffix(".json")
        if not record.is_file():
            return None
        data = json.loads(record.read_text())
        temp_read.update(data.get("temp_read", ()))
        temp_written.update(data.get("temp_written", ()))
        merged["written"].update(data["written"])
        merged["deleted"].update(data["deleted"])
        merged["unknown"].update(data["unknown"])
        for check, value in data["checks"].items():
            if check in merged["checks"] and merged["checks"][check] != value:
                merged["unknown"].add("conflicting-observation:" + check)
            merged["checks"].setdefault(check, value)
    # A temporary file is an intermediate only if some process of this export
    # wrote it; one that was already there is an input nothing recorded.
    def made(path):
        while path and path != "/":
            if path in temp_written:
                return True
            path = os.path.dirname(path)
        return False
    for path in sorted(temp_read):
        if not made(path):
            merged["unknown"].add("read-temporary:" + path)
    return {key: sorted(value) if isinstance(value, set) else value for key, value in merged.items()}


class Store:
    """Blobs by content; identities hold candidate check sets; entries describe outputs."""

    def __init__(self, root):
        self.root = Path(root)
        for sub in ("blobs", "identities", "entries"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def _blob(self, checksum: str) -> Path:
        return self.root / "blobs" / checksum[:2] / checksum

    def put_blob(self, source) -> str:
        checksum = file_digest(source)
        target = self._blob(checksum)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=target.parent)
            os.close(fd)
            shutil.copyfile(source, temporary)
            os.chmod(temporary, 0o444)
            os.replace(temporary, target)
        return checksum

    def get_blob(self, checksum: str, destination: Path, mode: int) -> bool:
        source = self._blob(checksum)
        if not source.is_file() or file_digest(source) != checksum:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.restore")
        shutil.copyfile(source, temporary)
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
        return True

    def _write_json(self, path: Path, value) -> None:
        # Insertion order kept: restored metadata (geometry dicts) re-enters
        # manifests, which must come out byte-identical to a fresh export.
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps(value))
        os.replace(temporary, path)

    def lookup(self, identity_key: str, roots: Roots):
        """(entry, checks) of the newest candidate whose checks all hold now, else None."""
        index = self.root / "identities" / identity_key[:2] / identity_key
        if not index.is_dir():
            return None
        candidates = sorted(index.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        values: dict[str, object] = {}
        for candidate in candidates:
            try:
                recorded = json.loads(candidate.read_text())
            except (OSError, ValueError):
                continue
            for check, expected in recorded["checks"].items():
                if check not in values:
                    try:
                        values[check] = evaluate(roots, check)
                    except ValueError:
                        values[check] = object()
                if values[check] != expected:
                    break
            else:
                entry = self.root / "entries" / recorded["entry"][:2] / (recorded["entry"] + ".json")
                try:
                    value = json.loads(entry.read_text())
                except (OSError, ValueError):
                    continue
                os.utime(candidate)
                return value, recorded["checks"]
        return None

    def record(self, identity_key: str, checks: dict, entry: dict) -> str:
        entry_key = stable({"identity": identity_key, "checks": checks})
        self._write_json(self.root / "entries" / entry_key[:2] / (entry_key + ".json"), entry)
        self._write_json(self.root / "identities" / identity_key[:2] / identity_key / (entry_key + ".json"),
                         {"checks": checks, "entry": entry_key})
        return entry_key


def still_valid(roots: Roots, checks: dict) -> bool:
    """Source observations must not have changed while the export ran."""
    for check, expected in checks.items():
        if check.startswith(("file|src:", "list|src:", "import|src:", "stat|src:")):
            if evaluate(roots, check) != expected:
                return False
    return True


def store_from_environment() -> Store | None:
    root = os.environ.get("CUTEAFD_EXPORT_CACHE", "")
    if not root or root == "off":
        return None
    return Store(Path(root))


def stats(event: str, **fields) -> None:
    path = os.environ.get("CUTEAFD_EXPORT_CACHE_STATS")
    print(f"export cache {event}: " + " ".join(f"{key}={value}" for key, value in fields.items()), flush=True)
    if path:
        with open(path, "a") as handle:
            handle.write(json.dumps({"event": event, **fields}) + "\n")


FORK_MODULE = "file|src:third_party/sparkinfer/b12x/"


def is_fork_module(check: str) -> bool:
    return check.startswith(FORK_MODULE) and check.endswith(".py")


class Executed:
    """The fork sources one program's compilation depended on.

    Every file whose code ran while it compiled (sys.monitoring PY_START,
    re-armed per program), each file's static import closure (module-level and
    function-local imports: a constant used by running code was imported from
    somewhere), and every module first imported meanwhile. A running module that
    imports by computed name (import_module, __import__, a package __getattr__)
    pulls in every loaded module of its package, so a lazily resolved constant
    is still keyed.
    """

    TOOL_NAME = "cuteafd-export-cache"

    def __init__(self, roots: Roots):
        self.roots = roots
        self.fork = str(roots.sparkinfer / "b12x")
        self.tool = None
        for tool in (4, 3):
            if sys.monitoring.get_tool(tool) is None:
                sys.monitoring.use_tool_id(tool, self.TOOL_NAME)
                self.tool = tool
                break
        self.files: set[str] = set()
        self.last_files: set[str] = set()
        self.closures: dict[str, frozenset] = {}
        self.unparsed: set[str] = set()
        self.loaded: set[str] = set()

    @property
    def available(self) -> bool:
        return self.tool is not None

    def _start(self, code, offset):
        filename = code.co_filename
        if filename.startswith(self.fork):
            self.files.add(filename)
        return sys.monitoring.DISABLE

    def begin(self) -> None:
        self.files = set()
        self.loaded = {name for name, module in sys.modules.items() if getattr(module, "__file__", None)}
        if self.tool is not None:
            sys.monitoring.register_callback(self.tool, sys.monitoring.events.PY_START, self._start)
            sys.monitoring.set_events(self.tool, sys.monitoring.events.PY_START)
            sys.monitoring.restart_events()

    def end(self) -> set[str]:
        if self.tool is None:
            # No monitoring slot: every loaded fork module counts (coarse, sound).
            return {module.__file__ for module in list(sys.modules.values())
                    if isinstance(getattr(module, "__file__", None), str) and module.__file__.startswith(self.fork)}
        sys.monitoring.set_events(self.tool, 0)
        result = set(self.files)
        self.last_files = set(self.files)
        for name, module in list(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if name not in self.loaded and isinstance(filename, str) and filename.startswith(self.fork):
                result.add(filename)
        pending = list(result)
        while pending:
            path = pending.pop()
            for dependency in self.closure(path):
                if dependency not in result:
                    result.add(dependency)
                    pending.append(dependency)
        return result

    def closure(self, path: str) -> frozenset:
        # The parsed imports never change; dynamic imports keep accruing.
        if path not in self.closures:
            self.closures[path] = frozenset(self._imports(path))
        result = self.closures[path] | frozenset(DYNAMIC_IMPORTS.get(path, ()))
        if path in self.unparsed:
            result |= frozenset(self._package_modules(path))  # loaded by now, not when parsed
        return result

    def _module_file(self, name: str):
        parts = name.split(".")
        if not parts or parts[0] != "b12x":
            return None
        base = Path(self.fork).parent.joinpath(*parts)
        for candidate in (base / "__init__.py", base.with_suffix(".py")):
            if candidate.is_file():
                return str(candidate)
        return None

    def _imports(self, path: str):
        import ast
        try:
            text = Path(path).read_text()
            tree = ast.parse(text)
        except (OSError, SyntaxError, ValueError):
            self.unparsed.add(path)  # unreadable: everything loaded near it counts
            return set()
        relative = Path(path).relative_to(Path(self.fork).parent).with_suffix("")
        parts = list(relative.parts)
        package = parts[:-1] if parts[-1] != "__init__" else parts[:-1]
        if parts[-1] == "__init__":
            package = parts[:-1]
        found = set()

        def add(name):
            pieces = name.split(".")
            for index in range(1, len(pieces) + 1):
                file = self._module_file(".".join(pieces[:index]))
                if file:
                    found.add(file)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[:len(package) - (node.level - 1)] if node.level > 1 else package
                    module = ".".join(base + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                add(module)
                for alias in node.names:
                    add(module + "." + alias.name)
        # Imports by computed name: every module this file's code ever imported
        # through importlib.import_module (prelude or program; recorded by the
        # wrapper installed at bootstrap). Lazy package __getattr__ resolves
        # through it. Other dynamic loaders key the file's whole loaded package.
        if "__import__(" in text or "spec_from_file_location" in text or "exec(" in text:
            self.unparsed.add(path)
        return found

    def _package_modules(self, path: str):
        directory = os.path.dirname(path)
        return {module.__file__ for module in list(sys.modules.values())
                if isinstance(getattr(module, "__file__", None), str)
                and module.__file__.startswith(directory + "/")}


# A fork pin bump rewrites the lock, which manifests embed, so whole-command
# entries key it. An exported program does not embed it: the fork files it
# compiled from are keyed by content (and lock verification proves the tree is
# the locked one), so one kernel edit re-exports only the programs that read it.
PROGRAM_NEUTRAL_CHECKS = frozenset({"file|src:third_party/sparkinfer.lock.json"})


class ProgramCache:
    """Per-program reuse inside one recorded export (export_b12x_dsv4_aot.py).

    A program's key is its identity (what the exporter passes: names, row
    capacities, geometry and the arguments its recipe can read) plus every
    observation the process had made before the program loop (the exporter and
    the modules it imported) and every observation made while the program
    compiled, including the source of every module loaded by then. Available
    only under a live recorder: without one nothing is reused or stored.
    """

    def __init__(self, output_dir):
        self.recorder = ACTIVE if ACTIVE is not None and ACTIVE.active else None
        self.store = store_from_environment() if self.recorder else None
        self.toolchain = toolchain_identity() if self.store else None
        self.enabled = bool(self.recorder and self.store and self.toolchain)
        self.output = Path(output_dir).resolve()
        self.prelude: set[str] = set()
        self.hits = self.misses = self.uncacheable = 0
        self.unidentified: list[str] = []

    def start(self) -> None:
        if self.enabled:
            self.recorder.modules()
            with self.recorder.lock:
                # Fork modules key a program only through what that program ran
                # (Executed below); everything else the exporter read keys all.
                self.prelude = {check for check in self.recorder.checks if not is_fork_module(check)}
                if set(self.recorder.unknown):
                    self.enabled = False  # the prelude itself escaped the record
            self.executed = Executed(self.recorder.roots)

    def key(self, identity: dict) -> str:
        return stable({"schema": SCHEMA, "layer": "program", "toolchain": self.toolchain, **identity})

    def lookup(self, identity: dict):
        """Restore a hit's files into the output directory; return its metadata, else None."""
        if not self.enabled:
            return None
        key = self.key(identity)
        with paused():
            found = self.store.lookup(key, self.recorder.roots)
            if found is None:
                return None
            entry, checks = found
            for name, item in entry["files"].items():
                if not self.store.get_blob(item["blob"], self.output / name, item["mode"]):
                    return None
        # Restored files are this export's outputs exactly as if it had written them.
        for name in entry["files"]:
            self.recorder.guarded(self.recorder.note_write, self.output / name)
        # Keep the reused program's inputs in the invocation's record.
        with self.recorder.lock:
            for check, value in checks.items():
                self.recorder.checks.setdefault(check, value)
        self.hits += 1
        return entry["metadata"]

    def scope(self):
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext(None)
        self.executed.begin()

        def closing(scope):
            for path in self.executed.end():
                self.recorder.note_read(path)
        return self.recorder.scope(closing)

    def store_program(self, identity: dict, scope, files, metadata) -> None:
        if not self.enabled or scope is None:
            return
        trace = os.environ.get("CUTEAFD_EXPORT_CACHE_TRACE")
        if trace:  # diagnostics: which fork sources key each program
            with open(trace, "a") as handle:
                handle.write(json.dumps({"label": identity.get("label"), "executed": sorted(self.executed.last_files),
                                         "keyed": sorted(name for name in scope.checks if is_fork_module(name))}) + "\n")
        if scope.unknown:
            self.uncacheable += 1
            return
        with self.recorder.lock:
            names = (self.prelude | scope.checks) - PROGRAM_NEUTRAL_CHECKS
            checks = {name: self.recorder.checks[name] for name in names if name in self.recorder.checks}
        stored = {}
        with paused():
            for name in files:
                path = self.output / name
                stored[name] = {"blob": self.store.put_blob(path), "mode": path.stat().st_mode & 0o777}
            self.store.record(self.key(identity), checks, {"files": stored, "metadata": metadata})
        self.misses += 1

    def report(self) -> None:
        if self.enabled:
            stats("programs", layer="program", hits=self.hits, misses=self.misses, uncacheable=self.uncacheable,
                  unidentified=len(self.unidentified),
                  **({"example": self.unidentified[0].replace(" ", "_")} if self.unidentified else {}))


class Unidentifiable(Exception):
    """A recipe captured a value the cache cannot name; that program is never reused."""


def recipe_identity(value, depth=0):
    """A stable description of what a program recipe (a closure) captured.

    Functions and classes are named, not hashed: their code is a recorded file
    read. Plain data, dataclasses, enums, dtypes and partials are described by
    value. Anything else (an object whose repr is an address) is unidentifiable.
    """
    import dataclasses
    import enum
    import functools
    import types
    if depth > 12:
        raise Unidentifiable("recipe nesting too deep")
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return [type(value).__name__, value if not isinstance(value, bytes) else value.hex()]
    if isinstance(value, (list, tuple, frozenset, set)):
        items = [recipe_identity(item, depth + 1) for item in value]
        return [type(value).__name__, sorted(items, key=stable) if isinstance(value, (set, frozenset)) else items]
    if isinstance(value, dict):
        return ["dict", sorted(([recipe_identity(k, depth + 1), recipe_identity(v, depth + 1)]
                                for k, v in value.items()), key=stable)]
    if isinstance(value, enum.Enum):
        return ["enum", type(value).__module__ + "." + type(value).__qualname__, value.name]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return ["dataclass", type(value).__module__ + "." + type(value).__qualname__,
                [[field.name, recipe_identity(getattr(value, field.name), depth + 1)]
                 for field in dataclasses.fields(value)]]
    if isinstance(value, functools.partial):
        return ["partial", recipe_identity(value.func, depth + 1), recipe_identity(value.args, depth + 1),
                recipe_identity(value.keywords, depth + 1)]
    if isinstance(value, types.MethodType):
        return ["method", recipe_identity(value.__func__, depth + 1), recipe_identity(value.__self__, depth + 1)]
    if isinstance(value, types.FunctionType):
        closure = [recipe_identity(cell.cell_contents, depth + 1) for cell in (value.__closure__ or ())]
        return ["function", value.__module__, value.__qualname__, closure,
                recipe_identity(value.__defaults__, depth + 1), recipe_identity(value.__kwdefaults__, depth + 1)]
    if isinstance(value, (type, types.ModuleType, types.BuiltinFunctionType)):
        return ["named", getattr(value, "__module__", None) or "", getattr(value, "__qualname__", None) or value.__name__]
    text = repr(value)
    if " at 0x" in text or text.startswith("<"):
        # torch dtypes and similar print by name; anything address-like is not a value.
        module = type(value).__module__
        if module.startswith("torch") and text.startswith("torch."):
            return ["repr", text]
        raise Unidentifiable(f"recipe captured an unnamed {type(value).__qualname__}")
    return ["repr", type(value).__module__ + "." + type(value).__qualname__, text]
