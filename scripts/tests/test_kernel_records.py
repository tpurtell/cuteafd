"""CPU-only kernel identity and change reporting regressions."""
import importlib.util
import json
from pathlib import Path
import struct

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("kernel_records", ROOT / "scripts/build/kernel-records.py")
records = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(records)


@pytest.mark.parametrize("changed", ["ir", "identity", "option", "pipeline", "signature"])
def test_backend_cache_keys_lowering_inputs_not_export_placement(tmp_path, changed):
    from types import SimpleNamespace
    class Option:
        pass
    option = SimpleNamespace(value="old")
    dsl = SimpleNamespace(compile_options=SimpleNamespace(options={Option: option}))
    def compile(dsl, module, sig="sig", function_name="kernel", func_type="executor",
                pipeline="pipeline", no_jit_engine=True, no_cache=True):
        pass
    cache = records.BackendCache(tmp_path, {"compiler": "old"})
    before = cache.key(dsl, compile, None, (), {}, "ir")
    kwargs, ir = {}, "ir"
    if changed == "ir":
        ir = "new ir"
    elif changed == "identity":
        cache.identity = {"compiler": "new"}
    elif changed == "option":
        option.value = "new"
    elif changed == "pipeline":
        kwargs["pipeline"] = "new"
    else:
        kwargs["sig"] = "new"
    assert before and before != cache.key(dsl, compile, None, (), kwargs, ir)
    assert cache.key(dsl, compile, None, (), {"no_cache": False}, ir) is None


def test_backend_identity_ignores_frontend_and_worker_provenance(tmp_path, compiler_inputs, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "python/tools/lib"))
    try:
        before = records.backend_identity()
        for name in ("CUDA_VISIBLE_DEVICES", "CUTEAFD_EXPORT_JOBS", "CUTEAFD_EXPORT_SLOT_DIR",
                     "CUTEAFD_DEV_ARTIFACTS", "CUTEAFD_KACHE", "CUTEAFD_EXL3_ROUTE_BLOCKS"):
            monkeypatch.setenv(name, "changed")
        assert records.backend_identity() == before
        monkeypatch.setenv("CUTEAFD_EXPORT_IMAGE_ID", "sha256:changed")
        assert records.backend_identity() != before
    finally:
        sys.path.pop(0)


def test_change_report_classifies_and_flags_exact_groups(tmp_path):
    old = {"programs/glmf_mhc.o": {"group": "programs", "ir_sha256": "a", "object_sha256": "x",
        "source_reads_sha256": {"kernel.py": "old", "unchanged.py": "same"}, "resources": [{"kernels": {"k": {"registers": 32}}}],
        "scratch_bytes_at_capacity": {"64": {"workspace": 16}}},
        "exl3/old.o": {"group": "exl3/old"}}
    new = {"programs/glmf_mhc.o": {**old["programs/glmf_mhc.o"], "object_sha256": "y",
        "source_reads_sha256": {"kernel.py": "new", "unchanged.py": "same"}, "resources": [{"kernels": {"k": {"registers": 40}}}],
        "scratch_bytes_at_capacity": {"64": {"workspace": 32}}},
        "exl3/new.o": {"group": "exl3/new"}}
    report = records.change_report(old, new, ["programs"])
    assert len(report["changed"]) == len(report["added"]) == len(report["removed"]) == 1
    changed = report["changed"][0]
    assert changed["object_changed"] and not changed["ir_changed"]
    assert changed["triggering_sources"] == ["kernel.py"]
    assert changed["resource_deltas"]["[0].kernels.k.registers"]["delta"] == 8
    assert changed["scratch_deltas"]["64.workspace"]["delta"] == 16
    assert report["leaks"] == ["exl3/new.o", "exl3/old.o"]
    records.write_report(tmp_path, report)
    markdown = (tmp_path / "CHANGES.md").read_text()
    assert "UNEXPECTED CHANGES" in markdown
    assert "cumulative" in markdown and "Sources: kernel.py." in markdown
    assert "unchanged.py" not in markdown
    assert json.loads((tmp_path / "CHANGES.json").read_text()) == report


def test_source_reads_include_actual_frames_not_every_import(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("SCALE = 3\ndef kernel():\n    return SCALE\n")
    unused = tmp_path / "unused.py"
    unused.write_text("UNUSED = True\n")
    spec = importlib.util.spec_from_file_location("read_test_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = records.Recorder(tmp_path, tmp_path / "output")
    import sys
    recorder.imported = set(sys.modules)
    previous = sys.getprofile()
    sys.setprofile(recorder._profile)
    try:
        assert module.kernel() == 3
    finally:
        sys.setprofile(previous)
    assert recorder.reads() == {"kernel.py": records.digest(source.read_bytes())}
    assert ("kernel.py", "kernel") in recorder.functions


def cuda_elf():
    names = b"\0.shstrtab\0.strtab\0.symtab\0.text.kernel\0.nv.info\0.nv.info.kernel\0.nv.shared.kernel\0"
    symbol_names = b"\0kernel\0"
    symbols = bytes(24) + struct.pack("<IBBHQQ", 1, 0, 0, 4, 0, 0)
    infos = struct.pack("<BBHII", 4, 47, 8, 1, 64)
    thread_info = struct.pack("<BBHIII", 4, 16, 12, 128, 1, 1)
    sections = [("", 0, b"", 0, 0), (".shstrtab", 3, names, 0, 0),
                (".strtab", 3, symbol_names, 0, 0), (".symtab", 2, symbols, 2, 0),
                (".text.kernel", 1, b"code", 0, 1), (".nv.info", 1, infos, 0, 0),
                (".nv.info.kernel", 1, thread_info, 0, 0), (".nv.shared.kernel", 8, bytes(1024), 0, 0)]
    blob = bytearray(64 + 64 * len(sections))
    blob[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", blob, 40, 64)
    struct.pack_into("<HHH", blob, 58, 64, len(sections), 1)
    for index, (name, kind, data, link, info) in enumerate(sections):
        struct.pack_into("<IIQQQQIIQQ", blob, 64 + index * 64,
                         names.index(name.encode() + b"\0") if name else 0, kind, 0, 0,
                         len(blob), len(data), link, info, 1, 0)
        if kind != 8:
            blob += data
    return bytes(blob)


def test_cuda_resources_match_elf_symbol_regcount_and_thread_attributes():
    assert records.elf_resources(cuda_elf()) == {
        "kernel": {"registers": 64, "static_smem_bytes": 1024, "max_threads": 128}}
    with pytest.raises(ValueError, match="ELF64"):
        records.elf_resources(b"not a cubin")


@pytest.mark.parametrize("field", ["abi_sha256", "plan", "planner_metadata"])
def test_metadata_only_changes_are_reported_and_checked(field):
    report = records.change_report({"k": {field: "old"}}, {"k": {field: "new"}}, [])
    assert len(report["changed"]) == 1
    assert report["leaks"] == ["k"]
    assert not report["changed"][0]["ir_changed"]


def test_construct_captures_recipe_and_scratch_callback_sources(tmp_path):
    source = tmp_path / "recipe.py"
    source.write_text("def construct():\n    return 5\n")
    spec = importlib.util.spec_from_file_location("record_recipe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = records.Recorder(tmp_path, tmp_path / "output")
    recorder.start_monitoring()
    try:
        assert recorder.construct(module.construct) == 5
        assert "recipe.py" in recorder.reads()
    finally:
        recorder.stop_monitoring()


def test_local_package_policy_is_a_recorded_input(tmp_path):
    import sys
    import types
    package = types.ModuleType("package_policy")
    package_path = tmp_path / "package_policy.py"
    package_path.write_text("TILE = 64\n")
    package.__file__ = str(package_path)
    recipe = tmp_path / "recipe.py"
    recipe.write_text("def caller(package):\n    import sys\n    return sys._getframe()\n")
    spec = importlib.util.spec_from_file_location("local_recipe", recipe)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = records.Recorder(tmp_path, tmp_path / "output")
    recorder.imported = set(sys.modules)
    recorder._profile(module.caller(package), "call", None)
    assert "package_policy.py" in recorder.reads()


def test_package_tuning_and_arguments_change_job_global_identity():
    context = {key: None for key in (
        "exporter", "exporter_sha256", "args", "comparison_script_sha256", "export_sources_sha256",
        "sms", "capability", "toolchain", "environment", "build_kind")}
    context.update(exporter="package_exl3_aot.py", exporter_sha256="old", args=["--bits", "4"])
    before = records.stable_hash(records.job_global_inputs(context))
    assert before != records.stable_hash(records.job_global_inputs({**context, "exporter_sha256": "new"}))
    assert before != records.stable_hash(records.job_global_inputs({**context, "args": ["--bits", "3"]}))


def test_stage0_candidates_require_globals_and_all_recorded_reads(tmp_path):
    source = tmp_path / "package.py"
    source.write_text("TILE = 64\n")
    inputs = {"exporter": "a", "args": ["--bits", "4"]}
    previous = {"stage0_inputs": inputs, "source_reads_sha256": {"package.py": records.digest(source.read_bytes())}}
    assert records.job_candidates(tmp_path, previous, inputs) == []
    source.write_text("TILE = 128\n")
    assert records.job_candidates(tmp_path, previous, inputs) == ["package.py"]
    assert records.job_candidates(tmp_path, previous, {**inputs, "exporter": "b"}) == ["job-global-inputs"]
    assert records.job_candidates(tmp_path, previous, None) == ["job-global-inputs"]
    assert records.job_candidates(tmp_path, {"stage0_inputs": inputs}, inputs) == ["missing-recorded-reads"]


@pytest.mark.parametrize("cached", ["__pycache__/package.cpython-312.pyc", "package.pyc", "__pycache__/cached"])
def test_stage0_ignores_legacy_bytecode_reads_but_keeps_source_changes(tmp_path, cached):
    source = tmp_path / "package.py"
    source.write_text("TILE = 64\n")
    bytecode = tmp_path / cached
    bytecode.parent.mkdir(parents=True, exist_ok=True)
    bytecode.write_bytes(b"old cache")
    inputs = {"exporter": "a"}
    previous = {"stage0_inputs": inputs, "source_reads_sha256": {
        "package.py": records.digest(source.read_bytes()), cached: records.digest(bytecode.read_bytes())}}
    bytecode.write_bytes(b"new cache")
    assert records.job_candidates(tmp_path, previous, inputs) == []
    bytecode.unlink()
    assert records.job_candidates(tmp_path, previous, inputs) == []
    source.write_text("TILE = 128\n")
    assert records.job_candidates(tmp_path, previous, inputs) == ["package.py"]


def test_recorder_import_keeps_source_without_bytecode_reads(tmp_path, monkeypatch, fake_cutlass):
    import py_compile
    import sys
    source = tmp_path / "recorded_import.py"
    source.write_text("TILE = 64\n")
    bytecode = Path(py_compile.compile(str(source), doraise=True))
    recorder = records.Recorder(tmp_path, tmp_path / "output").install()
    try:
        spec = importlib.util.spec_from_file_location("recorded_import", source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "recorded_import", module)
        spec.loader.exec_module(module)
        assert module.TILE == 64
        assert recorder.reads() == {"recorded_import.py": records.digest(source.read_bytes())}
        assert records.source_name(bytecode, tmp_path) is None
        legacy = tmp_path / "recorded_import.pyc"
        legacy.write_bytes(bytecode.read_bytes())
        assert records.source_name(legacy, tmp_path) is None
    finally:
        stop_recorder(recorder)


@pytest.fixture
def compiler_inputs(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    compiler = SimpleNamespace(_runtime_toolchain_key=lambda: "toolchain",
                               _compile_environment_key=lambda: "environment")
    monkeypatch.setitem(sys.modules, "b12x._lib", SimpleNamespace(compiler=compiler))
    monkeypatch.setenv("CUTEAFD_EXPORT_IMAGE_ID", "sha256:test-image")
    ptxas = tmp_path / "ptxas"
    ptxas.write_bytes(b"compiler")
    monkeypatch.setattr(records.shutil, "which", lambda _: str(ptxas))
    monkeypatch.setattr(records.subprocess, "check_output", lambda *args, **kwargs: "ptxas test")
    build = tmp_path / "scripts/build"
    build.mkdir(parents=True)
    (build / "kernel-input-rules.json").write_bytes((ROOT / "scripts/build/kernel-input-rules.json").read_bytes())
    for script in ("kernel-records.py", "compare-sm-exports.py", "export-placement-inputs.py"):
        (build / script).write_text("def report():\n    return 'old'\n")
    exporter = tmp_path / "python/tools/aot/export.py"
    exporter.parent.mkdir(parents=True)
    exporter.write_text("TILE = 64\n")
    payload = {"exporter": "export.py", "args": ["--geometry", "glmf"], "sms": 188}
    return payload, exporter


def test_orchestration_edits_keep_all_publication_jobs_reusable(tmp_path, compiler_inputs, monkeypatch):
    payload, exporter = compiler_inputs
    spec = importlib.util.spec_from_file_location(
        "identity_publisher", ROOT / "scripts/build/export-placement-inputs.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    # Job enumeration is tested separately against installed metadata; replay
    # identity must not need retired checkpoints in this synthetic source tree.
    monkeypatch.setattr(publisher, "live_configs", lambda: [])
    jobs = publisher.export_jobs()
    for _, name, _ in jobs:
        (exporter.parent / name).write_bytes(exporter.read_bytes())
    previous = []
    for _, name, args in jobs:
        job = {**payload, "exporter": name, "args": args}
        previous.append((job, {"stage0_inputs": records.stage0_inputs(tmp_path, job, (12, 0)),
            "source_reads_sha256": {str(exporter.relative_to(tmp_path)): records.digest(exporter.read_bytes())}}))
    for script in ("kernel-records.py", "compare-sm-exports.py", "export-placement-inputs.py",
                   "build-wip-artifacts.sh", "wip-native-store.sh"):
        (tmp_path / "scripts/build" / script).write_text("def report():\n    return 'new rsync/reporting'\n")
        for job, prior in previous:
            assert records.job_candidates(tmp_path, prior, records.stage0_inputs(tmp_path, job, (12, 0))) == []
    for job, prior in previous:
        changed = {**job, "args": [*job["args"], "--changed-exporter-arg"]}
        assert records.job_candidates(tmp_path, prior, records.stage0_inputs(tmp_path, changed, (12, 0))) == ["job-global-inputs"]


def test_orchestration_paths_do_not_key_replay_but_executed_helpers_do(tmp_path, compiler_inputs, monkeypatch):
    payload, exporter = compiler_inputs
    helper = tmp_path / "scripts/build/build-wip-artifacts.sh"
    helper.write_text("old assembly")
    inputs = records.stage0_inputs(tmp_path, payload, (12, 0))
    prior = {"stage0_inputs": inputs, "source_reads_sha256": {
        str(exporter.relative_to(tmp_path)): records.digest(exporter.read_bytes())}}
    for name in (*records.PROVENANCE_ENVIRONMENT_NAMES, "CUTEAFD_DEV_ARTIFACTS", "CUTEAFD_KACHE",
                 "CUTEAFD_KACHE_MODE", "CUTEAFD_KACHE_CACHE_DIR", "CUTEAFD_KACHE_SHIM_DIR"):
        monkeypatch.setenv(name, "/changed/" + name)
    helper.write_text("new assembly")
    assert records.job_candidates(tmp_path, prior, records.stage0_inputs(tmp_path, payload, (12, 0))) == []
    prior["source_reads_sha256"][str(helper.relative_to(tmp_path))] = records.digest(b"old assembly")
    assert records.job_candidates(tmp_path, prior, records.stage0_inputs(tmp_path, payload, (12, 0))) == [
        "scripts/build/build-wip-artifacts.sh"]


@pytest.mark.parametrize("changed", ["exporter", "args", "global", "helper", "environment", "toolchain"])
def test_compile_inputs_still_invalidate_stage0(tmp_path, monkeypatch, compiler_inputs, changed):
    payload, exporter = compiler_inputs
    inputs = records.stage0_inputs(tmp_path, payload, (12, 0))
    prior = {"stage0_inputs": inputs, "source_reads_sha256": {
        str(exporter.relative_to(tmp_path)): records.digest(exporter.read_bytes())}}
    assert records.job_candidates(tmp_path, prior, inputs) == []
    if changed == "exporter":
        exporter.write_text("TILE = 128\n")
    elif changed == "args":
        payload = {**payload, "args": ["--geometry", "glm"]}
    elif changed == "global":
        (tmp_path / "scripts/build/geometry.json").write_text('{"tile": 128}')
    elif changed == "helper":
        (tmp_path / "scripts/build/dev-toolchain.py").write_text("TOOLCHAIN = 'changed'\n")
    elif changed == "environment":
        monkeypatch.setenv("CUTEAFD_EXL3_ROUTE_BLOCKS", "changed")
    else:
        monkeypatch.setenv("CUTEAFD_EXPORT_IMAGE_ID", "sha256:new-toolchain")
    assert records.job_candidates(tmp_path, prior, records.stage0_inputs(tmp_path, payload, (12, 0))) == ["job-global-inputs"]


@pytest.fixture
def fake_cutlass(monkeypatch):
    import sys
    import types
    class JitCompiledFunction:
        def export_to_c(self, file_path, file_name, function_prefix=""):
            (Path(file_path) / (file_name + ".o")).write_bytes(cuda_elf())
    class BaseDSL:
        def generate_original_ir(*args, **kwargs):
            return ("same IR",)
        def compile_and_cache(*args, **kwargs):
            return JitCompiledFunction()
    for name in ("cutlass", "cutlass.base_dsl", "cutlass.base_dsl.dsl", "cutlass.base_dsl.jit_executor"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["cutlass.base_dsl.dsl"].BaseDSL = BaseDSL
    sys.modules["cutlass.base_dsl.jit_executor"].JitCompiledFunction = JitCompiledFunction
    return BaseDSL


def load_recipe(path, text):
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stop_recorder(recorder):
    recorder.stop_monitoring()
    recorder.active = False
    for cls, method, original in reversed(recorder.restores):
        setattr(cls, method, original)


@pytest.mark.parametrize("binding", ["local", "global", "cell"])
def test_each_program_rescans_recipe_tuning_modules(tmp_path, monkeypatch, fake_cutlass, binding):
    import sys
    import types
    bodies = {
        "local": "def recipe(policy, compile):\n    return compile(None)\n",
        "global": "def recipe(unused, compile):\n    return compile(policy)\n",
        "cell": "def recipe(value, compile):\n    policy = value\n    def closure():\n        return policy\n    return compile(None)\n",
    }
    recipe_module = load_recipe(tmp_path / "recipe.py", bodies[binding])
    policies = []
    for name in ("tuning_a", "tuning_b"):
        path = tmp_path / (name + ".py")
        path.write_text("TILE = 64\n")
        policy = types.ModuleType(name)
        policy.__file__ = str(path)
        monkeypatch.setitem(sys.modules, name, policy)
        policies.append(policy)
    del policy
    recorder = records.Recorder(tmp_path, tmp_path / "output").install()
    try:
        recipe_module.policy = policies[0]
        recipe_module.recipe(policies[0], fake_cutlass.generate_original_ir)
        first = next(iter(recorder.modules.values()))["source_reads_sha256"]
        assert "tuning_a.py" in first and "tuning_b.py" not in first
        recipe_module.policy = policies[1]
        recipe_module.recipe(policies[1], fake_cutlass.generate_original_ir)
        second = next(iter(recorder.modules.values()))["source_reads_sha256"]
        assert "tuning_b.py" in second
        assert "tuning_a.py" in second  # Program records are conservative cumulative observations.
        assert {"tuning_a.py", "tuning_b.py"} <= recorder.reads().keys()
    finally:
        stop_recorder(recorder)


def test_monitoring_restart_observes_same_code_in_later_recorder(tmp_path):
    module = load_recipe(tmp_path / "recipe.py", "def recipe():\n    return 5\n")
    for _ in range(2):
        recorder = records.Recorder(tmp_path, tmp_path / "output")
        recorder.start_monitoring()
        try:
            assert recorder.monitoring_id in (3, 4)
            assert module.recipe() == 5
            assert "recipe.py" in recorder.reads()
        finally:
            recorder.stop_monitoring()


def test_monitoring_disables_repeated_code_entry_and_releases_tool(tmp_path):
    source = tmp_path / "recipe.py"
    source.write_text("def recipe():\n    return 5\n")
    spec = importlib.util.spec_from_file_location("one_shot_recipe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = records.Recorder(tmp_path, tmp_path / "output")
    recorder.start_monitoring()
    tool_id = recorder.monitoring_id
    try:
        assert module.recipe() == 5
        recorder.frames.clear()
        assert module.recipe() == 5
        assert "recipe.py" not in recorder.frames
    finally:
        recorder.stop_monitoring()
    import sys
    assert sys.monitoring.get_tool(tool_id) is None


@pytest.fixture
def fake_worker(tmp_path, monkeypatch, fake_cutlass):
    import importlib.metadata
    import sys
    import types
    build = tmp_path / "scripts/build"
    build.mkdir(parents=True)
    for name in ("compare-sm-exports.py", "kernel-records.py"):
        (build / name).write_bytes((ROOT / "scripts/build" / name).read_bytes())
    monkeypatch.setenv("CUTEAFD_SOURCE_ROOT", str(ROOT))
    spec = importlib.util.spec_from_file_location("installed_worker", build / "compare-sm-exports.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    spec = importlib.util.spec_from_file_location("installed_records", build / "kernel-records.py")
    worker_records = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_records)
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    for name in ("b12x", "b12x._lib", "b12x._lib.compile_pool", "b12x._lib.compiler",
                 "b12x.integration", "b12x.integration.cuteafd", "_pinned_sparkinfer", "torch"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["b12x._lib.compile_pool"]._initialize_worker = lambda *args: None
    sys.modules["b12x._lib.compiler"].b12x_package_fingerprint = lambda: None
    class AotProgram:
        def export_to_c(*args, **kwargs):
            pass
    sys.modules["b12x.integration.cuteafd"].AotProgram = AotProgram
    sys.modules["_pinned_sparkinfer"].REVISION = "pin"
    sys.modules["_pinned_sparkinfer"].LOCK_DATA = {"source_tree_sha256": "tree"}
    sys.modules["torch"].cuda = types.SimpleNamespace(
        get_device_properties=lambda ordinal: types.SimpleNamespace(multi_processor_count=188),
        is_initialized=lambda: False)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fake")
    monkeypatch.setattr(worker_records, "stage0_inputs", lambda *args: {"fake": True})
    monkeypatch.setattr(worker_records, "install_backend_cache", lambda *args: None)
    monkeypatch.setattr(worker, "load_file", lambda name, path:
                        worker_records if name == "placement_kernel_records" else types.SimpleNamespace(
                            cuda_images=lambda blob: [(0, blob)]))
    exporter = tmp_path / "python/tools/aot/x.py"
    exporter.parent.mkdir(parents=True)
    exporter.write_text("# Fake backend; real worker and recorder install.\n")
    recipe = load_recipe(tmp_path / "recipe.py", "def recipe(compile, output):\n    ir = compile.generate_original_ir(None)[0]\n    result = compile.compile_and_cache(None, ir)\n    result.export_to_c(output, 'kernel')\n")
    after = load_recipe(tmp_path / "after.py", "def validate():\n    return 7\n")
    def run_path(*args, **kwargs):
        assert worker.PROGRAM_RECORDER.monitoring_id in (3, 4)
        recipe.recipe(fake_cutlass, worker.PROGRAM_RECORDER.output)
        assert after.validate() == 7
    monkeypatch.setattr(worker.runpy, "run_path", run_path)
    def run(payload):
        try:
            worker.worker(payload)
        finally:
            if worker.PROGRAM_RECORDER is not None:
                stop_recorder(worker.PROGRAM_RECORDER)
                worker.PROGRAM_RECORDER = None
    return run


def test_job_reads_include_helpers_after_last_program(tmp_path, fake_worker):
    output = tmp_path / "output"
    output.mkdir()
    fake_worker({"sms": 188, "exporter": "x.py", "args": [],
                 "output": str(output), "program_records": True})
    summary = json.loads((output / "comparison-context.json").read_text())
    assert {"recipe.py", "after.py", "scripts/build/compare-sm-exports.py",
            "scripts/build/kernel-records.py"} <= summary["source_reads_sha256"].keys()
    for script in ("compare-sm-exports.py", "kernel-records.py"):
        path = tmp_path / "scripts/build" / script
        original = path.read_text()
        path.write_text(original.replace("def worker(payload):", "def worker(payload):\n    compile_policy = 'changed'", 1)
                        if script == "compare-sm-exports.py" else original.replace(
                            "def construct(self, function, *args, **kwargs):",
                            "def construct(self, function, *args, **kwargs):\n        compile_policy = 'changed'", 1))
        assert records.job_candidates(tmp_path, summary, summary["stage0_inputs"]) == [f"scripts/build/{script}"]
        path.write_text(original)


@pytest.mark.parametrize("changed", ["args", "environment"])
def test_changed_identity_runs_traced_with_unchanged_reads_and_ir(tmp_path, monkeypatch, changed, fake_worker):
    spec = importlib.util.spec_from_file_location("traced_publisher", ROOT / "scripts/build/export-placement-inputs.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    source = tmp_path / "recipe.py"
    previous = tmp_path / "previous"
    directory = previous / ".workers/programs"
    directory.mkdir(parents=True)
    inputs = {"args": ["old"], "environment": {"MODE": "old"}}
    seed = tmp_path / "seed"
    seed.mkdir()
    fake_worker({"sms": 188, "exporter": "x.py", "args": [],
                 "output": str(seed), "program_records": True})
    before = json.loads((seed / "program-records.json").read_text())["kernel.o"]
    context = {"stage0_inputs": inputs, "source_reads_sha256": {
        "recipe.py": records.digest(source.read_bytes())}}
    (directory / "comparison-context.json").write_text(json.dumps(context))
    (directory / "program-records.json").write_text(json.dumps({"kernel.o": before}))
    files = {str(path.relative_to(previous)): records.digest(path.read_bytes()) for path in directory.iterdir()}
    (previous / "INPUTS.json").write_text(json.dumps({"files": files}))
    current = {**inputs, changed: ["new"] if changed == "args" else {"MODE": "new"}}
    monkeypatch.setattr(publisher.kernel_records, "stage0_inputs", lambda *args: current)
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    monkeypatch.setattr(publisher, "check_job_sm170", lambda *args, **kwargs: {})
    observed = []
    def compile_worker(command, **kwargs):
        payload = json.loads(command[-1])
        assert payload["program_records"] is True
        assert "prior_source_records" not in payload
        output = Path(payload["output"])
        fake_worker(payload)
        worker_context = json.loads((output / "comparison-context.json").read_text())
        after = json.loads((output / "program-records.json").read_text())["kernel.o"]
        assert after["ir_sha256"] == before["ir_sha256"]
        observed.append("recipe.py" in worker_context["source_reads_sha256"])
    monkeypatch.setattr(publisher.subprocess, "run", compile_worker)
    summary = publisher.compile_job(("programs", "x.py", current["args"]),
                                    tmp_path / "staging", previous, {}, 1)
    assert observed == [True]
    assert summary["execution_stats"]["stage0_skipped"] is False


def test_job_replay_checksums_and_verify_skips_reject_ir_drift(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("replay_publisher", ROOT / "scripts/build/export-placement-inputs.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    previous = tmp_path / "previous"
    directory = previous / ".workers/programs"
    directory.mkdir(parents=True)
    context = {"stage0_inputs": {"global": "same"}, "source_reads_sha256": {"kernel.py": "unused"}}
    (directory / "comparison-context.json").write_text(json.dumps(context))
    (directory / "program-records.json").write_text(json.dumps({"kernel.o": {"ir_sha256": "old"}}))
    files = {str(path.relative_to(previous)): records.digest(path.read_bytes()) for path in directory.iterdir()}
    (previous / "INPUTS.json").write_text(json.dumps({"files": files}))
    assert publisher.verified_job_files(previous, "programs")[0] == directory
    monkeypatch.setattr(publisher.kernel_records, "stage0_inputs", lambda *args: context["stage0_inputs"])
    monkeypatch.setattr(publisher.kernel_records, "job_candidates", lambda *args: [])
    def compile_worker(command, **kwargs):
        output = Path(json.loads(command[-1])["output"])
        (output / "program-records.json").write_text(json.dumps({"kernel.o": {"ir_sha256": "changed"}}))
    monkeypatch.setattr(publisher.subprocess, "run", compile_worker)
    with pytest.raises(ValueError, match="--verify-skips changed IR"):
        publisher.compile_job(("programs", "x.py", []), tmp_path / "staging", previous, {}, 1, verify_skips=True)
    (directory / "program-records.json").write_text("{}")
    with pytest.raises(ValueError, match="corrupt replay"):
        publisher.verified_job_files(previous, "programs")


def test_export_policy_environment_changes_invalidate_identity(monkeypatch):
    monkeypatch.delenv("CUTEAFD_BUILD_KIND", raising=False)
    monkeypatch.delenv("CUTEAFD_EXL3_ROUTE_BLOCKS", raising=False)
    before = records.export_environment()
    monkeypatch.setenv("CUTEAFD_BUILD_KIND", "release")
    assert records.export_environment() != before
    monkeypatch.delenv("CUTEAFD_BUILD_KIND")
    monkeypatch.setenv("CUTEAFD_EXL3_ROUTE_BLOCKS", "glmf=16")
    assert records.export_environment() != before


def test_export_environment_covers_every_source_read(monkeypatch):
    import ast
    # _export_cache.py is cache tooling like kernel-records.py: it reads only its
    # own control variables and replays recorded names, never export semantics.
    paths = [*sorted((ROOT / "python/tools/aot").glob("*.py")),
             *sorted(path for path in (ROOT / "python/tools/lib").glob("*.py") if path.name != "_export_cache.py"),
             ROOT / "scripts/build/compare-sm-exports.py",
             ROOT / "scripts/build/export-placement-inputs.py"]
    names = set()
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = ast.unparse(node.func.value)
                if ((owner == "os.environ" and node.func.attr in ("get", "setdefault"))
                        or (owner == "os" and node.func.attr == "getenv")
                        or (owner == "env" and node.func.attr == "get")):
                    assert node.args and isinstance(node.args[0], ast.Constant), (path, node.lineno)
                    names.add(node.args[0].value)
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                if ast.unparse(node.value) in ("os.environ", "env"):
                    if isinstance(node.slice, ast.Constant):
                        names.add(node.slice.value)
                    else:
                        # The comparison context uses this fixed two-name loop.
                        assert (path.name == "compare-sm-exports.py"
                                and ast.unparse(node.slice) == "key"), (path, node.lineno)
    assert names <= records.EXPORT_ENVIRONMENT_NAMES
    assert names <= records.export_environment().keys()
    for name in names - records.PROVENANCE_ENVIRONMENT_NAMES - {"CUTEAFD_SOURCE_ROOT", "CUTEAFD_SPARKINFER_SOURCE_DIR",
                          "CUTEAFD_SPARKINFER_LOCK_FILE", "TRITON_CACHE_DIR",
                          "CUDA_VISIBLE_DEVICES", "B12X_COMPILE_DISK_CACHE",
                          "B12X_COMPILE_MEMORY_CACHE", "CUTEAFD_EXPORT_NARROW_AOT",
                          "CUTEAFD_EXPORT_HC_LAGGED", "TRITON_DISABLE_LINE_INFO", "CUTEAFD_EXPORT_CACHE"}:
        before = records.export_environment()
        with monkeypatch.context() as patch:
            patch.setenv(name, "environment-coverage-sentinel")
            assert records.export_environment() != before, name


@pytest.mark.parametrize("filename", ["comparison-plans.json", "program-records.json", "manifest.json"])
def test_verify_skips_rejects_same_ir_metadata_drift(tmp_path, filename):
    spec = importlib.util.spec_from_file_location("metadata_publisher", ROOT / "scripts/build/export-placement-inputs.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    previous, current = tmp_path / "previous", tmp_path / "current"
    previous.mkdir()
    current.mkdir()
    metadata = {"ir_sha256": "same", "scratch_bytes": 64, "abi": ["ptr", "i32"],
                "geometry": {"rows": 80}, "params": {"tile": 128}}
    def document(record):
        return {"kernel.o": record} if filename == "program-records.json" else record
    (previous / filename).write_text(json.dumps(document(metadata)))
    (current / filename).write_text(json.dumps(document(metadata)))
    publisher.verify_job_metadata(previous, current, {}, "test")
    (current / filename).write_text(json.dumps(document({**metadata, "scratch_bytes": 128})))
    with pytest.raises(ValueError, match="placement metadata"):
        publisher.verify_job_metadata(previous, current, {}, "test")


def test_sm170_mismatch_is_reported_as_hard_failure(tmp_path):
    report = records.change_report({}, {"programs/kernel.o": {
        "object_sha256": "188", "sm170_object_sha256": "170"}})
    assert report["sm170_mismatches"] == ["programs/kernel.o"]
    records.write_report(tmp_path, report)
    assert "HARD FAILURE" in (tmp_path / "CHANGES.md").read_text()


def test_sm170_recompiles_only_changed_routed_objects(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("sm170_publisher", ROOT / "scripts/build/export-placement-inputs.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    job = ("fp8/test", "package_fp8_moe_aot.py", [])
    previous, staging = tmp_path / "previous", tmp_path / "staging"
    directory = staging / job[0]
    directory.mkdir(parents=True)
    programs = {"same.o": {"object_sha256": "same"}, "changed.o": {"object_sha256": "new"}}
    (directory / "program-records.json").write_text(json.dumps(programs))
    previous.mkdir()
    prior = {job[0] + "/" + name: {**record, "object_sha256": "old" if name == "changed.o"
                                   else record["object_sha256"]} for name, record in programs.items()}
    (previous / "INPUTS.json").write_text(json.dumps({"program_records": prior}))
    def publication(path, jobs):
        assert path == staging, "previous records must come from INPUTS, not historical worker schemas"
        return {job[0] + "/" + name: record for name, record in programs.items()}
    monkeypatch.setattr(publisher, "publication_records", publication)
    calls = []
    def worker(command, **kwargs):
        payload = json.loads(command[-1])
        calls.append(payload)
        (Path(payload["output"]) / "program-records.json").write_text(json.dumps({"changed.o": {"object_sha256": "new"}}))
    monkeypatch.setattr(publisher.subprocess, "run", worker)
    result = publisher.check_job_sm170(job, staging, previous, 1, {})
    assert result["sm170_programs"] == 1
    assert calls[0]["sms"] == 170 and calls[0]["selected_objects"] == ["changed"]
    output = json.loads((directory / "program-records.json").read_text())
    assert output["changed.o"]["sm170_object_sha256"] == "new"
    assert "sm170_object_sha256" not in output["same.o"]


def test_verify_metadata_normalizes_only_pin_and_route_checksum(tmp_path):
    spec = importlib.util.spec_from_file_location("pin_publisher", ROOT / "scripts/build/export-placement-inputs.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    old, new = tmp_path / "old", tmp_path / "new"
    lock = {"revision": "new-pin", "source_tree_sha256": "new-tree"}
    for directory, revision in ((old, "old-pin"), (new, "new-pin")):
        directory.mkdir()
        route = {"sparkinfer_revision": revision, "source_tree_sha256": revision + "-tree", "scratch": 64}
        (directory / "routes.json").write_text(json.dumps(route))
        manifest = {"sparkinfer_revision": revision,
                    "route_preparation": {"manifest": "routes.json", "sha256": records.digest(json.dumps(route).encode())}}
        (directory / "v41_exl3.json").write_text(json.dumps(manifest))
    publisher.verify_job_metadata(old, new, lock, "exl3")
    (new / "routes.json").write_text(json.dumps({"sparkinfer_revision": "new-pin", "scratch": 128}))
    with pytest.raises(ValueError, match="routes.json"):
        publisher.verify_job_metadata(old, new, lock, "exl3")


def test_publisher_writes_sm170_report_before_failing(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("mismatch_publisher", ROOT / "scripts/build/export-placement-inputs.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    job = ("programs", "export_b12x_dsv4_aot.py", [])
    monkeypatch.setattr(publisher, "export_jobs", lambda: [job])
    monkeypatch.setattr(publisher, "source_hashes", lambda: {})
    lock = json.loads((ROOT / "third_party/sparkinfer.lock.json").read_text())
    def worker(job, staging, *args):
        directory = staging / "programs"
        directory.mkdir()
        (directory / "dsv4_programs.json").write_text('{"programs": []}')
        (directory / "program-records.json").write_text(json.dumps({"kernel.o": {
            "object_sha256": "188", "sm170_object_sha256": "170"}}))
        return {"sparkinfer_revision": lock["revision"], "source_tree_sha256": lock["source_tree_sha256"],
                "cuda_initialized": False}
    monkeypatch.setattr(publisher, "compile_job", worker)
    original_load = publisher.compare.load_file
    def load(name, path):
        if name == "placement_filesystem":
            import types
            return types.SimpleNamespace(check_path=lambda path: None)
        return original_load(name, path)
    monkeypatch.setattr(publisher.compare, "load_file", load)
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="SM 170 / 188 OBJECT MISMATCH"):
        publisher.publish(destination, tmp_path / "staging", 1, 1)
    assert json.loads((destination / "CHANGES.json").read_text())["sm170_mismatches"] == ["programs/kernel.o"]
    assert "HARD FAILURE" in (destination / "CHANGES.md").read_text()


@pytest.mark.parametrize("content_changed", [False, True])
def test_publication_route_content_identity_is_pin_independent(tmp_path, content_changed):
    spec = importlib.util.spec_from_file_location("route_publisher", ROOT / "scripts/build/export-placement-inputs.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    sides = []
    raw_hashes = []
    for name, revision, scratch in (("old", "old-pin", 64), ("new", "new-pin", 128 if content_changed else 64)):
        staging = tmp_path / name
        directory = staging / "exl3" / "m1"
        directory.mkdir(parents=True)
        route = directory / "routes.json"
        route.write_text(json.dumps({"sparkinfer_revision": revision,
                                    "source_tree_sha256": revision + "-tree", "scratch": scratch}))
        raw = records.digest(route.read_bytes())
        raw_hashes.append(raw)
        (directory / "v41_exl3.json").write_text(json.dumps({
            "sparkinfer_revision": revision, "capacity": 1, "unique_execution_buffer_bytes": 64,
            "objects": [], "route_preparation": {"manifest": "routes.json", "sha256": raw}}))
        (staging / "exl3" / "program-records.json").write_text(json.dumps({"m1/core.o": {"object_sha256": "same"}}))
        sides.append(publisher.publication_records(staging, [("exl3", "package_exl3_aot.py", [])]))
    assert raw_hashes[0] != raw_hashes[1]
    for side, raw in zip(sides, raw_hashes):
        assert side["exl3/m1/core.o"]["planner_metadata"]["route_preparation"]["sha256"] == raw
    report = records.change_report(*sides, expected=[])
    assert bool(report["changed"]) == content_changed
    assert report["leaks"] == (["exl3/m1/core.o"] if content_changed else [])
    if content_changed:
        assert report["changed"][0]["planner_metadata_changed"]


@pytest.mark.parametrize("digests", [(None, None), (None, "same"), ("same", None), ("", "same")])
def test_legacy_route_records_keep_raw_checksum_strict(digests):
    sides = []
    for raw, content in zip(("old", "new"), digests):
        route = {"manifest": "routes.json", "sha256": raw}
        if content is not None:
            route["content_sha256"] = content
        sides.append({"core.o": {"planner_metadata": {"route_preparation": route}}})
    report = records.change_report(*sides, expected=[])
    assert report["leaks"] == ["core.o"]
    assert report["changed"][0]["planner_metadata_changed"]


def test_identity_and_reports_are_deterministic():
    assert records.stable_hash({"ir": "a", "target": [12, 0]}) == records.stable_hash({"target": [12, 0], "ir": "a"})
    assert records.stable_hash({"ir": "a"}) != records.stable_hash({"ir": "b"})
    assert records.change_report({"k": {}}, {"k": {}})["changed"] == []
