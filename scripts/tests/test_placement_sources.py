"""placement-sources.json covers every engine file the placement exporters can read.

The compiled tier of a publication's fingerprint hashes exactly the files in
scripts/build/placement-sources.json. This test makes that list sound: it walks
the full static import closure (module level and function level, transitively)
of every placement exporter and of the comparison worker that runs them, plus
every repo file those modules name by a literal path, and fails if any is
missing from the list. An exporter that newly imports or opens a repo file
fails here until the list is updated, so files outside the list (for example
python/tools/qualify) cannot change compiled programs unnoticed.
"""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = json.loads((ROOT / 'scripts/build/placement-sources.json').read_text())
JOBS = json.loads((ROOT / 'scripts/build/placement-jobs.json').read_text())
# The worker puts these on sys.path before running an exporter (compare-sm-exports.py worker()).
SEARCH = (ROOT / 'python/tools/lib', ROOT / 'python/tools/aot')
ORCHESTRATION = ('scripts/build/compare-sm-exports.py', 'scripts/build/export-placement-inputs.py',
                 'scripts/build/kernel-records.py')
# Repo-relative path literals the closure names that are not compile inputs, each for a stated reason.
NOT_INPUTS = (
    'third_party/',                                   # the fork: pinned by sparkinfer.lock.json's tree hash
    'scripts/build/assert-build-filesystem.py',       # output-path safety check
    'scripts/build/export-cache.py',                  # cache wrapper around the same worker
    'scripts/build/placement-',                       # placement-sources/jobs.json: hashed by name in the tier
    # Job selection: their only effect is the job list, hashed as placement-jobs.json (selection tier too).
    'benchmarks/', 'examples/configs', 'scripts/lib/', 'scripts/build/release-scope.py',
    'rust/crates/cuteafd-loader/src/serving_capacity/spark_memory.json',
    # Fallback for baselines predating python/tools/lib/spark_packages.py (routed_package_layouts).
    'rust/crates/cuteafd-loader/src/plan/experts.rs',
    # Paths used only by non-placement export modes of the same modules (CMake, audio/vision towers, scope CLI).
    'native/families/', 'native/shared/', 'python/reference/', 'scripts/bench/',
)
PREFIXES = ('python/', 'scripts/', 'rust/', 'native/', 'examples/', 'benchmarks/', 'docker/')


def module_path(name, directory):
    """A repo-local module file for an absolute import name, or None (stdlib, fork, torch...)."""
    head = name.split('.')[0]
    for base in (directory, *SEARCH):
        for candidate in (base / f'{head}.py', base / head / '__init__.py'):
            if candidate.is_file():
                parts = name.split('.')
                target = base.joinpath(*parts)
                for full in (target.with_suffix('.py'), target / '__init__.py'):
                    if full.is_file():
                        return full
                return candidate
    return None


def literal_paths(tree):
    """Repo-relative paths named by string constants, joined with / or not."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if value.startswith(PREFIXES) and ' ' not in value and '\n' not in value:
                found.add(value.rstrip('/'))
    return found


def closure():
    """(modules, named paths) reachable from every placement exporter and the worker."""
    entries = {ROOT / path for path in ORCHESTRATION}
    for kind in ('all', 'v4'):
        for _, exporter, _ in JOBS[kind]:
            entries.add(ROOT / 'python/tools/aot' / exporter)
    pending, modules, named = list(entries), set(), set()
    while pending:
        path = pending.pop()
        if path in modules:
            continue
        modules.add(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        named |= literal_paths(tree)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module, *(f'{node.module}.{alias.name}' for alias in node.names)]
            elif isinstance(node, ast.ImportFrom) and node.level:
                names = [(node.module or '')]
            for name in names:
                found = module_path(name, path.parent) if name else None
                if found is not None and found.is_relative_to(ROOT) and 'third_party' not in found.parts:
                    pending.append(found)
    return modules, named


def relative(path):
    return str(path.relative_to(ROOT))


def test_placement_job_specs_match_what_the_exporters_derive():
    spec = importlib.util.spec_from_file_location('placement_jobs_check', ROOT / 'scripts/build/export-placement-inputs.py')
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    assert JOBS == publisher.derived_jobs(), (
        'scripts/build/placement-jobs.json is stale: python3 scripts/build/export-placement-inputs.py --write-jobs')


@pytest.mark.parametrize('kind', ['all', 'v4'])
def test_compiled_tier_covers_the_full_static_closure_of_the_exporters(kind):
    modules, named = closure()
    listed = set(SOURCES['compiled'][kind])
    missing_modules = sorted(relative(path) for path in modules if relative(path) not in listed)
    assert not missing_modules, f'add to placement-sources.json compiled.{kind}: {missing_modules}'
    files = {path for path in named if (ROOT / path).is_file()}
    missing_files = sorted(path for path in files if path not in listed and not path.startswith(NOT_INPUTS))
    assert not missing_files, f'exporters name repo files outside compiled.{kind}: {missing_files}'


def test_listed_sources_exist_and_tiers_are_disjoint():
    for kind in ('all', 'v4'):
        compiled = SOURCES['compiled'][kind]
        selection = SOURCES['selection'][kind]['files']
        assert compiled == sorted(set(compiled)) and all((ROOT / path).is_file() for path in compiled + selection)
        assert not set(compiled) & set(selection)
    assert set(SOURCES['compiled']['v4']) <= set(SOURCES['compiled']['all'])


def test_qualify_bench_and_hf_tools_are_outside_the_closure():
    modules, _ = closure()
    assert not [path for path in modules if path.is_relative_to(ROOT / 'python/tools') and
                path.relative_to(ROOT / 'python/tools').parts[0] in ('qualify', 'bench', 'hf')]


def test_closure_walk_sees_module_level_function_level_and_transitive_imports(tmp_path, monkeypatch):
    lib, aot = tmp_path / 'python/tools/lib', tmp_path / 'python/tools/aot'
    for directory in (lib, aot):
        directory.mkdir(parents=True)
    (aot / 'export_x_aot.py').write_text('import helper_a\ndef run():\n    from helper_b import value\n')
    (lib / 'helper_a.py').write_text('import os\nimport helper_c\n')
    (lib / 'helper_b.py').write_text('PATH = "scripts/build/layout.json"\n')
    (lib / 'helper_c.py').write_text('')
    (lib / 'unused.py').write_text('')
    monkeypatch.setattr(__import__('sys').modules[__name__], 'ROOT', tmp_path)
    monkeypatch.setattr(__import__('sys').modules[__name__], 'SEARCH', (lib, aot))
    monkeypatch.setattr(__import__('sys').modules[__name__], 'ORCHESTRATION', ())
    monkeypatch.setattr(__import__('sys').modules[__name__], 'JOBS', {'all': [['x', 'export_x_aot.py', []]], 'v4': []})
    modules, named = closure()
    assert {relative(path) for path in modules} == {'python/tools/aot/export_x_aot.py', 'python/tools/lib/helper_a.py',
                                                    'python/tools/lib/helper_b.py', 'python/tools/lib/helper_c.py'}
    assert 'scripts/build/layout.json' in named


def _publisher():
    spec = importlib.util.spec_from_file_location('reattest_publisher', ROOT / 'scripts/build/export-placement-inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_publication(publisher, path, fingerprint):
    (path / 'share').mkdir(parents=True)
    (path / 'share/PROGRAMS.json').write_text('{"programs": [1]}')
    (path / 'CHANGES.json').write_text('{"leaks": []}')
    files = {'share/PROGRAMS.json': publisher.compare.sha((path / 'share/PROGRAMS.json').read_bytes())}
    lock = json.loads((ROOT / 'third_party/sparkinfer.lock.json').read_text())
    (path / 'INPUTS.json').write_text(json.dumps({'schema': 1, 'engine_commit': 'a' * 40, 'sparkinfer_revision': lock['revision'],
                                                  'source_tree_sha256': lock['source_tree_sha256'], **fingerprint,
                                                  'exports': [], 'files': files, 'program_records': {}}, indent=2))


def test_reattest_copies_compiled_bytes_and_records_the_checkouts_selection_tier(tmp_path, monkeypatch):
    publisher = _publisher()
    monkeypatch.setattr(publisher.compare, 'load_file', lambda name, path: type('F', (), {'check_path': staticmethod(lambda p: None)}))
    ours = publisher.source_hashes('all')
    source, output = tmp_path / 'store' / ('a' * 40), tmp_path / 'store' / ('b' * 40)
    _source_publication(publisher, source, {**ours, 'selection_sources_sha256': {'examples/configs/old.config': 'x'}})
    publisher.reattest(output, source)
    metadata = json.loads((output / 'INPUTS.json').read_text())
    assert metadata['selection_sources_sha256'] == ours['selection_sources_sha256']
    assert metadata['engine_sources_sha256'] == ours['engine_sources_sha256']
    assert metadata['reattested_from']['engine_commit'] == 'a' * 40
    assert (output / 'share/PROGRAMS.json').read_bytes() == (source / 'share/PROGRAMS.json').read_bytes()
    assert metadata['files'] == json.loads((source / 'INPUTS.json').read_text())['files']
    # A different compiled tier, a modified source file or a current link is refused.
    for case in ('compiled', 'modified', 'current'):
        other = tmp_path / case / ('a' * 40)
        fingerprint = {**ours, 'engine_sources_sha256': {'x': 'y'}} if case == 'compiled' else ours
        _source_publication(publisher, other, fingerprint)
        if case == 'modified':
            (other / 'share/PROGRAMS.json').write_text('tampered')
        target = tmp_path / case / ('current' if case == 'current' else 'c' * 40)
        with pytest.raises(ValueError, match='compiled tier differs|differs from its INPUTS|land-pr'):
            publisher.reattest(target, other)
        assert not (target / 'INPUTS.json').exists()
