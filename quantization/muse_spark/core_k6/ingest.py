#!/usr/bin/env python3
"""Place a published Muse checkpoint on six Sparks through SparkNest's API.

Weights stream directly from their immutable export into their assigned Spark's
managed store. There is no second raptor weight copy, archive registration,
rsync, custom serving protocol, or change to cluster configuration. Final
snapshot names and refs are published only after independent remote checksums.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlencode

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'quantization'))
if __package__ in (None, ''):
    __package__ = 'muse_spark.core_k6'
from kimi_k3.uniform_k2.storage import atomic_json, canonical, file_identity, sha256_file, verify_receipt
from . import publish
from .recipe import TARGET_REPO

HOSTS = ('ostrich', 'dodo', 'emu', 'kiwi', 'rhea', 'moa')
ADDRESSES = {host: f'10.55.0.{i + 1}' for i, host in enumerate(HOSTS)}
NODE_IDS = {host: i + 2 for i, host in enumerate(HOSTS)}
MOUNT = Path('/mnt/sparknest')
HUB = '/hf-home/hub'
CHUNK = 4 * 1024 * 1024
SCHEMA = 'cuteafd-muse-sparknest-ingest-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def safe_relative(value):
    if (not isinstance(value, str) or not value or '\\' in value or '\0' in value
            or PurePosixPath(value).is_absolute() or any(p in ('', '.', '..') for p in value.split('/'))):
        raise ValueError('unsafe publication relative path')
    return value


def safe_namespace(value):
    if not isinstance(value, str) or not value.startswith(HUB + '/') or posixpath.normpath(value) != value or '\0' in value:
        raise ValueError('unsafe SparkNest namespace')
    return value


def load_inputs(root, export=None):
    """Bind only remotely verified published files to the original owned export."""
    root = Path(root).expanduser().resolve(strict=True)
    plan = json.loads((root / 'plan.json').read_text())
    export = Path(export or plan['export']).expanduser().resolve(strict=True)
    identity = publish._complete_plan(export, root, TARGET_REPO)
    marker_path = root / 'publication-complete.json'
    marker = json.loads(marker_path.read_text())
    if marker.get('repo_id') != TARGET_REPO or marker.get('identity') != identity:
        raise ValueError('ingest requires the matching completed public upload')
    revision = marker.get('revision', '')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('invalid completed publication revision')
    snapshot = Path(marker['local_snapshot']).expanduser().absolute()
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.name != revision:
        raise ValueError('completed local snapshot is missing or unsafe')
    if snapshot.parent.name != 'snapshots' or snapshot.parent.parent.name != 'models--' + TARGET_REPO.replace('/', '--'):
        raise ValueError('completed local snapshot repository differs')
    files = []
    for directory, dirs, names in os.walk(snapshot, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in dirs):
            raise ValueError('symlink publication snapshot directory')
        for name in sorted(names):
            cached = Path(directory) / name
            relative = safe_relative(cached.relative_to(snapshot).as_posix())
            source = export / relative
            if (cached.is_symlink() or source.is_symlink() or not cached.is_file() or not source.is_file()
                    or not os.path.samefile(source, cached)):
                raise ValueError('published snapshot must hardlink the owned export')
            if relative.endswith('.safetensors'):
                if relative not in publish._expected_shards():
                    raise ValueError('unreviewed published weight shard')
                receipt = json.loads(publish._receipt_path(export, relative).read_text())
                if receipt.get('identity') != identity or receipt.get('filename') != relative:
                    raise ValueError('published weight receipt differs')
                verify_receipt(source, receipt)
                sha = receipt['sha256']
            else:
                sha = sha256_file(source)
            if not re.fullmatch(r'[0-9a-f]{64}', sha):
                raise ValueError('invalid published file checksum')
            files.append(dict(relative=relative, source=str(source), size=source.stat().st_size,
                sha256=sha, file_identity=file_identity(source), weight=relative.endswith('.safetensors')))
    if len(files) != marker.get('files'):
        raise ValueError('published snapshot file count differs')
    manifest = [[f['relative'], f['size'], f['sha256']] for f in sorted(files, key=lambda f: f['relative'])]
    if digest(manifest) != marker.get('manifest_sha256'):
        raise ValueError('published snapshot manifest differs')
    if {f['relative'] for f in files if f['weight']} != publish._expected_shards():
        raise ValueError('published checkpoint shard inventory differs')
    ref_source = snapshot.parent.parent / 'refs/main'
    if ref_source.is_symlink() or ref_source.read_bytes() != revision.encode():
        raise ValueError('completed local HF ref does not name the published revision')
    return dict(root=str(root), export=str(export), revision=revision, files=files,
        publication_marker_sha256=sha256_file(marker_path), manifest_sha256=marker['manifest_sha256'],
        identity=identity, ref_source=str(ref_source))


def make_plan(inputs):
    """Deterministic largest-first byte balance; stable host-order breaks ties."""
    revision = inputs['revision']
    repo = 'models--' + TARGET_REPO.replace('/', '--')
    snapshot = f'{HUB}/{repo}/snapshots/{revision}'
    prefix = 'muse-ingest-' + revision[:12] + '-' + inputs['manifest_sha256'][:12]
    totals = {host: 0 for host in HOSTS}
    rows = []
    for original in sorted(inputs['files'], key=lambda f: (-f['size'], f['relative'])):
        host = min(HOSTS, key=lambda h: (totals[h], HOSTS.index(h)))
        totals[host] += original['size']
        relative = safe_relative(original['relative'])
        namespace = safe_namespace(snapshot + '/' + relative)
        parent, leaf = posixpath.split(namespace)
        temporary = f'{parent}/.{leaf}.{prefix}-{original["sha256"][:16]}.incomplete'
        rows.append(dict(original, host=host, namespace=namespace, temporary=temporary))
    weight_hosts = {f['host'] for f in rows if f['weight']}
    if weight_hosts != set(HOSTS):
        raise ValueError('each of the six Sparks must receive a weight shard')
    ref_path = Path(inputs['ref_source'])
    ref = dict(relative='refs/main', source=str(ref_path), size=ref_path.stat().st_size,
        sha256=sha256_file(ref_path), file_identity=file_identity(ref_path), weight=False,
        host=HOSTS[0], namespace=f'{HUB}/{repo}/refs/main', is_ref=True)
    ref['temporary'] = f'{HUB}/{repo}/refs/.main.{prefix}-{ref["sha256"][:16]}.pending'
    return dict(schema=SCHEMA, repo_id=TARGET_REPO, revision=revision,
        publication_marker_sha256=inputs['publication_marker_sha256'],
        published_manifest_sha256=inputs['manifest_sha256'], producer=inputs['identity'],
        code_sha256=sha256_file(Path(__file__)), root=inputs['root'], export=inputs['export'],
        snapshot_namespace=snapshot, files=sorted(rows, key=lambda f: f['relative']),
        ref=ref, host_bytes=totals)


class APIError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f'SparkNest API returned HTTP {status}')


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, *, timeout=30):
        super().__init__('localhost', timeout=timeout)
        self.socket_path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


class NativeTransport:
    """Existing SparkNest UI transfer API; credentials never enter logs/argv."""
    def __init__(self, *, unix_socket='/srv/sparknest/api.sock', mount=MOUNT,
            upload_timeout=3600, ssh_timeout=600, settle_timeout=120,
            http_factory=http.client.HTTPConnection, unix_factory=UnixHTTPConnection):
        self.mount = Path(mount)
        self.upload_timeout, self.ssh_timeout = upload_timeout, ssh_timeout
        self.settle_timeout = settle_timeout
        self._http_factory = http_factory
        self._unix_factory = unix_factory
        self.unix_socket = unix_socket
        self._token = None

    def __repr__(self):
        return '<native SparkNest transport>'

    def _local(self, method, endpoint, body=None):
        connection = self._unix_factory(self.unix_socket, timeout=30)
        try:
            data = None if body is None else canonical(body).encode()
            connection.request(method, endpoint, body=data,
                headers={} if data is None else {'Content-Type': 'application/json'})
            return self._response(connection)
        finally:
            connection.close()

    @staticmethod
    def _response(connection):
        response = connection.getresponse()
        content = response.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            raise ValueError('oversized SparkNest control response')
        if not 200 <= response.status < 300:
            raise APIError(response.status)
        return json.loads(content)

    def authenticate(self):
        value = self._local('GET', '/v1/web')
        token = value.get('token')
        if not isinstance(token, str) or not re.fullmatch(r'[0-9a-f]{40}', token):
            raise ValueError('SparkNest did not supply a valid API credential')
        self._token = token

    def _connection(self, host):
        if host not in ADDRESSES:
            raise ValueError('Spark outside the ingest allowlist')
        if self._token is None:
            raise ValueError('SparkNest transport is not authenticated')
        return self._http_factory(ADDRESSES[host], 7401, timeout=30)

    def _remote(self, host, method, endpoint, body=None):
        connection = self._connection(host)
        try:
            connection.request(method, endpoint,
                body=None if body is None else canonical(body).encode(),
                headers={'Authorization': 'Bearer ' + self._token, 'Content-Type': 'application/json'})
            return self._response(connection)
        finally:
            connection.close()

    def readiness(self):
        self.authenticate()
        result = {}
        for host in HOSTS:
            status = self._remote(host, 'GET', '/v1/status')
            if status.get('me') != NODE_IDS[host] or status.get('serving') is not True:
                raise ValueError(f'assigned Spark is not serving: {host}')
            result[host] = 'ready'
        if not self.mount.is_mount() or not os.access(self.mount / HUB.lstrip('/'), os.R_OK | os.W_OK | os.X_OK):
            raise ValueError('managed HF hub is not mounted and writable')
        return result

    def metadata(self, namespace, *, follow=True):
        namespace = safe_namespace(namespace)
        for _ in range(16):
            try:
                value = self._local('GET', '/v1/ls?' + urlencode({'path': namespace}))
            except APIError as error:
                if error.status == 404:
                    return None
                raise
            entries = value.get('entries', [])
            if (len(entries) != 1 or entries[0].get('kind') == 'directory'
                    or entries[0].get('name') != posixpath.basename(namespace)):
                raise ValueError('expected one regular managed file')
            entry = dict(entries[0], resolved_namespace=namespace)
            if entry.get('kind') != 'symlink' or not follow:
                return entry
            target = entry.get('target')
            if not isinstance(target, str):
                raise ValueError('invalid managed symlink')
            namespace = safe_namespace(posixpath.normpath(posixpath.join(posixpath.dirname(namespace), target)))
        raise ValueError('managed symlink chain exceeds the bound')

    def stable(self, namespace, *, follow=True):
        deadline = time.monotonic() + self.settle_timeout
        while True:
            entry = self.metadata(namespace, follow=follow)
            if entry is None or not entry.get('writing_on'):
                return entry
            if time.monotonic() >= deadline:
                raise TimeoutError('managed file remains owned by a live writer; left untouched')
            time.sleep(1)

    def mkdir(self, host, namespace):
        self._remote(host, 'POST', '/v1/mkdir', {'path': safe_namespace(namespace)})

    def upload(self, item):
        connection = self._connection(item['host'])
        source = Path(item['source'])
        verify_receipt(source, item)
        deadline = time.monotonic() + self.upload_timeout
        checksum = hashlib.sha256()
        sent = 0
        try:
            endpoint = '/v1/upload?' + urlencode({'path': safe_namespace(item['temporary'])})
            connection.putrequest('PUT', endpoint)
            connection.putheader('Authorization', 'Bearer ' + self._token)
            connection.putheader('Content-Type', 'application/octet-stream')
            connection.putheader('Content-Length', str(item['size']))
            connection.endheaders()
            with source.open('rb') as stream:
                while data := stream.read(CHUNK):
                    if time.monotonic() > deadline:
                        raise TimeoutError('bounded SparkNest upload expired')
                    connection.send(data)
                    checksum.update(data)
                    sent += len(data)
            verify_receipt(source, item)
            if sent != item['size'] or checksum.hexdigest() != item['sha256']:
                raise ValueError('owned source bytes changed during upload')
            result = self._response(connection)
            if result.get('path') != item['temporary'] or result.get('bytes') != sent:
                raise ValueError('SparkNest upload acknowledgement differs')
            return result
        finally:
            # A cancellation closes the body stream. Recovery must observe
            # writing_on=null before it can inspect or remove its own partial.
            connection.close()

    def remote_sha256(self, host, namespace):
        if host not in HOSTS:
            raise ValueError('hash host outside ingest allowlist')
        mounted = self.mount / safe_namespace(namespace).lstrip('/')
        command = shlex.join(['sha256sum', '--', str(mounted)])
        result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3', host, command],
            text=True, capture_output=True, check=True, timeout=self.ssh_timeout)
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)\n?', result.stdout)
        if match is None or match[2] != str(mounted):
            raise ValueError('remote checksum response differs')
        return match[1]

    def seal(self, namespace):
        subprocess.run(['nest', 'seal', safe_namespace(namespace)], check=True,
            stdout=subprocess.DEVNULL, timeout=120)

    def remove_temporary(self, namespace):
        subprocess.run(['nest', 'rm', safe_namespace(namespace)], check=True,
            stdout=subprocess.DEVNULL, timeout=120)

    def replicate_asset(self, namespace, host):
        if host not in HOSTS:
            raise ValueError('asset destination outside ingest allowlist')
        subprocess.run(['nest', 'replicate', safe_namespace(namespace), '--hosts', host, '--wait'],
            check=True, stdout=subprocess.DEVNULL, timeout=600)

    def rename(self, temporary, final):
        source = self.mount / safe_namespace(temporary).lstrip('/')
        target = self.mount / safe_namespace(final).lstrip('/')
        # Linux renameat2(..., RENAME_NOREPLACE), implemented by SparkNest's
        # FUSE adapter. Unlike os.rename, this cannot replace a racing final.
        libc = ctypes.CDLL(None, use_errno=True)
        function = libc.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        if function(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError('managed final appeared during namespace publication')
            raise OSError(error, os.strerror(error))

    def read_ref(self, namespace):
        entry = self.metadata(namespace)
        if entry is None:
            return None
        if entry.get('kind') != 'regular' or entry.get('writing_on') or entry.get('size', 257) > 256:
            raise ValueError('managed HF ref is unsafe or still being written')
        path = self.mount / safe_namespace(entry['resolved_namespace']).lstrip('/')
        try:
            with path.open('rb') as stream:
                data = stream.read(257)
        except FileNotFoundError:
            return None
        if len(data) > 256:
            raise ValueError('oversized managed HF ref')
        return data

    def inventory(self, namespace):
        root = self.mount / safe_namespace(namespace).lstrip('/')
        files = set()
        for directory, dirs, leaves in os.walk(root, followlinks=False):
            if any((Path(directory) / name).is_symlink() for name in dirs):
                raise ValueError('unexpected managed snapshot directory symlink')
            files.update((Path(directory) / name).relative_to(root).as_posix() for name in leaves)
        return files

    def offline_snapshot(self, repo_id, revision):
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id, revision=revision, cache_dir=str(self.mount / HUB.lstrip('/')),
            local_files_only=True)
        expected = self.mount / HUB.lstrip('/') / ('models--' + repo_id.replace('/', '--')) / 'snapshots' / revision
        if Path(path).absolute() != expected:
            raise ValueError('managed HF offline lookup resolved a different snapshot')
        return str(expected)


class Ingestor:
    def __init__(self, plan, transport, *, state_root=None):
        self.plan, self.transport = plan, transport
        self.root = Path(state_root or plan['root'])
        self.state_path = self.root / 'ingest-state.json'
        self.plan_hash = digest(plan)
        self.state = dict(schema=SCHEMA, plan_sha256=self.plan_hash, files={}, intents={}, started=now())
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state.get('schema') != SCHEMA or self.state.get('plan_sha256') != self.plan_hash:
                raise ValueError('ingest recovery state differs from immutable placement plan')

    def save(self):
        self.state['updated'] = now()
        atomic_json(self.state_path, self.state)

    def event(self, event, **fields):
        # Only scalar evidence and public paths; never transport/credential repr.
        print(canonical(dict(event=event, time=now(), **fields)), flush=True)

    def _geometry(self, item, entry):
        if entry is None or entry.get('kind') != 'regular' or entry.get('writing_on'):
            raise ValueError('managed file is absent, nonregular or still being written')
        hosts = entry.get('hosts', [])
        if item['weight'] and (len(hosts) != 1 or hosts != [item['host']]):
            raise ValueError('weight shard must have exactly its assigned Spark copy')
        if not item['weight'] and item['host'] not in hosts:
            raise ValueError('assigned Spark lacks the managed asset')
        if entry.get('size') != item['size']:
            raise ValueError('managed file size differs from published artifact')

    def _verify(self, item, namespace, *, existing=False):
        entry = self.transport.stable(namespace, follow=namespace != item['temporary'])
        if entry is None:
            raise ValueError('managed file vanished before verification')
        preexisting = sorted(entry.get('hosts', [])) if existing else []
        if not item['weight'] and item['host'] not in entry.get('hosts', []):
            if entry.get('kind') != 'regular' or entry.get('writing_on') or entry.get('size') != item['size']:
                raise ValueError('existing managed asset geometry differs')
            # The assigned Spark can read an existing shared object through its
            # mount before receiving its own copy. Preserve every other owner.
            if self.transport.remote_sha256(item['host'], namespace) != item['sha256']:
                raise ValueError('existing shared asset checksum differs; left untouched')
            self.transport.replicate_asset(entry.get('resolved_namespace', namespace), item['host'])
            entry = self.transport.stable(namespace, follow=namespace != item['temporary'])
        self._geometry(item, entry)
        if self.transport.remote_sha256(item['host'], namespace) != item['sha256']:
            raise ValueError('managed file checksum differs from published artifact')
        if not entry.get('sealed') and not item.get('is_ref'):
            self.transport.seal(entry.get('resolved_namespace', namespace))
            entry = self.transport.stable(namespace, follow=namespace != item['temporary'])
        self._geometry(item, entry)
        if not entry.get('sealed') and not item.get('is_ref'):
            raise ValueError('managed file did not become immutable')
        return dict(sha256=item['sha256'], size=item['size'], host=item['host'],
            namespace=item['namespace'], owners=sorted(entry['hosts']), weight=item['weight'],
            preexisting_owners=preexisting)

    def _discard_partial(self, item):
        temporary = item['temporary']
        intent = self.state['intents'].get(temporary)
        if not intent or intent.get('relative') != item['relative'] or intent.get('host') != item['host']:
            raise ValueError('unrecognized staging file; left untouched')
        entry = self.transport.stable(temporary, follow=False)
        if entry is None:
            return
        if (entry.get('kind') != 'regular' or entry.get('writing_on')
                or entry.get('hosts') != [item['host']]):
            raise ValueError('staging owner differs or remains live; left untouched')
        self.transport.remove_temporary(temporary)
        if self.transport.metadata(temporary, follow=False) is not None:
            raise ValueError('owned incomplete staging file was not removed')

    def ensure(self, item):
        final = self.transport.metadata(item['namespace'])
        if final is not None:
            record = self._verify(item, item['namespace'], existing=True)
            if self.transport.metadata(item['temporary'], follow=False) is not None:
                self._discard_partial(item)
            return record
        temporary = self.transport.metadata(item['temporary'], follow=False)
        if temporary is not None:
            if item['temporary'] not in self.state['intents']:
                raise ValueError('staging file predates this committed task intent; left untouched')
            entry = self.transport.stable(item['temporary'], follow=False)
            if entry is not None:
                same = (entry.get('kind') == 'regular' and entry.get('hosts') == [item['host']]
                    and entry.get('size') == item['size'] and not entry.get('writing_on'))
                if same:
                    same = self.transport.remote_sha256(item['host'], item['temporary']) == item['sha256']
                if not same:
                    self._discard_partial(item)
                    temporary = None
        if temporary is None:
            self.transport.mkdir(item['host'], posixpath.dirname(item['namespace']))
            self.state['intents'][item['temporary']] = dict(relative=item['relative'], host=item['host'], started=now())
            self.save()  # Durable ownership precedes the first remote write.
            self.transport.upload(item)
        self._verify(item, item['temporary'])
        try:
            self.transport.rename(item['temporary'], item['namespace'])
        except FileExistsError:
            # A racing final is never overwritten. It must independently pass
            # the exact same immutable-artifact and ownership checks.
            record = self._verify(item, item['namespace'], existing=True)
            self._discard_partial(item)
            return record
        return self._verify(item, item['namespace'])

    def run(self):
        ref = self.transport.read_ref(self.plan['ref']['namespace'])
        if ref not in (None, self.plan['revision'].encode()):
            raise ValueError('managed refs/main names another revision; left untouched')
        for item in self.plan['files']:
            record = self.ensure(item)
            previous = self.state['files'].get(item['relative'])
            if previous:
                record['preexisting_owners'] = previous.get('preexisting_owners', [])
            self.state['files'][item['relative']] = record
            self.save()
            self.event('managed_file_verified', relative=item['relative'], host=item['host'],
                bytes=item['size'], completed=len(self.state['files']), total=len(self.plan['files']))
        expected = {item['relative'] for item in self.plan['files']}
        if self.transport.inventory(self.plan['snapshot_namespace']) != expected:
            raise ValueError('managed snapshot contains unexpected or missing files')
        if set(self.state['files']) != expected:
            raise ValueError('managed completion evidence is incomplete')
        # Ref publication is deliberately separated by the complete-file barrier.
        self.state['snapshot_verified'] = now()
        self.save()
        self.state['ref'] = self.ensure(self.plan['ref'])
        self.save()
        if self.transport.read_ref(self.plan['ref']['namespace']) != self.plan['revision'].encode():
            raise ValueError('managed HF ref does not name the published revision')
        snapshot = self.transport.offline_snapshot(self.plan['repo_id'], self.plan['revision'])
        # Re-read metadata after the ref barrier. Weights remain one copy;
        # shared assets may retain unrelated preexisting owners.
        for item in self.plan['files']:
            entry = self.transport.stable(item['namespace'])
            self._geometry(item, entry)
            if not entry.get('sealed'):
                raise ValueError('managed final is no longer immutable')
            self.state['files'][item['relative']]['owners'] = sorted(entry['hosts'])
        result = dict(schema=SCHEMA, plan_sha256=self.plan_hash, repo_id=self.plan['repo_id'],
            revision=self.plan['revision'], published_manifest_sha256=self.plan['published_manifest_sha256'],
            managed_snapshot=snapshot, files=self.state['files'], ref=self.state['ref'],
            file_count=len(self.plan['files']), weight_shards=sum(item['weight'] for item in self.plan['files']),
            weight_copies=sum(len(row['owners']) for row in self.state['files'].values() if row['weight']),
            snapshot_bytes=sum(item['size'] for item in self.plan['files']), host_bytes=self.plan['host_bytes'],
            finished=now())
        if result['weight_copies'] != result['weight_shards']:
            raise ValueError('managed checkpoint has more than one weight copy')
        atomic_json(self.root / 'ingest-complete.json', result)
        self.event('ingest_complete', file_count=result['file_count'], weight_shards=result['weight_shards'],
            weight_copies=result['weight_copies'], bytes=result['snapshot_bytes'])
        return result


def preflight(plan, transport):
    """Read-only: no directory, plan, lock, remote file or ref creation."""
    for item in [*plan['files'], plan['ref']]:
        verify_receipt(Path(item['source']), item)
        if not os.access(item['source'], os.R_OK):
            raise ValueError('published source is not readable')
    ready = transport.readiness()
    ref = transport.read_ref(plan['ref']['namespace'])
    if ref not in (None, plan['revision'].encode()):
        raise ValueError('existing managed refs/main names a different revision')
    path = Path(plan['root']) / 'ingest-plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('immutable ingest plan differs; do not reuse its cached state')
    return dict(status='passed', repo_id=plan['repo_id'], revision=plan['revision'],
        files=len(plan['files']), bytes=sum(item['size'] for item in plan['files']),
        host_bytes=plan['host_bytes'], sparks=ready, plan_sha256=digest(plan))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--preflight', '--dry-run', dest='preflight', action='store_true')
    parser.add_argument('--upload-timeout', type=int, default=3600)
    parser.add_argument('--ssh-timeout', type=int, default=600)
    parser.add_argument('--settle-timeout', type=int, default=120)
    parser.add_argument('--run-timeout', type=int, default=86400)
    args = parser.parse_args()
    if min(args.upload_timeout, args.ssh_timeout, args.settle_timeout, args.run_timeout) <= 0:
        parser.error('all operation timeouts must be positive')
    plan = make_plan(load_inputs(args.root, args.export))
    transport = NativeTransport(upload_timeout=args.upload_timeout, ssh_timeout=args.ssh_timeout,
        settle_timeout=args.settle_timeout)
    checked = preflight(plan, transport)
    if args.preflight:
        print(canonical(checked), flush=True)
        return
    root = Path(plan['root'])
    with (root / 'ingest.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another ingest supervisor holds this task') from error
        path = root / 'ingest-plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('ingest plan changed during admission')
        if not path.exists():
            atomic_json(path, plan)
        def interrupted(signum, _frame):
            raise InterruptedError(f'ingest interrupted by signal {signum}; resume from durable plan')
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, interrupted)
        def expired(_signum, _frame):
            raise TimeoutError('bounded ingest run expired; resume from durable plan')
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(args.run_timeout)
        try:
            Ingestor(plan, transport).run()
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    main()
