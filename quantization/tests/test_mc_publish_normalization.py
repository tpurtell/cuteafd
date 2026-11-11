"""Only additive literal LFS rules may differ from approved Hub trees."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mc_publish import verified_remote_tree


def git_entry(raw):
    return dict(bytes=len(raw), blob=hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest())


@pytest.fixture
def normalized(tmp_path, monkeypatch):
    import huggingface_hub
    import upload_model
    original = b'*.safetensors filter=lfs diff=lfs merge=lfs -text\n'
    expected = {'.gitattributes': git_entry(original),
                'index.json': dict(bytes=123, blob='a' * 64),
                'config.json': dict(bytes=12, blob='b' * 40)}
    actual = dict(expected)
    path = tmp_path / 'attributes'
    raw = original + b'index.json filter=lfs diff=lfs merge=lfs -text\n'
    path.write_bytes(raw)
    actual['.gitattributes'] = git_entry(raw)
    monkeypatch.setattr(upload_model, '_remote', lambda *args: actual)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda *args, **kwargs: path)
    return SimpleNamespace(token=False), expected, actual, original, path


def test_accepts_server_added_lfs_rule_and_records_delta(normalized):
    api, expected, actual, original, _ = normalized
    published, delta = verified_remote_tree(api, 'repo', 'commit', expected, original)
    assert published == actual
    assert delta['approved'] == expected['.gitattributes']
    assert delta['published'] == actual['.gitattributes']
    assert delta['added_lfs_rules'] == ['index.json filter=lfs diff=lfs merge=lfs -text']


@pytest.mark.parametrize('delta', [
    b'*.json filter=lfs diff=lfs merge=lfs -text\n',
    b'config.json filter=lfs diff=lfs merge=lfs -text\n',
    b'missing.json filter=lfs diff=lfs merge=lfs -text\n',
    b'index.json filter=lfs diff=lfs merge=lfs -text\n' * 2,
    b'index.json filter=lfs diff=lfs merge=lfs -text',
    b'index.json -filter -diff -merge text\n',
])
def test_rejects_other_attribute_changes(normalized, delta):
    api, expected, actual, original, path = normalized
    raw = original + delta
    path.write_bytes(raw)
    actual['.gitattributes'] = git_entry(raw)
    with pytest.raises(ValueError, match='server-normalized attributes'):
        verified_remote_tree(api, 'repo', 'commit', expected, original)


def test_rejects_rewritten_original_rules(normalized):
    api, expected, actual, original, path = normalized
    raw = path.read_bytes().replace(b'*.safetensors', b'*.bin')
    path.write_bytes(raw)
    actual['.gitattributes'] = git_entry(raw)
    with pytest.raises(ValueError, match='server-normalized attributes'):
        verified_remote_tree(api, 'repo', 'commit', expected, original)


@pytest.mark.parametrize('lost_reply', [False, True])
def test_publish_and_lost_reply_accept_only_normalized_attributes(tmp_path, monkeypatch, lost_reply):
    import huggingface_hub
    from test_mc_package import publisher_fixture
    publisher, output, state, packet, api = publisher_fixture(tmp_path, monkeypatch)
    api.token = False
    raw = (output / '.gitattributes').read_bytes() + b'config.json filter=lfs diff=lfs merge=lfs -text\n'
    path = tmp_path / 'normalized-attributes'
    path.write_bytes(raw)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda *args, **kwargs: path)
    create = api.create_commit

    def normalized_commit(*args, **kwargs):
        result = create(*args, **kwargs)
        api.files['.gitattributes'] = git_entry(raw)
        if lost_reply:
            raise RuntimeError('lost normalized commit reply')
        return result

    monkeypatch.setattr(api, 'create_commit', normalized_commit)
    if lost_reply:
        with pytest.raises(RuntimeError, match='lost normalized commit reply'):
            publisher.publish(output, state, packet, api=api)
    result = publisher.publish(output, state, packet, api=api)
    assert result['status'] == 'uploaded'
    assert result['files']['.gitattributes'] == git_entry(raw)
    assert result['server_normalized_attributes']['added_lfs_rules'] == [
        'config.json filter=lfs diff=lfs merge=lfs -text']
    assert api.commits == 1


@pytest.mark.parametrize('change', ['payload', 'missing', 'extra'])
def test_keeps_every_other_file_strict(normalized, change):
    api, expected, actual, original, _ = normalized
    if change == 'payload':
        actual['index.json'] = dict(bytes=123, blob='c' * 64)
    elif change == 'missing':
        del actual['config.json']
    else:
        actual['extra'] = dict(bytes=1, blob='d' * 40)
    with pytest.raises(ValueError, match='inventory differs'):
        verified_remote_tree(api, 'repo', 'commit', expected, original)
