from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ernie55_flash.storage import RoutedWorkspace, Stages, commit_wavefront


class Routes:
    latent = torch.arange(30, dtype=torch.float32).reshape(10, 3) / 7
    bounds = [0, 10, 12, 12]

    def expert(self, index):
        rows = (torch.arange(10), torch.tensor([2, 6]), torch.tensor([], dtype=torch.long))[index]
        return rows, (rows + 1) * .125

    def calibration(self, index):
        rows, gamma = self.expert(index)
        return rows, gamma, {"natural_rows": len(rows)}


def test_ordered_replay_uses_bounded_buffers_for_hot_and_cold_experts():
    routes = Routes()
    ws = RoutedWorkspace(routes, "cpu", chunk_rows=3, stage_bytes=36)
    pointers = [t.data_ptr() for t in (ws.sum, ws.staging.host_chunk, ws.staging.staged)]
    visits = []
    def load(i):
        visits.append(i)
        return lambda inputs: inputs + (1 + i)
    expected = torch.zeros_like(routes.latent)
    for i in (0, 1):
        rows, gamma = routes.expert(i)
        expected[rows] += (routes.latent[rows] + (1 + i)) * gamma[:, None]
    for _ in range(2):
        assert torch.equal(ws.replay(load), expected)
        chunks, _, _ = ws.calibration_inputs(0)
        assert torch.equal(torch.cat([chunk.clone() for chunk in chunks(10)]), routes.latent)
    assert visits == [0, 1, 0, 1]
    assert pointers == [t.data_ptr() for t in (ws.sum, ws.staging.host_chunk, ws.staging.staged)]


def test_only_summed_wavefront_survives_resume(tmp_path):
    store = Stages(tmp_path, {"corpus": "one"})
    def interrupted(start):
        for index in range(start, 35):
            if index == 34:
                raise OSError("interrupted")
            yield index, torch.tensor([index], dtype=torch.float32)
    with pytest.raises(OSError):
        commit_wavefront(store, "next", interrupted, 35)
    starts = []
    def finish(start):
        starts.append(start)
        for i in range(start, 35):
            yield i, torch.tensor([i], dtype=torch.float32)
    reopened = Stages(tmp_path, store.identity)
    result = commit_wavefront(reopened, "next", finish, 35)
    assert starts == [32]
    assert {e["hash_algorithm"] for e in result["entries"]} == {"xxh3-128"}
    assert len({e["filename"] for e in result["entries"]}) == 2
    assert commit_wavefront(reopened, "next", finish, 35) == result
    assert starts == [32]
    assert reopened.read_batch("next", 34)["state"].item() == 34
    with pytest.raises(ValueError, match="identity"):
        Stages(tmp_path, {"corpus": "other"}).load("next")
    with pytest.raises(ValueError, match="non-finite"):
        commit_wavefront(store, "bad", lambda _: [(0, torch.tensor([float("nan")]))], 1)
