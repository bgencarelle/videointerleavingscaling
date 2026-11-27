import importlib.util
import sys

import pytest
import torch

from faceloop.models import ModelManager
from faceloop.models.irn import IRN, Engine, _Down, _Up

from test_models import _fake_irn_weights


def _irn(tmp_path, scale=2):
    _fake_irn_weights(tmp_path, scale)
    return ModelManager(tmp_path).get_irn(f"irn-x{scale}", torch.device("cpu"))


def test_export_modules_match_the_torch_path(tmp_path):
    m = _irn(tmp_path, 4)
    x = torch.rand(1, 3, 32, 48)
    assert torch.allclose(_Down(m.net)(x), m._down_raw(x), atol=1e-6)
    lr = torch.rand(1, 3, 8, 12)
    want = m._up_raw(lr)
    assert torch.allclose(_Up(__import__("copy").deepcopy(m.net), 4, 8, 12)(lr), want, atol=1e-6)
    # traced at a fixed size (as for Core ML) they still match
    tr = torch.jit.trace(_Down(m.net).eval(), x, check_trace=False)
    assert torch.allclose(tr(x), m._down_raw(x), atol=1e-6)


def test_fastest_matching_engine_wins(tmp_path, monkeypatch):
    m = _irn(tmp_path)
    calls = []

    def cands(self, d, h, w):
        good = lambda: Engine("fast", "", lambda x: (calls.append("fast"), self._down_raw(x) if d == "down" else self._up_raw(x))[1])
        wrong = lambda: Engine("wrong", "", lambda x: torch.zeros_like(self._down_raw(x)))
        broken = lambda: (_ for _ in ()).throw(RuntimeError("no Neural Engine here"))
        yield "wrong", "wrong one", wrong
        yield "broken", "broken one", broken
        yield "fast", "the fast one", good
    monkeypatch.setattr(IRN, "_candidates", cands)
    # torch is made slow so a matching candidate wins
    real = IRN._torch_engine

    def slow_torch(self, d):
        e = real(self, d)
        f = e.fn
        e.fn = lambda x: (__import__("time").sleep(0.05), f(x))[1]
        return e
    monkeypatch.setattr(IRN, "_torch_engine", slow_torch)
    x = torch.rand(3, 3, 24, 32)
    y = m.down(x)
    assert y.shape == (3, 3, 12, 16)
    eng = m.engines[("down", 24, 32)]
    assert eng.name == "fast"
    note = m.notes[-1]
    assert "the fast one" in note and "wrong one: off by" in note and "broken one: couldn't run" in note
    n = len(calls)
    m.down(x)  # chosen once per size, then reused
    assert len(calls) == n + 1 and len(m.notes) == 1

    # a new process (new manager) reuses the saved choice without testing again
    assert (tmp_path / "irn" / "coreml" / "engines.json").exists()
    m2 = ModelManager(tmp_path).get_irn("irn-x2", torch.device("cpu"))
    timed = []
    monkeypatch.setattr(IRN, "_choose_from", lambda *a: timed.append(1))
    assert torch.allclose(m2.down(x), y, atol=1e-6)
    assert not timed and m2.engines[("down", 24, 32)].name == "fast"
    assert "chosen on an earlier run" in m2.notes[-1]
    # a different size is tested from scratch
    with pytest.raises(AttributeError):
        m2.down(torch.rand(1, 3, 16, 16))  # (the stub above returns None)
    assert timed


def test_no_candidates_no_tuning(tmp_path):
    m = _irn(tmp_path)
    m.down(torch.rand(1, 3, 16, 16))
    assert m.engines[("down", 16, 16)].name == "torch" and not m.notes


@pytest.mark.skipif(importlib.util.find_spec("coremltools") is None, reason="coremltools not installed")
def test_converts_for_core_ml(tmp_path):
    from faceloop.models.coreml import convert
    m = _irn(tmp_path)
    pkg = convert(_Up(m.net, 2, 12, 16), (1, 3, 12, 16), tmp_path / "up.mlpackage", "fp16")
    assert pkg.exists()
    assert convert(_Up(m.net, 2, 12, 16), (1, 3, 12, 16), pkg, "fp16") == pkg  # cached
    if sys.platform == "darwin":
        from faceloop.models.coreml import CoreMLRunner
        lr = torch.rand(2, 3, 12, 16)
        got = CoreMLRunner(pkg, "all", "fp16")(lr)
        assert float((got - m._up_raw(lr)).abs().mean()) < 2 / 255


def test_reference_is_full_precision(tmp_path):
    _fake_irn_weights(tmp_path, 2)
    mm = ModelManager(tmp_path)
    full = mm.get_irn("irn-x2", torch.device("cpu"))
    half = mm.get_irn("irn-x2", torch.device("cpu"), "fp16")
    x = torch.rand(2, 3, 16, 24)
    if half.dtype == torch.float16:
        assert torch.allclose(half._reference("down", x), full._down_raw(x), atol=1e-6)
    assert torch.allclose(half._reference("up", x), full._up_raw(x), atol=1e-6)
