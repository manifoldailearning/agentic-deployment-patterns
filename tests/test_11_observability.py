import json

import pytest

from adp.core import FakeLLM, LLMResponse, Usage
from adp.observability import Metrics, TracedLLM, Tracer


def test_nested_spans_share_trace_and_link_parents():
    t = Tracer()
    with t.span("request") as root:
        with t.span("retrieve") as child:
            with t.span("rank"):
                pass
    assert {s.trace_id for s in t.spans} == {root.trace_id}
    assert child.parent_id == root.span_id
    assert t.tree(root.trace_id) == [
        "request [ok]", "  retrieve [ok]", "    rank [ok]"]


def test_separate_requests_get_separate_traces():
    t = Tracer()
    with t.span("a"):
        pass
    with t.span("b"):
        pass
    assert len({s.trace_id for s in t.spans}) == 2


def test_errors_mark_span_and_propagate():
    t = Tracer()
    with pytest.raises(ValueError):
        with t.span("tool"):
            raise ValueError("bad")
    assert t.spans[0].status == "error"
    assert "ValueError" in t.spans[0].attrs["error"]


def test_traced_llm_records_tokens_cost_and_latency(tmp_path):
    t, m = Tracer(), Metrics()
    resp = LLMResponse(text="hi", usage=Usage(1000, 100),
                       model="m-large")
    llm = TracedLLM(FakeLLM([resp, resp]), t, m, "p-v2")
    llm.complete([])
    llm.complete([])
    s = t.spans[0]
    assert s.attrs["model"] == "m-large"
    assert s.attrs["prompt_version"] == "p-v2"
    assert s.attrs["cost_usd"] == pytest.approx(0.0045)
    assert m.counters["llm.calls"] == 2
    assert m.counters["llm.cost_usd"] == pytest.approx(0.009)
    out = tmp_path / "spans.jsonl"
    t.export_jsonl(str(out))
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert rows[0]["name"] == "llm.complete" and "ms" in rows[0]


def test_percentiles():
    m = Metrics()
    for v in range(1, 101):
        m.observe("lat", v)
    assert m.percentile("lat", 50) == 50
    assert m.percentile("lat", 95) == 95
    assert m.percentile("missing", 99) == 0.0
