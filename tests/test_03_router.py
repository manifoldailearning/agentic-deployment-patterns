from adp.core import FakeLLM
from adp.router import Router

H = {"order_status": lambda t: "S", "refund": lambda t: "R"}


def make(script, **kw):
    q = []
    r = Router(
        FakeLLM(script), H, lambda t: q.append(t) or "H", **kw
    )
    return r, q


def test_confident_valid_intent_routes_to_handler():
    r, q = make(['{"intent": "refund", "confidence": 0.9}'])
    res = r.route("refund pls")
    assert (res.route, res.output, res.attempts) == (
        "refund",
        "R",
        1,
    )
    assert q == []


def test_low_confidence_goes_to_human():
    r, q = make(['{"intent": "refund", "confidence": 0.5}'])
    res = r.route("hmm")
    assert res.route == "fallback:low_confidence" and q == ["hmm"]


def test_unknown_intent_goes_to_human():
    r, q = make(['{"intent": "cancel", "confidence": 0.99}'])
    assert r.route("x").route == "fallback:unknown_intent"


def test_invalid_json_is_repaired_once():
    r, _ = make(
        [
            "not json",
            '{"intent": "order_status", "confidence": 0.8}',
        ]
    )
    res = r.route("where")
    assert res.route == "order_status" and res.attempts == 2


def test_repeated_invalid_output_falls_back():
    r, q = make(["nope", '{"intent": "refund", "confidence": 7}'])
    res = r.route("x")
    assert res.route == "fallback:invalid_output" and q == ["x"]


def test_repair_prompt_includes_validation_error():
    llm = FakeLLM(
        ["oops", '{"intent": "refund", "confidence": 0.9}']
    )
    Router(llm, H, lambda t: "H").route("x")
    assert llm.calls[1][-1].content.startswith("Invalid:")
