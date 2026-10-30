from support_tools import build_registry

from adp.core import FakeClock, FakeLLM, LLMResponse, Usage, call, say
from adp.loop import AgentLoop, Budget


def run(llm, **budget):
    return AgentLoop(llm, build_registry(), Budget(**budget)).run("t")


def test_completes_with_answer():
    llm = FakeLLM([call("lookup_order", order_id="ORD-100001"),
                   say("shipped")])
    r = run(llm)
    assert (r.status, r.stop_reason, r.answer) == (
        "completed", "answer", "shipped")
    assert r.steps == 2 and r.tool_results[0].ok


def test_tool_result_is_fed_back_to_model():
    llm = FakeLLM([call("lookup_order", order_id="ORD-100001"),
                   say("ok")])
    run(llm)
    last = llm.calls[1][-1]
    assert last.role == "tool" and "shipped" in last.content


def test_step_budget():
    n = iter(range(100))
    llm = FakeLLM(responder=lambda m, t: call(
        "lookup_order", order_id=f"ORD-1000{next(n):02d}"))
    r = run(llm, max_steps=3, max_tool_errors=99)
    assert r.stop_reason == "step_budget" and r.steps == 3


def test_repeated_identical_call_is_stopped():
    llm = FakeLLM(responder=lambda m, t: call(
        "lookup_order", order_id="ORD-100001"))
    r = run(llm, max_repeats=2)
    assert r.stop_reason == "repeated_call" and r.steps == 3


def test_consecutive_tool_errors_stop_the_run():
    llm = FakeLLM(responder=lambda m, t: call(
        "lookup_order", order_id="bad"))
    r = run(llm, max_tool_errors=2, max_repeats=9)
    assert r.stop_reason == "tool_errors" and r.steps == 2


def test_token_and_cost_budgets():
    big = LLMResponse(text="x", usage=Usage(30_000, 10))
    assert run(FakeLLM([big])).stop_reason == "token_budget"
    pricey = LLMResponse(text="x", usage=Usage(9_000, 9_000))
    r = run(FakeLLM([pricey]), max_cost_usd=0.05)
    assert r.stop_reason == "cost_budget" and r.cost_usd > 0.05


def test_time_budget_uses_injected_clock():
    clock = FakeClock()

    def slow(m, t):
        clock.advance(40)
        return call("lookup_order", order_id="ORD-100001",
                    call_id=str(clock.now()))

    loop = AgentLoop(FakeLLM(responder=slow), build_registry(),
                     Budget(max_seconds=60, max_repeats=9),
                     clock=clock)
    r = loop.run("t")
    assert r.stop_reason == "time_budget" and r.steps == 2
