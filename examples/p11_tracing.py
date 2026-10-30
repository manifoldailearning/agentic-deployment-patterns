"""Pattern 11: one trace per request, one span per step."""

from support_tools import build_registry

from adp.core import FakeLLM, call, say
from adp.loop import AgentLoop
from adp.observability import Metrics, TracedLLM, Tracer

tracer, metrics = Tracer("support-agent"), Metrics()
llm = TracedLLM(
    FakeLLM([call("lookup_order", order_id="ORD-100001"),
             say("It shipped.")]),
    tracer, metrics, prompt_version="support-v3")
loop = AgentLoop(llm, build_registry())

with tracer.span("request", ticket="T-881") as root:
    result = loop.run("Where is ORD-100001?")
    root.attrs["stop_reason"] = result.stop_reason

print("\n".join(tracer.tree(root.trace_id)))
for s in tracer.spans[:1]:
    print({k: s.attrs[k] for k in ("model", "tokens_in",
                                   "cost_usd")})
print("calls", metrics.counters["llm.calls"],
      "cost", round(metrics.counters["llm.cost_usd"], 6))
