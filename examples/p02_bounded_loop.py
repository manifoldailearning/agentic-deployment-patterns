"""Pattern 02: an agent loop that always terminates."""

from support_tools import build_registry

from adp.core import FakeLLM, call, say
from adp.loop import AgentLoop, Budget

llm = FakeLLM([
    call("lookup_order", order_id="ORD-100001"),
    say("Your order ORD-100001 has shipped."),
])
loop = AgentLoop(llm, build_registry(), Budget(max_steps=4))
r = loop.run("Where is order ORD-100001?")
print(r.status, r.stop_reason, r.steps, "->", r.answer)

# A model stuck repeating the same call is stopped, not looped.
stuck = FakeLLM(responder=lambda m, t: call(
    "lookup_order", order_id="ORD-100002"))
r = AgentLoop(stuck, build_registry()).run("status?")
print(r.status, r.stop_reason, "after", r.steps, "steps")
