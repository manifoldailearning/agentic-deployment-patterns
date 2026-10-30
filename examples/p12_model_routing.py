"""Pattern 12: cache, route by difficulty, fall back."""

from adp.core import FakeClock, FakeLLM, Message
from adp.resilience import CircuitBreaker
from adp.routing import (
    CachedLLM,
    FallbackLLM,
    ModelRouter,
    length_and_keyword_policy,
)

small = FakeLLM(responder=lambda m, t: "small answer",
                model="small")
large = FakeLLM(responder=lambda m, t: "large answer",
                model="large")
router = ModelRouter({"small": small, "large": large},
                     length_and_keyword_policy())
cached = CachedLLM(router, model="router-v1", clock=FakeClock())

for q in ["How do I reset 2FA?", "How do I reset 2FA?",
          "I need a refund and want to escalate"]:
    print(q, "->", cached.complete([Message("user", q)]).text)
print("cache hits", cached.hits, "routed", router.routed)


class Down:
    def complete(self, messages, tools=None):
        raise TimeoutError("provider timeout")


clock = FakeClock()
fb = FallbackLLM(
    [("primary", Down()), ("secondary", large)],
    {"primary": CircuitBreaker(2, 30, clock),
     "secondary": CircuitBreaker(2, 30, clock)})
for _ in range(3):
    fb.complete([Message("user", "hi")])
    print("served by", fb.last_used,
          "| primary breaker", fb.breakers["primary"].state)
