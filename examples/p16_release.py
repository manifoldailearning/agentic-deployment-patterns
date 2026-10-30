"""Pattern 16: canary a new prompt, roll back on bad health."""

from adp.release import ReleaseManager, RolloutPolicy, Variant

v1 = Variant("support-v3", "model-a")
v2 = Variant("support-v4", "model-a")
rm = ReleaseManager(v1, RolloutPolicy(min_requests=20))
rm.start_canary(v2, percent=10)


def run(v: Variant) -> str:
    if v.prompt_version == "support-v4":
        raise RuntimeError("tool schema mismatch")  # bad release
    return "ok"


served = {"ok": 0, "err": 0}
for i in range(1000):
    try:
        rm.serve(f"req-{i}", run)
        served["ok"] += 1
    except RuntimeError:
        served["err"] += 1
print(served)
print("\n".join(rm.state.history))
print("stable is", rm.state.stable.prompt_version)

rm.set_shadow(Variant("support-v5", "model-b"))
rm.serve("req-x", lambda v: v.prompt_version)
print("shadow diffs:", rm.shadow_diffs)
