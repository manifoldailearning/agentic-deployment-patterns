import pytest

from adp.release import ReleaseManager, RolloutPolicy, Variant, bucket

V1, V2 = Variant("p1", "m"), Variant("p2", "m")


def test_variant_ids_are_content_addressed():
    assert V1.id == Variant("p1", "m").id != V2.id
    assert Variant("p1", "m", 0.7).id != V1.id


def test_bucketing_is_stable_and_proportional():
    assert bucket("r-1") == bucket("r-1")
    rm = ReleaseManager(V1)
    rm.start_canary(V2, 10)
    share = sum(rm.choose(f"r{i}") == V2
                for i in range(10_000)) / 10_000
    assert 0.08 < share < 0.12


def test_canary_percent_is_bounded():
    with pytest.raises(ValueError):
        ReleaseManager(V1).start_canary(V2, 80)


def test_auto_rollback_on_error_rate():
    rm = ReleaseManager(V1, RolloutPolicy(min_requests=5))
    rm.start_canary(V2, 50)

    def run(v):
        if v == V2:
            raise RuntimeError("bad")
        return "ok"
    for i in range(200):
        try:
            rm.serve(f"r{i}", run)
        except RuntimeError:
            pass
    assert rm.state.canary is None and rm.state.stable == V1
    assert rm.state.history[-1].startswith(f"rolled back {V2.id}")


def test_auto_rollback_on_eval_regression():
    rm = ReleaseManager(V1, RolloutPolicy(min_requests=1))
    rm.start_canary(V2, 50)
    for _ in range(10):
        rm.record_eval(V1.id, True)
        rm.record_eval(V2.id, False)
    rm.stats[V2.id].requests = 5
    assert rm.check_canary().startswith("eval pass")
    assert rm.state.canary is None


def test_healthy_canary_is_promoted():
    rm = ReleaseManager(V1, RolloutPolicy(min_requests=5))
    rm.start_canary(V2, 20)
    for i in range(100):
        rm.serve(f"r{i}", lambda v: "ok")
    rm.promote()
    assert rm.state.stable == V2 and rm.state.canary is None


def test_shadow_runs_but_is_never_served():
    rm = ReleaseManager(V1)
    rm.set_shadow(V2)
    out, vid = rm.serve("r1", lambda v: v.prompt_version)
    assert (out, vid) == ("p1", V1.id)
    assert rm.shadow_diffs == [("r1", "p1", "p2")]

    def shadow_breaks(v):
        if v == V2:
            raise RuntimeError("x")
        return "ok"
    assert rm.serve("r2", shadow_breaks)[0] == "ok"
