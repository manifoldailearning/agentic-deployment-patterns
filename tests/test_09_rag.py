from kb import DOCS

from adp.core import FakeLLM
from adp.rag import PermissionAwareRetriever, User, answer

R = PermissionAwareRetriever(DOCS)
AGENT = User("a", frozenset({"agents"}))
LEAD = User("l", frozenset({"leads"}))
NOBODY = User("n", frozenset({"guests"}))


def ids(user, q):
    return [d.id for d in R.search(user, q).docs]


def test_acl_applied_before_ranking():
    assert "int-9" not in ids(AGENT, "refund override codes")
    assert ids(LEAD, "refund override codes")[0] == "int-9"
    r = R.search(AGENT, "override")
    assert r.filtered_out["int-9"] == "no_access"


def test_stale_and_unreviewed_docs_excluded():
    r = R.search(LEAD, "2FA reset refund policy")
    assert r.filtered_out["kb-031"] == "stale"
    assert r.filtered_out["draft-4"] == "unreviewed"


def test_ranking_prefers_matching_doc():
    assert ids(AGENT, "reset two factor")[0] == "kb-112"


def test_answer_requires_valid_citations():
    ok = FakeLLM(["Use Settings [doc:kb-112]."])
    text, cited = answer(ok, R, AGENT, "reset two factor")
    assert cited == ["kb-112"]
    bad = FakeLLM(["Codes are 1234 [doc:int-9]."])
    text, cited = answer(bad, R, AGENT, "reset two factor")
    assert cited == [] and "couldn't" in text
    none = FakeLLM(["Just do it."])
    assert answer(none, R, AGENT, "reset two factor")[1] == []


def test_no_permitted_source_means_no_model_call():
    llm = FakeLLM([])
    text, cited = answer(llm, R, NOBODY, "refund")
    assert cited == [] and llm.calls == []
