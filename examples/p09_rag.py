"""Pattern 09: filter by permission before ranking."""

from kb import DOCS

from adp.core import FakeLLM
from adp.rag import PermissionAwareRetriever, User, answer

retriever = PermissionAwareRetriever(DOCS, max_age_days=365)
agent = User("u1", frozenset({"agents"}))
lead = User("u2", frozenset({"leads"}))

for who in (agent, lead):
    r = retriever.search(who, "refund override approval")
    print(who.id, [d.id for d in r.docs], r.filtered_out)

llm = FakeLLM(["Open Settings, then Security [doc:kb-112]."])
print(answer(llm, retriever, agent, "how to reset 2FA"))
