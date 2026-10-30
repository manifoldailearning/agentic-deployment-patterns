"""Shared example knowledge base with access groups."""

from adp.rag import Doc

ALL = frozenset({"agents", "leads"})
DOCS = [
    Doc(
        "kb-112",
        "To reset two factor authentication open "
        "Settings then Security and choose reset 2FA.",
        ALL,
    ),
    Doc(
        "kb-207",
        "Refunds over 200 dollars need lead approval "
        "before they are issued.",
        ALL,
    ),
    Doc(
        "int-9",
        "Internal escalation contacts and refund "
        "override codes for leads.",
        frozenset({"leads"}),
    ),
    Doc(
        "kb-031",
        "Old 2FA reset steps for the legacy app.",
        ALL,
        age_days=900,
    ),
    Doc(
        "draft-4",
        "Draft: new refund policy, not approved.",
        ALL,
        reviewed=False,
    ),
]
