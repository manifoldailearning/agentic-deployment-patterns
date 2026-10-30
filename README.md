# Agentic AI Deployment Patterns for Production

Sixteen patterns for taking AI agents from a working demo to a
service you can run, measure, change and roll back safely.

This is the companion code for the book **Agentic AI Deployment
Patterns for Production** by Nachiketh Murthy. Each pattern is a
small, readable module with an example and tests. Everything runs
offline: a deterministic `FakeLLM` stands in for the model, so
the tests need no API keys and give the same result every time.

## Quick start

```bash
git clone https://github.com/manifoldailearning/agentic-deployment-patterns
cd agentic-deployment-patterns
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q                                   # all tests
PYTHONPATH=src:examples python examples/p02_bounded_loop.py
```

Run the evaluation gate against the reference assistant:

```bash
PYTHONPATH=src:examples python -m adp.evals evals/cases.jsonl \
  support_system:build_system --baseline evals/baseline.json
```

Serve it:

```bash
PYTHONPATH=src:examples uvicorn adp.serving.main:app --port 8000
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"text": "Where is order ORD-100001?"}'
```

or with Docker: `docker compose up --build`.

## Pattern map

| # | Pattern | Module | Problem it solves |
|---|---------|--------|-------------------|
| 01 | Typed tool contracts | `adp.tools` | Model-proposed calls with bad arguments |
| 02 | Bounded agent loop | `adp.loop` | Runaway loops, cost and time overruns |
| 03 | Workflow-first router | `adp.router` | Using an agent where a workflow will do |
| 04 | Durable execution | `adp.durable` | Lost progress and repeated side effects after a crash |
| 05 | Human approval gate | `adp.approval` | Unreviewed write actions |
| 06 | Retries, idempotency, circuit breaker | `adp.resilience` | Flaky dependencies and duplicate writes |
| 07 | Async queue with backpressure | `adp.queue` | Bursty load and slow runs |
| 08 | Guardrails | `adp.guardrails` | Prompt injection, PII, unsafe output |
| 09 | Permission-aware RAG | `adp.rag` | Leaking documents a user may not see |
| 10 | Evaluation gate | `adp.evals` | Shipping regressions |
| 11 | Tracing and metrics | `adp.observability` | Debugging a bad answer after the fact |
| 12 | Caching and model routing | `adp.routing` | Cost, latency and provider outages |
| 13 | Bounded supervisor | `adp.multiagent` | Multi-agent sprawl |
| 14 | Tool gateway | `adp.gateway` | Direct, unaudited access to systems |
| 15 | Service layer | `adp.serving` | Health, timeouts, overload, configuration |
| 16 | Release management | `adp.release` | Unsafe prompt and model changes |

## Layout

```
src/adp/        the patterns (one package each)
examples/       runnable examples and the reference assistant
evals/          evaluation cases and the accepted baseline
tests/          one test module per pattern
```

## Using a real model

Patterns depend only on the small `LLM` protocol in
`adp/core/llm.py`: a `complete(messages, tools)` method that
returns an `LLMResponse`. Write an adapter for your provider
that maps to and from those types, and pass it wherever the
examples pass `FakeLLM`. Keep `FakeLLM` for unit tests.

## License

MIT © 2026 Nachiketh Murthy
