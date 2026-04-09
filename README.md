---
title: SRE Incident Response OpenEnv
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - openenv
---

# SRE Incident Response - OpenEnv Environment

This repository provides an **OpenEnv-compatible** environment where an agent acts as an on-call **Site Reliability Engineer (SRE)**. The environment exposes a simple HTTP API (FastAPI) that supports episodic rollouts via `/reset` and `/step`, plus grading via `/grader`.

## What’s in this repo

- **Environment server**: `app/main.py` (FastAPI, OpenEnv-style endpoints)
- **Task logic**: `app/tasks/` (three incident scenarios)
- **Inference runner**: `inference.py`
  - Uses an OpenAI model if `OPENAI_API_KEY` is set
  - Otherwise falls back to a deterministic, no-network policy

## Run locally (Docker)

Build and run:

```bash
docker build -t sre-incident-env .
docker run --rm -p 7860:7860 sre-incident-env
```

Then check:

```bash
curl http://localhost:7860/health
curl http://localhost:7860/tasks
```

## Run locally (Python)

Install:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
```

## Inference / evaluation

Run inference against a running environment:

```bash
python inference.py --base-url http://localhost:7860
```

Notes:

- **Exit codes**: by default `inference.py` exits **0** if it completes (even if tasks fail), to avoid “runner failed” false negatives. Use `--strict-exit` if you want non-zero on failed tasks.
- **Auto-start**: if `--base-url` is `http://localhost:7860` and the server isn’t running, `inference.py` will try to start the local server automatically.

## Hugging Face Spaces

This repo is set up for **Docker Spaces** (see `Dockerfile`). The server binds to port **7860**, and `/health` is used as a health check.

## Meta x PyTorch Hackathon submission notes

If the hackathon evaluator runs `inference.py` directly, this repo is designed to:

- Install cleanly from `requirements.txt`
- Bring up the environment server (Docker or local)
- Run `inference.py` without crashing or returning a non-zero code just because a baseline policy didn’t “pass”