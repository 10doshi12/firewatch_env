# 🔥 FirewatchEnv — Quickstart Guide

> Get from zero to running your first AI SRE agent in under 5 minutes.

---

## What is FirewatchEnv?

FirewatchEnv is an **RL training environment** for autonomous SRE incident response, built for the [Meta PyTorch OpenEnv Hackathon India 2026](https://github.com/meta-pytorch/OpenEnv). Your AI agent acts as an on-call Site Reliability Engineer — it receives simulated microservice telemetry (OTel-compatible metrics, Prometheus-style alerts, log excerpts) and must **diagnose and remediate the root cause** before the SLO error budget runs out.

**Key highlights:**
- Single container, no Kubernetes — runs on 2 vCPUs / 8 GB RAM
- Three difficulty tiers (Easy → Medium → Hard) with adversarial prompt injection in Task 3
- Outcome-only reward function — the agent can't game the grader; it must actually fix the system

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| **Python** | 3.10+ | [python.org](https://www.python.org/downloads/) |
| **uv** | latest | `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Git** | any | [git-scm.com](https://git-scm.com/) |
| **Docker** | latest *(optional — only for containerized runs)* | [docker.com](https://docs.docker.com/get-docker/) |

---

## 1 — Clone & Install

```bash
git clone https://huggingface.co/spaces/10doshi12/firewatch-env
cd firewatch-env
```

> **Important:** All commands below should be run from inside the `firewatch_env/` directory, which contains the actual environment code.

```bash
cd firewatch_env
uv sync            # installs all Python dependencies from pyproject.toml + uv.lock
```

This installs:
- `openenv-core[core]` ≥ 0.2.2 — FastAPI server + HTTP client types
- `pydantic` ≥ 2.0 — data models
- `openai` ≥ 1.0 — LLM inference via OpenAI-compatible API
- `python-dotenv` — `.env` file loading

---

## 2 — Configure Environment Variables

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# --- LLM Provider (HuggingFace Router) ---
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=hf_your_huggingface_token_here

# --- Server URL (usually auto-detected — leave commented for local dev) ---
# SPACE_URL=https://10doshi12-firewatch-env.hf.space
```

Get your HF token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (requires a **Pro** or **Enterprise** plan for router access to gated models).

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | HuggingFace Router endpoint (`https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Yes | Model on HF Hub (e.g. `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`) |
| `HF_TOKEN` | No* | HuggingFace API token. *If omitted, inference runs a deterministic rule-based fallback agent (no LLM calls).* |
| `SPACE_URL` | No | Override server URL. Auto-detected in order: `localhost:8000` → `localhost:7860` → HF Space |

---

## 3 — Start the Server

```bash
uv run server
```

The FastAPI server starts on **http://localhost:8000** with these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment — `{"difficulty": "easy", "seed": 42}` |
| `/step` | POST | Execute action — `{"action": {"action_type": "fetch_logs", "target_service": "auth-service"}}` |
| `/state` | GET | Get current environment state |
| `/schema` | GET | Action / observation JSON schemas |
| `/ws` | WS | WebSocket for persistent sessions |

### Quick smoke test (new terminal):

```bash
# Reset an easy episode
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy", "seed": 42}'

# Take an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "fetch_logs", "target_service": "cache"}}'

# Check current state
curl http://localhost:8000/state
```

---

## 4 — Run the Inference Agent

With the server running in one terminal, open a **second terminal**:

```bash
cd firewatch_env
python inference.py
```

This runs your agent across all three tasks sequentially:

| Task | Difficulty | Services | Red Herrings | Max Ticks | Seed |
|------|-----------|----------|-------------|-----------|------|
| `task_easy` | Easy | 3 | 0 | 20 | 42 |
| `task_medium` | Medium | 5 | 1 | 30 | 137 |
| `task_hard` | Hard | 7 | 3 (1 adversarial) | 40 | 256 |

### Expected Output

```
[START] task=task_easy env=firewatch-env model=x-ai/grok-4.1-fast
[STEP] step=1 action=fetch_logs:cache reward=-0.14 done=false error=null
[STEP] step=2 action=rollback_deploy:cache reward=-0.14 done=false error=null
...
[END] success=true steps=4 score=0.96 rewards=-0.14,-0.14,-0.14,1.86
```

Each `[STEP]` line shows the action taken, intermediate reward, and whether the episode ended. The `[END]` line reports the final graded score (0.0–1.0).

---

## 5 — Docker (Alternative)

Build and run the environment as a Docker container:

```bash
# From the firewatch_env/ directory
docker build -t firewatch-env ./server
docker run -p 7860:7860 firewatch-env
```

The server will be available at **http://localhost:7860**. Set `SPACE_URL=http://localhost:7860` when running `inference.py` (or let auto-detection find it).

---

## 6 — Deploy to HuggingFace Spaces

```bash
openenv validate          # must pass with zero errors
openenv push --repo-id 10doshi12/firewatch-env
```

Your environment will be live at `https://10doshi12-firewatch-env.hf.space`.

---

## Project Structure

```
firewatch_env/
├── models.py              # Pydantic models (FirewatchAction, SystemObservation, etc.)
├── simulation.py          # ServiceMesh + generate_episode() + fault physics
├── actions.py             # ActionHandler — all 17 action types
├── rewards.py             # RewardEngine + grade() + EpisodeResult
├── config.py              # Constants, TASKS dict, topology (pure data)
├── client.py              # OpenEnv-generated WebSocket client
├── inference.py           # LLM agent loop (stdout eval format)
├── openenv.yaml           # OpenEnv spec definition
├── .env.example           # Environment variable template
├── Dockerfile             # Multi-stage Docker build
├── pyproject.toml         # Dependencies & entry points
├── server/
│   ├── app.py             # FastAPI application (entry point)
│   └── firewatch_env_environment.py  # Environment wiring
└── tests/
    ├── test_integration.py
    ├── test_simulation.py
    └── test_inference.py
```

---

## Action Space Reference

### Investigation Actions (read-only)

| Action | Description |
|--------|-------------|
| `fetch_logs` | Populates `recent_logs` on the target service |
| `get_metrics_detail` | Returns 3-tick metric trend summary |
| `trace_dependencies` | Returns full upstream/downstream dependency chain |
| `strace_process` | System-call level process inspection |
| `profiler_dump` | CPU/memory profiler output |
| `check_gc_pressure` | GC pause times and heap pressure |
| `trace_distributed_request` | End-to-end distributed trace |
| `inspect_thread_pool` | Thread pool utilization and deadlock detection |
| `inspect_commit_diff` | Recent deployment diff |

### Remediation Actions (mutate state)

| Action | Description |
|--------|-------------|
| `restart_service` | Resets OOM state; wrong if `error_rate < 0.10` |
| `rollback_deploy` | Halts bad deployment progression |
| `revert_config` | Restores connection pool / config settings |
| `scale_replicas` | Increases memory headroom |
| `circuit_break` | Suppresses cascade for 3 ticks |
| `traffic_shift` | Redirects traffic away from degraded service |

### Meta Actions

| Action | Description |
|--------|-------------|
| `declare_resolved` | Terminates episode and triggers grading |
| `escalate` | Records escalation (no state change) |

---

## Fault Types

| Fault | Signal in Logs | Correct Remediation |
|-------|---------------|---------------------|
| `oom` | OOMKilled, exit code 137 | `restart_service` |
| `bad_deploy` | Error spike post-deployment SHA | `rollback_deploy` |
| `config_drift` | HikariCP pool exhaustion, 30s timeouts | `revert_config` |
| `network_partition` | Connection refused, circuit breaker OPEN | `circuit_break` or `restart_service` |
| `memory_leak` | Gradual latency increase, slow memory growth | `scale_replicas` → `restart_service` |

---

## Scoring

The grader produces a score between **0.0 and 1.0** based on four components:

| Component | Weight | What it Measures |
|-----------|--------|-----------------|
| Recovery | 40% | Did system health improve? |
| Speed | 25% | How quickly was MTTM achieved? |
| Precision | 20% | Were wrong actions avoided? |
| SLO | 15% | How much error budget remained? |

---

## Running Tests

```bash
cd firewatch_env
uv run pytest tests/                                  # all tests
uv run pytest tests/test_integration.py               # integration only
uv run pytest tests/test_simulation.py                # simulation logic
uv run pytest tests/test_integration.py::test_reset_deterministic  # single test
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `uv: command not found` | Install uv: `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `openenv-core` import error | Run `uv sync` inside `firewatch_env/` |
| Server won't start | Check port 8000 isn't in use: `lsof -i :8000` |
| `inference.py` can't find server | Server auto-detection probes `localhost:8000` → `localhost:7860`. Ensure the server is running. |
| LLM API errors / 401 | Verify `HF_TOKEN` in `.env`. Without it, the rule-based fallback agent runs (no LLM). |
| Score is 0.0 | Agent didn't call `declare_resolved` or SLO budget hit 0%. Check action logs. |
| Docker build fails | Ensure Docker Desktop is running. Build from `firewatch_env/`: `docker build -t fw ./server` |

---

## Next Steps

- **Swap the model**: Change `MODEL_NAME` in `.env` to test different HF-hosted models (e.g. `Qwen/Qwen2.5-72B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`)
- **Tune the agent**: Edit `SYSTEM_PROMPT` and `_recovery_hint()` in `inference.py` to improve decision-making
- **Add actions**: Extend `actions.py` with new diagnostic or remediation actions
- **Custom tasks**: Define new scenarios in `config.py` and `openenv.yaml`
- **Benchmark**: Compare scores across models to find the best SRE agent

---

*FirewatchEnv — Meta PyTorch OpenEnv Hackathon India 2026*
