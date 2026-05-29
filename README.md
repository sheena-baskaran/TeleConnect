# TeleConnect — Churn Model + Retention Agent

A two-part system for a telecom retention team:

1. **Part 1 — Churn model.** Clean a deliberately messy customer dataset, do EDA, train and
   compare two model families, and export the best one as a callable artifact.
2. **Part 2 — Retention agent.** Wire that model into a tool-calling LLM agent that retention
   reps talk to in natural language, evaluate it with an automated + LLM-as-judge pipeline, and
   deploy it as a clickable demo.

The two parts connect: `predict_churn()` from Part 1 becomes one of the agent's tools in Part 2.

> **🔗 Live demo:** _<add your Streamlit Cloud URL here after deploying — see [Deployment](#deployment)>_

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │  Part 1  (notebooks/churn_model.ipynb)        │
   raw CSV  ───────►│  clean → EDA → train (LogReg vs XGBoost) →    │
 (messy legacy)     │  export joblib pipeline + metadata.json       │
                    └───────────────┬───────────────────────────────┘
                                    │  artifacts
                                    ▼
   src/data_cleaning.py ─┐   models/churn_pipeline.joblib
   src/features.py ──────┼──►  + model_metadata.json
                         │
                         ▼
              src/predict.py  ──  predict_churn(dict) -> {prob, tier, factors}
                         ▲
                         │ (one of 5 tools)
   ┌─────────────────────┴───────────────────────────────────────────┐
   │  Part 2  agent (src/agent/)                                       │
   │   orchestrator.py  ── provider-agnostic tool-calling loop          │
   │   tools.py         ── TOOL_REGISTRY: lookup_customer, predict_churn,│
   │                       get_retention_offers, log_interaction,        │
   │                       escalate_to_supervisor                        │
   │   prompts.py       ── system prompt (chaining / ambiguity / escalate)│
   │   mock_db.py       ── customer lookup (cleaned CSV) + offer catalog  │
   │   src/llm_client.py── Anthropic client  ||  deterministic mock       │
   └───────────────┬───────────────────────────────┬───────────────────┘
                   │                               │
                   ▼                               ▼
   app/streamlit_app.py                   eval/ (run_eval.py)
   chat UI + visible tool timeline        test_suite + metrics + LLM-judge → scorecard
```

**Key design principle — no train/serve skew.** Cleaning (`src/data_cleaning.py`) and feature
engineering (`src/features.py`) are *shared modules* imported by both the training notebook and
the inference path (`src/predict.py`). The same raw value is normalized identically whether it's
in the training set or arriving live from the agent.

### Repository layout
```
data/        raw + cleaned dataset (cleaned also backs lookup_customer)
notebooks/   churn_model.ipynb (Part 1) + _build_notebook.py (reproducible builder)
models/      exported churn_pipeline.joblib + model_metadata.json
src/         data_cleaning, features, predict, llm_client + agent/ package
eval/        test_suite, metrics, judge, run_eval → results/scorecard.md
app/         streamlit_app.py (live demo)
```

---

## Setup & run (Windows PowerShell)

Requires Python 3.13 (3.11+ should work). Run these from the project folder:

```powershell
# 1. Create and activate the virtual environment
python -m venv .venv
Set-ExecutionPolicy -Scope Process -Bypass    # allows activation in THIS window only
.venv\Scripts\Activate.ps1                    # prompt now starts with (.venv)

# 2. Install dependencies
pip install -r requirements.txt               # runtime: app + eval
pip install -r requirements-dev.txt           # + notebook & plotting (only to re-run Part 1)

# 3. Run the demo (no API key needed — uses the built-in mock agent)
$env:FORCE_MOCK_LLM = "1"
streamlit run app/streamlit_app.py            # opens http://localhost:8501
```

Open **http://localhost:8501**, click an example in the sidebar (or type a message), and expand
the **🔧 Tool chain** under each answer to see every tool call, its input, and its output.

```powershell
# Run the evaluation suite (writes eval/results/scorecard.md)
python -m eval.run_eval --no-judge            # automated metrics, fast & free

# Re-run Part 1 (rebuilds + executes the notebook, re-exports the model)
python notebooks/_build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/churn_model.ipynb
```

> **macOS/Linux:** activate with `source .venv/bin/activate`; set the key with
> `export FORCE_MOCK_LLM=1` instead of `$env:...`.

**API key (optional).** The agent + LLM-judge use Anthropic Claude. With a funded
`ANTHROPIC_API_KEY` (copy `.env.example` to `.env`) they run live; otherwise the system uses a
deterministic **mock agent + judge** so everything still runs end-to-end. Mock mode is labeled in
the UI and the scorecard.

---

## Part 1 — Churn model (highlights)

Full narrative is in [`notebooks/churn_model.ipynb`](notebooks/churn_model.ipynb).

**1.1 Data quality.** Beyond missing values, the cleaner catches: inconsistent categorical
encodings (`Male/M/F/Female/Other`, `Yes/Y/no/N`, `fiber/Fiber optic`, `BT/CC` aliases),
impossible negatives (age, tenure, charges, data usage), sentinel placeholders (age `999`,
charges `9999`, tenure/tickets `500`, satisfaction `99`), an out-of-scale satisfaction column
(0–10 with values >10), a `total_charges` decimal-shift error (max 218k → repaired by
reconciling against `tenure × monthly`), and 50 duplicate customer IDs. Each issue is logged with
affected-row counts and a before/after summary table. Impossible/sentinel values become `NaN` so
the pipeline's imputer handles them identically at train and serve time.

**1.2 EDA.** Churn rate ≈ **36.5%**. Because the target is binary and features are mixed-type, I
rank associations with **mutual information** (a non-parametric, type-agnostic ranker) and
corroborate with **point-biserial** (continuous↔binary) and **Cramér's V** (categorical↔binary).
Top drivers: contract type, tenure, satisfaction. Five engineered features (with reasoning) incl.
`charges_per_tenure_month`, `tickets_per_tenure_year`, `expected_vs_actual_charges_gap`,
`tenure_bucket`, `has_bundle`.

**1.3 Models.** **Logistic Regression** (interpretable linear baseline) vs **XGBoost** (captures
interactions), same preprocessing pipeline. Imbalance handled with class weights (SMOTE trade-off
discussed, not blindly applied). **Metric stance:** *recall + PR-AUC matter most* for churn — a
missed churner (false negative) is far costlier than a wasted incentive; **accuracy is misleading**
at a 36% base rate and **ROC-AUC is optimistic** under imbalance.

**1.4 Visuals.** Confusion matrix, ROC + PR curves (both models), feature importance (SHAP →
permutation fallback). Every chart titled with its takeaway.

**1.5 Export.** Best pipeline saved to `models/churn_pipeline.joblib` + `model_metadata.json`
(thresholds, per-feature directions/stats, importances, per-category churn lift). `predict_churn`
returns `{churn_probability, risk_tier, top_risk_factors}` and degrades gracefully on partial input.

---

## Part 2 — Retention agent (highlights)

**2.1 Orchestration.** A provider-agnostic tool-calling loop (`src/agent/orchestrator.py`)
iterates a single `TOOL_REGISTRY` (`src/agent/tools.py`). **Adding a sixth tool = one registry
entry** (schema + impl); the orchestration layer doesn't change. Five tools: `lookup_customer`,
`predict_churn` (the real Part 1 model), `get_retention_offers` (a designed catalog filtered by
risk tier × contract), `log_interaction` (JSONL, production-realistic schema), and
`escalate_to_supervisor`. The system prompt encodes tool-chain order, asking for a customer ID
when one is missing, escalation triggers, conflicting-signal handling, and rep-facing synthesis
(never a raw data dump). Every tool call is traced (name, order, input, output, latency) for the
UI and evals.

**2.2 Evaluation.**
- **Test suite** (`eval/test_suite.py`): 14 dataclass cases across single-tool, multi-step,
  ambiguous, out-of-scope, escalation, model-disagreement, and adversarial/edge categories — each
  with expected tool calls (name/order/params), quality criteria, and a category label. Customer
  IDs are real rows chosen so the model's prediction fits the scenario.
- **Automated metrics** (`eval/metrics.py`, deterministic, no LLM): **tool-selection accuracy**
  (right tools, right order, no forbidden calls), **parameter-extraction accuracy** (e.g. the
  `customer_id` actually passed), **response completeness** (required elements present). Latency &
  tokens are reported for cost-awareness.
- **LLM-as-judge** (`eval/judge.py`): a *separate* model scores **factual correctness, tool-use
  appropriateness, actionability, and hallucination**, each with explicit **1/3/5 anchors** (not a
  binary, not an unanchored 1–10). Judge-reliability is discussed in the module: positivity bias
  (countered with anchors), prompt sensitivity (fixed rubric, temp 0), independence (different
  model than the agent), and calibration (the deterministic metrics cross-check the judge; we
  recommend periodic human-label agreement, e.g. Cohen's κ).

**2.3 Deploy.** `app/streamlit_app.py` — chat UI where every answer shows its **ordered tool
chain with inputs/outputs**. See [Deployment](#deployment).

### 2.4 Results & analysis

Run `python -m eval.run_eval` to regenerate [`eval/results/scorecard.md`](eval/results/scorecard.md).

> The committed scorecard is from **mock mode** (no funded key was available during the build).
> The **automated metrics are real** — they grade the (mock) agent's actual tool use. The
> **LLM-judge scores are placeholders** until a funded `ANTHROPIC_API_KEY` is set, at which point
> the same pipeline produces real, anchored judge scores. With the live LLM agent, the
> mock-specific failures below are expected to resolve.

**Mock-mode aggregate:** 14 cases · overall pass **71%** · tool-selection **0.89** ·
parameter-extraction **0.91** · completeness **1.0**. Ambiguous, out-of-scope, and escalation
categories pass 100%.

**Two things it did well**
1. **Escalation on a legal threat** (`escalate_legal`): given "…getting a lawyer to sue us," the
   agent correctly chose `escalate_to_supervisor` *instead of* attempting a retention offer, and
   produced a context summary for the handoff — exactly the boundary behavior we want.
2. **Full high-risk chain** (`chain_high_risk`): it chained `lookup_customer → predict_churn →
   get_retention_offers` in order, then synthesized a rep-facing recommendation naming a specific
   offer and a talking point — not a data dump.

**Two failure cases (root cause → fix)**
1. **Model-disagreement case fails** (`disagreement_low_model_bad_profile`). *Root cause:* the
   deterministic mock reports the model's low-risk tier but doesn't *reason* about the conflict
   with the customer's low satisfaction + high support contact, so it misses the required "flag
   the tension" behavior. *Fix:* (a) with the live LLM, the system prompt's "conflicting signals"
   section handles this; (b) to make even the mock safe, add a rule that compares `risk_tier`
   against satisfaction/ticket thresholds and surfaces a caution. The deeper, production fix is a
   **calibrated probability + an override rule** so low-confidence "low risk" predictions on poor
   profiles are flagged automatically.
2. **Outcome logging is skipped** (`chain_then_log`). *Root cause:* the mock's chain ends at
   synthesis and never calls `log_interaction`. *Fix:* the live LLM calls it per the prompt; for
   the mock, add a post-synthesis step that logs when the message states an outcome. Production
   fix: make logging a **non-optional final step** in the orchestrator for any completed
   recommendation, rather than relying on the model to remember.

**Production roadmap (CI/CD at scale).** Run `eval/run_eval.py` as a CI gate on every PR that
touches the agent, prompt, or tools: cache agent runs, shard cases across workers, and fail the
build if overall pass rate or any judge dimension regresses past a threshold. Replace the single
LLM-judge call with a small **judge panel + majority vote** to reduce variance, track judge-vs-
human agreement on a fixed golden set to catch judge drift, pin model versions, and record
per-run latency/token cost to dashboards so quality and spend are both observable over time.

---

## Deployment

Hosted on **Streamlit Community Cloud** (free):

1. Push this repo to GitHub (see below).
2. At [share.streamlit.io](https://share.streamlit.io) → **New app**, point it at this repo,
   branch `main`, main file `app/streamlit_app.py`. (Advanced settings → Python 3.13.)
3. In the app's **Secrets**, paste the contents of `.streamlit/secrets.toml.example` with a
   **funded** `ANTHROPIC_API_KEY`. (Omit the key, or set `FORCE_MOCK_LLM = "1"`, for a zero-cost
   public demo running the mock agent.)
4. Deploy → copy the URL into the [Live demo](#) line at the top of this README.

---

## Design decisions & documented assumptions

- **LLM provider:** Anthropic Claude (`claude-sonnet-4-6` agent, a *different* judge model for
  independence), behind a thin abstraction so swapping providers is one class.
- **`satisfaction_score` scale = 0–10**; values >10 (esp. 99) treated as sentinels.
- **`internet_service`: `No` = no service; `None`/`nan`/blank = missing** (imputed separately).
- **`monthly_charges == 15.0` (×110)** kept but flagged — plausibly a real low-tier price;
  nulling on a hunch would destroy signal. Documented, not hidden.
- **Risk-tier thresholds** are population quantiles of predicted probability; in production I'd
  set them by retention economics (offer cost vs. saved-customer value).

## Honest limitations / what I'd revisit with more time

- `top_risk_factors` is a transparent *approximation* (z-score × association × importance), not a
  causal or full-SHAP attribution.
- The committed scorecard is mock-mode; real judge numbers need a funded key.
- No temporal validation — I avoided leak-prone time features rather than guess the observation
  window; with more time I'd build a time-based split and add recency features.
- Next steps: probability calibration, cross-validated hyperparameter search, a judge panel,
  and feature-drift monitoring once the agent is live.

## Commit history

The repo is built in meaningful increments (scaffold → cleaning modules → Part 1 model → agent →
eval → app → docs) so the commit log shows how the project came together.
