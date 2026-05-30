AI/ML Engineer

Take-Home Assessment

────────────────────────────────────────

Estimated Time: 7–9 hours


Evaluation Criteria

We are evaluating how you think, not whether you finish everything. Four things matter most:

Judgment. Depth on fewer tasks is worth more than surface coverage of all of them. If you run out of time, tell us what you would have done next.

Communication. Your markdown cells, code comments, and written explanations carry as much weight as your outputs. We want to follow your reasoning without having to reverse-engineer it.

Production awareness. We notice when you handle edge cases, validate assumptions, version your artifacts, and write code that another engineer could extend without a walkthrough.

Honesty. Naming what you don’t know, where your approach has limitations, or what you’d revisit with more time earns more credit than glossing over gaps.

What to Submit

Two links:

1. A GitHub repository (public, or invite ishanky) containing your full codebase with commit history and a README that covers architecture, setup, and design decisions.

2. A live demo URL for the agent you build in Part 2 — hosted on Streamlit Cloud, Hugging Face Spaces, Railway, or any platform of your choosing. We just need a working link.

Scenario

You are an AI/ML Engineer at TeleConnect, a mid-size telecommunications company serving approximately 5,000 customers. Customer churn is increasing, and leadership has asked for two things:

1. 1. A predictive model that identifies customers likely to churn.

2. 2. An AI-powered agent that retention representatives can use in real time — one that runs your churn model, retrieves customer data, and helps reps decide on the right intervention.

These two parts are designed to connect. The model you build in Part 1 becomes a live tool that your agent calls in Part 2.


Part 1 — Churn Model

Time Budget: ~4 hours

Deliverable: Jupyter Notebook (.ipynb) with clear markdown narrative throughout


Context. The marketing team has provided a customer dataset (test_datafile.csv) extracted from several legacy systems. The data has known quality issues stemming from system migrations, manual entry errors, and inconsistent formatting across source systems. Treat it accordingly.

1.1 — Data Quality Assessment and Cleaning

Systematically identify all data quality issues in the dataset. Missing values are only the starting point — look for inconsistent encodings, impossible values, sentinel placeholders, and semantic outliers.

For every issue you find, document what the problem is, how many rows are affected, and what cleaning strategy you applied and why. Produce a summary table showing before-and-after statistics for each affected column.

We are specifically evaluating your ability to catch problems that aren’t obvious at first glance.

1.2 — Exploratory Data Analysis

Analyze the overall churn rate and how it relates to other features in the dataset. Identify the five features most strongly associated with churn, and justify your choice of correlation or association method — the target is binary, and not all features are continuous.

Visualize at least three meaningful relationships (for example: churn by contract type, churn by tenure, churn by satisfaction score). We value insight over aesthetics.

Propose at least two engineered features that could improve model performance. Explain the reasoning behind each.

1.3 — Model Building and Evaluation

Train at least two models from distinctly different families — for example, a tree-based model and a linear model, or a boosted ensemble and a neural network. For each, briefly explain why you selected it, where its structural strengths lie for this problem, and where it might fall short.

Evaluate using multiple metrics, then take a position: which single metric matters most for a churn prediction use case, and why? Which metrics are less useful or potentially misleading here?

If class imbalance is present, address it — or explain why you chose not to.

1.4 — Visualization

Include clear, purposeful visuals throughout your notebook. Every chart should make a specific point — title it with what the reader should take away.

At minimum, include a confusion matrix, an ROC or Precision-Recall curve, and a feature importance plot. Clean presentation is a plus, but clarity is the requirement.

1.5 — Export Your Model for Part 2

Save your best model as a callable artifact (pickle, joblib, ONNX, or equivalent) along with any preprocessing pipeline it requires. Part 2 depends on this.

Write a Python function with the following signature:

def predict_churn(customer_data: dict) -> dict:

"""

Accepts a dictionary of customer features. Returns:

{

"churn_probability": float, # 0.0 to 1.0

"risk_tier": str, # "high", "medium", or "low"

"top_risk_factors": list # top 3 features driving this prediction

}

"""

This function becomes one of the tools your agent uses in Part 2.


Part 2 — Retention Agent: Build, Evaluate, Deploy

Time Budget: ~4 hours

Deliverable: A working Python project with a live hosted demo, an evaluation suite, and a README explaining your architecture


Context. TeleConnect’s retention team wants an AI assistant that helps representatives handle at-risk customers in real time. The agent should retrieve customer data, run your churn model, surface relevant retention offers, and synthesize a recommendation — all through natural language. Your job is to build it, evaluate it, and deploy it.

2.1 — Agent Orchestration with Tool Calling

Build a working agent using any LLM provider (OpenAI, Anthropic, open-source, or otherwise) with the following tools:


Tool Purpose Notes

predict_churn Your model from Part 1. Accepts customer features, returns churn probability, risk tier, and top risk factors. Wire up the actual model you trained. If not working, mock it — but say so.

lookup_customer Retrieves a customer profile by ID. Returns demographics, contract, tenure, charges, satisfaction. Mock implementation backed by the dataset.

get_retention_offers Returns retention offers filtered by risk tier and contract type. Mock with a sensible offer catalog of your own design.

log_interaction Records the outcome of a retention conversation. Mock, but schema should be production-realistic.

escalate_to_supervisor Transfers the case to a human supervisor with context summary. For situations the agent should not handle alone.


The agent should:

• Accept a natural language message from a retention rep and chain tools in the correct order (look up customer, run churn prediction, retrieve relevant offers, synthesize a recommendation).

• Handle ambiguous or incomplete inputs gracefully — for example, “I have a high-risk customer on the phone” with no customer ID.

• Recognize when to escalate — a customer threatening legal action, a complex dispute, or a scenario outside the agent’s toolset.

• Produce a final response that is useful to a human representative, not a raw data dump.


What we are evaluating: how you design tool schemas and descriptions; how the agent handles multi-step reasoning and tool chaining; how it responds to errors, ambiguity, and conflicting signals; and whether the codebase is structured so that adding a sixth tool would not require rewriting the orchestration layer.

2.2 — Evaluation Framework and LLM-as-Judge

Design and implement a complete evaluation pipeline for your agent.

a) Test Suite

Build a structured test suite (JSON, YAML, or Python dataclass) with a minimum of 12 cases spanning the following categories:

• Single-tool happy path, multi-step chaining, ambiguous input, out-of-scope request, escalation trigger, model disagreement (your model flags low risk, but the profile shows warning signs), and adversarial or edge cases of your choosing.

• For each case, define: the user input, expected tool calls (name, order, key parameters), quality criteria for a good response, and a category label for aggregate reporting.

b) Automated Metrics

Implement at least three automated evaluation metrics. Select from tool selection accuracy, parameter extraction accuracy, response completeness, hallucination detection, or latency/token cost — or define your own. Choose what you believe matters most and explain why.

c) LLM-as-Judge Evaluator

Build a working LLM-as-judge that accepts a test case alongside the agent’s actual output, uses a separate LLM call to evaluate the response, and returns structured rubric-based scores — not binary pass/fail, and not an unanchored 1–10 scale.

At minimum, evaluate: factual correctness, tool use appropriateness, actionability for the representative, and hallucination.

Each scoring dimension must include anchors. Rather than “score tone from 1 to 5,” define what a 1 looks like, what a 3 looks like, and what a 5 looks like.

Address the meta-question: how do you know your judge is reliable? Even a brief discussion is valuable — consider inter-rater consistency, positivity bias, prompt sensitivity, or calibration against human labels.

2.3 — Deploy and Share the Link

Deploy your agent as a live, hosted application accessible in a browser. We are not looking for local setup instructions — we want a URL.

Suitable hosting options include Streamlit Community Cloud, Hugging Face Spaces, Railway, Render, Fly.io, or any platform you are comfortable with. The requirements:

• The URL works when we open it.

• A representative can type a message and receive the agent’s full response.

• The interface makes tool calls visible — what was called, in what order, and what each tool returned. Do not hide the orchestration; showing it is part of the deliverable.


We will also review your GitHub commit history to understand how the project came together. Commit naturally as you work. A trail of meaningful commits tells us far more than a single bulk upload.

2.4 — Results and Analysis

Run your agent against your full test suite and present:

• An aggregate scorecard showing overall pass rates, per-category breakdowns, and average LLM-judge scores by dimension.

• At least two examples where the agent performed well, with analysis of what it did right.

• At least two failure cases, with root-cause analysis and a specific, actionable fix you would make.

• A brief production roadmap (one paragraph): what would you change to run this evaluation pipeline in CI/CD at scale?


Submission Checklist

Deliverable Part

Jupyter notebook with markdown narrative 1

Cleaning code or cleaned dataset 1

Data quality summary table (before/after) 1

Three or more EDA visualizations with written takeaways 1

Two or more models trained, compared, and evaluated 1

Exported model artifact and predict_churn function 1 → 2

Agent orchestration code with tool definitions 2

Structured test suite (12+ cases) 2

Automated evaluation metrics 2

LLM-as-judge pipeline with anchored scoring rubric 2

Live demo URL 2

Results scorecard with success and failure analysis 2

GitHub repository with commit history and README Both

Additional Notes

Tooling. Use any libraries or frameworks you prefer. We evaluate your reasoning and decisions, not your stack. If you choose something uncommon, a brief explanation is appreciated.

AI assistants. You are welcome to use agentic engineering/coding techniques. If you do, the bar for design quality and understanding goes up accordingly.

Commit history. We review it. It does not need to be pristine — we are looking for how