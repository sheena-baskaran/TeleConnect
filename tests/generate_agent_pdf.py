"""
Generate docs/TeleConnect_Agent_Part2_Guide.pdf

A full end-to-end technical document explaining how the Part 2 retention agent
works, step by step -- architecture, tool chain, orchestrator loop, eval framework,
Streamlit app, and a complete annotated conversation walkthrough.

Run:
    python tests/generate_agent_pdf.py
Output:
    docs/TeleConnect_Agent_Part2_Guide.pdf
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fpdf import FPDF
from fpdf.enums import XPos, YPos

OUT = ROOT / "docs" / "TeleConnect_Agent_Part2_Guide.pdf"
OUT.parent.mkdir(exist_ok=True)

# ---- Colour palette --------------------------------------------------------
DARK_BLUE  = (23,  55,  94)
MID_BLUE   = (31,  97, 141)
LIGHT_BLUE = (214, 234, 248)
ORANGE     = (202, 111,  30)
GREEN      = (30,  132,  73)
DARK_GREY  = (44,  62,  80)
MID_GREY   = (127, 140, 141)
LIGHT_GREY = (245, 246, 250)
WHITE      = (255, 255, 255)
RED        = (169,  50,  38)


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)

    # ---- header / footer ---------------------------------------------------
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*MID_GREY)
        self.cell(0, 8, "TeleConnect Retention Agent -- Part 2 Technical Guide", align="L",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*LIGHT_BLUE)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MID_GREY)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # ---- helpers -----------------------------------------------------------
    def cover_page(self):
        self.add_page()
        # dark banner
        self.set_fill_color(*DARK_BLUE)
        self.rect(0, 0, 210, 80, "F")
        self.set_y(18)
        self.set_font("Helvetica", "B", 26)
        self.set_text_color(*WHITE)
        self.cell(0, 12, "TeleConnect", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "B", 18)
        self.cell(0, 10, "Retention Agent -- End-to-End Guide", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(180, 200, 220)
        self.cell(0, 8, "Part 2: How the AI Agent Works, Step by Step", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(90)
        self.set_text_color(*DARK_GREY)

    def h1(self, text):
        self.ln(4)
        self.set_fill_color(*DARK_BLUE)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 9, f"  {text}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK_GREY)
        self.ln(3)

    def h2(self, text):
        self.ln(3)
        self.set_text_color(*MID_BLUE)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*MID_BLUE)
        self.line(20, self.get_y(), 190, self.get_y())
        self.set_text_color(*DARK_GREY)
        self.ln(3)

    def h3(self, text):
        self.ln(2)
        self.set_text_color(*ORANGE)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK_GREY)
        self.ln(1)

    def body(self, text, indent=0):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK_GREY)
        self.set_x(20 + indent)
        self.multi_cell(170 - indent, 5.5, text)
        self.ln(1)

    def bullet(self, text, level=1):
        indent = (level - 1) * 8 + 4
        dot = "*" if level == 1 else "-"
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK_GREY)
        self.set_x(20 + indent)
        self.cell(5, 5.5, dot)
        self.multi_cell(165 - indent, 5.5, text)

    def code_block(self, lines, label=""):
        self.ln(2)
        if label:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*MID_GREY)
            self.cell(0, 5, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_fill_color(*LIGHT_GREY)
        self.set_draw_color(*MID_GREY)
        height = len(lines) * 5 + 6
        self.rect(20, self.get_y(), 170, height, "DF")
        self.set_y(self.get_y() + 3)
        self.set_font("Courier", "", 8)
        self.set_text_color(50, 50, 50)
        for line in lines:
            self.set_x(23)
            self.cell(0, 5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK_GREY)
        self.ln(3)

    def note_box(self, text, color=LIGHT_BLUE, text_color=MID_BLUE):
        self.ln(2)
        self.set_fill_color(*color)
        lines = text.split("\n")
        h = len(lines) * 5.5 + 6
        self.rect(20, self.get_y(), 170, h, "F")
        self.set_y(self.get_y() + 3)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*text_color)
        self.set_x(23)
        self.multi_cell(165, 5.5, text)
        self.set_text_color(*DARK_GREY)
        self.ln(3)

    def step_box(self, number, title, description):
        self.ln(2)
        self.set_fill_color(*MID_BLUE)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        self.cell(10, 8, str(number), fill=True, align="C")
        self.set_fill_color(*LIGHT_BLUE)
        self.set_text_color(*DARK_BLUE)
        self.cell(160, 8, f"  {title}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(30)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK_GREY)
        self.multi_cell(160, 5, description)
        self.ln(2)

    def two_col_row(self, left, right, bold_left=False):
        self.set_font("Helvetica", "B" if bold_left else "", 9)
        self.set_x(20)
        self.cell(55, 6, left)
        self.set_font("Helvetica", "", 9)
        self.multi_cell(115, 6, right)

    def table_header(self, cols, widths):
        self.set_fill_color(*DARK_BLUE)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        for col, w in zip(cols, widths):
            self.cell(w, 7, f" {col}", fill=True)
        self.ln()
        self.set_text_color(*DARK_GREY)

    def table_row(self, cells, widths, shade=False):
        if shade:
            self.set_fill_color(*LIGHT_GREY)
        self.set_font("Helvetica", "", 8.5)
        for cell, w in zip(cells, widths):
            self.cell(w, 6, f" {cell}", fill=shade)
        self.ln()


# ============================================================================
def build(pdf: PDF):

    # ---- COVER -------------------------------------------------------------
    pdf.cover_page()

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*DARK_BLUE)
    pdf.cell(0, 8, "What this document covers", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.body(
        "This guide explains -- step by step -- how the TeleConnect AI retention agent works "
        "end to end. It covers the agent's architecture, every tool it uses, the conversation "
        "flow, the orchestration loop, the evaluation framework, and the Streamlit demo app. "
        "It is written for a technical reviewer who wants to understand both the design "
        "decisions and the exact code that runs them."
    )
    pdf.ln(2)
    items = [
        ("Section 1", "System overview and the big picture"),
        ("Section 2", "The five tools -- schemas, implementations, and why each exists"),
        ("Section 3", "The orchestration loop -- how the agent calls tools step by step"),
        ("Section 4", "The LLM provider abstraction -- Anthropic, Ollama, and Mock"),
        ("Section 5", "Complete annotated conversation walkthrough"),
        ("Section 6", "Edge cases -- ambiguity, escalation, out-of-scope, unknown ID"),
        ("Section 7", "The evaluation framework -- test suite, metrics, LLM-as-judge"),
        ("Section 8", "The Streamlit demo app -- UI, tool timeline, deployment"),
        ("Section 9", "File map -- every file and what it does"),
    ]
    for num, desc in items:
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(*MID_BLUE)
        pdf.set_x(20)
        pdf.cell(32, 6, num)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*DARK_GREY)
        pdf.cell(0, 6, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ======================================================================
    # SECTION 1 -- Overview
    # ======================================================================
    pdf.add_page()
    pdf.h1("Section 1 -- System Overview")

    pdf.body(
        "The TeleConnect system has two parts that work together. Part 1 trains a churn "
        "prediction model from messy customer data. Part 2 wires that model into a live AI "
        "agent that retention representatives talk to in natural language. The model from "
        "Part 1 becomes one of the agent's tools in Part 2 -- they are explicitly connected."
    )

    pdf.h2("1.1 The Big Picture")
    pdf.code_block([
        " Retention rep types a message",
        "          |",
        "          v",
        " Streamlit app  (app/streamlit_app.py)",
        "          |",
        "          v",
        " RetentionAgent.run()  (src/agent/orchestrator.py)",
        "          |",
        "          v",
        " LLM Client  <-- reads SYSTEM PROMPT (src/agent/prompts.py)",
        " (Anthropic / Ollama / Mock)  src/llm_client.py",
        "          |",
        "    [tool_use block returned]",
        "          |",
        "          v",
        " execute_tool()  (src/agent/tools.py  -- TOOL_REGISTRY)",
        "    |           |           |           |           |",
        "lookup_    predict_   get_       log_       escalate_",
        "customer   churn      offers     interaction supervisor",
        "    |           |",
        "cleaned     sklearn pipeline",
        "CSV         (models/churn_pipeline.joblib)",
        "          |",
        "    [tool_result fed back to LLM]",
        "          |",
        "   ... loop until no more tool_use ...",
        "          |",
        "          v",
        " Final text response -> displayed in Streamlit + tool timeline shown",
    ], "Architecture flow")

    pdf.h2("1.2 Two Distinct 'Models' -- Do Not Confuse Them")
    pdf.body("There are two very different things called 'model' in this project:")
    widths = [48, 72, 50]
    pdf.table_header(["Component", "What it is", "Uses API?"], widths)
    rows = [
        ("Churn model", "Sklearn LogisticRegression pipeline (.joblib file)", "No -- local"),
        ("LLM (agent brain)", "Claude / Ollama / Mock -- drives conversation", "Optional"),
    ]
    for i, r in enumerate(rows):
        pdf.table_row(r, widths, shade=(i % 2 == 0))
    pdf.ln(3)
    pdf.note_box(
        "The churn model always runs locally and for free -- no API key, no network. "
        "Only the LLM (the conversation brain) optionally calls an external API. "
        "Without any key the system uses a deterministic mock that still demonstrates "
        "the correct tool chain."
    )

    pdf.h2("1.3 Key Design Principles")
    pdf.bullet("No train/serve skew: src/data_cleaning.py and src/features.py are shared "
               "by the notebook (training) and src/predict.py (inference). The same raw value "
               "gets the same transform at both stages.")
    pdf.bullet("Extensibility: adding a 6th tool = one entry in TOOL_REGISTRY. The "
               "orchestrator iterates the registry; nothing else changes.")
    pdf.bullet("Always runnable: FORCE_MOCK_LLM=1 runs the full system end-to-end with "
               "no API key, so the demo and evals are never blocked.")
    pdf.bullet("Honest labeling: mock mode is labeled everywhere -- in the Streamlit "
               "sidebar, the scorecard banner, and the README.")

    # ======================================================================
    # SECTION 2 -- The Five Tools
    # ======================================================================
    pdf.add_page()
    pdf.h1("Section 2 -- The Five Tools")

    pdf.body(
        "All tools live in src/agent/tools.py. Each has two parts: (1) a JSON Schema "
        "that tells the LLM what the tool does and what arguments it takes, and (2) a "
        "Python implementation function. The TOOL_REGISTRY dict maps tool names to both."
    )

    pdf.h2("2.1 lookup_customer")
    pdf.h3("Purpose")
    pdf.body("Retrieves a customer's profile by their account ID. This is always the FIRST "
             "tool called -- the agent cannot predict churn or get offers until it has a profile.")
    pdf.h3("Input")
    pdf.code_block(['{"customer_id": "TC-001096"}'])
    pdf.h3("Output")
    pdf.code_block([
        '{',
        '  "found": true,',
        '  "customer_id": "TC-001096",',
        '  "demographics": {"age": 32, "gender": "Male"},',
        '  "contract": {"contract_type": "Month-to-month", "payment_method": "Credit card"},',
        '  "tenure_months": 10,',
        '  "charges": {"monthly_charges": 69.24, "total_charges": 656.42},',
        '  "services": {"internet_service": "DSL", "phone_service": "Yes", ...},',
        '  "satisfaction_score": 7.8,',
        '  "_features": { ...full feature dict for predict_churn... }',
        '}',
    ])
    pdf.h3("Implementation")
    pdf.body("Backed by data/cleaned_customers.csv (the cleaned dataset). The _features "
             "field is a flat dict of all customer features -- the agent passes this directly "
             "to predict_churn as customer_data, so it doesn't need to re-extract fields.")
    pdf.h3("Error handling")
    pdf.body("If the ID is not found it returns {found: false, error: '...', hint: '...'} "
             "so the agent can relay the problem to the rep and ask them to re-check.")

    pdf.h2("2.2 predict_churn")
    pdf.h3("Purpose")
    pdf.body("Runs the REAL trained sklearn pipeline from Part 1 on the customer's features. "
             "This is the bridge between Part 1 and Part 2.")
    pdf.h3("Input")
    pdf.code_block(['{"customer_data": { ...the _features dict from lookup_customer... }}'])
    pdf.h3("Output")
    pdf.code_block([
        '{',
        '  "churn_probability": 0.76,',
        '  "risk_tier": "high",',
        '  "top_risk_factors": [',
        '    "Month-to-month contract (no lock-in)",',
        '    "High support-ticket count (4)",',
        '    "Low tenure (10 months)"',
        '  ]',
        '}',
    ])
    pdf.h3("How it works internally")
    pdf.body("The call chain inside predict_churn():")
    pdf.bullet("clean_record(customer_data) -- same cleaning as training (no skew)")
    pdf.bullet("add_engineered_features() -- same 5 features as training")
    pdf.bullet("pipeline.predict_proba(X) -- the saved sklearn pipeline (lru_cache: loaded once)")
    pdf.bullet("Probability thresholds from model_metadata.json -> risk_tier")
    pdf.bullet("Feature direction + importance heuristic -> top_risk_factors (top 3)")
    pdf.note_box(
        "If the model artifact is missing (not yet trained), the tool returns a clearly "
        "labeled mock prediction with a warning -- per the brief: 'mock it, but say so.'"
    )

    pdf.add_page()
    pdf.h2("2.3 get_retention_offers")
    pdf.h3("Purpose")
    pdf.body("Returns retention offers filtered by the customer's risk tier and optionally "
             "their contract type. Offer aggressiveness is matched to risk -- high risk "
             "unlocks stronger discounts; low risk returns margin-preserving options.")
    pdf.h3("Input")
    pdf.code_block(['{"risk_tier": "high", "contract_type": "Month-to-month"}'])
    pdf.h3("Offer catalog (designed by us -- brief asked for our own design)")
    widths = [48, 72, 50]
    pdf.table_header(["Offer", "Description", "Eligible tiers"], widths)
    offers = [
        ("20% Loyalty Discount", "20% off bill for 12 mo in exchange for 1-yr commitment", "high, medium"),
        ("Free Upgrade + 2yr Lock", "Free tier upgrade if moves to 2-year contract", "high, medium"),
        ("Service Recovery Credit", "Bill credit + 90 days priority support", "high only"),
        ("Free Add-On (6 mo)", "Complimentary streaming/security add-on", "medium, low"),
        ("Data/Minutes Boost", "Doubled allowance at no cost for 6 months", "medium, low"),
        ("Goodwill Check-In", "Proactive review, no financial concession", "low only"),
    ]
    for i, r in enumerate(offers):
        pdf.table_row(r, widths, shade=(i % 2 == 0))

    pdf.h2("2.4 log_interaction")
    pdf.h3("Purpose")
    pdf.body("Records the outcome of a retention conversation for analytics and audit. "
             "The brief required this to be mocked but with a production-realistic schema.")
    pdf.h3("Schema (appended to data/interaction_log.jsonl)")
    pdf.code_block([
        '{',
        '  "interaction_id": "INT-20260529182345123456",',
        '  "logged_at_utc": "2026-05-29T18:23:45.123456+00:00",',
        '  "customer_id": "TC-001096",',
        '  "rep_id": "REP-007",',
        '  "churn_risk_tier": "high",',
        '  "churn_probability": 0.76,',
        '  "offers_presented": ["RET-LOYALTY-20", "RET-CONTRACT-SWITCH"],',
        '  "outcome": "accepted",  // accepted|declined|escalated|pending|callback',
        '  "notes": "Customer agreed to 1-year contract after loyalty discount offer",',
        '  "schema_version": "1.0"',
        '}',
    ])

    pdf.h2("2.5 escalate_to_supervisor")
    pdf.h3("Purpose")
    pdf.body("Transfers the case to a human supervisor with a full context summary. "
             "Used for situations outside the agent's remit.")
    pdf.h3("When the agent escalates (per system prompt)")
    pdf.bullet("Customer threatens legal or regulatory action (lawyer, lawsuit, ombudsman)")
    pdf.bullet("Complex billing dispute the agent cannot resolve")
    pdf.bullet("Customer is highly distressed or asks for something no tool handles")
    pdf.h3("Output")
    pdf.code_block([
        '{',
        '  "escalated": true,',
        '  "ticket": {',
        '    "escalation_id": "ESC-20260529182345123456",',
        '    "customer_id": "TC-003356",',
        '    "reason": "Customer threatened legal action over disputed charges",',
        '    "context_summary": "Customer on Month-to-month, 61% churn prob ...",',
        '    "urgency": "high",',
        '    "status": "queued_for_supervisor"',
        '  }',
        '}',
    ])

    # ======================================================================
    # SECTION 3 -- Orchestration Loop
    # ======================================================================
    pdf.add_page()
    pdf.h1("Section 3 -- The Orchestration Loop (Step by Step)")

    pdf.body(
        "The orchestrator (src/agent/orchestrator.py) runs a generic tool-calling loop. "
        "It is completely provider-agnostic -- it does not know or care whether the LLM is "
        "Claude, Ollama, or the mock. It just sends messages, executes tools, and loops."
    )

    pdf.h2("3.1 The Loop in Detail")
    steps = [
        ("1", "Rep message received",
         "The rep's text is appended to the messages list as a 'user' role message. "
         "The conversation history (prior turns) is included so the LLM has context."),
        ("2", "System prompt sent",
         "The SYSTEM_PROMPT from src/agent/prompts.py is sent with every LLM call. "
         "It encodes tool-chain order, ambiguity handling, escalation triggers, and "
         "response format. This is the agent's permanent instruction set."),
        ("3", "LLM responds",
         "The LLM returns one or more content blocks. Each block is either: "
         "(a) text -- the LLM is thinking or synthesizing, or "
         "(b) tool_use -- the LLM wants to call a specific tool with specific arguments."),
        ("4", "Tool_use? Execute tools",
         "For each tool_use block: look up the tool in TOOL_REGISTRY, call its "
         "implementation with the provided arguments, capture the result (always a dict), "
         "record the call in the trace (order, name, input, output, latency_ms)."),
        ("5", "Tool result fed back",
         "All tool results are appended to messages as 'user' role tool_result blocks "
         "(Anthropic format). This is the standard multi-turn tool-calling protocol."),
        ("6", "Loop repeats",
         "Steps 2-5 repeat. The LLM now sees the tool outputs and can either call "
         "more tools or produce a final text response."),
        ("7", "No tool_use? Done",
         "When the LLM returns only text blocks (stop_reason = 'end_turn'), the loop "
         "exits. The final text is the agent's response to the rep."),
        ("8", "Max turns safety",
         "If the loop runs for more than 8 turns without a final answer, it exits with "
         "a fallback message asking the rep to rephrase. This prevents infinite loops."),
    ]
    for num, title, desc in steps:
        pdf.step_box(num, title, desc)

    pdf.h2("3.2 The Trace Record")
    pdf.body(
        "Every tool call is recorded as a ToolCallRecord with: order (1, 2, 3...), "
        "name, input dict, output dict, and latency_ms. This trace is returned in "
        "AgentResult and consumed by: (a) the Streamlit app to render the tool timeline, "
        "and (b) the eval harness to compute tool-selection accuracy."
    )
    pdf.code_block([
        "AgentResult(",
        "  final_text='**Retention summary for TC-001096**...',",
        "  tool_calls=[",
        "    ToolCallRecord(order=1, name='lookup_customer', input={...}, output={...}, latency_ms=2.1),",
        "    ToolCallRecord(order=2, name='predict_churn',   input={...}, output={...}, latency_ms=0.8),",
        "    ToolCallRecord(order=3, name='get_retention_offers', ...),",
        "  ],",
        "  is_mock=True,",
        "  total_latency_ms=215.3,",
        "  input_tokens=0, output_tokens=0  # 0 in mock mode",
        ")",
    ], "AgentResult structure")

    # ======================================================================
    # SECTION 4 -- LLM Provider Abstraction
    # ======================================================================
    pdf.add_page()
    pdf.h1("Section 4 -- LLM Provider Abstraction (src/llm_client.py)")

    pdf.body(
        "The entire codebase speaks one normalised format (Anthropic-style content blocks). "
        "All three client classes implement a single method: respond(system, messages, tools). "
        "Swapping providers = one class. The orchestrator, tools, and app never change."
    )

    pdf.h2("4.1 Provider Selection Logic")
    pdf.code_block([
        "LLM_PROVIDER env var  +  FORCE_MOCK_LLM",
        "",
        "FORCE_MOCK_LLM=1  -->  MockClient  (always wins, overrides everything)",
        "LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY set  -->  AnthropicClient",
        "LLM_PROVIDER=ollama  -->  OllamaClient",
        "(nothing set)  -->  MockClient",
    ], "Provider resolution order")

    pdf.h2("4.2 The Three Clients")

    pdf.h3("AnthropicClient")
    pdf.body("Uses the Anthropic Python SDK. Sends the system prompt, messages, and tool "
             "schemas to Claude (claude-sonnet-4-6 for the agent, claude-opus-4-8 for the "
             "judge). Parses the response into normalised text/tool_use blocks.")
    pdf.bullet("Requires: pip install anthropic + funded ANTHROPIC_API_KEY")
    pdf.bullet("Agent model: AGENT_MODEL env (default claude-sonnet-4-6)")
    pdf.bullet("Judge model: JUDGE_MODEL env (default claude-opus-4-8 -- different for independence)")

    pdf.h3("OllamaClient")
    pdf.body("Talks to Ollama's /api/chat endpoint using Python's standard library urllib "
             "(zero new dependencies). Translates between Anthropic-style blocks and "
             "Ollama's OpenAI-compatible tool format. Ollama itself wraps llama.cpp and "
             "downloads GGUF model weights.")
    pdf.bullet("Requires: install Ollama (https://ollama.com) + ollama pull qwen2.5:3b-instruct")
    pdf.bullet("Set: LLM_PROVIDER=ollama, OLLAMA_MODEL=qwen2.5:3b-instruct")
    pdf.bullet("Also works with a remote Ollama via OLLAMA_BASE_URL (e.g. ngrok tunnel)")
    pdf.bullet("Speed measured: 3b ~ 2 min/request on CPU, 7b ~ 10 min/request on CPU")

    pdf.h3("MockClient")
    pdf.body("A deterministic rule-based stand-in. It does NOT call any LLM -- it uses "
             "pattern matching (regex for customer IDs, keyword lists for escalation) to "
             "decide which tool to call next and synthesise a structured final response.")
    pdf.body("It correctly handles: the full tool chain, ambiguity (no ID -> ask), "
             "legal threats (-> escalate), out-of-scope (-> decline), unknown IDs "
             "(-> stop after lookup). The mock judge returns placeholder scores.")
    pdf.note_box(
        "The mock has known limitations: it cannot reason about conflicting signals "
        "(model score vs profile quality), and it does not reliably call log_interaction "
        "at the end. Both are documented as mock-specific gaps -- the real LLM handles them "
        "correctly per the system prompt."
    )

    # ======================================================================
    # SECTION 5 -- Full Conversation Walkthrough
    # ======================================================================
    pdf.add_page()
    pdf.h1("Section 5 -- Annotated Conversation Walkthrough")

    pdf.body(
        "Below is a complete trace of a real agent session for a high-risk customer. "
        "Each step shows exactly what message is sent, what the LLM returns, what tool "
        "is called, and what comes back. This is the exact sequence the code runs."
    )

    pdf.h2("Scenario: 'Customer TC-001096 called in saying they might cancel'")

    # Step 1
    pdf.h3("Step 1 -- Rep message arrives")
    pdf.code_block([
        "messages = [",
        '  {"role": "user",',
        '   "content": "Customer TC-001096 called in saying they might cancel. What should I do?"}',
        "]",
    ])
    pdf.body("The orchestrator appends the rep's message and calls LLM.respond() with "
             "the system prompt + messages + all 5 tool schemas.")

    # Step 2
    pdf.h3("Step 2 -- LLM returns: call lookup_customer")
    pdf.code_block([
        "LLM response blocks:",
        '[{"type": "tool_use",',
        '  "id": "tu_001",',
        '  "name": "lookup_customer",',
        '  "input": {"customer_id": "TC-001096"}}]',
        "",
        'stop_reason = "tool_use"  --> loop continues',
    ])
    pdf.body("The LLM recognised the customer ID and chose the correct first tool. "
             "The system prompt says: 'Call lookup_customer FIRST whenever you have a customer ID.'")

    # Step 3
    pdf.h3("Step 3 -- execute_tool('lookup_customer', {customer_id: 'TC-001096'})")
    pdf.code_block([
        "Tool result (from cleaned_customers.csv):",
        '{',
        '  "found": true, "customer_id": "TC-001096",',
        '  "tenure_months": 10, "contract": {"contract_type": "Month-to-month"},',
        '  "satisfaction_score": 7.8, "charges": {"monthly_charges": 69.24},',
        '  "_features": {all feature fields...}',
        '}',
        "",
        "Recorded: ToolCallRecord(order=1, name='lookup_customer', latency_ms=2.1)",
    ])
    pdf.body("The result is appended as a tool_result block. The loop sends this back to the LLM.")

    # Step 4
    pdf.h3("Step 4 -- LLM returns: call predict_churn")
    pdf.code_block([
        '[{"type": "tool_use",',
        '  "name": "predict_churn",',
        '  "input": {"customer_data": {<the _features dict>}}}]',
    ])
    pdf.body("The LLM extracted the _features dict from the previous tool result and "
             "passed it to predict_churn. This is multi-step reasoning -- the output of "
             "step 3 becomes the input of step 4.")

    # Step 5
    pdf.h3("Step 5 -- execute_tool('predict_churn', {customer_data: {...}})")
    pdf.code_block([
        "Tool result (from sklearn pipeline -- the real trained model):",
        '{',
        '  "churn_probability": 0.76,',
        '  "risk_tier": "high",',
        '  "top_risk_factors": [',
        '    "Month-to-month contract (no lock-in)",',
        '    "High support-ticket count (4)",',
        '    "Low tenure (10 months)"',
        '  ]',
        '}',
        "",
        "Recorded: ToolCallRecord(order=2, name='predict_churn', latency_ms=0.8)",
    ])
    pdf.body("The churn model runs locally (no API). Pipeline: clean -> engineer features "
             "-> impute -> scale -> LogisticRegression.predict_proba -> threshold.")

    # Step 6
    pdf.h3("Step 6 -- LLM returns: call get_retention_offers")
    pdf.code_block([
        '[{"type": "tool_use",',
        '  "name": "get_retention_offers",',
        '  "input": {"risk_tier": "high", "contract_type": "Month-to-month"}}]',
    ])
    pdf.body("The LLM extracted risk_tier='high' from the predict_churn result and "
             "contract_type='Month-to-month' from the lookup result. It correctly "
             "combined information across two prior tool calls.")

    # Step 7
    pdf.h3("Step 7 -- execute_tool('get_retention_offers', ...)")
    pdf.code_block([
        "Tool result (filtered offer catalog):",
        '{',
        '  "risk_tier": "high", "count": 3,',
        '  "offers": [',
        '    {"offer_id": "RET-LOYALTY-20", "name": "20% Loyalty Discount (12 mo)",',
        '     "description": "20% off the monthly bill for 12 months in exchange for a 1-year commitment.",',
        '     "est_monthly_cost": 14.0},',
        '    {"offer_id": "RET-CONTRACT-SWITCH", "name": "Free Upgrade + 2-Year Lock-In", ...},',
        '    {"offer_id": "RET-SERVICE-RECOVERY", "name": "Service Recovery Credit", ...}',
        '  ]',
        '}',
    ])

    pdf.add_page()
    # Step 8
    pdf.h3("Step 8 -- LLM synthesises the final response")
    pdf.code_block([
        'stop_reason = "end_turn"  --> loop exits',
        "",
        "Final response text:",
        "**Retention summary for TC-001096**",
        "",
        "- **Churn risk:** HIGH (76% probability)",
        "- **Top risk factors:** Month-to-month contract, High support-ticket count (4), Low tenure (10 months)",
        "- **Recommended offer:** 20% Loyalty Discount (12 mo) -- 20% off the monthly",
        "  bill for 12 months in exchange for a 1-year commitment.",
        "- **Alternatives:** Free Upgrade + 2-Year Lock-In, Service Recovery Credit",
        "",
        "**Suggested approach:** Lead with empathy, acknowledge the customer's concerns,",
        "and present the recommended offer framed around the value they'll keep.",
        "Confirm acceptance and log the outcome.",
    ])
    pdf.body("The LLM synthesised information from 3 tool calls into a rep-facing "
             "recommendation -- not a data dump. Specific offer, specific reason, talking point.")

    pdf.h2("5.2 Full Turn Summary")
    widths = [10, 48, 60, 52]
    pdf.table_header(["#", "Tool called", "Key input", "Key output"], widths)
    rows = [
        ("1", "lookup_customer", "customer_id: TC-001096", "profile + _features dict"),
        ("2", "predict_churn", "_features from step 1", "prob=0.76, tier=high, 3 factors"),
        ("3", "get_retention_offers", "tier=high, contract=Month-to-month", "3 eligible offers"),
        ("4", "(synthesis)", "all prior results", "rep-facing recommendation"),
    ]
    for i, r in enumerate(rows):
        pdf.table_row(r, widths, shade=(i % 2 == 0))

    # ======================================================================
    # SECTION 6 -- Edge Cases
    # ======================================================================
    pdf.add_page()
    pdf.h1("Section 6 -- Edge Cases and How the Agent Handles Them")

    cases = [
        ("6.1 Ambiguous input -- no customer ID",
         "Input: 'I have a high-risk customer on the phone!'",
         "Expected: ask for the customer ID, call NO tools",
         "How: system prompt says 'If the rep describes a customer but gives no customer ID, "
         "ask for it before doing anything else.' The LLM returns a text block (not tool_use). "
         "The mock client checks: no TC-XXXXX pattern in the message -> returns a clarification request.",
         "No tools called. Response: 'Happy to help. Could you give me the customer ID (TC-XXXXX)?'"),
        ("6.2 Unknown customer ID",
         "Input: 'Look up customer TC-999999 and tell me their churn risk.'",
         "Expected: call lookup_customer, get not-found, stop -- do NOT predict",
         "How: lookup_customer returns {found: false, error: 'No customer found with id TC-999999'}. "
         "The system prompt says: 'If lookup returns an error, tell the rep plainly and ask them to re-check.' "
         "The mock specifically checks the last tool result for found=false and returns a text response. "
         "predict_churn is NEVER called -- fabricating a prediction for an unknown customer would be hallucination.",
         "Tools: [lookup_customer]. Response mentions 'not found' and asks rep to re-check."),
        ("6.3 Legal / regulatory threat -- escalation",
         "Input: 'Customer TC-003356 says they will get a lawyer and sue us.'",
         "Expected: escalate_to_supervisor, do NOT offer discounts",
         "How: the word 'lawyer' triggers the escalation path (word-boundary regex, not substring). "
         "The system prompt: 'Escalate when the customer threatens legal or regulatory action.' "
         "The agent calls escalate_to_supervisor with the customer ID, reason, and a context summary. "
         "It does NOT call get_retention_offers -- offering a discount to a customer threatening legal action "
         "could be construed as an admission.",
         "Tools: [escalate_to_supervisor]. Response confirms escalation and context handoff."),
        ("6.4 Out-of-scope request",
         "Input: 'What is the weather like in London today?'",
         "Expected: politely decline, call NO tools",
         "How: the out-of-scope keyword list catches 'weather' (whole-word match). "
         "System prompt: 'If asked something unrelated to retention, politely decline and redirect.' "
         "No tools are called at all.",
         "Tools: []. Response redirects to retention help."),
        ("6.5 Model disagreement -- low score, bad profile",
         "Input: 'The model says TC-002360 is low risk, but they seem really unhappy.'",
         "Expected: surface the tension -- low model score but low satisfaction + high tickets",
         "How: this is a REAL LLM behavior (not mock). The system prompt section 'Conflicting signals' "
         "says: 'If the model's risk tier disagrees with the profile, say so explicitly and recommend "
         "the more cautious action.' A real LLM reads this and reasons about it. The mock skips it "
         "(documented limitation). With Claude or Ollama, the agent flags the conflict.",
         "Note: this case FAILS in mock mode (0%) and PASSES with a real LLM (documented in eval)."),
    ]

    for title, inp, expected, how, result in cases:
        pdf.h2(title)
        pdf.two_col_row("Input:", inp, bold_left=True)
        pdf.two_col_row("Expected:", expected, bold_left=True)
        pdf.two_col_row("How:", how, bold_left=True)
        pdf.two_col_row("Result:", result, bold_left=True)
        pdf.ln(3)

    # ======================================================================
    # SECTION 7 -- Evaluation Framework
    # ======================================================================
    pdf.add_page()
    pdf.h1("Section 7 -- The Evaluation Framework")

    pdf.body(
        "The eval framework (eval/) implements Part 2.2 of the brief: a structured test suite, "
        "automated metrics, and an LLM-as-judge. It runs with no API key (mock mode) and "
        "produces a scorecard that proves the agent was tested -- not just 'tried'."
    )

    pdf.h2("7.1 The Test Suite (eval/test_suite.py)")
    pdf.body("14 dataclass test cases spanning all required categories. Each case has:")
    pdf.bullet("user_input -- the rep's message")
    pdf.bullet("expected_tools -- ordered list of (tool, key params)")
    pdf.bullet("must_not_call -- tools that must NOT be called")
    pdf.bullet("quality_criteria -- plain-language bar for a good response")
    pdf.bullet("response_must_contain -- substrings checked by the completeness metric")
    pdf.bullet("category -- for aggregate reporting")
    pdf.ln(2)
    widths = [52, 18, 100]
    pdf.table_header(["Category", "Cases", "What it tests"], widths)
    cats = [
        ("single_tool_happy_path", "1", "Simple lookup -- stop after one tool"),
        ("multi_step_chaining", "3", "Full lookup -> predict -> offers -> synthesize chain"),
        ("ambiguous_input", "2", "No customer ID -- ask before acting"),
        ("out_of_scope", "2", "Weather/password -- decline, call no tools"),
        ("escalation_trigger", "2", "Legal threat / regulatory complaint -> escalate"),
        ("model_disagreement", "1", "Low model score but bad profile -- flag the conflict"),
        ("adversarial_edge", "3", "Prompt injection, unknown ID, conflicting instruction"),
    ]
    for i, r in enumerate(cats):
        pdf.table_row(r, widths, shade=(i % 2 == 0))

    pdf.h2("7.2 Automated Metrics (eval/metrics.py)")
    pdf.body("Three deterministic metrics -- no LLM needed. Run in milliseconds. "
             "Used as a cross-check against the LLM judge.")

    pdf.h3("1. Tool Selection Accuracy (primary)")
    pdf.body("Combines: (a) no forbidden tool was called (hard requirement), "
             "(b) Jaccard similarity between expected and actual tool sets, "
             "(c) whether expected tools appear in the correct order (subsequence check). "
             "Score = 0.6 * jaccard + 0.4 * ordered_correct, halved if forbidden tool called.")

    pdf.h3("2. Parameter Extraction Accuracy")
    pdf.body("For each expected tool call that declares key params, checks whether "
             "the agent passed the correct value. E.g. did lookup_customer receive "
             "the exact customer_id mentioned in the input? "
             "Score = fraction of key params correctly extracted.")

    pdf.h3("3. Response Completeness")
    pdf.body("Checks that the final response contains all required substrings "
             "(e.g. 'high', 'offer', 'not found'). Fraction of required strings present. "
             "Catches cases where the agent called the right tools but gave a useless response.")

    pdf.h2("7.3 LLM-as-Judge (eval/judge.py)")
    pdf.body("A separate LLM call (deliberately a DIFFERENT model than the agent -- "
             "claude-opus-4-8 or qwen2.5:7b vs the agent's claude-sonnet-4-6 or 3b) "
             "scores the agent's response on four dimensions using anchored 1/3/5 rubric scores.")

    widths = [52, 30, 30, 30, 28]
    pdf.table_header(["Dimension", "Score 1", "Score 3", "Score 5", ""], widths)
    dims = [
        ("factual_correctness", "Contradicts tools", "Minor mismatch", "Fully grounded", ""),
        ("tool_use_appropriateness", "Wrong tools/order", "Questionable", "Perfect chain", ""),
        ("actionability", "Raw data dump", "Generic advice", "Specific + talking point", ""),
        ("hallucination", "Fabricates facts", "One unsupported detail", "No fabrication", ""),
    ]
    for i, r in enumerate(dims):
        pdf.table_row(r, widths, shade=(i % 2 == 0))
    pdf.ln(3)

    pdf.h3("Judge reliability (addressed in the brief)")
    pdf.bullet("Positivity bias: countered by explicit anchors (not 'good/bad' but concrete descriptions)")
    pdf.bullet("Prompt sensitivity: fixed rubric version, temperature=0")
    pdf.bullet("Independence: judge uses a DIFFERENT model than the agent")
    pdf.bullet("Calibration: deterministic metrics cross-check the judge; recommend "
               "periodic hand-labeling + Cohen's kappa measurement")

    pdf.h2("7.4 The Scorecard (eval/run_eval.py)")
    pdf.body("Run: python -m eval.run_eval [--no-judge]")
    pdf.body("Outputs: eval/results/scorecard.md + scorecard.json with:")
    pdf.bullet("Overall pass rate (14 cases, 71% in mock mode)")
    pdf.bullet("Per-category pass rate breakdown")
    pdf.bullet("Average automated metric scores")
    pdf.bullet("Average LLM-judge score by dimension")
    pdf.bullet("Per-case table (tools called, pass/fail, judge mean)")
    pdf.bullet("Cost report (latency per case, total tokens)")

    # ======================================================================
    # SECTION 8 -- Streamlit App
    # ======================================================================
    pdf.add_page()
    pdf.h1("Section 8 -- The Streamlit Demo App (app/streamlit_app.py)")

    pdf.body(
        "The Streamlit app is the Part 2.3 deliverable -- a clickable, hosted demo. "
        "It wraps the agent in a chat interface and, crucially, makes every tool call "
        "VISIBLE to the reviewer (required by the brief: 'showing it is part of the deliverable')."
    )

    pdf.h2("8.1 UI Layout")
    pdf.bullet("Left sidebar: mode banner (mock/live), example prompt buttons, tool list, clear button")
    pdf.bullet("Main area: chat history (user + assistant turns)")
    pdf.bullet("Under each assistant response: tool-call timeline (expandable)")
    pdf.bullet("Footer per response: latency, token count, model name")

    pdf.h2("8.2 The Tool-Call Timeline (the key differentiator)")
    pdf.body(
        "The brief says: 'The interface makes tool calls visible -- what was called, in what order, "
        "and what each tool returned. Do not hide the orchestration; showing it is part of the deliverable.'"
    )
    pdf.body("For each tool call in AgentResult.tool_calls:")
    pdf.code_block([
        "st.expander('1. lookup_customer  .  2.1 ms')",
        "  Input:  {'customer_id': 'TC-001096'}",
        "  Output: {'found': True, 'tenure_months': 10, ...}",
        "",
        "st.expander('2. predict_churn  .  0.8 ms')",
        "  Input:  {'customer_data': {...}}",
        "  Output: {'churn_probability': 0.76, 'risk_tier': 'high', ...}",
        "",
        "st.expander('3. get_retention_offers  .  0.3 ms')",
        "  Input:  {'risk_tier': 'high', 'contract_type': 'Month-to-month'}",
        "  Output: {'count': 3, 'offers': [...]}",
    ])

    pdf.h2("8.3 API Key and Secrets")
    pdf.body("The app reads env vars or Streamlit secrets. The provider is selected automatically:")
    pdf.bullet("FORCE_MOCK_LLM=1 -> mock agent (always works, no cost)")
    pdf.bullet("ANTHROPIC_API_KEY set -> real Claude agent")
    pdf.bullet("LLM_PROVIDER=ollama -> local Ollama agent (local demo only)")
    pdf.body("For Streamlit Cloud deployment, paste secrets in the app's Secrets manager "
             "(not in the repo -- the .gitignore excludes .streamlit/secrets.toml).")

    pdf.h2("8.4 Running Locally")
    pdf.code_block([
        "# Mock mode (always works, no API key):",
        "$env:FORCE_MOCK_LLM = '1'",
        ".venv\\Scripts\\python.exe -m streamlit run app/streamlit_app.py",
        "# Opens at http://localhost:8501",
        "",
        "# Ollama mode (real local LLM, no cost):",
        "$env:FORCE_MOCK_LLM = ''",
        "$env:LLM_PROVIDER = 'ollama'",
        "$env:OLLAMA_MODEL = 'qwen2.5:3b-instruct'",
        ".venv\\Scripts\\python.exe -m streamlit run app/streamlit_app.py",
    ])

    # ======================================================================
    # SECTION 9 -- File Map
    # ======================================================================
    pdf.add_page()
    pdf.h1("Section 9 -- Complete File Map")

    pdf.body("Every file in the project and what it does:")

    sections = [
        ("src/ -- source code", [
            ("src/data_cleaning.py", "Clean raw CSV data. Shared by notebook (training) and predict.py (inference)."),
            ("src/features.py", "Add 5 engineered features. Shared by notebook and predict.py."),
            ("src/predict.py", "predict_churn(dict) -> {probability, tier, factors}. The Part 1.5 function."),
            ("src/llm_client.py", "Provider abstraction: AnthropicClient, OllamaClient, MockClient."),
            ("src/agent/tools.py", "TOOL_REGISTRY: 5 tool schemas + implementations. Add 6th tool here."),
            ("src/agent/orchestrator.py", "Generic tool-calling loop. Provider-agnostic. Returns AgentResult."),
            ("src/agent/prompts.py", "SYSTEM_PROMPT: chain order, ambiguity, escalation, response format."),
            ("src/agent/mock_db.py", "lookup_customer (CSV-backed) + get_retention_offers (offer catalog)."),
        ]),
        ("eval/ -- evaluation", [
            ("eval/test_suite.py", "14 TestCase dataclasses covering all required categories."),
            ("eval/metrics.py", "3 deterministic metrics: tool_selection, param_extraction, completeness."),
            ("eval/judge.py", "LLM-as-judge: anchored 1/3/5 rubric, 4 dimensions, separate model."),
            ("eval/run_eval.py", "Orchestrates full eval -> scorecard.md + scorecard.json."),
            ("eval/results/scorecard.md", "Generated aggregate scorecard with per-case results."),
        ]),
        ("app/ -- demo", [
            ("app/streamlit_app.py", "Chat UI + tool-call timeline + mock/live mode banner."),
        ]),
        ("notebooks/ -- Part 1", [
            ("notebooks/churn_model.ipynb", "Full narrative notebook: cleaning, EDA, models, export."),
            ("notebooks/_build_notebook.py", "Programmatic notebook builder for reproducibility."),
        ]),
        ("models/ -- artifacts", [
            ("models/churn_pipeline.joblib", "Trained sklearn pipeline (ColumnTransformer + LogisticRegression)."),
            ("models/model_metadata.json", "Thresholds, feature directions/stats, importances, metrics."),
        ]),
        ("tests/ -- tests", [
            ("tests/test_predict.py", "5 tests for predict_churn (high/low/partial/empty/schema)."),
            ("tests/test_agent.py", "7 tests for agent tool chain (chain, ambiguity, escalation, OOS...)."),
            ("tests/test_eval_suite.py", "3 tests: categories present, metrics run, scorecard written."),
            ("tests/test_churn_model.py", "Full model evaluation report: algorithm, metrics, importances."),
            ("tests/run_all_tests.py", "Single-command runner: 15 tests, no pytest needed."),
            ("tests/generate_model_comparison.py", "Generates docs/model_comparison.md with all model metrics."),
            ("tests/generate_agent_pdf.py", "Generates this PDF document."),
        ]),
        ("docs/ -- generated reports", [
            ("docs/model_comparison.md", "Side-by-side metrics for LogisticRegression vs XGBoost."),
            ("docs/TeleConnect_Agent_Part2_Guide.pdf", "This document."),
        ]),
        ("data/ -- data files", [
            ("data/test_datafile.csv", "Raw dataset (5,050 rows, deliberately messy)."),
            ("data/cleaned_customers.csv", "Cleaned dataset (5,000 rows) -- also backs lookup_customer."),
        ]),
    ]

    for section_title, files in sections:
        pdf.h2(section_title)
        for fname, desc in files:
            pdf.set_font("Courier", "B", 8.5)
            pdf.set_text_color(*MID_BLUE)
            pdf.set_x(20)
            pdf.cell(75, 5.5, fname)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(*DARK_GREY)
            pdf.multi_cell(95, 5.5, desc)

    # final note
    pdf.ln(5)
    pdf.note_box(
        "Run tests/run_all_tests.py to verify the full system end-to-end.\n"
        "Run tests/generate_model_comparison.py to regenerate the model metrics report.\n"
        "Run python -m eval.run_eval to regenerate the agent scorecard.\n"
        "Run streamlit run app/streamlit_app.py to launch the live demo.",
        color=(235, 245, 235), text_color=GREEN
    )

    # ---- save ---------------------------------------------------------------
    pdf.output(str(OUT))
    print(f"Written: {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    pdf = PDF()
    build(pdf)