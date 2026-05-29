# TeleConnect Agent — Sample Chat Inputs

Use these prompts to test the Streamlit demo. Each covers a different scenario.

---

## 1. Happy Path — Full Retention Chain

### 1.1 High-Risk Customer (Expected: lookup → predict → offers → synthesize)
```
Customer TC-001096 called saying they might cancel. What should I do?
```

**Agent will:**
- Look up TC-001096 (month-to-month, low satisfaction 7.8, 4 support tickets)
- Predict: HIGH risk (76% churn probability)
- Get offers: 20% Loyalty Discount, Free Upgrade, Service Recovery Credit
- Synthesize: Recommend 20% discount tied to 1-year commitment

**Expected output:** Specific offer, talking point, top 3 risk factors

---

### 1.2 Low-Risk Customer (Expected: similar chain but margin-preserving offers)
```
Customer TC-004460 — is she a churn risk?
```

**Agent will:**
- Look up TC-004460 (two-year contract, high satisfaction 9.0, 0 tickets)
- Predict: LOW risk (19% churn probability)
- Get offers: Free Add-On Service, Data/Minutes Boost, Goodwill Check-In
- Synthesize: Goodwill check-in (no discount, preserve margin)

**Expected output:** Low risk confirmed, relationship-preserving action

---

### 1.3 Medium-Risk Customer
```
Can you check on customer TC-002360? They're on the verge of switching to a competitor.
```

**Agent will:**
- Look up TC-002360
- Predict: MEDIUM or HIGH risk
- Get medium-tier offers
- Synthesize with urgency

---

## 2. Ambiguous Input (Expected: agent asks for customer ID)

### 2.1 No Customer ID Given
```
I have a risky customer on the phone. What retention offers can I make?
```

**Agent will:** Ask for customer ID (TC-XXXXX format)

**Expected:** No tools called, clarification request

---

### 2.2 Vague Description
```
The person I'm talking to mentioned they're unhappy with service quality. What should I offer?
```

**Agent will:** Ask for customer ID

---

### 2.3 Name Only (Not Enough)
```
John Smith is threatening to leave. Help!
```

**Agent will:** Ask for customer ID, not name

---

## 3. Escalation Scenarios (Expected: escalate_to_supervisor tool)

### 3.1 Legal Threat
```
Customer TC-003356 is saying they will sue us over billing issues. What do I do?
```

**Agent will:**
- Recognize legal threat
- Call escalate_to_supervisor
- NOT call get_retention_offers (inappropriate)

**Expected:** Escalation ticket created, supervisor notified

---

### 3.2 Regulatory Threat
```
Customer TC-005000 says they're filing a complaint with the FCC. This is serious.
```

**Agent will:** Escalate (regulatory action trigger)

---

### 3.3 Complex Dispute
```
Customer TC-001234 has a billing dispute from 6 months ago that's never been resolved. They want a $500 credit. I don't have authority for that.
```

**Agent will:** Escalate (complex, high-value dispute)

---

### 3.4 Highly Distressed Customer
```
Customer TC-002500 is crying on the phone. They lost their job and can't afford service anymore. What offer would help?
```

**Agent will:** Escalate (hardship case, needs human judgment)

---

## 4. Out-of-Scope Requests (Expected: decline, NO tools called)

### 4.1 Weather Question
```
What's the weather like in San Francisco today?
```

**Agent will:** Politely decline, redirect to retention focus

---

### 4.2 Password Reset
```
The customer forgot their password. Can you help them reset it?
```

**Agent will:** Decline (not a retention tool), suggest they contact IT

---

### 4.3 Billing System Question
```
How do I process a refund in our billing system?
```

**Agent will:** Decline (internal process, not retention), suggest asking supervisor

---

### 4.4 Jokes / Small Talk
```
Tell me a joke to lighten the mood.
```

**Agent will:** Politely decline, redirect

---

## 5. Unknown / Invalid Customer ID (Expected: lookup fails, agent stops)

### 5.1 Non-Existent Customer
```
Look up customer TC-999999 and tell me their churn risk.
```

**Agent will:**
- Call lookup_customer(TC-999999)
- Get error: "No customer found with id 'TC-999999'"
- Tell rep to re-check ID
- NOT call predict_churn on non-existent customer

**Expected:** Error message, stops gracefully

---

### 5.2 Malformed ID
```
Can you pull up the profile for customer John-Smith-123?
```

**Agent will:** Lookup fails, ask for proper TC-XXXXX format

---

## 6. Conflicting Signals (Expected: real LLM flags the conflict, mock may miss it)

### 6.1 Model Says Low, But Profile Looks Bad
```
The model says TC-002360 is low risk, but they have only 2.7 out of 10 satisfaction and 4 support tickets. Should I trust the low score?
```

**Agent will (REAL Claude):**
- Look up customer (low satisfaction, high tickets)
- Predict: LOW risk (per model)
- **Flag the contradiction:** "Model says low, but satisfaction=2.7 and 4 tickets are warning signs. I'd treat them as MEDIUM risk."
- Recommend more aggressive offer

**Agent will (MOCK):** Just reports low risk (limitation documented)

**Note:** This is a test case that passes with real LLM, fails with mock

---

## 7. Chaining with Conditions

### 7.1 Decision Tree: "What offer matches this customer's budget?"
```
Customer TC-001096 said they can afford a maximum $15/month extra. What's the best retention offer?
```

**Agent will:**
- Lookup TC-001096
- Predict: HIGH risk
- Get offers: Loyalty Discount (14/month), Service Recovery (25/month), etc.
- Recommend Loyalty Discount (fits budget)

---

### 7.2 Multiple Customers in One Turn
```
I have three at-risk customers on calls: TC-001096, TC-004460, and TC-002500. Which ones should I prioritize?
```

**Agent will:**
- Handle only ONE customer (it doesn't multi-task)
- Ask: "Which customer should we focus on first?"

**Expected:** Agent stays focused on single-customer workflows

---

## 8. Real Customer IDs from the Dataset

Use these actual IDs (from `data/cleaned_customers.csv`) for realistic testing:

| ID | Profile | Expected Outcome |
|---|---|---|
| **TC-001096** | Month-to-month, 10m tenure, satisfaction 7.8, 4 tickets | HIGH risk (76%) |
| **TC-004460** | Two-year, 60m tenure, satisfaction 9.0, 0 tickets | LOW risk (19%) |
| **TC-002360** | Month-to-month, 20m tenure, satisfaction 2.7, 4 tickets | HIGH risk (conflicting signals) |
| **TC-003356** | One-year, 30m tenure, satisfaction 6.0 | MEDIUM risk |
| **TC-001234** | Month-to-month, 5m tenure, satisfaction 3.0 | HIGH risk (early churn) |
| **TC-005000** | Fiber optic, two-year, satisfaction 8.0 | LOW risk |
| **TC-002500** | DSL, month-to-month, satisfaction 2.0, 8 tickets | VERY HIGH risk |

---

## 9. Stress Tests (Edge Cases)

### 9.1 Very Old Customer
```
Check customer TC-000001. They've been with us since the beginning.
```

**Agent will:** Look up, predict (likely very sticky, low churn)

---

### 9.2 New Customer
```
TC-010000 just signed up last month. Are they at risk?
```

**Agent will:** Look up, predict (early tenure = higher risk, capture with tenure_bucket feature)

---

### 9.3 Heavy Complainer
```
Customer TC-003000 has called support 15 times this month. Should I offer something?
```

**Agent will:**
- Predict: HIGH (support ticket count is strong signal)
- Offer: Service Recovery Credit (includes 90 days priority support)

---

## 10. Production-Like Scenarios

### 10.1 Churn Predicted But Customer Sounds Happy
```
The model says TC-004500 is medium risk, but they just said they love our service and are thinking about upgrading. What's going on?
```

**Agent will (REAL):**
- Flag potential false positive
- Recommend: Data/Minutes Boost (low-cost, engagement signal)
- Note: Might be retention-happy churn signal (happy but month-to-month = flight risk)

---

### 10.2 Customer Wants to Switch
```
TC-002100 says a competitor is offering 40% off for 12 months. We're losing them. What can we do?
```

**Agent will:**
- Look up
- Predict churn risk
- Get best offers
- Acknowledge price competition
- Synthesize: "If price is the only issue, 20% Loyalty Discount may not win. This might be an escalation case."

---

### 10.3 Retention Success Story
```
We offered TC-001500 a deal last week and they accepted. Just confirming: what should I log?
```

**Agent will:**
- Look up TC-001500
- Log the interaction with outcome=accepted
- Suggest: "Great retention win! This interaction is now logged for analytics."

---

## 11. What NOT to Try (Known Limitations)

### 11.1 Chat Context — Agent Doesn't Remember Previous Turns
```
Turn 1: "Look up TC-001096"
[Agent returns profile]
Turn 2: "What offers should I make?" 
```

**Issue:** Agent forgot TC-001096 from Turn 1. You must repeat the ID.

**Correct:** "What offers for TC-001096?"

---

### 11.2 Multiple Tools Called Out of Order
```
"Get me an offer first, then look up the customer."
```

**Issue:** Agent ignores this and chains tools correctly per system prompt.

**Expected:** lookup → predict → offers (always)

---

### 11.3 Numerical Comparisons
```
"Which is riskier: TC-001096 or TC-004460?"
```

**Issue:** Agent doesn't do cross-customer comparison. Only single-customer workflows.

**Agent:** "I can help with one customer at a time. Which would you like to focus on?"

---

## 12. Testing the Tool Timeline (UI Feature)

After you send a message, **expand the "🔧 Tool Chain" section** to see:

```
1. lookup_customer   2.1 ms
   Input:  {customer_id: "TC-001096"}
   Output: {found: true, tenure_months: 10, satisfaction_score: 7.8, ...}

2. predict_churn   0.8 ms
   Input:  {customer_data: {...}}
   Output: {churn_probability: 0.76, risk_tier: "high", top_risk_factors: [...]}

3. get_retention_offers   0.3 ms
   Input:  {risk_tier: "high", contract_type: "Month-to-month"}
   Output: {count: 3, offers: [...]}
```

This is the **orchestration trace** — proof the agent chained tools correctly.

---

## Quick Test Checklist

Run these 5 prompts in order to verify the system end-to-end:

1. **Happy path:** `Customer TC-001096 called saying they might cancel. What should I do?`
   - ✓ Should chain: lookup → predict → offers → synthesize

2. **Ambiguous:** `I have a risky customer on the phone, what can I offer?`
   - ✓ Should ask for customer ID (no tools called)

3. **Escalation:** `Customer TC-003356 is saying they will get a lawyer.`
   - ✓ Should escalate (no offers)

4. **Out-of-scope:** `What's the weather today?`
   - ✓ Should decline (no tools called)

5. **Unknown ID:** `Look up customer TC-999999.`
   - ✓ Should tell you it wasn't found (stops after lookup)

✅ If all 5 work as expected, the agent is functioning correctly.

---

## How to Run

1. **Start Streamlit:**
   ```powershell
   .venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
   ```

2. **Open:** `http://localhost:8501`

3. **Copy a prompt** from above into the chat input box

4. **Hit Enter** and watch the tool timeline appear

5. **Expand 🔧 Tool Chain** to see every tool call in order

---

## Expected Behavior Summary

| Scenario | Expected Tool Chain | Chat Response Style |
|---|---|---|
| High-risk lookup | lookup → predict → offers | Urgent, specific offer, talking point |
| Low-risk lookup | lookup → predict → offers | Reassuring, margin-friendly offer |
| No ID given | (none) | Ask for customer ID |
| Legal threat | escalate_to_supervisor | Confirm escalation, hand-off context |
| Out-of-scope | (none) | Polite decline, redirect |
| Unknown ID | lookup (fails) | "Not found, please re-check" |
| Conflicting signals | lookup → predict | Flag the tension (real LLM only) |

---

**More examples?** Open `data/cleaned_customers.csv` and pick any customer ID (TC-XXXXX format) to test.
