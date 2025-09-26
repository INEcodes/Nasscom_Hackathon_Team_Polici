"""
Agentic AI using only an LLM (Flask app).
Endpoints:
- POST /v1/policy/parse
- POST /v1/nlp/intent
- POST /v1/nlp/sentiment
- POST /v1/agent/message
- POST /v1/payments/initiate
- POST /v1/notify/sms

Run:
export OPENAI_API_KEY="sk-..."
python app.py
"""
import os
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from pydantic import BaseModel
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # change to the model you have

app = Flask(__name__)
DB_PATH = "agentic_ai.db"


# ---------------------------
# DB utilities (very light)
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # policies: store parsed policies
    c.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            id TEXT PRIMARY KEY,
            policy_number TEXT,
            customer_name TEXT,
            customer_contact TEXT,
            policy_type TEXT,
            insurer_name TEXT,
            expiry_date TEXT,
            premium_amount REAL,
            no_claim_bonus_percent REAL,
            eligible_upsells TEXT,
            raw_parse TEXT,
            created_at TEXT
        )
    """)
    # sessions/conversations
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            message TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_policy(parsed_obj):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pid = str(uuid.uuid4())
    c.execute(
        "INSERT INTO policies (id, policy_number, customer_name, customer_contact, policy_type, insurer_name, expiry_date, premium_amount, no_claim_bonus_percent, eligible_upsells, raw_parse, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            pid,
            parsed_obj.get("policy_number"),
            parsed_obj.get("customer_name"),
            parsed_obj.get("customer_contact"),
            parsed_obj.get("policy_type"),
            parsed_obj.get("insurer_name"),
            parsed_obj.get("expiry_date"),
            parsed_obj.get("premium_amount"),
            parsed_obj.get("no_claim_bonus_percent"),
            json.dumps(parsed_obj.get("eligible_upsell") or []),
            json.dumps(parsed_obj),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return pid


def get_policy_by_number(policy_number):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT raw_parse FROM policies WHERE policy_number = ?", (policy_number,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


def log_conversation(session_id, role, message):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cid = str(uuid.uuid4())
    c.execute(
        "INSERT INTO conversations (id, session_id, role, message, created_at) VALUES (?,?,?,?,?)",
        (cid, session_id, role, message, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


# ---------------------------
# LLM helper
# ---------------------------
def call_llm(messages, temperature=0.0, max_tokens=800):
    """
    messages: [{'role':'system'|'user'|'assistant', 'content': '...'}, ...]
    """
    if openai.api_key is None:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )
    return resp.choices[0].message["content"]


# ---------------------------
# Prompt templates
# ---------------------------
SYSTEM_PARSER_PROMPT = """
You are a professional insurance document parser. Input is either raw text extracted from a policy PDF or a description. Output MUST be a strict JSON object only (no explanatory text) with these fields:
- policy_number (string or null)
- policy_type (string or null)
- insurer_name (string or null)
- customer_name (string or null)
- customer_contact (string or null, E.164 if possible)
- expiry_date (YYYY-MM-DD or null)
- premium_amount (number or null)
- no_claim_bonus_percent (number between 0-100 or null)
- asset_details (object with keys: make, model, year, registration_number - any can be null)
- coverage_summary (array of strings)
- eligible_upsell (array of strings)
- last_payment_date (YYYY-MM-DD or null)

If value is not present in the text, set it to null or an empty array/object as appropriate.
Make a best-effort extraction. STRICT JSON output and nothing else.
"""

SYSTEM_NLU_PROMPT = """
You are a compact NLU assistant. Given user text, return a JSON with:
- intent: one of [renew_now, renew_later_date, needs_discount, needs_human_agent, modify_policy, interested_in_upsell, switching_to_competitor, not_interested, callback_request, out_of_scope]
- confidence: 0.0 - 1.0
- entities: dict of any extracted slots (policy_number, followup_date, preferred_channel, language, upsell_choice, payment_method, contact)
Return only JSON, nothing else.
"""

SYSTEM_SENTIMENT_PROMPT = """
You are a sentiment classifier. Input: a short user utterance. Return JSON:
- sentiment: one of [positive, neutral, negative]
- score: 0.0 - 1.0 (confidence)
Return only JSON.
"""

SYSTEM_AGENT_POLICY = """
You are an Insurance Renewal & Upsell virtual agent. Behavior rules:
1. Always confirm customer identity (customer_name and policy_number) if not confirmed.
2. If expiry_date within 30 days, proactively offer renewal. Show premium and NCB.
3. Check eligible_upsell and propose simple, short add-ons if sentiment positive and user willing.
4. If user requests payment, call payment initiation (return payment_link).
5. Keep messages short, empathetic, and in the user's preferred language (English/Hindi). If language not provided, default English.
6. Respect opt-out and DND: if user says 'stop' or 'unsubscribe', log opt-out and confirm.
7. Output must be a JSON object with keys:
   - reply (string): text to send to the user
   - language (string): 'en' or 'hi'
   - action (string or null): one of [ask_for_missing_info, offer_renewal, initiate_payment, upsell_offer, schedule_callback, escalate_human, none]
   - action_payload (object or null): additional instructions (e.g., payment_link, amount, followup_date)
Return only JSON.
"""

# ---------------------------
# Endpoints
# ---------------------------

@app.route("/v1/policy/parse", methods=["POST"])
def parse_policy():
    """
    Accepts: { "text": "<extracted text from pdf or pasted policy content>" }
    Returns: parsed JSON (as described in SYSTEM_PARSER_PROMPT)
    """
    body = request.json or {}
    text = body.get("text") or body.get("file_url") or ""
    if not text:
        return jsonify({"error": "Provide 'text' (extracted PDF text) or 'file_url' (text)."}), 400

    messages = [{"role": "system", "content": SYSTEM_PARSER_PROMPT},
                {"role": "user", "content": text}]
    try:
        llm_out = call_llm(messages, temperature=0.0, max_tokens=600)
        # ensure it's JSON
        parsed = json.loads(llm_out)
    except Exception as e:
        # fallback: ask LLM to re-output only json
        try:
            retry_msg = [
                {"role": "system", "content": SYSTEM_PARSER_PROMPT},
                {"role": "user", "content": "Please re-output ONLY the JSON object. Previous attempt error: " + str(e) + "\n\nOriginal text:\n" + text}
            ]
            llm_out = call_llm(retry_msg, temperature=0.0, max_tokens=600)
            parsed = json.loads(llm_out)
        except Exception as e2:
            return jsonify({"error": "LLM parsing failed", "details": str(e2)}), 500

    # store in DB
    pid = save_policy(parsed)
    parsed["_db_id"] = pid
    return jsonify(parsed)


@app.route("/v1/nlp/intent", methods=["POST"])
def nlu_intent():
    """
    Request: { "text": "I will renew today" }
    Response: JSON {intent, confidence, entities}
    """
    body = request.json or {}
    text = body.get("text", "")
    if not text:
        return jsonify({"error": "text required"}), 400

    messages = [{"role": "system", "content": SYSTEM_NLU_PROMPT},
                {"role": "user", "content": text}]
    try:
        out = call_llm(messages, temperature=0.0, max_tokens=200)
        parsed = json.loads(out)
    except Exception as e:
        return jsonify({"error": "NLU LLM error", "details": str(e)}), 500

    return jsonify(parsed)


@app.route("/v1/nlp/sentiment", methods=["POST"])
def nlu_sentiment():
    body = request.json or {}
    text = body.get("text", "")
    if not text:
        return jsonify({"error": "text required"}), 400

    messages = [{"role": "system", "content": SYSTEM_SENTIMENT_PROMPT},
                {"role": "user", "content": text}]
    try:
        out = call_llm(messages, temperature=0.0, max_tokens=100)
        parsed = json.loads(out)
    except Exception as e:
        return jsonify({"error": "Sentiment LLM error", "details": str(e)}), 500

    return jsonify(parsed)


@app.route("/v1/agent/message", methods=["POST"])
def agent_message():
    """
    Main agent endpoint.
    Accepts:
    {
      "session_id": "optional session id",
      "policy_number": "optional",
      "language": "en" or "hi",
      "message": "user message text"
    }
    Returns:
    JSON described by SYSTEM_AGENT_POLICY
    """
    body = request.json or {}
    session_id = body.get("session_id") or str(uuid.uuid4())
    policy_number = body.get("policy_number")
    language = body.get("language", "en")
    user_text = body.get("message", "")

    if not user_text:
        return jsonify({"error": "message required"}), 400

    # log user message
    log_conversation(session_id, "user", user_text)

    # If policy_number provided, fetch parsed policy (if exists)
    policy = None
    if policy_number:
        policy = get_policy_by_number(policy_number)

    # 1) Ask the LLM for intent
    try:
        nlu_payload = call_llm(
            [{"role": "system", "content": SYSTEM_NLU_PROMPT},
             {"role": "user", "content": user_text}], temperature=0.0, max_tokens=200
        )
        nlu = json.loads(nlu_payload)
    except Exception:
        nlu = {"intent": "out_of_scope", "confidence": 0.5, "entities": {}}

    # 2) Sentiment
    try:
        sent_payload = call_llm(
            [{"role": "system", "content": SYSTEM_SENTIMENT_PROMPT},
             {"role": "user", "content": user_text}], temperature=0.0, max_tokens=80
        )
        sent = json.loads(sent_payload)
    except Exception:
        sent = {"sentiment": "neutral", "score": 0.5}

    # 3) Compose agent policy prompt with context
    policy_context = json.dumps(policy) if policy else "{}"
    agent_user_prompt = f"""
User message: {user_text}
Policy context: {policy_context}
NLU: {json.dumps(nlu)}
Sentiment: {json.dumps(sent)}
Preferred language: {language}

Following the agent behavior rules, produce the agent response JSON exactly with keys:
reply, language ('en' or 'hi'), action (ask_for_missing_info | offer_renewal | initiate_payment | upsell_offer | schedule_callback | escalate_human | none), action_payload (object|null).

Keep reply short and polite. If asking for missing info, be specific about which field is missing (e.g., policy_number or customer_contact). If offering renewal, include premium and expiry_date. If initiating payment, include payment link.
    """
    try:
        agent_out = call_llm(
            [{"role": "system", "content": SYSTEM_AGENT_POLICY},
            {"role": "user", "content": agent_user_prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        agent_json = json.loads(agent_out)
    except Exception as e:
        # Fallback short reply
        agent_json = {
            "reply": "I'm sorry, I'm having trouble right now. I'll escalate to a human agent.",
            "language": language,
            "action": "escalate_human",
            "action_payload": {"reason": str(e)}
        }

    # If action is initiate_payment and action_payload has amount, create a mock payment_link
    if agent_json.get("action") == "initiate_payment":
        payload = agent_json.get("action_payload") or {}
        amount = payload.get("amount") or (policy.get("premium_amount") if policy else None)
        payment_id = "pay_" + str(uuid.uuid4())[:8]
        payment_link = f"https://mockpay.example/pay/{payment_id}"
        agent_json["action_payload"] = {"payment_id": payment_id, "payment_link": payment_link, "amount": amount}
        # Log a "system" message
    # Log agent reply
    log_conversation(session_id, "agent", agent_json.get("reply", ""))

    # Return
    response = agent_json
    return jsonify(response)


@app.route("/v1/payments/initiate", methods=["POST"])
def payments_initiate():
    body = request.json or {}
    policy_number = body.get("policy_number")
    amount = body.get("amount")
    if not amount:
        return jsonify({"error": "amount required"}), 400
    payment_id = "pay_" + str(uuid.uuid4())[:8]
    link = f"https://mockpay.example/pay/{payment_id}"
    # in production: create idempotent record etc
    return jsonify({"id": payment_id, "payment_link": link})


@app.route("/v1/notify/sms", methods=["POST"])
def notify_sms():
    body = request.json or {}
    to = body.get("to")
    text = body.get("text")
    if not to or not text:
        return jsonify({"error": "to and text required"}), 400
    # mock send - in prod call provider
    return jsonify({"status": "sent", "id": "msg_" + str(uuid.uuid4())[:8]})


# ---------------------------
# Simple test / health
# ---------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})


if __name__ == "__main__":
    init_db()
    print("Starting agentic LLM-only service on port 5000")
    app.run(host="0.0.0.0", port=5000)
