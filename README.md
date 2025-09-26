# Nasscom_Hackathon_Team_Polici
Our demo code mimicking the inya developed agent



````markdown
# Agentic LLM-Only Service (Insurance Renewal & Upsell Agent)

This project implements a backend service using **Flask** and the **OpenAI API** to create a simple, stateful **Agentic AI** capable of handling insurance policy interactions, primarily focusing on renewals and upsells. It operates using a series of LLM calls for parsing, Natural Language Understanding (NLU), sentiment analysis, and the core agentic decision-making logic.

## 🚀 Features

* **Policy Document Parsing:** Extracts key fields from raw policy text (e.g., PDF content) into a structured JSON format.
* **Natural Language Understanding (NLU):** Classifies user intent (e.g., `renew_now`, `needs_discount`, `escalate_human`) and extracts entities.
* **Sentiment Analysis:** Determines the user's sentiment (`positive`, `neutral`, `negative`).
* **Conversational Agent (`/v1/agent/message`):** A core agent endpoint that uses policy context, NLU, and sentiment to generate a response, determine the next action (e.g., `offer_renewal`, `initiate_payment`), and maintain a short-term conversation.
* **Mock Utility Endpoints:** Includes mock endpoints for payment initiation and SMS notification.
* **SQLite Persistence:** Uses a lightweight SQLite database to store parsed policies and conversation history.

## 🛠️ Requirements

* Python 3.8+
* An OpenAI API Key

## ⚙️ Setup and Installation

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd agentic-llm-service
````

### 2\. Create a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3\. Install Dependencies

The project uses `Flask`, `openai`, `pydantic`, and `python-dotenv`.

```bash
pip install -r requirements.txt
# If you don't have a requirements.txt, you can install them directly:
# pip install Flask openai pydantic python-dotenv
```

### 4\. Configure Environment Variables

Create a file named `.env` in the root directory and add your OpenAI API Key.

**.env**

```
# Get your key from the OpenAI platform
OPENAI_API_KEY="sk-..."

# Optional: Change the LLM model. Default is gpt-4o-mini
# LLM_MODEL="gpt-4-turbo"
```

### 5\. Run the Application

The script will initialize the SQLite database (`agentic_ai.db`) and start the Flask server.

```bash
export OPENAI_API_KEY="sk-..."  # Use the key if you don't use a .env file
python Main.py
```

The service will be running at `http://0.0.0.0:5000`.

## 🔗 Endpoints

The API is structured with versioning (`/v1/`) and categorical groupings.

| Endpoint | Method | Description | Request Body Example |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | Simple health check. | N/A |
| `/v1/policy/parse` | `POST` | Uses the LLM to extract structured data from a raw policy text. | `{"text": "Policy #ABC123..."}` |
| `/v1/nlp/intent` | `POST` | Classifies user intent and extracts entities. | `{"text": "I want to pay now."}` |
| `/v1/nlp/sentiment` | `POST` | Determines the sentiment of the user's message. | `{"text": "That's great news!"}` |
| `/v1/agent/message` | `POST` | The main conversational agent. Uses all context to generate a reply and next action. | `{"session_id": "...", "policy_number": "...", "message": "..."}` |
| `/v1/payments/initiate` | `POST` | Mock endpoint to initiate a payment and generate a link. | `{"policy_number": "ABC123", "amount": 100.50}` |
| `/v1/notify/sms` | `POST` | Mock endpoint to send a notification (SMS). | `{"to": "+15551234567", "text": "Your policy is expiring."}` |

## 💡 Agent Workflow (`/v1/agent/message`)

The agent endpoint orchestrates the following sequence to generate a response:

1.  **Log:** The incoming user message is logged to the `conversations` table.
2.  **Context Fetch:** If a `policy_number` is provided, the parsed policy is fetched from the `policies` database table.
3.  **NLU:** The user message is sent to the LLM to determine **intent** (e.g., `renew_now`) and **entities**.
4.  **Sentiment:** The user message is sent to the LLM for **sentiment** analysis.
5.  **Agent Logic:** All gathered context (user message, policy data, NLU, sentiment, and a defined `SYSTEM_AGENT_POLICY` prompt) is sent to the core LLM.
6.  **Action Check:** The LLM's structured JSON output (`reply`, `action`, `action_payload`) is processed. A mock payment link is generated if the action is `initiate_payment`.
7.  **Log:** The agent's generated reply is logged.
8.  **Respond:** The final structured JSON response is returned to the client.

This design demonstrates a powerful pattern where the LLM performs all analytical and decision-making steps, acting as the brain for the entire agent architecture.

## 📝 Data Schema

The SQLite database (`agentic_ai.db`) uses two main tables:

### `policies`

Stores the structured data extracted from policy documents.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `TEXT` | UUID primary key. |
| `policy_number` | `TEXT` | The customer's policy number. |
| `customer_name` | `TEXT` | Customer's name. |
| `expiry_date` | `TEXT` | Policy expiry date (YYYY-MM-DD). |
| `premium_amount` | `REAL` | Renewal premium. |
| `eligible_upsells` | `TEXT` | JSON array of potential add-ons. |
| `raw_parse` | `TEXT` | Full JSON output from the parser LLM call. |

### `conversations`

Stores a simple log of messages for each session.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `TEXT` | UUID primary key. |
| `session_id` | `TEXT` | Identifier for the conversation session. |
| `role` | `TEXT` | `user` or `agent`. |
| `message` | `TEXT` | The text content of the message. |
| `created_at` | `TEXT` | Timestamp (ISO format). |

```
```
