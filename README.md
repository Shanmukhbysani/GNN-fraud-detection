# Customer Life Event Detection from Transactions using LLMs

Detects customer **life events** — house move, new baby, new job, wedding — from raw transaction history using a Large Language Model, and maps each event to relevant banking products. Built with an interactive UI.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shanmukhbysani/life-event-detector/blob/main/life_event_gradio.ipynb)

> Click the badge to run the interactive app in your browser.

---

## What it does

Banks hold every customer's transaction history. Buried in those merchant names are **life events** the bank could support at the right moment — home insurance when you move, a Junior ISA when a baby arrives. This system surfaces those events automatically.

**Flow:** transaction history → PII masking → structured LLM prompt → detected events with confidence, evidence, and product recommendations → interactive UI.

## Why it matters for a bank

| Lloyds use case | How this project addresses it |
|---|---|
| **Detect customer life events from transactions** | Core output — detected events with the exact transactions that triggered them |
| **Analyse transactions to surface financial needs** | Each event maps to recommended banking products |

## Data privacy (the key design decision)

Sending real customer data to an external LLM API would breach banking regulation (data residency, GDPR). So:

- **Prototype (this repo):** synthetic data + Groq cloud API, for demonstration only.
- **Production:** the same prompt runs on a **self-hosted open-source LLM (Llama 3 / Mistral)** inside the bank's private cloud — no customer data ever leaves the bank. PII is masked before any inference.

## How hallucination is controlled

- Low temperature (0.1) for consistent, factual output
- A **closed list** of allowed life events — the model can't invent new ones
- A hard rule: every detected event must **cite the transactions** that justify it
- Forced JSON output via the API's structured-output mode

## Tech stack

`Python` · `Pandas` · `Groq API (Llama 3)` · `Gradio` · `Prompt Engineering`

## How to run

**In Colab (easiest):**
1. Click the Colab badge above (or upload `life_event_gradio.ipynb`)
2. Run the single cell
3. Paste your free Groq API key when prompted (get one at console.groq.com/keys)
4. The interactive UI appears in the output — pick a customer and click **Analyse transactions**

**Live, permanent demo (Hugging Face Spaces):**
1. Create a free account at huggingface.co
2. New Space → SDK: Gradio
3. Add the app code as `app.py` and a `requirements.txt`
4. In *Settings → Secrets*, add `GROQ_API_KEY`
5. You get a permanent public URL — visitors use the app without needing their own key

## Limitations

- Real merchant strings are messier than synthetic ones — production needs a transaction-enrichment step first.
- Model-estimated confidence isn't a calibrated probability; I'd validate it against labelled outcomes.

## Author

**Shanmukh Bysani** — B.E. AI & Data Science, CBIT Hyderabad
