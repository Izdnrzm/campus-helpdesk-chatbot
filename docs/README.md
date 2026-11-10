# Campus IT Helpdesk Chatbot

A domain-specific chatbot that answers student IT support questions (passwords, Wi-Fi, VPN, MFA, printing, e-learning, tickets and more).

## How it works

1. **Own dataset** - `faq.csv` has 36 hand-written `Question`/`Answer` pairs.
2. **Retrieval** - the user's message is matched against the dataset with TF-IDF and cosine similarity.
3. **Generation (optional)** - the top matches are sent to the Gemini API, which writes the reply using *only* that context (RAG), reducing hallucinations. Without an API key the bot returns the best-matching answer directly.
4. **Fallback** - if nothing matches well, the bot points the student to the Helpdesk.

## Setup

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional - enable the LLM (free key from https://aistudio.google.com/apikey):

```bash
export GEMINI_API_KEY=your-key-here      # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
```

## Run

```bash
python chatbot.py
```

Example:

```
You: I forgot my password
Bot: Go to the self-service portal, click 'Forgot password', and enter your student ID...
```

## Project structure

| File | Purpose |
|------|---------|
| `faq.csv` | Custom Q&A dataset |
| `chatbot.py` | Retrieval + optional Gemini answer generation, CLI loop |
| `requirements.txt` | Python dependencies |
| `.env.example` | Shows the API key variable (never commit the real key) |

## Extending

- Add more rows to `faq.csv`; the bot picks them up on restart.
- Swap `build_llm()` / `answer()` to use OpenAI or Hugging Face instead of Gemini.
- Tune `MIN_SCORE` and `TOP_K` in `chatbot.py`.
