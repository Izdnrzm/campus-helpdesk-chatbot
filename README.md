# Campus IT Helpdesk RAG Chatbot

A domain-specific chatbot that answers student IT support questions (passwords, Wi-Fi, VPN, MFA, email, printing, e-learning, tickets) using **Retrieval-Augmented Generation (RAG)** over our own dataset, `faq.csv` (36 Question/Answer pairs).

**LLM provider:** Google Gemini API (free tier) - `gemini-embedding-001` for retrieval, `gemini-2.5-flash` for answers.

## Architecture

```
            faq.csv  (Question, Answer)
                |   one document per row
                v
   Gemini embeddings  -->  cached vectors (.cache/)
                                  |
 student question --> embed --> cosine similarity --> top-3 rows above threshold
                                                            |
                                       none relevant        v
                      "contact the Helpdesk"   <--   Gemini LLM answers ONLY from
                                                     the retrieved rows (+ last 3 chat turns)
                                                            |
                                                            v
                                              reply + cited FAQ rows
```

- **Grounded:** the system prompt forbids outside knowledge, so answers come from the dataset, not the model's guesses.
- **Follow-ups:** "and if that fails?" works because the last on-topic question and recent chat turns are used.
- **Fallbacks:** no API key -> offline TF-IDF retrieval returns the best stored answer; LLM error -> stored answer; nothing relevant found -> Helpdesk message.
- **Sources shown:** each reply lists the `faq.csv` rows it used (type `sources` to toggle).

## Setup

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free key at https://aistudio.google.com/apikey, then either set it in the terminal:

```bash
export GEMINI_API_KEY=your-key-here                    # PowerShell: $env:GEMINI_API_KEY="your-key-here"
```

or copy `.env.example` to `.env` and put the key there (`.env` is git-ignored).

## Run

```bash
python chatbot.py
```

Commands: `sources` (toggle citations), `reset` (clear history), `quit`.

The first run embeds the 36 rows and caches them in `.cache/`; later runs start instantly. Editing `faq.csv` rebuilds the cache automatically.

On GitHub: open the repo -> **Code** -> **Codespaces** -> **Create codespace**, then run the commands above in its terminal (add `GEMINI_API_KEY` under repo Settings -> Secrets and variables -> Codespaces).

## Evaluate retrieval

```bash
python eval.py
```

Runs 32 paraphrased student questions plus 4 off-topic ones and reports Hit@1 / Hit@3 and off-topic rejection. Run it with and without `GEMINI_API_KEY` to compare semantic embeddings against the keyword baseline.

Baseline (TF-IDF, no API): Hit@1 72%, Hit@3 88%, off-topic rejected 3/4. Its misses are synonym cases (e.g. "scam email" vs "phishing"); record your Gemini numbers in your report.

## Files

| File | Purpose |
|------|---------|
| `faq.csv` | Own dataset: Question and Answer columns |
| `rag.py` | RAG core: indexing, retrieval, grounded generation, fallbacks |
| `chatbot.py` | Command-line chat interface |
| `eval.py` | Retrieval evaluation |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for the API key (never commit the real key) |

## Limitations

- The dataset is small (36 rows) and the portal names and URLs are fictional examples; replace them with your institution's real details.
- Similarity thresholds (`MIN_SCORE_EMBED`, `MIN_SCORE_TFIDF` in `rag.py`) should be tuned on your own test questions.
- Gemini free-tier limits apply; the bot degrades to stored answers if they are hit.
