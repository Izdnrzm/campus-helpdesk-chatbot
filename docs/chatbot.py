"""Campus IT Helpdesk chatbot.

Pipeline (Retrieval-Augmented Generation):
  1. Load our own Question/Answer dataset from faq.csv.
  2. Retrieve the most similar Q&A pairs for the user's message (TF-IDF + cosine similarity).
  3. If a Gemini API key is set, ask the LLM to answer using ONLY those pairs.
     Otherwise fall back to returning the best-matching answer directly.

Run:  python chatbot.py
"""

import os
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "faq.csv")
TOP_K = 3                 # how many Q&A pairs to give the LLM as context
MIN_SCORE = 0.12          # below this similarity we say "I don't know"
GEMINI_MODEL = "gemini-2.0-flash"

# "campus"/"student" appear in almost every row, so they carry no signal.
STOP_WORDS = list(ENGLISH_STOP_WORDS | {"campus", "student", "students", "university"})

FALLBACK_MSG = (
    "Sorry, I don't have an answer for that. Please submit a ticket on the "
    "Helpdesk portal or visit the counter (Mon-Fri, 9 am-5 pm)."
)


def normalize(text):
    """Light cleanup so 'wifi', 'Wi-Fi' and 'wi fi' all match; common synonyms."""
    text = text.lower().replace("wi-fi", "wifi").replace("wi fi", "wifi")
    text = re.sub(r"\b(replies|reply|respond|response)\b", "reply", text)
    text = re.sub(r"\b(stolen|lost|broken)\b", "lost", text)
    return text


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Question", "Answer"]).reset_index(drop=True)
    return df


class Retriever:
    def __init__(self, df):
        self.df = df
        # Question is repeated so it outweighs the (longer) answer text.
        corpus = (df["Question"] + " " + df["Question"] + " " + df["Answer"]).tolist()
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize, stop_words=STOP_WORDS, ngram_range=(1, 2)
        )
        corpus = corpus
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, k=TOP_K):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).ravel()
        top = scores.argsort()[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]


def build_llm():
    """Return a Gemini model object, or None if no key / package available."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        print("[info] google-generativeai not installed; using retrieval-only mode.")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def answer(query, retriever, llm=None):
    hits = retriever.search(query)
    best_idx, best_score = hits[0]
    if best_score < MIN_SCORE:
        return FALLBACK_MSG

    if llm is None:  # retrieval-only mode
        return retriever.df.loc[best_idx, "Answer"]

    context = "\n\n".join(
        f"Q: {retriever.df.loc[i, 'Question']}\nA: {retriever.df.loc[i, 'Answer']}"
        for i, s in hits
        if s >= MIN_SCORE
    )
    prompt = (
        "You are a friendly Campus IT Helpdesk assistant. Answer the student's "
        "question using ONLY the reference Q&A below. Be concise. If the "
        "references do not contain the answer, reply exactly with: "
        f"\"{FALLBACK_MSG}\"\n\n"
        f"Reference Q&A:\n{context}\n\nStudent question: {query}\nAnswer:"
    )
    try:
        return llm.generate_content(prompt).text.strip()
    except Exception as exc:  # network/quota errors: degrade gracefully
        print(f"[warn] LLM call failed ({exc}); using retrieval-only answer.")
        return retriever.df.loc[best_idx, "Answer"]


def main():
    df = load_dataset()
    retriever = Retriever(df)
    llm = build_llm()
    mode = "LLM + retrieval (Gemini)" if llm else "retrieval-only"
    print(f"Campus IT Helpdesk bot [{mode}] - {len(df)} Q&A pairs loaded.")
    print("Type your question, or 'quit' to exit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue
        print(f"Bot: {answer(query, retriever, llm)}\n")


if __name__ == "__main__":
    sys.exit(main())
