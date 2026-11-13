"""Measure retrieval quality: does the right FAQ row appear in the top-k results?

Each test case is (paraphrased student question, exact Question text of the row that should be found).
Run:  python eval.py            (uses Gemini embeddings if GEMINI_API_KEY is set, else TF-IDF)
"""

import os

from rag import EmbeddingRetriever, TfidfRetriever, load_dataset

CASES = [
    ("I forgot my password", "How do I reset my campus account password?"),
    ("can't remember my login password", "How do I reset my campus account password?"),
    ("wifi keeps asking me to sign in again", "Why does the Wi-Fi keep asking me to log in again?"),
    ("how do I get on the campus wifi", "How do I connect to the campus Wi-Fi?"),
    ("set up VPN from home", "How do I connect to the campus VPN?"),
    ("VPN works but library site won't open", "The VPN connects but I cannot reach library resources. What should I do?"),
    ("how to enable two factor login", "How do I set up multi-factor authentication (MFA)?"),
    ("my phone got stolen can't do MFA", "I lost my phone and cannot use MFA. How do I get back in?"),
    ("my inbox is full", "My mailbox is full. What can I do?"),
    ("send my uni emails to gmail", "How do I forward my student email to my personal email?"),
    ("get Microsoft Office for free", "How do I install Microsoft Office for free?"),
    ("where can I download MATLAB", "Where can I download licensed software like MATLAB or AutoCAD?"),
    ("how do I print from my laptop", "How do I print from my laptop?"),
    ("add money to my print account", "How do I top up my printing credit?"),
    ("printing job is not moving", "My print job is stuck in the queue. What should I do?"),
    ("reserve a study room", "How do I book a computer lab or study room?"),
    ("when is the helpdesk open", "What are the opening hours of the IT Helpdesk?"),
    ("how do I raise a support request", "How do I submit a helpdesk ticket?"),
    ("how long until someone replies to my ticket", "How long does it take to get a reply to my ticket?"),
    ("where can I see my ticket progress", "How do I check the status of my ticket?"),
    ("I got locked out of my account", "My account is locked. How do I unlock it?"),
    ("someone hacked my account", "I think my account has been hacked. What should I do?"),
    ("how to spot a scam email", "How do I recognise a phishing email?"),
    ("my course is missing on the e-learning site", "I cannot see my enrolled course on the learning platform. Why?"),
    ("how do I hand in my assignment", "How do I submit an assignment online?"),
    ("upload to the assignment page fails", "My file upload keeps failing. What can I try?"),
    ("join a Teams lecture", "How do I join an online class on Zoom or Teams?"),
    ("I deleted a file on the cloud drive, can I get it back", "How can I recover a deleted file from cloud storage?"),
    ("my laptop is super slow", "My laptop is running very slowly. What can I do?"),
    ("can I borrow a laptop", "Can I borrow a laptop from the university?"),
    ("projector in lab is broken", "How do I report a broken computer or projector in a lab?"),
    ("change my recovery email", "How do I change my registered personal email for password resets?"),
]

OFF_TOPIC = ["what's the weather today", "best pizza near campus", "who won the football match", "tell me a joke"]


def main():
    df = load_dataset()
    if os.getenv("GEMINI_API_KEY"):
        from google import genai

        retriever = EmbeddingRetriever(df, genai.Client())
    else:
        retriever = TfidfRetriever(df)
    row_of = {q: i for i, q in enumerate(df["Question"])}

    hit1 = hit3 = 0
    for query, expected in CASES:
        ranked = [i for i, _ in retriever.search(query, k=3)]
        hit1 += ranked[0] == row_of[expected]
        hit3 += row_of[expected] in ranked
        if ranked[0] != row_of[expected]:
            print(f"  miss: {query!r} -> got {df.loc[ranked[0], 'Question']!r}")

    n = len(CASES)
    rejected = sum(retriever.search(q, k=1)[0][1] < retriever.min_score for q in OFF_TOPIC)
    print(f"\nRetriever: {retriever.name}")
    print(f"Hit@1: {hit1}/{n} = {hit1 / n:.0%}   Hit@3: {hit3}/{n} = {hit3 / n:.0%}")
    print(f"Off-topic correctly rejected: {rejected}/{len(OFF_TOPIC)}")


if __name__ == "__main__":
    main()
