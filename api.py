# ==========================================
# SUPPORT INTELLIGENCE API
# PHASE 4 + PHASE 5 + PHASE 6
# ==========================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, text

import pandas as pd
import joblib
import faiss
import pickle
import numpy as np
import requests
import re
import json


# ==========================================
# 1. FASTAPI APP
# ==========================================

app = FastAPI(title="Support Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 2. DATABASE
# ==========================================

engine = create_engine("sqlite:///tickets.db")


# ==========================================
# 3. LOAD ML MODELS
# ==========================================

try:
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    clf_queue = joblib.load("clf_queue.joblib")
    clf_prio = joblib.load("clf_prio.joblib")
    clf_category = joblib.load("clf_category.joblib")

    print("ML models loaded successfully.")

except Exception as e:
    print("Error loading ML models:", e)


# ==========================================
# 4. LOAD RAG RESOURCES
# ==========================================

from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

rag_index = faiss.read_index(
    "tickets_rag.index"
)

with open("rag_metadata.pkl", "rb") as f:
    rag_metadata = pickle.load(f)

print("RAG resources loaded successfully.")


# ==========================================
# 5. OLLAMA / QWEN3
# ==========================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:4b"


def ollama_chat(messages):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": 500
            }
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    content = data["message"]["content"].strip()

    # Remove Qwen reasoning if it appears
    if "</think>" in content:
        content = content.split(
            "</think>",
            1
        )[1].strip()

    return content


# ==========================================
# 6. PYDANTIC MODELS
# ==========================================

class TriageRequest(BaseModel):
    subject: str
    body: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    messages: List[ChatMessage]


# ==========================================
# 7. RAG SEARCH TOOL
# ==========================================

def search_similar_tickets(
    query: str,
    k: int = 3,
    threshold: float = 0.45
):

    query_vector = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = rag_index.search(
        query_vector,
        k
    )

    results = []

    for dist, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if float(dist) < threshold:
            continue

        meta = rag_metadata[idx]

        results.append({
            "id": meta["id"],
            "queue": meta.get("queue", ""),
            "priority": meta.get("priority", ""),
            "answer": meta.get("answer", "")
        })

    return results


# ==========================================
# 8. DATABASE STATISTICS TOOL
# ==========================================

def query_stats():

    df = pd.read_sql(
        "SELECT queue, priority FROM tickets",
        con=engine
    )

    return {
        "queue_counts":
            df["queue"].value_counts().to_dict(),

        "priority_counts":
            df["priority"].value_counts().to_dict()
    }


# ==========================================
# 9. CHAT LOGGING
# ==========================================

def log_chat(
    session_id: str,
    role: str,
    message: str,
    cited_ids: Optional[str] = None,
    tool_used: Optional[str] = None
):

    with engine.begin() as conn:

        conn.execute(
            text(
                """
                INSERT INTO chat_logs
                (
                    session_id,
                    role,
                    content,
                    cited_ticket_ids,
                    tool_used
                )
                VALUES
                (
                    :session_id,
                    :role,
                    :content,
                    :cited_ids,
                    :tool_used
                )
                """
            ),
            {
                "session_id": session_id,
                "role": role,
                "content": message,
                "cited_ids": cited_ids,
                "tool_used": tool_used
            }
        )


# ==========================================
# 10. CLEAN RAG ANSWERS
# ==========================================

def clean_rag_answers(
    search_results,
    user_query
):

    """
    Extract concise troubleshooting advice
    from retrieved support tickets while
    removing ticket-specific filler.
    """

    sentences = []

    for result in search_results:

        answer = str(
            result.get("answer", "")
        )

        if not answer:
            continue

        answer = re.sub(
            r"<[^>]+>",
            "",
            answer
        )

        answer = re.sub(
            r"\s+",
            " ",
            answer
        ).strip()

        parts = re.split(
            r"(?<=[.!?])\s+",
            answer
        )

        sentences.extend(parts)

    if not sentences:
        return []

    query_lower = user_query.lower()

    # --------------------------------------
    # Login / account questions
    # --------------------------------------

    if any(
        word in query_lower
        for word in [
            "login",
            "log in",
            "sign in",
            "password",
            "account access",
            "cannot access",
            "can't access"
        ]
    ):

        topic_words = [
            "login",
            "log in",
            "sign in",
            "password",
            "account",
            "access",
            "browser",
            "device",
            "cache",
            "update",
            "error",
            "credentials",
            "reset"
        ]

    # --------------------------------------
    # Payment questions
    # --------------------------------------

    elif any(
        word in query_lower
        for word in [
            "payment",
            "pay",
            "paid",
            "charged",
            "card",
            "declined",
            "billing"
        ]
    ):

        topic_words = [
            "payment",
            "pay",
            "card",
            "billing",
            "charged",
            "declined",
            "expiry",
            "expired",
            "balance",
            "bank",
            "payment method",
            "update",
            "error",
            "details"
        ]

    # --------------------------------------
    # Return / refund questions
    # --------------------------------------

    elif any(
        word in query_lower
        for word in [
            "return",
            "refund",
            "exchange",
            "wrong item",
            "wrong product"
        ]
    ):

        topic_words = [
            "return",
            "refund",
            "exchange",
            "product",
            "item",
            "order"
        ]

    # --------------------------------------
    # General troubleshooting
    # --------------------------------------

    else:

        topic_words = [
            "check",
            "verify",
            "try",
            "update",
            "error",
            "issue",
            "problem",
            "restart",
            "clear",
            "ensure",
            "make sure"
        ]

    # --------------------------------------
    # Remove ticket-specific / useless text
    # --------------------------------------

    unwanted_phrases = [

        "account number",
        "acc number",
        "phone number",
        "telephone number",
        "tel number",
        "email address",

        "suitable time",
        "available to call",
        "contact you",
        "call you",
        "contact us",
        "contact support",
        "contact the support",

        "arrange a call",
        "we can arrange",
        "we are available",

        "provide your",
        "share your",
        "please confirm your",

        "could you please confirm",
        "please provide",
        "please share",

        "at your convenience",

        "we may need",
        "let us know",
        "please let us know",

        "thank you for",
        "we apologize",
        "we regret",
        "i regret",
        "we appreciate",

        "you have already tried",

        "delve deeper",
        "resolve the issue directly",

        "aid us in identifying",
        "identify the root cause",
        "root cause and offering",
        "precise solution",
        "offering a precise solution",
        "help us identify",
        "help us determine",
        "further investigate",
        "investigate the issue"
    ]

    unwanted_starts = [

        "dear ",
        "hello ",
        "hi ",
        "thank you",
        "we apologize",
        "we regret",
        "i regret",
        "please let us know",
        "we are available",
        "we would be",
        "could you",
        "would you"
    ]

    candidates = []

    # --------------------------------------
    # Score sentences
    # --------------------------------------

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        lower = sentence.lower()

        # Remove unwanted phrases
        if any(
            phrase in lower
            for phrase in unwanted_phrases
        ):
            continue

        # Remove unwanted starts
        if any(
            lower.startswith(start)
            for start in unwanted_starts
        ):
            continue

        # Ignore extremely short sentences
        if len(sentence.split()) < 5:
            continue

        score = 0

        # Topic relevance
        for keyword in topic_words:

            if keyword in lower:
                score += 1

        # Actionable language
        actionable_words = [
            "check",
            "try",
            "clear",
            "update",
            "reset",
            "verify",
            "ensure",
            "make sure",
            "use",
            "switch",
            "change"
        ]

        for keyword in actionable_words:

            if keyword in lower:
                score += 2

        # Ignore irrelevant sentences
        if score == 0:
            continue

        # Remove transition words
        sentence = re.sub(
            r"^(also|however|additionally|furthermore)[,:]?\s*",
            "",
            sentence,
            flags=re.IGNORECASE
        ).strip()

        if sentence:

            sentence = (
                sentence[0].upper()
                + sentence[1:]
            )

        candidates.append(
            (score, sentence)
        )

    # Highest scoring first
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    useful_points = []

    for score, sentence in candidates:

        duplicate = False

        for existing in useful_points:

            if (
                sentence.lower()
                in existing.lower()
                or
                existing.lower()
                in sentence.lower()
            ):
                duplicate = True
                break

        if not duplicate:
            useful_points.append(sentence)

        if len(useful_points) >= 4:
            break

    return useful_points


# ==========================================
# 11. HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "ok"
    }


# ==========================================
# 12. STATS ENDPOINT
# ==========================================

@app.get("/stats")
def get_stats():

    return query_stats()


# ==========================================
# 13. TRIAGE ENDPOINT
# ==========================================

@app.post("/triage")
def triage_ticket(
    request: TriageRequest
):

    text_feature = (
        request.subject
        + " "
        + request.body
    )

    vec = tfidf.transform(
        [text_feature]
    )

    # Category
    predicted_category = (
        clf_category.predict(vec)[0]
    )

    # Queue
    predicted_queue = (
        clf_queue.predict(vec)[0]
    )

    queue_confidence = float(
        max(
            clf_queue.predict_proba(vec)[0]
        )
    )

    # Priority
    predicted_priority = (
        clf_prio.predict(vec)[0]
    )

    return {

        # Category
        "category": predicted_category,
        "predicted_category": predicted_category,

        # Queue / assigned team
        "predicted_queue": predicted_queue,
        "queue_confidence": queue_confidence,

        # Priority
        "predicted_priority": predicted_priority
    }


# ==========================================
# 14. CHAT ENDPOINT
# ==========================================

@app.post("/chat")
def chat_endpoint(
    request: ChatRequest
):

    # --------------------------------------
    # Get latest user message
    # --------------------------------------

    user_msg = (
        request.messages[-1].content
    )

    # Log user message
    log_chat(
        request.session_id,
        "user",
        user_msg
    )

    # --------------------------------------
    # Ask Qwen which tool to use
    # --------------------------------------

    router_prompt = """
You are the routing component of an AI customer support system.

You have exactly two tools.

TOOL 1: SEARCH_TICKETS

Use SEARCH_TICKETS for ANY customer support problem or request,
including but not limited to:

- login problems
- password problems
- account access
- payment problems
- card problems
- billing problems
- declined payments
- refunds
- returns
- exchanges
- wrong items
- wrong products
- damaged products
- orders
- delivery problems
- technical problems
- product problems
- troubleshooting
- requests asking how to solve a customer issue

TOOL 2: QUERY_STATS

Use QUERY_STATS ONLY when the user asks about numerical
information or statistics from the ticket dataset, such as:

- number of tickets
- ticket counts
- counts by queue
- counts by priority
- statistics
- dataset statistics

IMPORTANT:

If the user describes a customer problem, ALWAYS use
SEARCH_TICKETS, even if the problem involves returns,
refunds, exchanges, orders, payments, login, or products.

If the user asks a non-support question such as:
"What is the capital of France?"
do NOT use either tool.

For SEARCH_TICKETS, output exactly:

TOOL: SEARCH_TICKETS | <search query>

For statistics, output exactly:

TOOL: QUERY_STATS

Do not provide an answer.
Do not explain your choice.
Output only the tool command.
"""

    messages = [
        {
            "role": "system",
            "content": router_prompt
        },
        {
            "role": "user",
            "content": user_msg
        }
    ]

    try:

        router_output = ollama_chat(
            messages
        )

    except Exception as e:

        return {
            "answer":
                "I couldn't connect to the AI assistant. "
                "Please make sure Ollama and Qwen3 are running.",
            "cited_ticket_ids": "",
            "tool_used": "None"
        }

    # --------------------------------------
    # SEARCH_TICKETS
    # --------------------------------------

    if router_output.startswith(
        "TOOL: SEARCH_TICKETS"
    ):

        tool_used = (
            "search_similar_tickets"
        )

        parts = router_output.split(
            "|",
            1
        )

        if len(parts) > 1:

            search_query = (
                parts[1].strip()
            )

        else:

            search_query = user_msg

        search_results = (
            search_similar_tickets(
                search_query,
                k=3,
                threshold=0.45
            )
        )

        # ----------------------------------
        # No relevant results
        # ----------------------------------

        if not search_results:

            final_answer = (
                "I couldn't find sufficiently "
                "similar support tickets to answer "
                "this confidently."
            )

            cited_ticket_ids = ""

        else:

            # ----------------------------------
            # Clean retrieved answers
            # ----------------------------------

            useful_points = (
                clean_rag_answers(
                    search_results,
                    user_msg
                )
            )

            # ----------------------------------
            # If nothing useful survived cleaning
            # ----------------------------------

            if not useful_points:

                final_answer = (
                    "I couldn't find sufficiently "
                    "similar support tickets to answer "
                    "this confidently."
                )

            else:

                final_answer = (
                    "Based on similar support tickets, "
                    "you can try:\n\n"
                    + "\n".join(
                        f"- {point}"
                        for point in useful_points
                    )
                )

            # ----------------------------------
            # Citation IDs
            # ----------------------------------

            cited_ticket_ids = ", ".join(
                str(result["id"])
                for result in search_results
            )


    # ======================================
    # QUERY_STATS
    # ======================================

    elif router_output.startswith(
        "TOOL: QUERY_STATS"
    ):

        tool_used = "query_stats"

        stats = query_stats()

        queue_counts = (
            stats["queue_counts"]
        )

        priority_counts = (
            stats["priority_counts"]
        )

        # ----------------------------------
        # Queue count question
        # ----------------------------------

        if any(
            word in user_msg.lower()
            for word in [
                "queue",
                "queues"
            ]
        ):

            lines = []

            for queue, count in (
                queue_counts.items()
            ):

                lines.append(
                    f"- {queue}: {count:,} tickets"
                )

            final_answer = (
                "The ticket counts by support queue are:\n\n"
                + "\n".join(lines)
            )

        # ----------------------------------
        # Priority count question
        # ----------------------------------

        elif any(
            word in user_msg.lower()
            for word in [
                "priority",
                "priorities"
            ]
        ):

            lines = []

            for priority, count in (
                priority_counts.items()
            ):

                lines.append(
                    f"- {priority}: {count:,} tickets"
                )

            final_answer = (
                "The ticket counts by priority are:\n\n"
                + "\n".join(lines)
            )

        # ----------------------------------
        # General stats
        # ----------------------------------

        else:

            queue_lines = []

            for queue, count in (
                queue_counts.items()
            ):

                queue_lines.append(
                    f"- {queue}: {count:,}"
                )

            priority_lines = []

            for priority, count in (
                priority_counts.items()
            ):

                priority_lines.append(
                    f"- {priority}: {count:,}"
                )

            final_answer = (
                "Ticket counts by support queue:\n\n"
                + "\n".join(queue_lines)
                + "\n\n"
                + "Ticket counts by priority:\n\n"
                + "\n".join(priority_lines)
            )

        cited_ticket_ids = ""


    # ======================================
    # UNKNOWN / NON-SUPPORT REQUEST
    # ======================================

    else:

        tool_used = "None"

        cited_ticket_ids = ""

        final_answer = (
            "I can only help with support tickets "
            "and information from the support ticket dataset."
        )


    # ======================================
    # LOG ASSISTANT RESPONSE
    # ======================================

    log_chat(
        request.session_id,
        "assistant",
        final_answer,
        cited_ids=cited_ticket_ids,
        tool_used=tool_used
    )


    # ======================================
    # RETURN RESPONSE
    # ======================================

    return {

        "answer": final_answer,

        "cited_ticket_ids":
            cited_ticket_ids,

        "tool_used":
            tool_used
    }