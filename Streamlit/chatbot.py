
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from sql_tools import answer_database_question


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


SYSTEM_PROMPT = """
You are Banking Risk Copilot, an assistant for an AI banking
fraud detection and credit risk platform.

Help users understand:
- Fraud detection and suspicious transactions
- Credit risk and loan default prediction
- Machine learning models and evaluation metrics
- SHAP explanations, feature engineering, and MLflow

Be clear, concise, and technically accurate.
Never invent a model prediction, transaction record, or metric.
Do not treat a risk score as a definitive financial decision.
If project data is unavailable, say so.

You are running locally through Ollama.
"""


def generate_answer(history):
    """Generate a normal chat answer using the local Ollama model."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Only send standard chat messages to Ollama.
    messages.extend(
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in history[-12:]
        if message.get("role") in ("user", "assistant")
    )

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 350,
            },
        },
        timeout=(10, 180),
    )

    response.raise_for_status()

    answer = response.json().get("message", {}).get("content", "")

    if not answer.strip():
        raise RuntimeError("Ollama returned an empty response.")

    return answer.strip()



def is_database_question(question: str) -> bool:

    """Detect questions that should query the banking database."""

    q = question.lower()

    # Explicit database references always use SQL.
    database_terms = (
        "database",
        "postgres",
        "sql query",
        "from customers",
        "from loans",
        "from transactions",
        "transaction",
        "merchant category",
        "merchant categories",
        "customer count",
        "number of customers",
        "credit score of customers",
        "largest loan",
        "largest loans",
        "loan amount",
        "grouped by",
        "average credit score",
        "total transactions",
        "count transactions",
        "how many customers",
        "how many loans",
        "customer records",
        "loan records",
        "show customers",
        "show loans",
        "show transactions",
        "fraudulent transactions",
        "fraud transactions",
        "is_fraud",
        "late payments",
        "previous defaults",
        "customer id",
        "customer ids",
        "customer_id",
        "transaction id",
        "transaction_id",
        "loan id",
        "loan_id",
    )

    # Data filtering and aggregation language.
    data_action_terms = (
        "show",
        "find",
        "give me",
        "list",
        "display",
        "retrieve",
        "count",
        "average",
        "sum",
        "maximum",
        "minimum",
        "greater than",
        "less than",
        "more than",
        "greater",
        "less",
        "where",
        "filter",
        "top ",
        "highest",
        "lowest",
        "greater than",
        ">",
        "<",
        ">=",
        "<=",
    )

    mentions_banking_data = any(
        term in q
        for term in (
            "customer",
            "customers",
            "loan",
            "loans",
            "transaction",
            "transactions",
            "merchant",
            "credit score",
            "credit_score",
            "fraud",
            "amount",
            "payment",
            "payments",
            "default",
            "income",
        )
    )

    explicit_database_request = any(term in q for term in database_terms)

    data_operation = any(term in q for term in data_action_terms)

    return explicit_database_request or (
        mentions_banking_data and data_operation
    )


def render_saved_message(message):
    """Display a chat message and any saved database results."""

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sql"):
            with st.expander("View generated SQL"):
                st.code(message["sql"], language="sql")

        if (
            message.get("columns")
            and message.get("rows") is not None
        ):
            if message["rows"]:
                dataframe = pd.DataFrame(
                    message["rows"],
                    columns=message["columns"],
                )
                st.dataframe(
                    dataframe,
                    width="stretch"
                )
            else:
                  st.info(
                 "No matching records were returned by this database query."
                 )
            


def render_chatbot():
    """Render the unified Banking Risk Copilot chat interface."""

    st.subheader("Banking Risk Copilot")
    st.caption(
        "Ask about banking risk, machine learning, or your PostgreSQL data."
    )

    # Initialize history once per Streamlit session.
    if "banking_chat_history" not in st.session_state:
        st.session_state.banking_chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your Banking Risk Copilot. "
                    "Ask about fraud detection, credit risk, SHAP, "
                    "MLflow, machine learning, customers, loans, "
                    "or transactions."
                ),
            }
        ]

    history = st.session_state.banking_chat_history

    # Sidebar history.
    with st.sidebar.expander("Chat History", expanded=True):
        if not history:
            st.caption("No conversations yet.")

        for index, message in enumerate(history):
            if message["role"] == "user":
                st.markdown(f"**You · {index + 1}**")
            else:
                st.markdown("**Banking Copilot**")

            st.markdown(message["content"])

            if message.get("sql"):
                with st.expander("SQL details"):
                    st.code(message["sql"], language="sql")

            st.divider()

        if st.button(
            "Clear chat history",
            key="clear_banking_history",
        ):
            st.session_state.banking_chat_history = []
            st.rerun()

    # Display the conversation in one chat interface.
    for message in history:
        render_saved_message(message)

    prompt = st.chat_input(
        "Ask about banking risk, ML, customers, loans, or transactions...",
        key="banking_copilot_input",
    )

    if prompt:
        # Save the user's message.
        history.append({
            "role": "user",
            "content": prompt,
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        assistant_message = {
            "role": "assistant",
            "content": "",
        }

        with st.chat_message("assistant"):
            try:
                if is_database_question(prompt):
                    # Database questions use the existing validated SQL pipeline.
                    with st.spinner(
                        "Generating SQL and querying PostgreSQL..."
                    ):
                        answer, sql, columns, rows = (
                            answer_database_question(prompt)
                        )

                    assistant_message.update({
                        "content": answer,
                        "sql": sql,
                        "columns": columns,
                        "rows": rows,
                        "mode": "sql",
                    })

                    st.markdown(answer)

                    if sql:
                        with st.expander("View generated SQL"):
                            st.code(sql, language="sql")

                    
                    if rows:
                     st.dataframe(
                           pd.DataFrame(rows, columns=columns),
                           width="stretch"
                        )
                    else:
                           st.info(
                               "No matching customer IDs were found for transaction "
                               "amounts greater than 50,000. Try a lower threshold "
                               "if appropriate."
                            )


                else:
                    # All other questions use normal Ollama chat.
                    with st.spinner("Llama 3.2 is thinking..."):
                        answer = generate_answer(history)

                    assistant_message["content"] = answer
                    st.markdown(answer)

            except requests.Timeout:
                assistant_message["content"] = (
                    "The model timed out. Please try again."
                )
                st.error(assistant_message["content"])

            except requests.ConnectionError:
                assistant_message["content"] = (
                    "Cannot connect to Ollama. Make sure Ollama is running."
                )
                st.error(assistant_message["content"])

            except requests.RequestException:
                assistant_message["content"] = (
                    "The Ollama request failed. Check the local model service."
                )
                st.error(assistant_message["content"])

            except Exception as exc:
                # Keep technical details in the terminal, not the UI.
                print(f"Banking Risk Copilot error: {exc}")

                assistant_message["content"] = (
                    "I couldn't complete that request. Please try again "
                    "or check the terminal for the detailed error."
                )
                st.error(assistant_message["content"])

        # Save the answer and rerun to update the complete chat.
        history.append(assistant_message)
        st.rerun()

    with st.expander("Model configuration"):
        st.write(f"**Model:** `{OLLAMA_MODEL}`")
        st.write(f"**Ollama URL:** `{OLLAMA_BASE_URL}`")
        st.write("**Database:** PostgreSQL read-only query pipeline")
