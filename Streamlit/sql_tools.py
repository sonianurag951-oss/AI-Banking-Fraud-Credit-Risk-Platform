
import json
import os
import re
from pathlib import Path

import psycopg
import requests
import sqlglot
from dotenv import load_dotenv
from sqlglot import exp


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

PGSCHEMA = os.getenv("PGSCHEMA", "public").lower()
MAX_ROWS = 100
QUERY_TIMEOUT_MS = 5000

ALLOWED_TABLES = {
    "customers": {
        "customer_id", "age", "income", "employment_type",
        "credit_score", "account_tenure", "city", "dependents",
    },
    "loans": {
        "loan_id", "customer_id", "loan_amount", "loan_term",
        "interest_rate", "monthly_income", "existing_debt",
        "previous_defaults", "late_payments", "default",
    },
    "transactions": {
        "transaction_id", "customer_id", "amount",
        "transaction_time", "merchant_category", "payment_method",
        "location", "device_type", "is_fraud",
    },
}

SAFE_ANONYMOUS_FUNCTIONS = {
    "coalesce", "nullif", "greatest", "least",
    "date_trunc", "date_part", "to_char",
    "lower", "upper", "trim", "round",
    "abs", "floor", "ceil", "substring",
    "count", "sum", "avg", "min", "max",
}


# ---------------------------------------------------------
# PostgreSQL connection
# ---------------------------------------------------------

def get_connection():
    required = ("PGDATABASE", "PGUSER", "PGPASSWORD")
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise RuntimeError(
            "Missing .env settings: " + ", ".join(missing)
        )

    return psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        connect_timeout=5,
        application_name="banking_risk_copilot",
    )


def schema_for_prompt():
    return """
customers(
    customer_id INTEGER, age INTEGER, income NUMERIC,
    employment_type VARCHAR, credit_score INTEGER,
    account_tenure INTEGER, city VARCHAR, dependents INTEGER
)

loans(
    loan_id INTEGER, customer_id INTEGER, loan_amount NUMERIC,
    loan_term INTEGER, interest_rate NUMERIC, monthly_income NUMERIC,
    existing_debt NUMERIC, previous_defaults INTEGER,
    late_payments INTEGER, "default" INTEGER
)

transactions(
    transaction_id INTEGER, customer_id INTEGER, amount NUMERIC,
    transaction_time TIMESTAMP, merchant_category VARCHAR,
    payment_method VARCHAR, location VARCHAR, device_type VARCHAR,
    is_fraud INTEGER
)
"""


# ---------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------

def call_ollama(messages, temperature=0, num_predict=400):
    url = f"{OLLAMA_BASE_URL}/api/chat"

    response = requests.post(
        url,
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        },
        timeout=(10, 180),
    )

    response.raise_for_status()

    data = response.json()
    answer = data.get("message", {}).get("content", "")

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Ollama returned an empty response.")

    return answer.strip()


# ---------------------------------------------------------
# Natural language -> SQL
# ---------------------------------------------------------

def generate_sql(question):
    prompt = f"""
You generate PostgreSQL SELECT queries for a banking analytics assistant.

Available schema:
{schema_for_prompt()}

Rules:
- Return one SQL query only. Do not use Markdown fences.
- Use only the tables and columns in the supplied schema.
- Use public.customers, public.loans, and public.transactions.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or other
  data-changing statements.
- Do not use CTEs or SELECT INTO.
- For customer credit-score questions, use public.customers.
- For fraud transaction questions, use public.transactions.
- For the largest transactions, select transaction_id, amount,
  and merchant_category, order by amount DESC, and use LIMIT 10.
- For "top N" questions, use LIMIT N.
- Use an appropriate ORDER BY when the question requests a ranking.
- If the question cannot be answered with this schema, return
  CANNOT_ANSWER and nothing else.
"""

    # The returned text is the actual SQL-generation response.
    raw_sql = call_ollama([
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ])

    print("DEBUG raw SQL:", repr(raw_sql))
    return raw_sql


# ---------------------------------------------------------
# Fallback SQL for common questions
# ---------------------------------------------------------

def get_fallback_sql(question):
    q = question.lower().strip()

    # Example: "top 10 customer ids where credit score > 500"
    greater_match = re.search(
        r"\b(?:credit\s*score|credit_score)\b"
        r"\s*(?:is\s*)?(?:greater\s+than|more\s+than|above|over|>=|>)"
        r"\s*(\d+)",
        q,
    )

    if greater_match and "customer" in q:
        threshold = int(greater_match.group(1))
        top_match = re.search(r"\btop\s+(\d+)\b", q)
        limit = (
            min(int(top_match.group(1)), MAX_ROWS)
            if top_match else MAX_ROWS
        )

        operator = ">=" if ">=" in greater_match.group(0) else ">"

        return f"""
            SELECT customer_id
            FROM public.customers
            WHERE credit_score {operator} {threshold}
            ORDER BY credit_score DESC
            LIMIT {limit}
        """

    # Example: "give me customer IDs whose credit score is 400"
    exact_match = re.search(
        r"\b(?:credit\s*score|credit_score)\b"
        r"\s*(?:is|equals?|of|=)?\s*(\d+)\b",
        q,
    )

    if exact_match and "customer" in q:
        score = int(exact_match.group(1))

        return f"""
            SELECT customer_id
            FROM public.customers
            WHERE credit_score = {score}
            ORDER BY customer_id
            LIMIT {MAX_ROWS}
        """

    # Example: "customers with credit score below 600"
    below_match = re.search(
        r"\b(?:credit\s*score|credit_score)\b"
        r"\s*(?:is\s*)?(?:less\s+than|below|under|<)\s*(\d+)",
        q,
    )

    if below_match and "customer" in q:
        threshold = int(below_match.group(1))

        return f"""
            SELECT customer_id
            FROM public.customers
            WHERE credit_score < {threshold}
            ORDER BY credit_score ASC
            LIMIT {MAX_ROWS}
        """

    # Example: "how many fraudulent transactions?"
    if "fraud" in q and any(
        word in q for word in ("how many", "count", "number")
    ):
        return """
            SELECT COUNT(*) AS fraud_count
            FROM public.transactions
            WHERE is_fraud = 1
        """

    # Example: "show top 10 largest transactions"
    if "transaction" in q and any(
        word in q
        for word in ("largest", "biggest", "highest amount", "top")
    ):
        top_match = re.search(r"\btop\s+(\d+)\b", q)
        limit = (
            min(int(top_match.group(1)), MAX_ROWS)
            if top_match else 10
        )

        return f"""
            SELECT transaction_id, amount, merchant_category
            FROM public.transactions
            ORDER BY amount DESC
            LIMIT {limit}
        """

    # Example: "customer IDs for transactions above amount 5000"
    amount_match = re.search(
        r"\b(?:amount\s*(?:is\s*)?)?"
        r"(?:greater\s+than|more\s+than|above|over|>)"
        r"\s*₹?\s*([\d,]+(?:\.\d+)?)",
        q,
    )

    if (
        amount_match
        and "customer" in q
        and "transaction" in q
    ):
        threshold = float(amount_match.group(1).replace(",", ""))

        return f"""
            SELECT DISTINCT customer_id
            FROM public.transactions
            WHERE amount > {threshold}
            ORDER BY customer_id
            LIMIT {MAX_ROWS}
        """

    return None


# ---------------------------------------------------------
# SQL validation
# ---------------------------------------------------------

def validate_sql(raw_sql):
    if not isinstance(raw_sql, str) or not raw_sql.strip():
        raise ValueError("The generated SQL is empty or malformed.")

    sql = raw_sql.strip()

    # Permit optional Markdown fences from the model.
    sql = re.sub(
        r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE
    )
    sql = re.sub(r"\s*```$", "", sql).strip()

    if not sql:
        raise ValueError("The generated SQL is empty.")

    # Allow one optional trailing semicolon.
    sql = sql.rstrip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()

    # Strictly reject multiple statements.
    if ";" in sql:
        raise ValueError("Only one SQL statement is allowed.")

    parsed = sqlglot.parse(sql, read="postgres")

    if len(parsed) != 1 or parsed[0] is None:
        raise ValueError("Only one SQL statement is allowed.")

    tree = parsed[0]

    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT statements are allowed.")

    if tree.args.get("with"):
        raise ValueError("CTEs are not allowed.")

    if tree.args.get("into"):
        raise ValueError("SELECT INTO is not allowed.")

    if tree.args.get("locks"):
        raise ValueError("Row-locking clauses are not allowed.")

    forbidden = (
        exp.Insert, exp.Update, exp.Delete, exp.Merge,
        exp.Create, exp.Drop, exp.Alter, exp.Command,
    )

    for node in tree.walk():
        if isinstance(node, forbidden):
            raise ValueError("A prohibited SQL operation was detected.")

    tables = list(tree.find_all(exp.Table))

    if not tables:
        raise ValueError("Query must reference an approved table.")

    for table in tables:
        table_name = table.name.lower()

        if table_name not in ALLOWED_TABLES:
            raise ValueError(
                f"Table '{table.name}' is not approved."
            )

        if table.db and table.db.lower() != PGSCHEMA:
            raise ValueError(
                "Queries to other schemas are not allowed."
            )

        if table.catalog:
            raise ValueError(
                "Catalog-qualified tables are not allowed."
            )

    for node in tree.find_all(exp.Anonymous):
        name = node.name.lower()

        if name not in SAFE_ANONYMOUS_FUNCTIONS:
            raise ValueError(
                f"Function '{name}' is not approved."
            )

    # Enforce a fixed, bounded top-level row limit.
    limit = tree.args.get("limit")

    if limit is None:
        tree = tree.limit(MAX_ROWS)
    else:
        expression = limit.expression

        if (
            not isinstance(expression, exp.Literal)
            or not expression.is_int
        ):
            raise ValueError("LIMIT must be a fixed integer.")

        requested_limit = int(expression.this)

        if requested_limit < 1:
            raise ValueError("LIMIT must be at least 1.")

        tree = tree.limit(min(requested_limit, MAX_ROWS))

    return tree.sql(dialect="postgres")


# ---------------------------------------------------------
# Execute read-only query
# ---------------------------------------------------------

def execute_query(sql):
    with get_connection() as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute(
                f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}ms'"
            )
            conn.execute("SET LOCAL lock_timeout = '2000ms'")

            cursor = conn.execute(sql)

            if cursor.description is None:
                raise ValueError("The query returned no result set.")

            columns = [item.name for item in cursor.description]
            records = cursor.fetchmany(MAX_ROWS)

    rows = []

    for record in records:
        row = {}

        for column, value in zip(columns, record):
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            elif value is not None and not isinstance(
                value, (str, int, float, bool)
            ):
                value = str(value)

            row[column] = value

        rows.append(row)

    return columns, rows


# ---------------------------------------------------------
# Summarize database results
# ---------------------------------------------------------

def summarize(question, sql, columns, rows):
    if not rows:
        return "The query succeeded but returned no matching rows."

    preview = {
        "columns": columns,
        "rows": rows[:30],
        "total_rows_returned": len(rows),
    }

    messages = [
        {
            "role": "system",
            "content": (
                "Explain the PostgreSQL query results clearly. "
                "Use only the supplied data. Do not invent figures. "
                "Do not claim the returned rows represent the full "
                "population when the query is limited."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"SQL executed: {sql}\n"
                f"Query results: {json.dumps(preview, default=str)}"
            ),
        },
    ]

    try:
        return call_ollama(
            messages, temperature=0.2, num_predict=350
        )
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        # The SQL result is still usable if the summary model fails.
        print(f"Summary generation failed: {exc}")
        return (
            f"The SQL query succeeded and returned {len(rows)} row(s). "
            "See the result table for the retrieved data."
        )


# ---------------------------------------------------------
# Main entry point called by the Streamlit app
# ---------------------------------------------------------

def answer_database_question(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a question.")

    fallback_sql = get_fallback_sql(question)

    # First attempt: generate and validate model SQL.
    try:
        raw_sql = generate_sql(question)

        if raw_sql.strip().upper().rstrip(";") == "CANNOT_ANSWER":
            raise ValueError(
                "This question cannot be answered using the approved schema."
            )

        safe_sql = validate_sql(raw_sql)

    except Exception as exc:
        print(
            "SQL generation/validation error:",
            type(exc).__name__,
            str(exc),
        )

        if fallback_sql is None:
            raise ValueError(
                "Could not generate valid SQL for this question. "
                "Check the terminal for the detailed error."
            ) from exc

        safe_sql = validate_sql(fallback_sql)

    print("Generated SQL:", safe_sql)

    # Execute only after validation. Database errors are not silently
    # converted into a different query.
    columns, rows = execute_query(safe_sql)

    answer = summarize(question, safe_sql, columns, rows)

    return answer, safe_sql, columns, rows
