import re
import json
import os
from pathlib import Path

import psycopg
import requests
import sqlglot
from dotenv import load_dotenv
from sqlglot import exp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
).rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

PGSCHEMA = os.getenv("PGSCHEMA", "public").lower()
MAX_ROWS = 100
QUERY_TIMEOUT_MS = 5000

# Actual schema retrieved from your PostgreSQL database.
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

# Only allow common, read-only functions when the SQL parser represents
# a function as an anonymous/custom function.
SAFE_ANONYMOUS_FUNCTIONS = {
    "coalesce", "nullif", "greatest", "least",
    "date_trunc", "date_part", "to_char",
    "lower", "upper", "trim", "round",
    "abs", "floor", "ceil", "substring",
    "extract", "count", "sum", "avg", "min", "max",
}


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


def call_ollama(messages, temperature=0, num_predict=400):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
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

    answer = response.json().get("message", {}).get("content", "")
    if not answer.strip():
        raise RuntimeError("Ollama returned an empty response.")
    return answer.strip()


def generate_sql(question):
    prompt = f"""
You write PostgreSQL SELECT queries for a banking analytics assistant.

Available schema:
{schema_for_prompt()}

IMPORTANT SQL RULES:
- For questions about transactions, use public.transactions.
- Do not join customers or loans unless the question requires their columns.
- For the 10 largest transactions, select transaction_id, amount,
  and merchant_category from public.transactions.
- Use ORDER BY amount DESC LIMIT 10 for the 10 largest transactions.
- Every alias must be defined before it is used.
- Do not invent table or column names.
- Return one valid PostgreSQL SELECT statement only.
"""
    return call_ollama([
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ])


def validate_sql(raw_sql):
    sql = raw_sql.strip()

    if not sql or "```" in sql:
        raise ValueError("The generated SQL is empty or malformed.")

    # Allow one optional trailing semicolon only.
    sql = sql.rstrip().removesuffix(";").strip()
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

    # Catch data-changing expressions nested inside the query as well.
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

    allowed = set(ALLOWED_TABLES)
    for table in tables:
        table_name = table.name.lower()

        if table_name not in allowed:
            raise ValueError(f"Table '{table.name}' is not approved.")

        if table.db and table.db.lower() != PGSCHEMA:
            raise ValueError("Queries to other schemas are not allowed.")

        if table.catalog:
            raise ValueError("Catalog-qualified tables are not allowed.")

    # Reject arbitrary/custom functions unless explicitly approved.
    for node in tree.find_all(exp.Anonymous):
        name = node.name.lower()
        if name not in SAFE_ANONYMOUS_FUNCTIONS:
            raise ValueError(f"Function '{name}' is not approved.")

    # Enforce a top-level row limit.
    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(MAX_ROWS)
    else:
        expression = limit.expression
        if not isinstance(expression, exp.Literal) or not expression.is_int:
            raise ValueError("LIMIT must be a fixed integer.")
        tree = tree.limit(min(int(expression.this), MAX_ROWS))

    return tree.sql(dialect="postgres")


def execute_query(sql):
    # The account should have SELECT privileges only. The transaction
    # is also explicitly read-only as a second layer of protection.
    with get_connection() as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute(
                f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}ms'"
            )
            conn.execute("SET LOCAL lock_timeout = '2000ms'")

            cursor = conn.execute(sql)
            if cursor.description is None:
                raise ValueError("The query did not return a result set.")

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


def summarize(question, sql, columns, rows):
    if not rows:
        return "The query succeeded but returned no matching rows."

    # Limit the data sent to the LLM; the UI can show all returned rows.
    result_preview = {
        "columns": columns,
        "rows": rows[:30],
        "total_rows_returned": len(rows),
    }

    return call_ollama([
        {
            "role": "system",
            "content": (
                "Explain the supplied PostgreSQL results clearly. "
                "Use only the supplied data. Never invent figures or "
                "claim that a sample of rows is the full population. "
                "For fraud or loan defaults, describe the stored labels "
                "without assuming their meaning unless established."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"SQL executed: {sql}\n"
                f"Query results: {json.dumps(result_preview, default=str)}"
            ),
        },
    ], temperature=0.2, num_predict=350)

def get_fallback_sql(question: str) -> str | None:
    q = question.lower()

   
    # 1. Count fraudulent transactions
    if (
        "fraud" in q
        and any(word in q for word in ["how many", "count", "number"])
    ):
        return """
            SELECT COUNT(*) AS fraud_count
            FROM public.transactions
            WHERE is_fraud = 1
        """

    # 2. High-income customers with low credit scores
    if (
        "customer" in q
        and ("income" in q or "earning" in q)
        and ("credit score" in q or "credit_score" in q)
        and ("low" in q or "below" in q)
    ):
        return """
            SELECT customer_id, income, credit_score
            FROM public.customers
            WHERE income > 100000
              AND credit_score < 600
            ORDER BY income DESC
            LIMIT 100
        """

        # Find customer IDs for transactions above a specified amount.
    q = question.lower()

    amount_patterns = [
        r"\bamount\s*(?:is\s*)?(?:greater\s+than|more\s+than|above|over|>=|>)\s*₹?\s*([\d,]+(?:\.\d+)?)",
        r"\b(?:greater\s+than|more\s+than|above|over|>=|>)\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(?:for\s+)?(?:transaction\s+)?amount\b",
    ]

    if (
        "customer" in q
        and "transaction" in q
        and any(word in q for word in ["id", "ids", "identifier"])
    ):
        for pattern in amount_patterns:
            match = re.search(pattern, q)

            if match:
                threshold_text = match.group(1).replace(",", "")

                # Accept only a valid, non-negative numeric threshold.
                if re.fullmatch(r"\d+(?:\.\d+)?", threshold_text):
                    threshold = float(threshold_text)

                    return f"""
                        SELECT DISTINCT customer_id
                        FROM public.transactions
                        WHERE amount > {threshold}
                        ORDER BY customer_id
                        LIMIT 100
                    """

    return None

# def answer_database_question(question):
    raw_sql = generate_sql(question)

    if raw_sql.strip().upper() == "CANNOT_ANSWER":
        raise ValueError(
            "This question cannot be answered using the approved schema."
        )

    safe_sql = validate_sql(raw_sql)
    print("Generated SQL:", safe_sql)
    columns, rows = execute_query(safe_sql)
    answer = summarize(question, safe_sql, columns, rows)

    return answer, safe_sql, columns, rows

def answer_database_question(question):
    fallback_sql = get_fallback_sql(question)
    
    raw_sql = generate_sql(question)

    try:
        safe_sql = validate_sql(raw_sql)
    except (ValueError, TypeError):
        if fallback_sql is None:
            raise
        safe_sql = validate_sql(fallback_sql)

    try:
        columns, rows = execute_query(safe_sql)
    except Exception:
        if fallback_sql is None or safe_sql.strip() == fallback_sql.strip():
            raise

        safe_sql = validate_sql(fallback_sql)
        columns, rows = execute_query(safe_sql)

    answer = summarize(question, safe_sql, columns, rows)
    return answer, safe_sql, columns, rows