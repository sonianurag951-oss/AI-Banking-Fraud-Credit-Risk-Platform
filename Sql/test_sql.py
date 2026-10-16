
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# test_sql.py is inside the Sql folder.
# parents[1] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

print("Project root:", PROJECT_ROOT)
print("Environment file exists:", ENV_FILE.exists())
print("Database configured:", bool(os.getenv("PGDATABASE")))

required = ["PGDATABASE", "PGUSER", "PGPASSWORD"]
missing = [key for key in required if not os.getenv(key)]

if missing:
    raise RuntimeError(
        f"Missing settings in {ENV_FILE}: {', '.join(missing)}"
    )

with psycopg.connect(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.environ["PGDATABASE"],
    user=os.environ["PGUSER"],
    password=os.environ["PGPASSWORD"],
    connect_timeout=5,
) as conn:
    with conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")

        for table in ("customers", "loans", "transactions"):
            cur.execute(
                f'SELECT COUNT(*) FROM public."{table}"'
            )
            print(f"{table}: {cur.fetchone()[0]} rows")

print("PostgreSQL connection successful!")
