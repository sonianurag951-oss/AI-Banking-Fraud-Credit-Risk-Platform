CREATE TABLE customers (
    customer_id      INTEGER PRIMARY KEY,
    age              INTEGER,
    income           NUMERIC(12, 2),
    employment_type  VARCHAR(50),
    credit_score     INTEGER,
    account_tenure   INTEGER,             
    city             VARCHAR(100),
    dependents       INTEGER
);

CREATE TABLE loans (
    loan_id           INTEGER PRIMARY KEY,
    customer_id       INTEGER NOT NULL,
    loan_amount       NUMERIC(14, 2),
    loan_term         INTEGER,            
    interest_rate     NUMERIC(5, 2),       
    monthly_income    NUMERIC(12, 2),
    existing_debt     NUMERIC(14, 2),
    previous_defaults INTEGER,
    late_payments     INTEGER,
    "default"         INTEGER,            
    CONSTRAINT fk_loans_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers (customer_id)
);

CREATE TABLE transactions (
    transaction_id    INTEGER PRIMARY KEY,
    customer_id       INTEGER NOT NULL,
    amount            NUMERIC(14, 2),
    transaction_time  TIMESTAMP,
    merchant_category VARCHAR(100),
    payment_method    VARCHAR(50),
    location          VARCHAR(100),
    device_type       VARCHAR(50),
    is_fraud          INTEGER,             
    CONSTRAINT fk_transactions_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers (customer_id)
);


select * from customers 
limit 5

SELECT
    table_schema,
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;

SELECT *
FROM public.loans
LIMIT 5;


SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('customers', 'loans', 'transactions')
ORDER BY table_name, ordinal_position;



CREATE ROLE banking_copilot
    LOGIN
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOINHERIT
    PASSWORD 'REPLACE_WITH_A_SECURE_PASSWORD';

GRANT CONNECT ON DATABASE "AI_Banking_Fraud"
    TO banking_copilot;

GRANT USAGE ON SCHEMA public
    TO banking_copilot;

GRANT SELECT ON TABLE
    public.customers,
    public.loans,
    public.transactions
TO banking_copilot;

SELECT COUNT(*) FROM public.customers;
SELECT COUNT(*) FROM public.loans;
SELECT COUNT(*) FROM public.transactions;


SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('customers', 'loans', 'transactions')
ORDER BY table_name, ordinal_position;


SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('customers', 'loans', 'transactions')
ORDER BY table_name, ordinal_position;


SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('customers', 'loans', 'transactions')
ORDER BY table_name, ordinal_position;

SELECT
    merchant_category,
    COUNT(*) AS transaction_count,
    SUM(amount) AS total_amount
FROM public.transactions
GROUP BY merchant_category
ORDER BY total_amount DESC
LIMIT 100;


SELECT
    transaction_id,
    amount,
    merchant_category
FROM public.transactions
ORDER BY amount DESC
LIMIT 10;


SELECT
    merchant_category,
    COUNT(*) AS transaction_count
FROM public.transactions
GROUP BY merchant_category
ORDER BY transaction_count DESC
LIMIT 100;

SELECT DISTINCT customer_id
FROM public.transactions
WHERE amount > 50000
ORDER BY customer_id
LIMIT 100;

-- Check the largest transaction amount
SELECT MAX(amount) AS max_transaction_amount
FROM public.transactions;

-- Check the transaction amount range
SELECT
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount,
    COUNT(*) AS total_transactions,
    COUNT(*) FILTER (WHERE amount > 50000) AS transactions_over_50000
FROM public.transactions;

-- Inspect the 10 largest transactions
SELECT transaction_id, customer_id, amount
FROM public.transactions
ORDER BY amount DESC
LIMIT 100;

SELECT
    MAX(amount) AS largest_transaction,
    COUNT(*) FILTER (WHERE amount > 50000) AS matching_transactions
FROM public.transactions;

SELECT DISTINCT customer_id
FROM public.transactions
WHERE amount > 50000
ORDER BY customer_id
LIMIT 100;

SELECT COUNT(*) AS fraud_count
FROM public.transactions
WHERE is_fraud = 1;

SELECT
    COUNT(*) AS total_transactions,
    MAX(amount) AS largest_transaction,
    COUNT(*) FILTER (
        WHERE amount > 50000
    ) AS matching_transactions
FROM public.transactions;