-- Database Schema: Investment Intelligence Platform
-- Star Schema Design

-- Dimension: Startup Information
CREATE TABLE IF NOT EXISTS dim_startup (
    startup_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category_list TEXT,
    status TEXT, -- e.g., 'acquired', 'closed', 'operating'
    country_code TEXT,
    state_code TEXT,
    city TEXT,
    founded_at DATE,
    first_funding_at DATE,
    last_funding_at DATE,
    description TEXT, -- Important for Deep-LLM
    embeddings_json TEXT -- Placeholder for future pre-computed embeddings
);

-- Dimension: Time (Derived from funding dates for faster analytics)
CREATE TABLE IF NOT EXISTS dim_time (
    date_key DATE PRIMARY KEY,
    year INTEGER,
    quarter INTEGER,
    month INTEGER
);

-- Fact: Funding Rounds
CREATE TABLE IF NOT EXISTS fact_funding_rounds (
    round_id INTEGER PRIMARY KEY AUTOINCREMENT,
    startup_id INTEGER,
    funding_round_type TEXT, -- e.g., 'seed', 'series-a'
    funding_round_code TEXT,
    raised_amount_usd REAL,
    participants INTEGER,
    is_first_round INTEGER,
    is_last_round INTEGER,
    FOREIGN KEY(startup_id) REFERENCES dim_startup(startup_id)
);

-- Indexes for performance
CREATE INDEX idx_startup_status ON dim_startup(status);
CREATE INDEX idx_funding_amount ON fact_funding_rounds(raised_amount_usd);
