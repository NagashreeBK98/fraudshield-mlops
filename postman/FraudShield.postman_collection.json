-- FraudShield Database Schema

CREATE TABLE IF NOT EXISTS predictions (
    id                SERIAL PRIMARY KEY,
    transaction_id    VARCHAR(50) UNIQUE NOT NULL,
    amount            DECIMAL(10,2) NOT NULL,
    category          VARCHAR(100),
    prediction        VARCHAR(20) NOT NULL,
    fraud_probability DECIMAL(6,4) NOT NULL,
    confidence        VARCHAR(10),
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_created_at ON predictions(created_at DESC);
CREATE INDEX idx_prediction ON predictions(prediction);

CREATE VIEW fraud_summary AS
SELECT
    COUNT(*) AS total_predictions,
    SUM(CASE WHEN prediction='FRAUD' THEN 1 ELSE 0 END) AS fraud_count,
    SUM(CASE WHEN prediction='LEGITIMATE' THEN 1 ELSE 0 END) AS legit_count,
    ROUND(AVG(CASE WHEN prediction='FRAUD' THEN 1.0 ELSE 0.0 END)*100, 2) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_transaction_amount
FROM predictions;