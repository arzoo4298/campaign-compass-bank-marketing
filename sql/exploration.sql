-- Campaign Compass SQL pack (DuckDB): run against bank-additional-full.csv.
-- CREATE TABLE bank_marketing AS SELECT * FROM read_csv('data/bank-additional-full.csv', delim=';', header=true);

-- 1. Establish the baseline conversion rate and class balance.
SELECT
    count(*) AS contacted_customers,
    countif(y = 'yes') AS subscriptions,
    round(100.0 * avg(CASE WHEN y = 'yes' THEN 1 ELSE 0 END), 2) AS conversion_pct
FROM bank_marketing;

-- 2. Compare previous-campaign outcomes, keeping the sample size visible.
SELECT
    poutcome AS previous_outcome,
    count(*) AS customers,
    countif(y = 'yes') AS subscriptions,
    round(100.0 * avg(CASE WHEN y = 'yes' THEN 1 ELSE 0 END), 2) AS conversion_pct
FROM bank_marketing
GROUP BY poutcome
ORDER BY conversion_pct DESC;

-- 3. Find segments worth investigating; suppress tiny groups from a naive ranking.
SELECT
    job,
    count(*) AS customers,
    countif(y = 'yes') AS subscriptions,
    round(100.0 * avg(CASE WHEN y = 'yes' THEN 1 ELSE 0 END), 2) AS conversion_pct
FROM bank_marketing
GROUP BY job
HAVING count(*) >= 500
ORDER BY conversion_pct DESC;

-- 4. Quantify contact pressure descriptively. Correlation does not prove fatigue:
-- customer intent and selection into repeated contact can confound this pattern.
SELECT
    CASE
        WHEN campaign = 1 THEN '1'
        WHEN campaign BETWEEN 2 AND 3 THEN '2-3'
        WHEN campaign BETWEEN 4 AND 6 THEN '4-6'
        ELSE '7+'
    END AS contacts_in_current_campaign,
    count(*) AS customers,
    round(100.0 * avg(CASE WHEN y = 'yes' THEN 1 ELSE 0 END), 2) AS conversion_pct
FROM bank_marketing
GROUP BY 1
ORDER BY min(campaign);

-- 5. Explore month/channel mix without mistaking mix differences for causal effects.
SELECT
    month,
    contact,
    count(*) AS customers,
    round(100.0 * avg(CASE WHEN y = 'yes' THEN 1 ELSE 0 END), 2) AS conversion_pct
FROM bank_marketing
GROUP BY month, contact
HAVING count(*) >= 100
ORDER BY month, contact;
