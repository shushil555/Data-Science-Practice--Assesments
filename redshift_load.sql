-- redshift_load.sql - PRT661 Life Expectancy Prediction
-- Loads the cleaned WHO data from S3 into Amazon Redshift for SQL analytics
-- and the dashboard. Based on Lab 2 (Module 8): Storing and Analyzing Data
-- by Using Amazon Redshift.
--
-- Run in the Redshift query editor. Replace <IAM_ROLE_ARN> with the role
-- that lets Redshift read the S3 bucket (never paste access keys here).

-- 1. Table for the processed dataset
CREATE TABLE IF NOT EXISTS life_expectancy (
    country                  VARCHAR(64),
    year                     SMALLINT,
    status                   VARCHAR(16),
    life_expectancy          DECIMAL(5,2),
    adult_mortality          DECIMAL(8,2),
    infant_deaths            DECIMAL(8,2),
    alcohol                  DECIMAL(6,2),
    percentage_expenditure   DECIMAL(12,2),
    hepatitis_b              DECIMAL(6,2),
    measles                  DECIMAL(10,2),
    bmi                      DECIMAL(6,2),
    under_five_deaths        DECIMAL(8,2),
    polio                    DECIMAL(6,2),
    total_expenditure        DECIMAL(6,2),
    diphtheria               DECIMAL(6,2),
    hiv_aids                 DECIMAL(6,2),
    gdp                      DECIMAL(14,2),
    population               DECIMAL(18,2),
    thinness_1_19            DECIMAL(6,2),
    thinness_5_9             DECIMAL(6,2),
    income_composition       DECIMAL(5,3),
    schooling                DECIMAL(5,2),
    log_gdp                  DECIMAL(8,4),
    log_population           DECIMAL(8,4),
    status_developed         SMALLINT
)
DISTSTYLE AUTO
SORTKEY (country, year);   -- most queries filter by country and year

-- 2. Load from S3 (processed layer)
COPY life_expectancy
FROM 's3://prt661-life-expectancy/processed/processed_data.csv'
IAM_ROLE '<IAM_ROLE_ARN>'
FORMAT AS CSV
IGNOREHEADER 1
EMPTYASNULL;   -- blank values (not yet imputed) load as NULL

-- 3. Check the load
SELECT COUNT(*) AS row_count, COUNT(DISTINCT country) AS countries
FROM life_expectancy;

-- 4. Analytics queries for the dashboard
-- KPI: average life expectancy, developed vs developing
SELECT status, ROUND(AVG(life_expectancy), 1) AS avg_life_expectancy
FROM life_expectancy
GROUP BY status;

-- Trend: global average by year (2000-2015)
SELECT year, ROUND(AVG(life_expectancy), 2) AS avg_life_expectancy
FROM life_expectancy
GROUP BY year
ORDER BY year;

-- Relationship: schooling band vs life expectancy
SELECT FLOOR(schooling / 2) * 2 AS schooling_band,
       ROUND(AVG(life_expectancy), 1) AS avg_life_expectancy,
       COUNT(*) AS n
FROM life_expectancy
GROUP BY 1
ORDER BY 1;

-- Drill-down: one country over time
SELECT year, life_expectancy, adult_mortality, schooling, gdp
FROM life_expectancy
WHERE country = 'Australia'
ORDER BY year;
