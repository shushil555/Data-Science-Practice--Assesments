# PRT661 - Life Expectancy Prediction

## Team Members
- Ashish (Data Engineer / ML Engineer / Docs Lead)
- Sadan (Data Engineer / Model Evaluator)
- Pramod (Data Engineer / ML Engineer)
- Sabin (Data Analyst / Visualisation Lead)
- Shushil (Data Analyst / ML Engineer)

## Project
Predicting life expectancy using WHO dataset (2000-2015).
193 countries, 2,938 rows, 22 columns.

## Dataset
WHO Life Expectancy dataset from Kaggle.

## Models Used
- Linear Regression (baseline) - R2=0.823
- Random Forest (in progress)
- XGBoost (in progress)

## Storage
Raw and processed data stored on Amazon S3:
- s3://prt661-life-expectancy/raw/
- s3://prt661-life-expectancy/processed/

## How to Run
1. Install requirements: pip install -r requirements.txt
2. Run pipeline: python data_pipeline.py
3. Open notebooks in /notebooks/ folder

## GitHub
https://github.com/shushil555/Data-Science-Practice--Assesments
