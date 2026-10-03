# PRT661 - Life Expectancy Prediction
Sydney Group 3, Theme 2: Predictive Analytics and Forecasting


## Team Members
- Shushil Paudel
- Ashish Dhakal
- Pramod Yadav
- Sabin Subedi
- Sadan Magar


## Project
Predicting life expectancy Which health, economic and social factors drive a country's life expectancy,?
WHO Life Expectancy dataset (Kaggle): 193 countries, 2000-2015, 2,938 rows, 22 columns.

## Dataset
Download "Life Expectancy Data.csv" from Kaggle:
https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

## Architecture
S3 raw/ -> data_pipeline.py -> S3 processed/ -> Amazon Redshift (dashboard queries) and model training.
All AWS access uses IAM roles; no keys are stored in this repository.


## Models Used
- Linear Regression (baseline) - R2=0.8195
- Random Forest 
- XGBoost 


## How to Run
1. pip install -r requirements.txt
2. python data_pipeline.py "data/Life Expectancy Data.csv"
3. python model_comparison.py
4. (Optional) python s3_setup.py, then run redshift_load.sql in the Redshift query editor

## Results (5-fold cross-validation grouped by country)
| Model | R² | RMSE (years) |
|---|---|---|
| Linear Regression (A2 baseline) | 0.794 | 4.18 |
| Random Forest (tuned) | 0.910 | 2.77 |
| XGBoost (tuned) - best | 0.915 | 2.69 |


## GitHub
https://github.com/shushil555/Data-Science-Practice--Assesments
