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
Source: WHO Life Expectancy dataset on Kaggle (KumarRajarshi, 2018):
https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

Included in `data/` so the project runs without a download:
- `data/Life Expectancy Data.csv` - original file, unchanged
- `data/processed_data.csv` - output of `data_pipeline.py` (2,928 rows, 183 countries), used by Redshift, the models and the dashboard

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
5. streamlit run dashboard.py  (opens the dashboard at http://localhost:8501)

## Results (5-fold cross-validation grouped by country)
| Model | R² | RMSE (years) |
|---|---|---|
| Linear Regression (A2 baseline) | 0.794 | 4.18 |
| Random Forest (tuned) | 0.910 | 2.77 |
| XGBoost (tuned) - best | 0.915 | 2.69 |


## GitHub
https://github.com/shushil555/Data-Science-Practice--Assesments


##How to Run Dashboard 

Open a terminal: Terminal > New Terminal
Install the libraries:
       pip install -r requirements.txt
 Start the dashboard:
       python -m streamlit run dashboard.py
