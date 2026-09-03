# data_pipeline.py
  import pandas as pd
  import numpy as np
  from sklearn.impute import KNNImputer

  # Step 1: Load data
    df = pd.read_csv('data/life_expectancy.csv')
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 2: Drop rows where Life Expectancy is missing
      df = df.dropna(subset=['Life expectancy'])
      print(f"After removing missing target: {df.shape[0]} rows")

      # Step 3: KNN imputation for numeric columns
        imputer = KNNImputer(n_neighbors=5)
        numeric_cols = df.select_dtypes(include=np.number).columns
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df[numeric_cols]),
            columns=numeric_cols
        )

        # Step 4: Log transform GDP and Population
        df_imputed['log_GDP'] = np.log1p(df_imputed['GDP'])
        df_imputed['log_Population'] = np.log1p(df_imputed['Population'])

        # Step 5: Save processed data
          df_imputed.to_csv('data/processed_data.csv', index=False)
          print("Pipeline complete. Processed data saved.")
