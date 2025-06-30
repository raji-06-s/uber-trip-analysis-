 Uber Trip Demand Forecasting using Machine Learning

 Project Objective

To analyze Uber trip data from April to September 2014 and forecast hourly demand using time series machine learning techniques. The goal is to identify usage patterns and build predictive models to help Uber improve operational efficiency, driver allocation, and pricing strategies.


 Dataset Description

- Source Files: `uber-raw-data-apr14.csv` to `uber-raw-data-sep14.csv`
- Additional File: `hourly_trip_data.csv` (hourly resampled and cleaned data)
- Features:
  - `Date/Time`: Timestamp of each ride
  - `Lat`, `Lon`: Geolocation
  - `Base`: Dispatching base ID

Rows Combined: Approx. 4.5 million records
Time Frame: April to September 2014
Format: .csv


Tools and Environment:
 Jupyter Notebook
Language: Python 3.7+
Core Libraries:
  - `pandas` – data manipulation
  - `numpy` – numerical operations
  - `matplotlib`, `seaborn` – visualizations
  - `scikit-learn` – RandomForest, GBTR, GridSearchCV
  - `xgboost` – gradient boosting model
  - `statsmodels` – time series decomposition


 Methodology

1. Data Preprocessing
   - Merged CSVs into one DataFrame
   - Converted `Date/Time` to `datetime64`
   - Set datetime index and resampled data using `.resample('h')`
   - Aggregated hourly trip counts

2. Feature Engineering
   - Extracted temporal features: `hour`, `day`, `weekday`, `month`
   - Created lag features using sliding window approach (24-hour windows)

3. Train-Test Split
   - Split at `2014-09-15 00:00:00`
   - Used `TimeSeriesSplit(n_splits=5)` for CV to maintain time order

4. Modeling
   - `RandomForestRegressor()` (baseline)
   - `GradientBoostingRegressor()` tuned using `GridSearchCV`
   - `XGBRegressor()` tuned with:
     - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`

5. Evaluation
   - Used `mean_absolute_percentage_error (MAPE)`
   - Visualization using custom `PlotPredictions()` and `PlotDecomposition()`



 Model Comparison (MAPE)

| Model               | MAPE (%) |
|--------------------|----------|
| Random Forest       | ~12.5    |
| Gradient Boosting   | ~10.2    |
| XGBoost (Best)      | ✅ ~8.4  |


 Visual Outputs

- 📊 Decomposed time series: trend, seasonality, residual
- 📈 Line charts of trip count by hour/day/week
- 📉 Model prediction vs actual plots (for RF, GBTR, XGBoost)
-  Model performance pie chart (MAPE share)



 Technical Learnings

- Time series resampling and lag window transformation
- Model tuning using `GridSearchCV`
- Handling large datasets in Jupyter using efficient vectorized code
- Building reusable visualization and prediction functions


 Future Improvements

- Integrate external data (weather, holidays, events)
- Use LSTM or Prophet for deep learning-based forecasting
- Develop a live dashboard using Streamlit or Dash
- Extend to multi-city or real-time Uber data


 Author

S. Rajeswari  
📧 raji.singaraj2005@gmail.com  
📞 8148242782  
GitHub Profile:(https://github.com/raji-06-s)



