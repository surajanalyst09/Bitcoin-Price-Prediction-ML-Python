# ============================================================
# BITCOIN PRICE PREDICTION USING MACHINE LEARNING IN PYTHON
# Date: 02 October 2026
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression 


# ============================================================
# 2. LOAD BITCOIN DATASET
# ============================================================

df = pd.read_csv("bitcoin.csv") 


# ============================================================
# 3. INITIAL DATASET EXPLORATION
# ============================================================

# print(df.head()) 
# print(df.shape) 
# print(df.info()) 

# print(df.isnull().sum()) 


# ============================================================
# 4. DATE CONVERSION
# ============================================================

df["Date"] = pd.to_datetime(df["Date"]) 
# print(df.info()) 


# ============================================================
# 5. DATE RANGE CHECK
# ============================================================

# print(df["Date"].min()) 
# print(df["Date"].max()) 


# ============================================================
# 6. DUPLICATE DATE CHECK
# ============================================================

# print(df["Date"].duplicated().sum()) 


# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

# print(df.describe()) 


# ============================================================
# 8. BITCOIN CLOSING PRICE OVER TIME
# ============================================================

# plt.plot(df["Date"], df["Close"]) 
# plt.xlabel("Date") 
# plt.ylabel("Close Price") 
# plt.title("Bitcoin Price Over Time") 
# plt.legend() 
# # save chart 
# plt.savefig("bitcoin_price_over_time1.png", dpi=300, bbox_inches="tight") 
# plt.show() 


# ============================================================
# 9. YEAR-WISE AVERAGE BITCOIN CLOSING PRICE
# ============================================================

# Chart/Analysis Name: Year-wise Average Bitcoin Closing Price 
# Date: 02 October 2026 

yearly_close = df.groupby(df["Date"].dt.year)["Close"].mean() 

print(yearly_close) 


# ============================================================
# 10. YEAR-WISE AVERAGE CLOSING PRICE CHART
# ============================================================

# charts 
# yearly_close.plot(kind="bar") 
# plt.xlabel("year") 
# plt.ylabel("Average closing price") 
# plt.title("year-wise Average Bitcoin closing price") 
# plt.legend() 
# # save chart 
# plt.savefig("year-wise_average_bitcoin_closing_price2.png", dpi=300, bbox_inches="tight") 
# plt.show() 


# ============================================================
# 11. DAILY RETURN ANALYSIS
# ============================================================

# Analysis Name: Bitcoin Daily Return Statistics 
# Date: 02 October 2026 

df["Daily_Return"] = df["Close"].pct_change() 

print("\n========== DAILY RETURN ANALYSIS ==========") 
print(df["Daily_Return"].describe()) 
print("===========================================") 


# ============================================================
# 12. DAILY RETURN DISTRIBUTION CHART
# ============================================================

# Chart Name: Bitcoin Daily Return Distribution 
# Date: 02 October 2026 

# df["Daily_Return"].plot(kind ="hist") 

# plt.xlabel("Daily Return") 
# plt.ylabel("Frequency") 
# plt.title("Bitcoin Daily Return Distribution") 
# plt.legend() 
# # save chart         
# plt.savefig("bitcoin_daily_return_distribution3.png", dpi=300, bbox_inches="tight")   
# plt.show() 


# ============================================================
# 13. 30-DAY ROLLING VOLATILITY ANALYSIS
# ============================================================

rolling_volatility = df["Daily_Return"].rolling(window=30).std() 

print("\n========== ROLLING VOLATILITY ANALYSIS ==========") 
print(rolling_volatility.describe()) 
print("===========================================") 


# ============================================================
# 14. ROLLING VOLATILITY CHART - INDEX BASED VERSION
# ============================================================

# charts 
# rolling_volatility.plot(kind="line") 
# plt.xlabel("Date")     
# plt.ylabel("Rolling Volatility")     
# plt.title("Bitcoin Rolling Volatility (30-day window)") 
# plt.show() 


# ============================================================
# 15. BITCOIN 30-DAY ROLLING VOLATILITY CHART
# ============================================================

# Chart Name: Bitcoin 30-Day Rolling Volatility 
# Date: 02 October 2026 

# plt.plot(df["Date"], rolling_volatility) 

# plt.xlabel("Date") 
# plt.ylabel("Rolling Volatility") 
# plt.title("Bitcoin 30-Day Rolling Volatility") 
# plt.legend() 
# # save chart 
# plt.savefig("bitcoin_30_day_rolling_volatility4.png", dpi=300, bbox_inches="tight")  
# plt.show() 


# ============================================================
# 16. FEATURE ENGINEERING - PREVIOUS DAY CLOSING PRICE
# ============================================================

# Feature Engineering: Previous Day Closing Price 
# Date: 02 October 2026 

df["previous_close"] = df["Close"].shift(1) 

print(df[["Date", "Close", "previous_close"]].head()) 


# ============================================================
# 17. FEATURE ENGINEERING - MOVING AVERAGES
# ============================================================

df["MA_7"] = df["Close"].rolling(window=7).mean() 

df["MA_30"] = df["Close"].rolling(window=30).mean() 

print(df[["Date", "Close", "MA_7", "MA_30"]].head())       


# ============================================================
# 18. TARGET VARIABLE - NEXT DAY CLOSING PRICE
# ============================================================

# Target Variable: Next-Day Bitcoin Closing Price 
# Date: 02 October 2026 

df["Target"] = df["Close"].shift(-1) 

print(df[["Date", "Close", "Target"]].head()) 
print(df[["Date", "Close", "Target"]].tail()) 


# ============================================================
# 19. PREPARE DATASET FOR MACHINE LEARNING
# ============================================================

# ML Dataset Preparation 
# Date: 02 October 2026 

ml_data = df.dropna().copy() 

print(ml_data.shape) 
print(ml_data.isnull().sum()) 


# ============================================================
# 20. DEFINE FEATURES AND TARGET
# ============================================================

X=ml_data[["Close", "previous_close", "MA_7", "MA_30", "Daily_Return"]] 

y=ml_data["Target"] 

print("X shape:", X.shape) 
print("y shape:", y.shape) 


# ============================================================
# 21. TRAIN-TEST SPLIT FOR TIME SERIES
# ============================================================

# Train-Test Split for Time-Series 
# Date: 02 October 2026 

split_index = int(len(X) * 0.8) 

X_train = X.iloc[:split_index] 

X_test = X.iloc[split_index:] 

y_train = y.iloc[:split_index] 

y_test = y.iloc[split_index:] 

print("X_train:", X_train.shape) 
print("X_test:", X_test.shape) 
print("y_train:", y_train.shape) 
print("y_test:", y_test.shape) 


# ============================================================
# 22. TRAIN LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression() 

model.fit(X_train, y_train) 


# ============================================================
# 23. PREDICT NEXT-DAY BITCOIN CLOSING PRICE
# ============================================================

# Prediction: Next-Day Bitcoin Closing Price 
# Date: 02 October 2026 

y_pred = model.predict(X_test) 

print("First 10 predicted prices:") 
print(y_pred[:10]) 

print("\n First 10 actual prices:") 
print(y_test.iloc[:10].values) 


# ============================================================
# 24. MODEL EVALUATION - MAE
# ============================================================

# Model Evaluation: Mean Absolute Error 
# Date: 02 October 2026 

from sklearn.metrics import mean_absolute_error 

mae = mean_absolute_error(y_test, y_pred) 

print("MAE:", mae) 


# ============================================================
# 25. MODEL EVALUATION - RMSE
# ============================================================

# Model Evaluation: Root Mean Squared Error 
# Date: 02 October 2026 

from sklearn.metrics import mean_squared_error 

rmse = mean_squared_error(y_test, y_pred) ** 0.5 

print("RMSE:", rmse)     


# ============================================================
# 26. MODEL EVALUATION - R² SCORE
# ============================================================

# Model Evaluation: R² Score 
# Date: 02 October 2026 

from sklearn.metrics import r2_score 

r2 = r2_score(y_test, y_pred) 

print("R² Score:", r2) 


# ============================================================
# 27. ACTUAL VS PREDICTED PRICE CHART
# ============================================================

# Chart Name: Actual vs Predicted Bitcoin Closing Price 
# Date: 02 October 2026 

# plt.plot(y_test.values, label="Actual Price") 
# plt.plot(y_pred, label="Predicted Price") 

# plt.xlabel("Test Observations") 
# plt.ylabel("Bitcoin Closing Price") 
# plt.title("Actual vs Predicted Bitcoin Closing Price") 
# plt.legend() 
# # Save Chart 
# plt.savefig("actual_vs_predicted_bitcoin_price.png", dpi=300, bbox_inches="tight") 
# plt.show() 


# ============================================================
# 28. BASELINE MODEL - MAE
# ============================================================

#mae  
baseline_pred = X_test["Close"] 

mae = mean_absolute_error(y_test, baseline_pred) 

print("Baseline MAE:", mae) 


# ============================================================
# 29. BASELINE MODEL - RMSE
# ============================================================

# rsme 
baseline_pred = X_test["Close"] 

rmse = mean_squared_error(y_test, baseline_pred) ** 0.5 

print("Baseline RMSE:", rmse) 


# ============================================================
# 30. BASELINE MODEL - R² SCORE
# ============================================================

# R2 
baseline_pred = X_test["Close"] 

r2 = r2_score(y_test, baseline_pred)     

print("Baseline R² Score:", r2) 


# ============================================================
# 31. LINEAR REGRESSION VS BASELINE COMPARISON CHART
# ============================================================

# Linear Regression vs Baseline chart: 

# plt.plot(y_test.values, label="Actual Price") 
# plt.plot(y_pred, label="Linear Regression") 
# plt.plot(baseline_pred, label="Baseline") 
# plt.xlabel("Test Observations") 
# plt.ylabel("Bitcoin Closing Price")          
# plt.title("Actual vs Linear Regression vs Baseline") 
# plt.legend() 
# # Save Chart 
# plt.savefig("bitcoin_model_vs_baseline6.png", dpi=300, bbox_inches="tight") 
# plt.show()     


# ============================================================
# 32. PROJECT CONCLUSION
# Date: 02 October 2026
# ============================================================

# ============================================================ 
# PROJECT CONCLUSION 
# Date: 02 October 2026 
# ============================================================ 
 
# Bitcoin's average closing price showed a strong overall upward 
# trend, with a slight decline in 2019 and approximately 12.8% lower 
# average closing price in 2022 compared with 2021. 
# 
# Daily return analysis showed an average daily return of 0.24% 
# with a standard deviation of 3.88%, indicating significant 
# day-to-day price volatility. 
# 
# The 30-day rolling volatility ranged from approximately 0.71% 
# to 9.13%, showing that Bitcoin's volatility changed considerably 
# over time. 
# 
# The Linear Regression model achieved: 
# MAE  = 1139.64 
# RMSE = 1645.25 
# R²   = 0.98905 
# 
# The Baseline model achieved: 
# MAE  = 1132.36 
# RMSE = 1641.01 
# R²   = 0.98911 
# 
# The Baseline model performed slightly better than Linear Regression 
# on the test dataset. Therefore, the selected Linear Regression 
# features did not provide a meaningful improvement over the 
# simple baseline approach. 
# 
# Overall, the project covered Bitcoin price analysis, volatility 
# analysis, feature engineering, time-series model training, 
# evaluation, and comparison with a baseline model.