# Bitcoin Price Prediction Using Machine Learning in Python

## Project Overview

This project analyzes historical Bitcoin price data and builds a Machine Learning model to predict the next day's Bitcoin closing price.

The project combines exploratory data analysis, return and volatility analysis, feature engineering, time-series train-test splitting, Linear Regression, model evaluation, and comparison with a simple baseline approach.

The objective is to understand Bitcoin's historical price behavior and evaluate whether Linear Regression can provide an improvement over a simple previous-close prediction strategy.

---

## Objectives

* Analyze historical Bitcoin price trends.
* Calculate year-wise average closing prices.
* Analyze daily Bitcoin returns and price volatility.
* Calculate 30-day rolling volatility.
* Create meaningful features for Machine Learning.
* Predict the next day's Bitcoin closing price.
* Evaluate the Linear Regression model using MAE, RMSE, and R².
* Compare the Machine Learning model with a simple baseline model.

---

## Dataset

The project uses historical Bitcoin market data containing daily price and trading-volume information.

### Dataset Features

| Column    | Description                  |
| --------- | ---------------------------- |
| Date      | Trading date                 |
| Open      | Opening price                |
| High      | Highest price during the day |
| Low       | Lowest price during the day  |
| Close     | Closing price                |
| Adj Close | Adjusted closing price       |
| Volume    | Trading volume               |

### Dataset Information

* **Total Records:** 2,713
* **Total Columns:** 7
* **Date Range:** 17 September 2014 – 19 February 2022
* **Missing Values:** None
* **Duplicate Dates:** None

---

## Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* VS Code

---

## Project Workflow

```text
Raw Bitcoin Dataset
        ↓
Data Loading
        ↓
Data Cleaning & Validation
        ↓
Exploratory Data Analysis
        ↓
Daily Return Analysis
        ↓
Rolling Volatility Analysis
        ↓
Feature Engineering
        ↓
Target Variable Creation
        ↓
Time-Series Train-Test Split
        ↓
Linear Regression Model
        ↓
Prediction
        ↓
Model Evaluation
        ↓
Baseline Comparison
        ↓
Final Analysis & Conclusion
```

---

## 1. Data Loading and Validation

The Bitcoin dataset was loaded using Pandas.

Initial validation included:

* Dataset preview
* Dataset shape
* Column and data-type inspection
* Missing-value checking
* Date range checking
* Duplicate-date checking
* Descriptive statistics

The `Date` column was converted into a datetime format to support time-based analysis.

---

## 2. Bitcoin Price Trend Analysis

The historical closing price was analyzed to understand Bitcoin's long-term price movement.

Year-wise average closing prices were calculated using the `Date` column.

### Key Observation

Bitcoin's average closing price showed a strong overall upward trend over the analyzed period. There was a slight decline in 2019, followed by substantial growth in subsequent years.

The average closing price in 2022 was approximately **12.8% lower than in 2021** within the available dataset period.

---

## 3. Daily Return Analysis

Daily returns were calculated using the percentage change in Bitcoin's closing price.

### Results

| Metric               |   Value |
| -------------------- | ------: |
| Average Daily Return |   0.24% |
| Standard Deviation   |   3.88% |
| Minimum Daily Return | -37.17% |
| Maximum Daily Return |  25.25% |

### Key Observation

The average daily return was approximately **0.24%**, while the standard deviation of **3.88%** indicates significant day-to-day price volatility.

The distribution of daily returns was concentrated around zero, with some large positive and negative movements.

---

## 4. 30-Day Rolling Volatility

A 30-day rolling standard deviation of daily returns was calculated to study how Bitcoin's volatility changed over time.

### Results

| Metric                    | Value |
| ------------------------- | ----: |
| Mean Rolling Volatility   | 3.53% |
| Median Rolling Volatility | 3.33% |
| Minimum                   | 0.71% |
| Maximum                   | 9.13% |

### Key Observation

Bitcoin's 30-day rolling volatility varied considerably over time, ranging from approximately **0.71% to 9.13%**.

This indicates that Bitcoin experienced periods of both relatively lower and higher market volatility.

---

## 5. Feature Engineering

Several features were created for the Machine Learning model.

### Features Used

* `Close` — Current day's closing price
* `previous_close` — Previous day's closing price
* `MA_7` — 7-day moving average
* `MA_30` — 30-day moving average
* `Daily_Return` — Daily percentage return

### Moving Averages

Two moving averages were created:

* **7-day moving average:** Short-term price trend
* **30-day moving average:** Broader short-term price trend

These features were used to provide the model with information about recent Bitcoin price behavior.

---

## 6. Target Variable

The target variable represents the **next day's Bitcoin closing price**.

The target was created by shifting the `Close` column one position backward:

```python
df["Target"] = df["Close"].shift(-1)
```

This means that the features from the current day are used to predict the closing price of the following day.

---

## 7. Machine Learning Dataset

After creating the required features and target variable, rows containing missing values were removed.

### Final ML Dataset

* **Rows:** 2,683
* **Features:** 5
* **Target:** Next-day closing price

### Feature Matrix

```text
Close
previous_close
MA_7
MA_30
Daily_Return
```

### Target

```text
Target
```

---

## 8. Time-Series Train-Test Split

Because Bitcoin data is time-dependent, the dataset was split chronologically instead of randomly.

* **80%:** Training data
* **20%:** Testing data

### Split Results

| Dataset |     Shape |
| ------- | --------: |
| X       | 2,683 × 5 |
| X_train | 2,146 × 5 |
| X_test  |   537 × 5 |
| y_train |     2,146 |
| y_test  |       537 |

The earlier observations were used for training, while the later observations were reserved for testing.

---

## 9. Linear Regression Model

A Linear Regression model from Scikit-learn was trained using the engineered features.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

The trained model was then used to generate next-day closing-price predictions for the test dataset.

---

## 10. Model Evaluation

The Linear Regression model was evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### Linear Regression Results

| Metric |   Result |
| ------ | -------: |
| MAE    | 1,139.64 |
| RMSE   | 1,645.25 |
| R²     |  0.98905 |

### Metric Interpretation

**MAE:**
The model's predictions differed from the actual next-day closing prices by approximately **1,139.64 price units on average** in absolute terms.

**RMSE:**
The RMSE was **1,645.25**, which gives greater weight to larger prediction errors.

**R²:**
The R² score of **0.98905** indicates that the model explained a high proportion of the variation in the test target under this particular feature setup and test period.

R² should not be interpreted as prediction accuracy.

---

## 11. Baseline Model

A simple baseline approach was created for comparison.

The baseline assumes:

> **Tomorrow's predicted closing price = Today's closing price**

The baseline prediction was therefore created using the `Close` value from the test dataset.

### Baseline Results

| Metric |   Result |
| ------ | -------: |
| MAE    | 1,132.36 |
| RMSE   | 1,641.01 |
| R²     |  0.98911 |

---

## 12. Linear Regression vs Baseline

The Machine Learning model was compared with the simple baseline.

| Model             |      MAE |     RMSE |      R² |
| ----------------- | -------: | -------: | ------: |
| Linear Regression | 1,139.64 | 1,645.25 | 0.98905 |
| Baseline          | 1,132.36 | 1,641.01 | 0.98911 |

The baseline produced slightly lower MAE and RMSE and a slightly higher R² score on the test dataset.

Therefore, with the selected features and test period, Linear Regression did not provide a meaningful improvement over the simple previous-close baseline.

---

## 13. Visualizations

The project includes visual analysis of:

* Bitcoin closing price over time
* Year-wise average Bitcoin closing price
* Daily return distribution
* 30-day rolling volatility
* Actual vs predicted closing prices
* Linear Regression vs baseline predictions

These visualizations help understand historical price movement, volatility, and model behavior.

---

## 14. Key Insights

1. Bitcoin's average closing price showed a strong overall upward trend across the analyzed period.
2. The dataset showed considerable day-to-day price volatility.
3. Daily returns were generally concentrated around zero but included several extreme movements.
4. 30-day rolling volatility changed significantly over time.
5. Moving averages and previous closing prices were used as predictive features.
6. The Linear Regression model achieved an R² of approximately 0.989 on the test dataset.
7. The simple previous-close baseline performed slightly better than Linear Regression on the selected evaluation metrics.
8. The results demonstrate that a relatively simple time-series baseline can be difficult to improve upon when the current closing price is already highly informative for next-day prediction.

---

## 15. Project Conclusion

This project provided practical experience in applying Python and Machine Learning to historical Bitcoin market data.

The analysis covered data validation, price trend analysis, daily return calculation, rolling volatility, feature engineering, time-series train-test splitting, Linear Regression modeling, model evaluation, and baseline comparison.

The results showed that Linear Regression performed similarly to the simple baseline, but did not provide a meaningful improvement on the selected test dataset.

---

## Project Structure

```text
Bitcoin Price Prediction using Machine Learning in Python/
│
├── bitcoin.csv
├── bitcoin_pridiction.py
├── README.md
│
├── bitcoin_price_over_time1.png
├── year-wise_average_bitcoin_closing_price2.png
├── bitcoin_daily_return_distribution3.png
├── bitcoin_30_day_rolling_volatility4.png
├── actual_vs_predicted_linear_regression.png
└── bitcoin_model_vs_baseline6.png
```

---

## Skills Demonstrated

* Python Programming
* Pandas Data Analysis
* Data Cleaning & Validation
* Exploratory Data Analysis
* Time-Series Analysis
* Feature Engineering
* Data Visualization
* Machine Learning
* Linear Regression
* Model Evaluation
* Baseline Model Comparison
* Business-Oriented Data Interpretation
