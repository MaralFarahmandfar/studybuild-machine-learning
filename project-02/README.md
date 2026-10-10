# 🚗 Traffic Volume Analysis

## 📌 Project Overview

This project analyzes hourly westbound traffic volume on Interstate 94 (I-94) between Minneapolis and St. Paul, Minnesota. The dataset contains traffic, weather, holiday, and time-related information collected from 2012 to 2018.

The main goal of this project is to explore the factors associated with variations in hourly traffic volume and identify meaningful patterns in traffic behavior across different times, weather conditions, and holidays.

### 📊 Dataset

The dataset contains **48,204 hourly observations** and **8 input features**, with `traffic_volume` as the target variable.

The data were collected from Minnesota Department of Transportation Automatic Traffic Recorder (ATR) station 301, located approximately midway between Minneapolis and St. Paul.

### 🧩 Features

| Feature | Type | Description |
|---|---|---|
| `holiday` | Categorical | U.S. national holidays and the Minnesota State Fair |
| `temp` | Continuous | Average temperature in Kelvin |
| `rain_1h` | Continuous | Amount of rainfall during the hour (mm) |
| `snow_1h` | Continuous | Amount of snowfall during the hour (mm) |
| `clouds_all` | Integer | Percentage of cloud cover |
| `weather_main` | Categorical | Short description of the prevailing weather condition |
| `weather_description` | Categorical | More detailed description of the weather condition |
| `date_time` | Date/Time | Local date and time of the observation |
| `traffic_volume` | Integer | Hourly westbound traffic volume reported by ATR station 301 |


## 🧹 Data Cleaning & Feature Engineering

The raw dataset was cleaned and prepared through a separate reproducible data-cleaning and feature-engineering process.

### 🔧 Data Cleaning

- Merged duplicate hourly records, reducing the dataset from **48,204 to 40,575 observations**.
- Identified and corrected invalid temperature values and converted temperature from **Kelvin to Celsius**.
- Detected and corrected an extreme rainfall outlier (**9,831.3 mm/h**).
- Normalized inconsistent weather labels.
- Removed the unreliable `holiday` feature due to excessive missing values.
- Preserved the original `traffic_volume` target without modification.
- Documented missing time periods and suspicious low traffic values without artificially changing the data.

### ⚙️ Feature Engineering

New features were created to capture important traffic patterns, including:

- Calendar features such as **year, month, weekday, hour, weekend, and season**.
- **Cyclical time features** for hour, weekday, and month.
- Weekday **rush-hour indicators**.
- Weather-related features such as **temperature, freezing, rain, snow, and precipitation**.
- Log-transformed precipitation features.

The final dataset contains **40,575 observations, 23 model features, the `traffic_volume` target, and 4 QA columns**. QA columns are used only for validation and are not used for model training.

📓 **Notebook:** `data_cleaning.csv`

## 📈 Linear Regression Modeling

The cleaned hourly data were modeled with ordinary least-squares linear regression on the 23 engineered features (numeric inputs standardized, `season` one-hot encoded, the `qa_` columns never used). To respect the time structure of the data, the oldest 80% of the hours (2012-10-02 to 2017-10-26, 32,460 rows) were used for training and the newest 20% (to 2018-09-30, 8,115 rows) for testing, with the scaler and encoder fitted on the training part only. The baseline model reaches a test R² of 0.857 (RMSE ≈ 745 vehicles/hour, against a standard deviation of ≈ 1,985 for the target), with almost identical training and test scores. Residual diagnostics, however, show a clear weakness: the errors follow a daily wave, consecutive hours have strongly correlated errors (Durbin-Watson 0.85), and the precipitation flags are almost perfectly collinear (VIF > 1,000). A single straight line cannot draw the double-peak daily traffic curve, so four optional variants were tested. Adding the second daily harmonic of the hour together with weekday/weekend interactions (V4) clearly gave the best model, cutting the test RMSE by about 19%.

| Model | Model inputs | Test R² | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| **A** – Baseline (23 features) | 25 | 0.857 | 744.9 | 559.2 |
| **V1** – Cyclical encoding only (raw `hour`, `month`, `day_of_week` dropped) | 22 | 0.838 | 792.5 | 605.5 |
| **V2** – Log-transformed target (`log1p`) | 25 | 0.620 | 1213.7 | 832.1 |
| **V3** – Trained without the 46 flagged low-volume hours | 25 | 0.857 | 744.6 | 558.9 |
| **V4** – Hour harmonics × weekend interactions | 31 | **0.906** | **603.9** | **426.0** |

*RMSE and MAE are in vehicles per hour, computed on the chronological test set.*

The decision to leave weather labels out of the features was also tested. The 37 textual weather descriptions were merged into six groups with a similar effect on driving conditions (clear/cloudy, reduced visibility such as mist, fog, haze and smoke, light rain/drizzle, heavy rain, snow/ice, and severe storms covering thunderstorms and squalls). Adding these groups to the baseline changed the test RMSE only from 744.9 to 744.0, and a partial F-test found no significant gain (p = 0.30); removing all weather information costs only about 2 RMSE units, so hour of day and weekday dominate hourly traffic in a linear model. These results support the choice to exclude the labels, with two caveats: a linear model cannot capture interactions such as fog mattering more during rush hour, and the severe-storm group has only 239 hours. 📓 **Full notebook:** `traffic-analysis_full.ipynb` (sections 1–6).
