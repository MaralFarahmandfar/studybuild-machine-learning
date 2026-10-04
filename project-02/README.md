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

This phase starts from the **raw CSV** (not from the EDA notebook's output), so it is independent and fully reproducible. Everything is implemented in `traffic_cleaning_feature_engineering_final.ipynb`, which ends with real `assert` checks that fail loudly if any step is broken or reordered.

**Guiding principles**

1. Measure first, fix second: every anomaly is counted and located before it is treated.
2. Invalid values are set to `NaN` **before** duplicate timestamps are merged, otherwise a median would hide them.
3. The target `traffic_volume` is **never modified** (verified by an assertion against the raw data).
4. Every treated anomaly leaves a trace in a `qa_*` flag column.

### 🔍 Problems found and how we solved them

| # | Problem (evidence) | Why it matters | Solution |
|---|---|---|---|
| 1 | **17 exact duplicate rows** | Same observation counted twice | Dropped. |
| 2 | **Duplicate timestamps**: 5,430 hours have 2–6 rows (7,612 extra rows). `traffic_volume` is identical inside every group (proved, 0 exceptions); only the weather report differs | Hours counted several times bias means and correlations; a model could see the same target repeatedly | Merged to **one row per hour**: median for numeric columns, most frequent value for weather labels. Result: 48,204 → **40,575 rows**. |
| 3 | **Weather label merging trap**: taking the mode of `weather_main` and `weather_description` separately creates impossible pairs (e.g. `Mist` + `light snow`) for 1,542 hours | Corrupts categorical features | Mode is taken on the **(main, description) pair**; an assertion checks every final pair exists in the raw data. |
| 4 | **Temperature = 0 K** (10 rows, two windows in Jan/Feb 2014) | Sensor failure; `temp − 273.15` gives −273 °C | Set to `NaN`, then time-based interpolation, flagged. |
| 5 | **Stuck temperature values**: `276.793 K` (78 rows) and `297.888 K` (43 rows) repeat for up to 33 and 43 consecutive hours. The first appears in the middle of the January 2014 cold wave at +3.6 °C while neighbours are −10 to −26 °C; the second stays flat for 43 hours in July with no daily cycle | Not catchable with range filters; silently wrong by up to ~30 °C | Detected from repeated off-grid values, set to `NaN` before merging, interpolated in time (max gap between valid neighbours 48 h), flagged with `qa_temp_imputed` (127 hours). |
| 6 | **Rainfall = 9,831.3 mm/h** (next highest real value is 55.63; neighbouring hours are 0) | One value dwarfs the whole column and blows up `log1p` | Set to `NaN` and imputed with the median of rows sharing the same label (`very heavy rain` ≈ 23.75 mm), flagged with `qa_rain_invalid`. Capping at an arbitrary value (e.g. 100) was rejected because it invents a value larger than any real observation. |
| 7 | **Inconsistent casing**: `Sky is Clear` (1,726 rows) vs `sky is clear` (11,665) | Duplicate categories after one-hot encoding | Lower-cased and stripped **before** merging (38 raw labels → 37). |
| 8 | **`holiday` column is incomplete**: only 61 rows are labelled, **all at 00:00** (53 dates); 3 federal holidays present in the data are not labelled at all (2013-01-21, 2014-07-04, 2016-01-18) | The other 23 hours of a holiday look like normal days | Rebuilt from the raw labels + the US federal calendar, applied to **all 24 hours**. Names match the original column. |
| 9 | **Missing hours**: 11,976 of 52,551 hourly slots (22.8 %) are absent; the largest gap is 307 days (Aug 2014 → Jun 2015) | A plain `shift(1)` lag may point months back; interpolating the target is invalid | The target is never filled. Lag features are built by exact timestamp (see below) and kept out of the main file. |
| 10 | **Suspicious target values**: 46 hours below 100 vehicles/h, in 9 contiguous clusters (summer 2015–2016, some at 1–10 vehicles) | Looks like sensor outage or road closure; cannot be told apart from the data | Left unchanged and flagged with `qa_low_volume`. |
| 11 | **`holiday` NaN misread as "missing data"** | 99.87 % "missing" is really "not a holiday" | Not imputed; converted to `is_holiday` / `holiday_name`. |

**Missing values:** raw data had 48,143 missing cells (all in `holiday`, structural). After cleaning there are **0 missing cells**.

### 🛠️ Feature Engineering

| Group | Features | Rationale |
|---|---|---|
| Calendar | `year`, `month`, `day_of_week`, `hour`, `is_weekend`, `season` | Basic temporal structure. `day`, `week_of_year` and `day_name` were skipped (redundant or collinear). |
| Cyclical encoding | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` | Hour 23 and hour 0 are adjacent in time but 23 apart numerically. |
| Rush hour | `is_rush_hour` (weekdays only, hours 6–8 and 15–17) | Weekends have no morning peak. The hour set was confirmed from the data (hours within 85 % of the weekday peak); mean traffic is 5,773 in rush hours vs 2,756 otherwise. |
| Holidays | `holiday_name`, `is_holiday`, `is_day_before_holiday`, `is_day_after_holiday`, `is_state_fair_opening` | Peak-hour traffic is ~0.56× normal on holidays, ~0.82× the day before and ~0.89× the day after. The State Fair opening day shows no effect (~1.00×), so it is kept as a separate flag instead of being mixed into `is_holiday`. |
| Weather | `temp_c`, `is_freezing`, `is_raining`, `is_snowing`, `precipitation_flag`, `rain_1h_log1p`, `snow_1h_log1p` | Only ~5 % of hours have rain, so a presence flag carries signal the raw value hides. Log transform reduces skew. |
| Original kept | `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description` | Dropping is left to the modeling phase. |

**Result:** 30 feature columns + the target + 4 `qa_*` columns, 40,575 rows.

### 🚩 QA / trace columns (do **not** train on these)

`qa_rain_invalid`, `qa_temp_imputed`, `qa_n_reports`, `qa_low_volume`.
`qa_low_volume` is derived from the target itself, so using it as a feature would be direct leakage. The ready-made list of columns is in `feature_lists.json`.

### 📦 Output files

| File | Content |
|---|---|
| `traffic_cleaning_feature_engineering_final.ipynb` | Full executed pipeline with audit, plots and assertions |
| `Metro_Traffic_Clean_FE.csv` | Clean dataset (40,575 × 36) for modeling |
| `feature_lists.json` | `FEATURES`, `TARGET`, `QA_COLS` |
| `Metro_Traffic_Forecast_Features.csv` | *Optional*, forecasting only: `lag_1h`, `lag_24h`, `lag_168h`, `roll_mean_24h` |

**Why lag/rolling features are separate:** they are built from the target and are valid only for forecasting, not for "traffic from weather and time" regression. They use exact timestamp matching (not `shift`) and a past-only time window, and NaNs are intentionally left unfilled.

### ⚠️ Known limitations

- 127 temperature values are interpolated (up to ~44 h between valid neighbours); this loses the daily cycle in those windows. They are marked by `qa_temp_imputed`.
- One isolated temperature jump (2013-02-03 18:00) looks suspicious but cannot be proven wrong, so it was left as is.
- The rain value 9,831.3 is replaced by a label-based estimate; this is one row out of 40,575 and is flagged.
- The 307-day gap splits the series in two; any model should be validated with this in mind.

### ✅ Recommendations for the modeling phase

1. Use a **time-based** train/test split, not a random one (neighbouring hours are nearly identical).
2. Exclude all `qa_*` columns from the features.
3. Fit scalers and encoders on the training set only.
4. For linear models use either raw `hour/day_of_week/month` **or** their sin/cos versions (collinearity); tree models do not need sin/cos.
5. Report errors separately for `qa_low_volume == 1` rows.
6. Rare `weather_main` classes (`Smoke`, `Squall`) should be grouped at encoding time.
7. EDA statistics computed before this step still include duplicated timestamps and the anomalies above, so they should be re-run on the cleaned file.
