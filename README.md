# HemaMatch - Blood Inventory Forecasting Module

An ARIMA-based machine learning module built on top of HemaMatch, a real-time blood donation and emergency matching system for Kenyan hospitals. This module shifts the system from reactive (alerting when stock is already low) to predictive (forecasting depletion days in advance).


## What This Does

Kenyan hospitals currently discover blood shortages only when stock has already run out. This module analyses historical blood consumption patterns and forecasts when each blood type will hit a critical level - giving hospital staff a warning window to act before the emergency hits.


## Live Dashboard

[View on Streamlit Cloud](#) — update this link after deployment


## Features

- Forecasts inventory for all 8 blood types across 3 hospitals
- ARIMA(2,1,2) time series model trained on historical consumption data
- Early warning alerts triggered when stock is projected to drop below 10 units
- Interactive controls - switch hospitals, adjust forecast window (3-30 days), filter blood types
- Light and dark mode support
- Model evaluation metrics (MAE and RMSE) displayed per blood type
- Synthetic data generated from published Kenyan blood bank consumption patterns


## Project Structure

```
hemamatch-ml/
├── .streamlit/
│   └── config.toml
├── data/
│   └── hemamatch_blood_inventory.csv
├── generate_blood_data.py
├── arima_forecast.py
├── dashboard.py
├── streamlit_dashboard.py
└── requirements.txt
```


## The Data

Real hospital blood bank data is protected under the Kenya Data Protection Act 2022 and is not accessible to student researchers. The dataset used here is synthetic - generated to closely mirror real Kenyan blood bank consumption patterns based on:

- Blood type prevalence from Kenya National Blood Transfusion Service (O+ approximately 40%, AB- approximately 1%)
- Weekend and public holiday usage spikes driven by higher accident and trauma rates
- Seasonal patterns - higher consumption during March-April long rains and November–December short rains
- Random emergency surge events simulating mass casualty incidents
- Restocking cycles modelled on typical donation drive frequency

The result is 2 years of daily data (2023–2024) across 8 blood types and 3 hospitals, totalling 17,544 rows. This approach is standard practice in healthcare ML research and is documented transparently here.


## Hospitals Covered

| Code | Hospital |
|------|----------|
| KNH | Kenyatta National Hospital, Nairobi |
| MKS_L5 | Machakos Level 5 Hospital |
| MSA_PGH | Mombasa Port Reitz Hospital |


## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/elmbondo/hemamatch-ml.git
cd hemamatch-ml
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate the dataset**
```bash
python generate_blood_data.py
```

**4. Run the ARIMA model (terminal output)**
```bash
python arima_forecast.py
```

**5. Launch the dashboard**
```bash
streamlit run streamlit_dashboard.py
```


## Model Details

| Parameter | Value |
|-----------|-------|
| Model | ARIMA(2,1,2) |
| Training window | 60 days |
| Forecast horizon | 7 days (adjustable up to 30) |
| Critical threshold | 10 units |
| Evaluation metrics | MAE and RMSE |

The model is retrained per blood type per hospital on each run. Rare blood types like B- and AB- show higher alert frequency due to lower baseline stock levels, which reflects real-world scarcity patterns in Kenyan hospitals.


## Sample Output

```
Blood Type   Current Stock    MAE     RMSE    Alert
O+           118.0            15.55   19.23   OK
O-           18.7             11.61   13.18   OK
A+           0.0              10.04   12.40   CRITICAL
B-           0.0               5.01    5.44   CRITICAL
AB+          5.8               4.31    4.73   CRITICAL
AB-          4.8               2.17    2.40   CRITICAL

EARLY WARNING ALERTS — KNH — Next 7 Days
  A+  - projected critical in 1 day(s)  (current stock: 0.0 units)
  B-  - projected critical in 1 day(s)  (current stock: 0.0 units)
  AB+ - projected critical in 1 day(s)  (current stock: 5.8 units)
  AB- - projected critical in 1 day(s)  (current stock: 4.8 units)
```


## Tech Stack

- Python - pandas, numpy, statsmodels, scikit-learn
- Streamlit - interactive dashboard
- Plotly - interactive charts
- ARIMA via statsmodels - time series forecasting


## Part of HemaMatch

This module is an ML extension of [HemaMatch](https://github.com/elmbondo/Hema_match) - a full-stack blood donation and emergency matching system built with React.js, Node.js, and PostgreSQL for Kenyan hospitals.


## Author

**Fidelmah Nthambi Mbondo**  
BSc Mathematics and Computer Science, JKUAT  
[GitHub](https://github.com/elmbondo)
