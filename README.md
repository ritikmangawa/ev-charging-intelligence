# EV Charging Demand Prediction & Station Utilization Intelligence

An end-to-end Machine Learning system for EV charging infrastructure, featuring demand prediction and utilization analytics.

## 🚀 Project Overview

The system addresses two major objectives:

1. **EV Charging Demand Prediction**
   Predicts the expected energy demand for an EV charging session based on temporal and historical patterns using an XGBoost Machine Learning pipeline.
2. **Station Utilization Intelligence**
   Analyzes historical charging sessions to identify peak charging hours, high-demand locations, and session-based utilization indicators.

## 🧠 Machine Learning Approach

This is a Classical Machine Learning project utilizing Scikit-learn and XGBoost.
- **Features**: Time-derived features (Hour, DayOfWeek, IsWeekend, etc.) and Historical features (PreviousChargerDemand, HistoricalLocationMean, etc.).
- **Data Leakage Prevention**: Features like session duration, end time, and future outcomes are strictly excluded from the predictive model.
- **Validation Strategy**: TimeSeriesSplit cross-validation, maintaining chronological order of EV sessions.
- **Final Model**: Tuned XGBoost.

## 📊 Application (Streamlit)

A Streamlit application provides a user interface for:
- Predicting session demand by selecting Charger configurations and timing.
- Viewing the interactive Station Utilization Dashboard.
- Checking model evaluation metrics and feature importance.

### How to Run Locally

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app/main.py
   ```

## 📁 Project Structure

```text
├── data/                       # Contains dataset, lookup CSVs, and analysis results
├── models/                     # Saved machine learning pipelines (joblib)
├── notebook/                   # Jupyter notebooks (EDA, Feature Eng., Training)
├── app/                        # Streamlit application files (main.py)
├── EV_Charging_Demand_Prediction.ipynb # Master original notebook
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 📈 Evaluation Metrics

The final XGBoost model achieved on the chronological test set:
- **MAE**: ~8.97
- **RMSE**: ~12.07
- **R²**: ~0.247

## ⚠️ Notes
* Utilization indicators in this project refer to Session Activity and Average Demand. They do *not* represent physical capacity percentage, as hardware capacities are not available in the dataset.
