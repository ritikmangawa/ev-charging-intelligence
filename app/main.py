import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime

# Page configuration
st.set_page_config(
    page_title="EV Charging Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set custom CSS for a modern, sleek aesthetic
st.markdown("""
<style>
    :root {
        --primary-color: #00ff88;
        --bg-color: #0e1117;
        --text-color: #e0e0e0;
        --panel-bg: #1e222b;
    }
    
    .main {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    .stMetric {
        background-color: var(--panel-bg);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border-left: 4px solid var(--primary-color);
        transition: transform 0.2s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
    }
    
    h1, h2, h3 {
        color: var(--primary-color) !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        border: none;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.5);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('models/ev_demand_pipeline.joblib')

@st.cache_data
def load_data():
    app_options = pd.read_csv('data/app_options.csv')
    charger_history = pd.read_csv('data/charger_history_lookup.csv')
    location_history = pd.read_csv('data/location_history_lookup.csv')
    charger_hour = pd.read_csv('data/charger_hour_lookup.csv')
    last_charger = pd.read_csv('data/last_charger_demand_lookup.csv')
    last_location = pd.read_csv('data/last_location_demand_lookup.csv')
    
    # Load analysis results for dashboard
    charger_util = pd.read_csv('data/charger_utilization.csv')
    location_util = pd.read_csv('data/location_utilization.csv')
    hourly_demand = pd.read_csv('data/hourly_demand.csv')
    feat_importance = pd.read_csv('data/feature_importance.csv')
    model_results = pd.read_csv('data/final_model_results.csv')
    
    return (app_options, charger_history, location_history, charger_hour, 
            last_charger, last_location, charger_util, location_util, 
            hourly_demand, feat_importance, model_results)

# Load resources
try:
    pipeline = load_model()
    (app_options, charger_history, location_history, charger_hour, 
     last_charger, last_location, charger_util, location_util, 
     hourly_demand, feat_importance, model_results) = load_data()
except Exception as e:
    st.error(f"Error loading required files. Please ensure the project structure is correct. Details: {e}")
    st.stop()


# Sidebar Navigation
st.sidebar.title("⚡ EV Dashboard")
page = st.sidebar.radio("Navigate", ["Demand Prediction", "Station Utilization Intelligence", "Model Performance"])

st.sidebar.markdown("---")
st.sidebar.markdown("Built for EV Charging Intelligence.")

if page == "Demand Prediction":
    st.title("🔋 EV Charging Demand Prediction")
    st.write("Enter the session details below to predict expected charging demand.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # We use the app_options to provide valid combinations
            # Start by selecting a ChargerID
            charger_ids = sorted(app_options['ChargerID'].unique())
            selected_charger = st.selectbox("Charger ID", charger_ids)
            
            # Filter options based on selected charger
            charger_data = app_options[app_options['ChargerID'] == selected_charger].iloc[0]
            
            selected_company = st.text_input("Charger Company", value=str(charger_data['ChargerCompany']), disabled=True)
            selected_location = st.text_input("Location", value=str(charger_data['Location']), disabled=True)
            selected_type = st.text_input("Charger Type", value=str(charger_data['ChargerType']), disabled=True)
            
        with col2:
            st.markdown("##### Session Timing")
            selected_date = st.date_input("Charging Date", value=datetime.date.today())
            selected_time = st.time_input("Charging Start Time", value=datetime.time(12, 0))
            
        submit = st.form_submit_button("Predict Demand 🚀")
        
    if submit:
        # Extract base values
        c_id = int(selected_charger)
        c_comp = int(charger_data['ChargerCompany'])
        c_loc = charger_data['Location']
        c_type = int(charger_data['ChargerType'])
        
        # Temporal features
        dt_val = datetime.datetime.combine(selected_date, selected_time)
        hour = dt_val.hour
        day_of_week = dt_val.weekday() # 0 = Monday, 6 = Sunday
        day_of_month = dt_val.day
        week_of_year = dt_val.isocalendar()[1]
        month = dt_val.month
        year = dt_val.year
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Historical features
        prev_charger = last_charger[last_charger['ChargerID'] == c_id]['PreviousChargerDemand']
        prev_charger_val = prev_charger.values[0] if not prev_charger.empty else last_charger['PreviousChargerDemand'].median()
        
        prev_loc = last_location[last_location['Location'] == c_loc]['PreviousLocationDemand']
        prev_loc_val = prev_loc.values[0] if not prev_loc.empty else last_location['PreviousLocationDemand'].median()
        
        hist_charger = charger_history[charger_history['ChargerID'] == c_id]['HistoricalChargerMean']
        hist_charger_val = hist_charger.values[0] if not hist_charger.empty else charger_history['HistoricalChargerMean'].median()
        
        hist_loc = location_history[location_history['Location'] == c_loc]['HistoricalLocationMean']
        hist_loc_val = hist_loc.values[0] if not hist_loc.empty else location_history['HistoricalLocationMean'].median()
        
        c_hour = charger_hour[(charger_hour['ChargerID'] == c_id) & (charger_hour['Hour'] == hour)]['ChargerHourHistoricalMean']
        c_hour_val = c_hour.values[0] if not c_hour.empty else hist_charger_val
        
        # Build dataframe exactly as expected by the pipeline
        feature_order = [
            'ChargerID', 'ChargerCompany', 'Location', 'ChargerType',
            'Hour', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Month',
            'Year', 'IsWeekend', 'PreviousChargerDemand', 'PreviousLocationDemand',
            'HistoricalChargerMean', 'HistoricalLocationMean', 'ChargerHourHistoricalMean'
        ]
        
        input_data = pd.DataFrame([[
            c_id, c_comp, c_loc, c_type,
            hour, day_of_week, day_of_month, week_of_year, month,
            year, is_weekend, prev_charger_val, prev_loc_val,
            hist_charger_val, hist_loc_val, c_hour_val
        ]], columns=feature_order)
        
        # Predict
        try:
            prediction = pipeline.predict(input_data)[0]
            st.success("Prediction Successful!")
            st.markdown(f"### ⚡ Expected Charging Demand: **{prediction:.2f} kWh**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

elif page == "Station Utilization Intelligence":
    st.title("📊 Station Utilization Intelligence")
    
    st.markdown("Analyze historical charging sessions to identify active chargers, locations, and peak hours. *Note: Utilization indicators are based on session activity and demand, not physical charger capacity.*")
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    total_demand = charger_util['TotalDemand'].sum() if 'TotalDemand' in charger_util.columns else 0
    avg_demand = charger_util['AverageDemand'].mean() if 'AverageDemand' in charger_util.columns else 0
    total_sessions = charger_util['SessionCount'].sum() if 'SessionCount' in charger_util.columns else 0
    
    with col1:
        st.metric("Total Recorded Demand (kWh)", f"{total_demand:,.0f}")
    with col2:
        st.metric("Total Sessions", f"{total_sessions:,.0f}")
    with col3:
        st.metric("Average Session Demand", f"{avg_demand:.2f} kWh")
    with col4:
        st.metric("Active Chargers", f"{len(charger_util)}")
        
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Hourly Peak Demand")
        if 'Hour' in hourly_demand.columns and 'AverageDemand' in hourly_demand.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1e222b')
            ax.set_facecolor('#1e222b')
            ax.plot(hourly_demand['Hour'], hourly_demand['AverageDemand'], marker='o', color='#00ff88', linewidth=2)
            ax.set_xlabel("Hour of Day", color='#e0e0e0')
            ax.set_ylabel("Average Demand", color='#e0e0e0')
            ax.tick_params(colors='#e0e0e0')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
            ax.grid(color='#444', linestyle='--', alpha=0.5)
            st.pyplot(fig)
        else:
            st.info("Hourly demand chart data not formatted as expected.")
            
    with col_chart2:
        st.subheader("🏢 Location Activity (Top 5)")
        if 'Location' in location_util.columns and 'SessionCount' in location_util.columns:
            top_locs = location_util.sort_values('SessionCount', ascending=False).head(5)
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1e222b')
            ax.set_facecolor('#1e222b')
            bars = ax.bar(top_locs['Location'], top_locs['SessionCount'], color='#00C9FF')
            ax.set_xlabel("Location Type", color='#e0e0e0')
            ax.set_ylabel("Total Sessions", color='#e0e0e0')
            ax.tick_params(colors='#e0e0e0')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
            st.pyplot(fig)
        else:
            st.info("Location utilization chart data not formatted as expected.")

    st.markdown("---")
    
    st.subheader("🔥 Top 10 Highest-Demand Chargers")
    if 'ChargerID' in charger_util.columns and 'TotalDemand' in charger_util.columns:
        top_chargers = charger_util.sort_values('TotalDemand', ascending=False).head(10)
        # Beautify dataframe
        st.dataframe(top_locs if False else top_chargers.style.background_gradient(cmap='viridis', subset=['TotalDemand', 'SessionCount']), use_container_width=True)

elif page == "Model Performance":
    st.title("🎯 Model Performance & Intelligence")
    
    st.markdown("### Final Model Evaluation Metrics (XGBoost)")
    
    col1, col2, col3 = st.columns(3)
    
    # Try to extract metrics safely
    try:
        mae = model_results['MAE'].values[0] if 'MAE' in model_results.columns else "N/A"
        rmse = model_results['RMSE'].values[0] if 'RMSE' in model_results.columns else "N/A"
        r2 = model_results['R2'].values[0] if 'R2' in model_results.columns else "N/A"
        
        with col1:
            st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}" if isinstance(mae, float) else mae)
        with col2:
            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}" if isinstance(rmse, float) else rmse)
        with col3:
            st.metric("R² Score", f"{r2:.4f}" if isinstance(r2, float) else r2)
    except Exception as e:
        st.error("Could not parse final_model_results.csv correctly.")
        
    st.markdown("---")
    
    st.subheader("🧩 Feature Importance")
    st.write("Visualizing the most influential features driving the XGBoost predictions.")
    
    if not feat_importance.empty and 'Feature' in feat_importance.columns and 'Importance' in feat_importance.columns:
        top_features = feat_importance.sort_values('Importance', ascending=True).tail(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1e222b')
        ax.set_facecolor('#1e222b')
        
        ax.barh(top_features['Feature'], top_features['Importance'], color='#92FE9D')
        ax.set_xlabel("Importance Score", color='#e0e0e0')
        ax.tick_params(colors='#e0e0e0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
            
        st.pyplot(fig)
    else:
        st.info("Feature importance data is missing or incorrectly formatted.")
