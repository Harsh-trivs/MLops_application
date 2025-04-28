import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go  # Add this import

# Configuration
API_BASE_URL = "http://localhost:8000"  # Update with your FastAPI server URL

st.set_page_config(layout="wide")
st.title("Dynamic Forecasting Visualization")

# Initialize session state
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
if 'current_date' not in st.session_state:
    st.session_state.current_date = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'drift_dates' not in st.session_state:
    st.session_state.drift_dates = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'window_size' not in st.session_state:  # Add this for the time window
    st.session_state.window_size = 30

# Sidebar for model initialization
with st.sidebar:
    st.header("Model Configuration")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    window_size = st.number_input("Window Size", min_value=1, value=7)
    threshold = st.number_input("Threshold", min_value=0.0, value=8.0, step=0.01)
    
    if st.button("Initialize Model") and uploaded_file is not None:
        try:
            # Send file to FastAPI endpoint
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {"window_size": window_size, "threshold": threshold}
            
            response = requests.post(
                f"{API_BASE_URL}/model_init",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                st.success("Model initialized successfully!")
                st.session_state.model_initialized = True
                
                # Get the last trained date
                response = requests.get(f"{API_BASE_URL}/last_trained_date")
                if response.status_code == 200:
                    last_trained_date = datetime.strptime(response.json()["last_trained_date"], "%Y-%m-%d")
                    st.session_state.current_date = last_trained_date + timedelta(days=1)
                    st.session_state.predictions = []
                    st.session_state.drift_dates = []
                else:
                    st.error("Could not get last trained date")
            else:
                st.error(f"Error initializing model: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Main content
if st.session_state.model_initialized:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Forecast Visualization")
        chart_placeholder = st.empty()
        
    with col2:
        st.header("Controls")
        if st.button("Start Simulation"):
            st.session_state.is_running = True
        
        if st.button("Pause Simulation"):
            st.session_state.is_running = False
        
        if st.button("Reset"):
            st.session_state.predictions = []
            st.session_state.drift_dates = []
            st.session_state.is_running = False
        
        st.metric("Current Date", st.session_state.current_date.strftime("%Y-%m-%d") if st.session_state.current_date else "N/A")
        
        drift_count = len([p for p in st.session_state.predictions if p.get('drift_detected', False)])
        st.metric("Drift Events Detected", drift_count)
        
        # Add control for time window size
        st.session_state.window_size = st.slider("View Window (days)", 7, 365, 30)
    
    # Display the current state of the chart with scrolling
    if len(st.session_state.predictions) > 0:
        df = pd.DataFrame(st.session_state.predictions)
        df['date'] = pd.to_datetime(df['date'])
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['predicted'],
            name='Predicted',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['actual'],
            name='Actual',
            line=dict(color='green')
        ))
        
        # Highlight drift points
        if len(st.session_state.drift_dates) > 0:
            drift_df = df[df['date'].isin(pd.to_datetime(st.session_state.drift_dates))]
            fig.add_trace(go.Scatter(
                x=drift_df['date'],
                y=drift_df['predicted'],
                mode='markers',
                name='Drift Detected',
                marker=dict(color='red', size=8)
            ))
        
        # Set up scrolling window
        if len(df) > st.session_state.window_size:
            min_date = df['date'].iloc[-st.session_state.window_size]
        else:
            min_date = df['date'].iloc[0]
        
        max_date = df['date'].iloc[-1] + timedelta(days=1)
        
        fig.update_layout(
            xaxis=dict(
                range=[min_date, max_date],
                rangeslider=dict(visible=True),
                type="date"
            ),
            title="Predicted vs Actual Demand (Scrolling View)",
            xaxis_title="Date",
            yaxis_title="Demand",
            hovermode="x unified"
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Simulation logic
    if st.session_state.is_running and st.session_state.current_date:
        try:
            # Get prediction for current date
            response = requests.get(
                f"{API_BASE_URL}/predict_for_date",
                params={"current_date": st.session_state.current_date.strftime("%Y-%m-%d")}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Store prediction
                prediction = {
                    "date": st.session_state.current_date.strftime("%Y-%m-%d"),
                    "predicted": float(data.get("predicted", 0)),
                    "actual": float(data.get("actual", 0)),
                    "error": float(data.get("error", 0)),
                    "drift_detected": bool(data.get("drift_detected", False))
                }
                
                st.session_state.predictions.append(prediction)
                
                # Update drift dates if detected
                if prediction["drift_detected"]:
                    st.session_state.drift_dates.append(prediction["date"])
                
                # Update current date
                st.session_state.current_date += timedelta(days=1)
                
                # Rerun to update the display
                st.rerun()
            
            else:
                st.error(f"Error getting prediction: {response.json().get('error', 'Unknown error')}")
                st.session_state.is_running = False
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.is_running = False

else:
    st.warning("Please initialize the model first using the sidebar controls.")

# Display raw data if available
if len(st.session_state.predictions) > 0:
    with st.expander("Show Raw Data"):
        st.dataframe(pd.DataFrame(st.session_state.predictions))