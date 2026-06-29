"""
dashboard.py

Streamlit web interface for the EdgeOps AI platform.
Provides a real-time view of the MLOps registry, active deployment state,
and manual override controls for the pipeline, optimizer, and edge rollbacks.
Now integrates real-time telemetry stream simulation with concept drift injection.
"""

import sqlite3
import pandas as pd
import streamlit as st
import requests
import importlib

# Local imports
import config
import database
import ml_pipeline
import optimizer
import data_streamer  
importlib.reload(config) # Forces Streamlit to grab the freshest config.py on every refresh

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="EdgeOps AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# DATA FETCHING UTILITIES
# ---------------------------------------------------------
@st.cache_data(ttl=5) # Refresh cache every 5 seconds to show live updates
def fetch_registry_data() -> pd.DataFrame:
    """Fetches all registered model versions from SQLite."""
    with database.get_db_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM model_versions ORDER BY version_number DESC", conn)
    return df

@st.cache_data(ttl=5)
def fetch_deployment_state() -> pd.Series:
    """Fetches the singleton deployment state row."""
    with database.get_db_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM deployment_state WHERE id = 1", conn)
    return df.iloc[0] if not df.empty else pd.Series()

def check_api_status() -> bool:
    """Pings the FastAPI Edge Node to see if it is alive."""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=1)
        return response.status_code == 200
    except:
        return False
    
import json # Make sure json is imported at the top of your script!

#----------------------------------------------------------------------

def render_drift_alarm():
    """Reads the Evidently AI drift state and displays an alarm if triggered."""
    drift_file = config.BASE_DIR / "drift_state.json"
    if drift_file.exists():
        try:
            with open(drift_file, "r") as f:
                state = json.load(f)
            
            if state["drift_detected"] or state["drift_share"] >= config.DRIFT_THRESHOLD:
                st.error(f"🚨 **SYSTEM ALARM: Data Drift Detected!** The live factory data has shifted significantly ({state['drift_share']*100:.1f}% of features drifted). The Optimizer is spinning up a new model to compensate.")
            else:
                st.info(f"✅ **Data Stream Healthy:** No drift detected in the most recent log check ({state['drift_share']*100:.1f}% shift).")
        except:
            pass # Failsafe in case file is currently being written

# ---------------------------------------------------------
# UI COMPONENTS
# ---------------------------------------------------------
def render_sidebar():
    """Renders the MLOps control panel in the sidebar."""
    st.sidebar.title("⚡ EdgeOps AI")
    st.sidebar.caption("Automated MLOps & Edge Deployment")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Pipeline Controls")
    
    if st.sidebar.button("Run Initial Pipeline", use_container_width=True):
        with st.spinner("Training base model and initializing registry..."):
            database.init_db()
            ml_pipeline.run_pipeline()
        st.sidebar.success("Pipeline executed successfully!")
        st.rerun()

    if st.sidebar.button("🌊 Simulate Live Data Stream", use_container_width=True, type="primary"):
        with st.spinner("Generating 800 IoT readings & injecting machine drift..."):
            data_streamer.start_stream(cycles=1, interval_seconds=0)
        st.sidebar.success("Telemetry streamed! Pipeline gated evaluation finished.")
        st.rerun()
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Maintenance & Tuning")
    
    if st.sidebar.button("Force Optimization Cycle", use_container_width=True):
        with st.spinner("Running HPO and evaluating candidate..."):
            optimizer.run_optimization_cycle(force=True)
        st.sidebar.success("Optimization cycle complete!")
        st.rerun()

    if st.sidebar.button("⚠️ Log Component Maintenance", use_container_width=True):
        with st.spinner("Archiving old baseline statistics..."):
            optimizer.trigger_hard_reset()
        st.sidebar.error("Hard Reset Enabled! AI is tracking new hardware modifications.")
        st.rerun()

    if st.sidebar.button("Restore Factory Default Baseline", use_container_width=True):
        optimizer.clear_hard_reset()
        st.sidebar.success("Factory default baseline restored!")
        st.rerun()
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚨 Emergency Controls")
    if st.sidebar.button("⏪ Rollback Deployment", use_container_width=True):
        with st.spinner("Rolling back to stable fallback..."):
            success = database.rollback_deployment()
            if success:
                optimizer.trigger_edge_reload()
                st.sidebar.success("Rollback successful. Edge API reloaded.")
            else:
                st.sidebar.error("Rollback failed. No stable model found.")
        st.rerun()

def render_header():
    """Renders the main title and live API status."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏭 Predictive Maintenance Control Center")
        st.markdown("Monitor live model performance, inject drift, and manage edge deployments.")
    with col2:
        st.write("") # Spacing
        st.write("")
        if check_api_status():
            st.success("🟢 **Edge API:** Online & Ready")
        else:
            st.error("🔴 **Edge API:** Offline")

def render_active_deployment(state: pd.Series, registry: pd.DataFrame):
    """Renders the top KPI metrics for the currently active edge model."""
    st.markdown("### 🌐 Active Edge Deployment")
    
    if state.empty or pd.isna(state.get("current_model_id")):
        st.warning("No active deployment found. Run the pipeline to initialize the system.")
        return
        
    current_id = state["current_model_id"]
    stable_id = state.get("stable_model_id", "None")
    current_model = registry[registry["id"] == current_id].iloc[0]
    
    # Use a styled container for the metrics
    with st.container(border=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Active Version", f"v{current_model['version_number']}")
        col2.metric("ROC-AUC", f"{current_model['roc_auc']:.4f}")
        col3.metric("F1 Score", f"{current_model['f1_score']:.4f}")
        col4.metric("Accuracy", f"{current_model['accuracy']:.4f}")
        col5.metric("Fallback Target", f"v{stable_id}")

def render_performance_chart(registry: pd.DataFrame):
    """Renders a line chart showing model improvement over time."""
    if registry.empty:
        return
        
    st.markdown("### 📈 Optimization History")
    # Sort chronologically and set index to version number for the X-axis
    chart_data = registry.sort_values("version_number").set_index("version_number")
    
    # Display line chart tracking our core metrics
    st.line_chart(
        chart_data[["roc_auc", "f1_score"]], 
        height=250,
        color=["#2ecc71", "#3498db"] # Green and Blue
    )

def render_model_registry(registry: pd.DataFrame):
    """Renders the historical table of all trained models."""
    st.markdown("### 📚 Model Artifact Registry")
    
    if registry.empty:
        st.info("Registry is empty.")
        return
        
    # Format the dataframe for cleaner UI presentation
    display_df = registry.copy()
    display_df["version_number"] = display_df["version_number"].apply(lambda x: f"v{x}")
    display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Highlight the deployed model row
    deployment_state = fetch_deployment_state()
    active_id = deployment_state.get("current_model_id") if not deployment_state.empty else None
    
    def highlight_active(row):
        if row["id"] == active_id:
            return ['background-color: rgba(46, 204, 113, 0.2)'] * len(row)
        return [''] * len(row)
        
    styled_df = display_df.style.apply(highlight_active, axis=1)
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": "System ID",
            "version_number": "Version",
            "model_path": "Artifact Path",
            "created_at": "Timestamp",
            "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
            "f1_score": st.column_config.NumberColumn("F1 Score", format="%.4f"),
            "precision": None, # Hide less important metrics to keep it clean
            "recall": None
        }
    )

def main():
    """Main Streamlit execution loop."""
    render_sidebar()
    
    try:
        registry_df = fetch_registry_data()
        state_series = fetch_deployment_state()
        
        render_header()
        render_drift_alarm()
        st.markdown("---")
        
        render_active_deployment(state_series, registry_df)
        
        # Split the bottom into two columns: Chart on left, Table on right
        st.write("") # Spacing
        render_performance_chart(registry_df)
        render_model_registry(registry_df)
        
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        st.warning("⚠️ Database tables are missing or empty. Please click 'Run Initial Pipeline' in the sidebar to build the schema and train your first model!")

if __name__ == "__main__":
    main()