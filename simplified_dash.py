import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import random
import requests

# Set page configuration
st.set_page_config(
    page_title="Simplified HSC Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for font styling
st.markdown("""
    <style>
    /* Import Space Grotesk Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Apply Space Grotesk font to everything */
    html, body, [class*="st"], [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* Navigation buttons styling */
    .stButton>button {
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
        padding-left: 10px !important;
        width: 100% !important;
    }
    
    /* File upload success styling */
    .upload-success {
        background-color: #f0f7ff !important;
        border-left: 4px solid #4257B2 !important;
        padding: 1rem !important;
        border-radius: 4px !important;
        margin: 1rem 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize prediction storage structure
def initialize_experiment_data():
    """Initialize empty data structures for a new experiment"""
    # Create empty prediction storage
    hsc_predictions = []
    lineage_predictions = []
    
    return hsc_predictions, lineage_predictions

# Initialize session state variables
if 'experiments' not in st.session_state:
    st.session_state.experiments = []
    
if 'current_experiment' not in st.session_state:
    st.session_state.current_experiment = None

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Welcome"

if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = {}

# Force Welcome page to be the default on each page load
if 'page_just_loaded' not in st.session_state:
    st.session_state.current_page = "Welcome"  # Reset to Welcome page on each fresh load
    st.session_state.page_just_loaded = True

# Sidebar navigation
with st.sidebar:
    st.header("Osiris v.1")
    st.markdown("---")
    
    # Create New Experiment button
    if st.button("➕ New Experiment", use_container_width=True, key="create_experiment_btn"):
        st.session_state.show_create_dialog = True
    
    # Show experiment creation dialog
    if st.session_state.get('show_create_dialog', False):
        with st.form("new_experiment_form"):
            experiment_name = st.text_input("Experiment Name", key="new_experiment_name")
            submitted = st.form_submit_button("Create Experiment")
            
            if submitted and experiment_name:
                # Create a new experiment
                experiment_id = len(st.session_state.experiments) + 1
                new_experiment = {
                    "id": experiment_id,
                    "name": experiment_name,
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Add the new experiment to the list
                st.session_state.experiments.append(new_experiment)
                
                # Initialize empty prediction data for the new experiment
                st.session_state[f"hsc_predictions_{experiment_id}"] = []
                st.session_state[f"lineage_predictions_{experiment_id}"] = []
                
                # Initialize data upload status for this experiment
                st.session_state.data_uploaded[experiment_id] = False
                
                # Set this as the current experiment
                st.session_state.current_experiment = experiment_id
                st.session_state.current_page = "Dashboard"
                
                # Hide the dialog
                st.session_state.show_create_dialog = False
                st.rerun()
    
    # Display experiment tabs if any exist
    if st.session_state.experiments:
        st.markdown("### My Experiments")
        
        for experiment in st.session_state.experiments:
            # Create a button for each experiment
            if st.button(f"📊 {experiment['name']}", use_container_width=True, key=f"experiment_{experiment['id']}"):
                st.session_state.current_experiment = experiment["id"]
                st.session_state.current_page = "Dashboard"
                st.rerun()

# Display the appropriate page based on the current_page value
if st.session_state.current_page == "Welcome":
    # Welcome page with centered content
    st.markdown("""
    <style>
    .welcome-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 2rem;
        max-width: 800px;
        margin: 0 auto;
        margin-top: 3rem;
    }
    .welcome-heading {
        font-weight: 600;
        font-size: 2.5rem;
        color: #4257B2;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="welcome-container">
        <h1 class='welcome-heading'>Hello, Neo!</h1>
        <h3>👈 Click '➕ New Experiment' to get started.</h3>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == "Dashboard":
    # Check if an experiment is selected
    if st.session_state.current_experiment is not None:
        # Find the selected experiment
        selected_exp = next((exp for exp in st.session_state.experiments if exp["id"] == st.session_state.current_experiment), None)
        
        if selected_exp:
            # Display experiment-specific dashboard
            st.title(f"{selected_exp['name']}")
            
            # Create experiment-specific keys for session state
            hsc_predictions_key = f"hsc_predictions_{selected_exp['id']}"
            lineage_predictions_key = f"lineage_predictions_{selected_exp['id']}"
            
            # Initialize prediction data for this experiment if it doesn't exist
            if hsc_predictions_key not in st.session_state:
                st.session_state[hsc_predictions_key] = []
                st.session_state[lineage_predictions_key] = []
            
            # File uploader section
            st.markdown("### Upload Data")
            uploaded_file = st.file_uploader(
                "Upload your `.tsv` file here", 
                type=["tsv"],
                key=f"uploader_{selected_exp['id']}"
            )
            
            # Backend URL
            backend_url = "http://localhost:8080/predict"
            
            if uploaded_file is not None:
                # Display success message
                st.markdown(f"""
                <div class="upload-success">
                    <h4 style="margin-top: 0; color: #4257B2;">✅ File uploaded successfully</h4>
                    <p style="margin-bottom: 0;"><strong>File:</strong> {uploaded_file.name}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Run prediction button
                if st.button("Run Prediction", key=f"predict_btn_{selected_exp['id']}"):
                    try:
                        with st.spinner("Processing your data..."):
                            # Send the file as multipart/form-data
                            files = {"file": uploaded_file}
                            response = requests.post(backend_url, files=files)
                            
                            if response.status_code == 200:
                                data = response.json()
                                st.success(data["message"])
                                
                                # Store predictions in session state
                                st.session_state[hsc_predictions_key] = data.get("hsc_predictions", [])
                                st.session_state[lineage_predictions_key] = data.get("lineage_predictions", [])
                                
                                # Mark data as uploaded
                                st.session_state.data_uploaded[selected_exp['id']] = True
                                st.rerun()
                            else:
                                st.error(f"Server returned an error: {response.status_code}")
                                st.json(response.json())
                    except Exception as e:
                        st.error(f"Could not connect to backend: {e}")
            
            # Display predictions if data has been uploaded and processed
            if st.session_state.data_uploaded.get(selected_exp['id'], False):
                st.markdown("---")
                
                # Get predictions from session state
                hsc_predictions = st.session_state[hsc_predictions_key]
                lineage_predictions = st.session_state[lineage_predictions_key]
                
                # Display HSC predictions
                st.markdown("### HSC Fate Prediction")
                
                if hsc_predictions:
                    # Create columns for displaying predictions
                    cols = st.columns(len(hsc_predictions))
                    
                    for i, result in enumerate(hsc_predictions):
                        with cols[i]:
                            # Calculate a color based on probability (higher = more intense)
                            prob = float(result['probability'])
                            color_intensity = int(prob * 255)
                            color = f"rgba(66, 87, 178, {prob})"
                            
                            st.markdown(f"""
                            <div style="padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; text-align: center;">
                                <h3 style="color: {color}; font-size: 1.5rem; margin-bottom: 0.5rem;">{result['class']}</h3>
                                <div style="font-size: 2rem; font-weight: bold;">{prob:.1%}</div>
                                <div style="font-size: 0.8rem; color: #666;">probability</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Display Lineage predictions
                st.markdown("### Lineage Bias Prediction")
                
                if lineage_predictions:
                    # Create columns for displaying predictions
                    cols = st.columns(len(lineage_predictions))
                    
                    # Color map for different lineages
                    color_map = {
                        "Myeloid": "#4257B2",  # Blue
                        "Lymphoid": "#00CC96", # Green
                        "Erythroid": "#FF4B4B" # Red
                    }
                    
                    for i, result in enumerate(lineage_predictions):
                        with cols[i]:
                            # Get color based on lineage type
                            lineage_type = result['class']
                            color = color_map.get(lineage_type, "#888888")
                            prob = float(result['probability'])
                            
                            st.markdown(f"""
                            <div style="padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; text-align: center;">
                                <h3 style="color: {color}; font-size: 1.5rem; margin-bottom: 0.5rem;">{lineage_type}</h3>
                                <div style="font-size: 2rem; font-weight: bold;">{prob:.1%}</div>
                                <div style="font-size: 0.8rem; color: #666;">probability</div>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        # Show default dashboard when no experiment is selected
        st.title("HSC Dashboard")
        st.markdown("### Welcome to the HSC Dashboard")
        st.markdown("Create a new experiment or select an existing one from the sidebar to get started.")
        st.info("👈 Click on '➕ New Experiment' in the sidebar to create your first experiment.")

# Footer
st.markdown("---")
st.caption("© 2025 Osiris Bio")

# Main function to run the app
if __name__ == "__main__":
    pass
