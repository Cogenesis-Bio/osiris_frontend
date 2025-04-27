import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import datetime
import random
from sklearn.preprocessing import MinMaxScaler
import streamlit_shadcn_ui as ui

# Set page configuration
st.set_page_config(
    page_title="Osiris-1 HSC Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for font styling and interactive elements
st.markdown("""
    <style>
    /* Import Space Grotesk, Inter, and Jomolhari Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=Jomolhari&display=swap');
    
    /* Apply Space Grotesk font to absolutely everything */
    html, body, div, span, applet, object, iframe,
    h1, h2, h3, h4, h5, h6, p, blockquote, pre,
    a, abbr, acronym, address, big, cite, code,
    del, dfn, em, img, ins, kbd, q, s, samp,
    small, strike, strong, sub, sup, tt, var,
    b, u, i, center,
    dl, dt, dd, ol, ul, li,
    fieldset, form, label, legend,
    table, caption, tbody, tfoot, thead, tr, th, td,
    article, aside, canvas, details, embed, 
    figure, figcaption, footer, header, hgroup, 
    menu, nav, output, ruby, section, summary,
    time, mark, audio, video, [class*="st"], [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* Streamlit specific elements */
    .stApp, .stButton, .stTextInput, .stSelectbox, .stTab, .stRadio, .stCheckbox, 
    .stDataFrame, .stTable, .stHeader, .stMarkdown, .stMetric {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* Override for main header to use Jomolhari */
    .main-header {
        font-family: 'Jomolhari', serif !important;
        font-weight: 400;
        font-size: 2.5rem;
        color: #4257B2;
        text-align: center;
    }
    
    .sub-header {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500;
        font-size: 1.5rem;
        color: #5C5C5C;
    }
    
    /* Force Space Grotesk on all buttons */
    button, .stButton > button {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500;
    }
    
    /* File uploader styling and hover effects */
    .stFileUploader > button {
        transition: all 0.3s ease !important;
        border: 2px dashed #4257B2 !important;
        border-radius: 8px !important;
        background-color: rgba(66, 87, 178, 0.05) !important;
        color: #4257B2 !important;
        font-weight: 500 !important;
        position: relative !important;
        padding: 1.2rem 1rem !important;
        cursor: pointer !important;
    }
    
    .stFileUploader > button:before {
        content: '📄 ' !important;
        margin-right: 5px !important;
    }
    
    .stFileUploader > button:after {
        content: ' (or drag and drop)' !important;
        font-size: 0.8em !important;
        opacity: 0.7 !important;
    }
    
    .stFileUploader > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(66, 87, 178, 0.2) !important;
        background-color: rgba(66, 87, 178, 0.1) !important;
        border-style: solid !important;
    }
    
    /* Add a subtle pulse animation to the upload button */
    @keyframes subtle-pulse {
        0% { box-shadow: 0 0 0 0 rgba(66, 87, 178, 0.4); }
        70% { box-shadow: 0 0 0 6px rgba(66, 87, 178, 0); }
        100% { box-shadow: 0 0 0 0 rgba(66, 87, 178, 0); }
    }
    
    .stFileUploader > button:hover {
        animation: subtle-pulse 2s infinite;
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






# Helper functions for future AI model integration

def generate_protocol_response(user_input, experiment_data):
    """
    Generate a response for the protocol recommendations chat based on user input and experiment data.
    This is a simple keyword-based response system that will be replaced with a more sophisticated AI model in the future.
    
    Parameters:
    -----------
    user_input : str
        The user's message
    experiment_data : DataFrame
        The current experiment data
        
    Returns:
    --------
    str
        A response message
    """
    # Get the latest data point
    latest_data = experiment_data.iloc[-1]
    
    # Convert user input to lowercase for easier matching
    user_input_lower = user_input.lower()
    
    # Check for specific keywords and provide relevant responses
    if any(word in user_input_lower for word in ["hello", "hi", "hey", "greetings"]):
        return "Hello! How can I help with your HSC expansion protocol today?"
    
    elif any(word in user_input_lower for word in ["self-renewal", "self renewal", "renewal"]):
        sr_score = latest_data["Self_Renewal_Score"]
        if sr_score > 75:
            return f"Your current self-renewal score is {sr_score:.1f}, which is excellent! I recommend maintaining your current cytokine concentrations, particularly SCF and TPO levels."
        elif sr_score > 50:
            return f"Your current self-renewal score is {sr_score:.1f}, which is good but could be improved. Consider increasing SCF by 10% and ensuring TPO is at 50ng/mL to enhance self-renewal capacity."
        else:
            return f"Your current self-renewal score is {sr_score:.1f}, which is below optimal levels. I recommend increasing both SCF and TPO by 20%, and reducing differentiation-inducing cytokines like GM-CSF if present in your media."
    
    elif any(word in user_input_lower for word in ["multipotency", "multipotent", "potency"]):
        mp_score = latest_data["Multipotency_Score"]
        if mp_score > 75:
            return f"Your current multipotency score is {mp_score:.1f}, which indicates excellent maintenance of HSC potential! Your current cytokine balance is working well."
        elif mp_score > 50:
            return f"Your current multipotency score is {mp_score:.1f}, which is reasonable but could be improved. Consider adding IL-6 at low concentration (10ng/mL) to your media to enhance multipotency."
        else:
            return f"Your current multipotency score is {mp_score:.1f}, which suggests your HSCs may be losing multipotency. I recommend a complete media change with fresh cytokines, particularly ensuring a balance of SCF, TPO, and FLT3L to support multipotency."
    
    elif any(word in user_input_lower for word in ["myeloid", "granulocyte", "macrophage"]):
        myeloid_pct = latest_data["Myeloid_Percentage"]
        if myeloid_pct > 70:
            return f"Your culture shows a strong myeloid bias ({myeloid_pct:.1f}%). To reduce this bias, consider decreasing G-CSF and GM-CSF if present, and slightly increasing FLT3L to promote lymphoid potential."
        elif myeloid_pct < 30:
            return f"Your culture shows low myeloid output ({myeloid_pct:.1f}%). To increase myeloid differentiation, consider adding GM-CSF at 10ng/mL or increasing IL-3 concentration."
        else:
            return f"Your myeloid percentage ({myeloid_pct:.1f}%) is within a balanced range. Current cytokine conditions appear appropriate for balanced lineage output."
    
    elif any(word in user_input_lower for word in ["lymphoid", "lymphocyte", "b cell", "t cell"]):
        lymphoid_pct = latest_data["Lymphoid_Percentage"]
        if lymphoid_pct > 70:
            return f"Your culture shows a strong lymphoid bias ({lymphoid_pct:.1f}%). To balance lineage output, consider adding IL-3 at low concentration to promote myeloid differentiation."
        elif lymphoid_pct < 30:
            return f"Your culture shows low lymphoid output ({lymphoid_pct:.1f}%). To increase lymphoid differentiation, consider adding FLT3L and IL-7 to your media."
        else:
            return f"Your lymphoid percentage ({lymphoid_pct:.1f}%) is within a balanced range. Current conditions appear appropriate for balanced lineage output."
    
    elif any(word in user_input_lower for word in ["erythroid", "red", "erythrocyte", "rbc"]):
        erythroid_pct = latest_data["Erythroid_Percentage"]
        if erythroid_pct > 50:
            return f"Your culture shows a strong erythroid bias ({erythroid_pct:.1f}%). To reduce erythroid differentiation, consider decreasing EPO concentration by 50% in your next media change."
        elif erythroid_pct < 10:
            return f"Your culture shows very low erythroid output ({erythroid_pct:.1f}%). If erythroid potential is desired, consider adding EPO at 3U/mL to your media."
        else:
            return f"Your erythroid percentage ({erythroid_pct:.1f}%) is within an acceptable range. Current conditions appear appropriate."
    
    elif any(word in user_input_lower for word in ["cytokine", "growth factor", "medium", "media"]):
        return "For optimal HSC expansion, I recommend a base medium of StemSpan SFEM II with the following cytokines: SCF (100ng/mL), TPO (50ng/mL), FLT3L (100ng/mL), and IL-6 (20ng/mL). Adjust based on your specific goals: increase SCF and TPO for self-renewal, or add lineage-specific cytokines for directed differentiation."
    
    elif any(word in user_input_lower for word in ["protocol", "recommend", "suggestion", "advice"]):
        sr_score = latest_data["Self_Renewal_Score"]
        mp_score = latest_data["Multipotency_Score"]
        myeloid_pct = latest_data["Myeloid_Percentage"]
        lymphoid_pct = latest_data["Lymphoid_Percentage"]
        erythroid_pct = latest_data["Erythroid_Percentage"]
        
        # Determine the main issue to address
        if sr_score < 50:
            return "Based on your current data, I recommend focusing on improving self-renewal capacity. Increase SCF to 150ng/mL and TPO to 100ng/mL. Ensure your cells are at optimal density (5-10 × 10^4 cells/mL) and perform a 50% media change every 2 days rather than complete media changes."
        elif mp_score < 50:
            return "Your data indicates declining multipotency. I recommend a complete media change with fresh cytokines: SCF (100ng/mL), TPO (50ng/mL), FLT3L (100ng/mL), and IL-6 (10ng/mL). Also, reduce culture density if currently above 2 × 10^5 cells/mL to minimize paracrine differentiation signals."
        elif max(myeloid_pct, lymphoid_pct, erythroid_pct) > 70:
            # Determine which lineage is dominant
            dominant = "myeloid" if myeloid_pct > 70 else "lymphoid" if lymphoid_pct > 70 else "erythroid"
            return f"Your culture shows a strong {dominant} bias. To rebalance, I recommend adjusting cytokines: {'reduce G-CSF and GM-CSF' if dominant == 'myeloid' else 'reduce IL-7 and FLT3L' if dominant == 'lymphoid' else 'reduce EPO by 50%'}. A partial media change with rebalanced cytokines should help restore multipotency."
        else:
            return "Your current protocol appears to be working well with balanced lineage output. Continue with your current cytokine regimen and schedule. For optimal results, ensure you're performing media changes every 2-3 days and maintaining cell density between 5-20 × 10^4 cells/mL."
    
    else:
        return "I'm not sure I understand your question. Could you ask about specific aspects of your HSC protocol? I can provide recommendations on cytokines, media composition, self-renewal enhancement, or lineage balancing based on your current data."





def calculate_self_renewal_score(proliferation_rate, cd34_expression):
    """
    Calculate self-renewal score based on proliferation rate and CD34 expression.
    This is a placeholder for future AI model integration.
    """
    # Normalize inputs to 0-1 scale
    norm_prolif = min(max(proliferation_rate / 100, 0), 1)
    norm_cd34 = min(max(cd34_expression / 100, 0), 1)
    
    # Simple weighted average - to be replaced with ML model
    score = (norm_prolif * 0.6) + (norm_cd34 * 0.4)
    return score * 100  # Convert to 0-100 scale

def calculate_multipotency_score(lineage_markers):
    """
    Calculate multipotency score based on lineage marker diversity.
    This is a placeholder for future AI model integration.
    """
    # Count number of lineages with significant expression
    significant_lineages = sum(1 for marker in lineage_markers.values() if marker > 20)
    
    # Calculate evenness of distribution (Shannon diversity index-inspired)
    total = sum(lineage_markers.values())
    if total == 0:
        return 0
    
    proportions = [marker/total for marker in lineage_markers.values() if marker > 0]
    evenness = -sum(p * np.log(p) for p in proportions) / np.log(len(proportions)) if proportions else 0
    
    # Combine metrics - to be replaced with ML model
    score = (significant_lineages / len(lineage_markers) * 0.5) + (evenness * 0.5)
    return score * 100  # Convert to 0-100 scale

def get_protocol_recommendation(self_renewal_score, multipotency_score, lineage_bias, constraints=None):
    """
    Generate protocol recommendations based on scores and lineage bias.
    This is a placeholder for future AI model integration.
    """
    recommendations = []
    
    # Self-renewal recommendations
    if self_renewal_score < 40:
        recommendations.append({
            "type": "self-renewal",
            "action": "Add 20 ng/mL SCF",
            "rationale": "Boosts self-renewal capacity",
            "evidence": "Based on 12 studies",
            "confidence": 0.85
        })
        recommendations.append({
            "type": "self-renewal",
            "action": "Increase TPO to 50 ng/mL",
            "rationale": "Enhances HSC maintenance",
            "evidence": "Based on 8 studies",
            "confidence": 0.78
        })
    
    # Multipotency recommendations
    if multipotency_score < 50:
        recommendations.append({
            "type": "multipotency",
            "action": "Add 10 ng/mL FLT3L",
            "rationale": "Promotes lymphoid differentiation potential",
            "evidence": "Based on 15 studies",
            "confidence": 0.82
        })
    
    # Lineage bias adjustments
    if "myeloid" in lineage_bias and lineage_bias["myeloid"] > 70:
        recommendations.append({
            "type": "lineage",
            "action": "Reduce SCF by 20%",
            "rationale": "Balances lymphoid potential",
            "evidence": "Based on 7 studies",
            "confidence": 0.75
        })
    elif "lymphoid" in lineage_bias and lineage_bias["lymphoid"] > 70:
        recommendations.append({
            "type": "lineage",
            "action": "Add 5 ng/mL IL-3",
            "rationale": "Enhances myeloid differentiation",
            "evidence": "Based on 10 studies",
            "confidence": 0.8
        })
    
    # Apply constraints if provided
    if constraints:
        if "no_small_molecules" in constraints and constraints["no_small_molecules"]:
            recommendations = [r for r in recommendations if "small molecule" not in r["action"].lower()]
        if "budget" in constraints:
            # Simple budget filter - would be more sophisticated in real implementation
            if constraints["budget"] < 200:
                recommendations = recommendations[:2]  # Limit to top 2 recommendations
    
    return recommendations

def generate_sample_data(days=30):
    """Generate sample data for demonstration purposes"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
    
    # Create base trends with some randomness
    self_renewal_trend = np.linspace(40, 75, days) + np.random.normal(0, 5, days)
    multipotency_trend = np.linspace(30, 65, days) + np.random.normal(0, 7, days)
    
    # Ensure values are within reasonable ranges
    self_renewal_trend = np.clip(self_renewal_trend, 0, 100)
    multipotency_trend = np.clip(multipotency_trend, 0, 100)
    
    # Create lineage markers with some correlation to the scores
    myeloid_bias = 70 - (multipotency_trend - 30) * 0.5 + np.random.normal(0, 5, days)
    lymphoid_bias = 30 + (multipotency_trend - 30) * 0.5 + np.random.normal(0, 5, days)
    erythroid_bias = np.random.normal(15, 3, days)
    
    # Ensure percentages sum to 100
    total = myeloid_bias + lymphoid_bias + erythroid_bias
    myeloid_bias = (myeloid_bias / total) * 100
    lymphoid_bias = (lymphoid_bias / total) * 100
    erythroid_bias = (erythroid_bias / total) * 100
    
    # Protocol changes - simulate 3 protocol changes
    protocol_changes = []
    change_days = [7, 15, 22]
    changes = [
        {"day": 7, "change": "Added 10 ng/mL FLT3L", "target": "Increase lymphoid potential"},
        {"day": 15, "change": "Reduced SCF by 15%", "target": "Balance lineage output"},
        {"day": 22, "change": "Added 5 ng/mL IL-6", "target": "Boost proliferation"}
    ]
    
    # Generate gene expression data for HSC-related genes
    # Define a list of important HSC-related genes
    hsc_genes = [
        "CD34", "KIT", "GATA2", "RUNX1", "TAL1", "BMI1", "HOXA9", 
        "MEIS1", "MECOM", "MYB", "GATA1", "PU.1", "CEBPA", "FLT3", "MPL"
    ]
    
    # Generate expression values for each gene
    # Some genes will have higher expression based on lineage bias
    gene_expression = {}
    
    # Stem cell maintenance genes (correlated with self-renewal score)
    gene_expression["CD34"] = (60 + self_renewal_trend * 0.3 + np.random.normal(0, 15, days)).astype(int)
    gene_expression["KIT"] = (70 + self_renewal_trend * 0.25 + np.random.normal(0, 12, days)).astype(int)
    gene_expression["BMI1"] = (50 + self_renewal_trend * 0.35 + np.random.normal(0, 10, days)).astype(int)
    gene_expression["HOXA9"] = (45 + self_renewal_trend * 0.4 + np.random.normal(0, 8, days)).astype(int)
    
    # Myeloid lineage genes (correlated with myeloid bias)
    gene_expression["PU.1"] = (30 + myeloid_bias * 0.7 + np.random.normal(0, 15, days)).astype(int)
    gene_expression["CEBPA"] = (25 + myeloid_bias * 0.8 + np.random.normal(0, 10, days)).astype(int)
    
    # Lymphoid lineage genes (correlated with lymphoid bias)
    gene_expression["FLT3"] = (20 + lymphoid_bias * 0.9 + np.random.normal(0, 12, days)).astype(int)
    gene_expression["IL7R"] = (15 + lymphoid_bias * 0.8 + np.random.normal(0, 10, days)).astype(int)
    
    # Erythroid lineage genes (correlated with erythroid bias)
    gene_expression["GATA1"] = (25 + erythroid_bias * 1.2 + np.random.normal(0, 15, days)).astype(int)
    gene_expression["KLF1"] = (20 + erythroid_bias * 1.0 + np.random.normal(0, 12, days)).astype(int)
    
    # Multipotency-related genes (correlated with multipotency score)
    gene_expression["GATA2"] = (40 + multipotency_trend * 0.5 + np.random.normal(0, 10, days)).astype(int)
    gene_expression["RUNX1"] = (35 + multipotency_trend * 0.45 + np.random.normal(0, 8, days)).astype(int)
    gene_expression["TAL1"] = (30 + multipotency_trend * 0.4 + np.random.normal(0, 12, days)).astype(int)
    gene_expression["MYB"] = (45 + multipotency_trend * 0.3 + np.random.normal(0, 15, days)).astype(int)
    gene_expression["MECOM"] = (25 + multipotency_trend * 0.35 + np.random.normal(0, 10, days)).astype(int)
    
    # Ensure all gene expression values are positive
    for gene in gene_expression:
        gene_expression[gene] = np.clip(gene_expression[gene], 0, None)
    
    # Create the dataframe
    data = {
        "Date": dates,
        "Self_Renewal_Score": self_renewal_trend,
        "Multipotency_Score": multipotency_trend,
        "Myeloid_Percentage": myeloid_bias,
        "Lymphoid_Percentage": lymphoid_bias,
        "Erythroid_Percentage": erythroid_bias,
        "CD34_Expression": 60 + np.random.normal(0, 10, days),
        "Proliferation_Rate": 50 + np.random.normal(0, 15, days)
    }
    
    # Add gene expression data to the dataframe
    for gene, values in gene_expression.items():
        data[f"Gene_{gene}"] = values
    
    df = pd.DataFrame(data)
    
    # Add protocol change markers
    df["Protocol_Change"] = False
    for change in changes:
        df.loc[change["day"], "Protocol_Change"] = True
        df.loc[change["day"], "Change_Description"] = change["change"]
        df.loc[change["day"], "Change_Target"] = change["target"]
    
    return df, changes

# Initialize experiments list if it doesn't exist
if 'experiments' not in st.session_state:
    st.session_state.experiments = []
    
# Track current experiment
if 'current_experiment' not in st.session_state:
    st.session_state.current_experiment = None

# Initialize current page if it doesn't exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Welcome"  # Default to Welcome page



# Sidebar navigation
with st.sidebar:
    st.markdown("<h1 style='font-size: 2rem; padding-top: 0.1rem; padding-bottom: 0.1rem;'>Osiris-1</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation buttons styling
    st.markdown("""
    <style>
        .stButton>button {
            text-align: left !important;
            justify-content: flex-start !important;
            display: flex !important;
            align-items: center !important;
            padding-left: 10px !important;
            width: 100% !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main navigation buttons
    
    # Create New Experiment button
    if st.button("➕ New Experiment", use_container_width=True, key="create_experiment_btn"):
        st.session_state.show_create_dialog = True
    
    # Show experiment creation dialog
    if st.session_state.get('show_create_dialog', False):
        with st.form("new_experiment_form"):
            experiment_name = st.text_input("Experiment Name", key="new_experiment_name")
            submitted = st.form_submit_button("Create Experiment")
            
            if submitted and experiment_name:
                # Create a new experiment with timestamp and storage for experiment-specific data
                experiment_id = len(st.session_state.experiments) + 1
                new_experiment = {
                    "id": experiment_id,
                    "name": experiment_name,
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "data": None
                }
                
                # Add to experiments list
                st.session_state.experiments.append(new_experiment)
                
                # Generate experiment-specific data keys
                exp_data_key = f"data_{experiment_id}"
                exp_protocol_key = f"protocol_changes_{experiment_id}"
                
                # Initialize experiment-specific data
                if exp_data_key not in st.session_state:
                    st.session_state[exp_data_key], st.session_state[exp_protocol_key] = generate_sample_data()
                
                # Set as current experiment
                st.session_state.current_experiment = experiment_id
                st.session_state.current_page = "Dashboard"  # Take user to Dashboard page
                
                # Hide the dialog
                st.session_state.show_create_dialog = False
                st.rerun()
    
    # Display experiment tabs if any exist
    if st.session_state.experiments:
        st.markdown("### My Experiments")
        
        for experiment in st.session_state.experiments:
            # Create a button for each experiment with the same styling as navigation buttons
            if st.button(f"📊 {experiment['name']}", use_container_width=True, key=f"experiment_{experiment['id']}"):
                st.session_state.current_experiment = experiment["id"]
                st.session_state.current_page = "Dashboard"  # Keep on Dashboard page
                st.rerun()
    
    # Settings section header
    st.markdown("### Settings")
    
    # Account button under Settings section
    if st.button("👤 Account", use_container_width=True, key="nav_account"):
        st.session_state.current_page = "Account"
        st.session_state.current_experiment = None
        st.rerun()



# Display the appropriate page based on the current_page value
if st.session_state.current_page == "Welcome":
    # Add custom CSS for centering content
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
    .welcome-header {
        margin-bottom: 2rem;
    }
    .welcome-heading {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600;
        font-size: 2.5rem;
        color: #4257B2;
        text-align: center;
    }
    .welcome-content {
        margin-bottom: 3rem;
    }
    .welcome-button {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Welcome page with centered content
    st.markdown("""
    <div class="welcome-container" style="padding-bottom: 0.1rem;">
        <div class="welcome-header">
            <h1 class='welcome-heading'>Good morning, Christian.</h1>
            <h2>Welcome to Osiris!</h2>
            <h3>Click "New Experiment" to get started.</h3>
        </div>
    </div>
    <script>
        const targetElement = document.getElementById('welcome-heading');
        const text = "Good morning, Christian";
        let i = 0;

        function typeText() {
            if (i < text.length) {
                targetElement.textContent += text.charAt(i);
                i++;
                setTimeout(typeText, 80);
            }
        }

        typeText();
    </script>
    """, unsafe_allow_html=True)
    

elif st.session_state.current_page == "Dashboard":
    # Check if an experiment is selected
    if st.session_state.current_experiment is not None:
        # Find the selected experiment
        selected_exp = next((exp for exp in st.session_state.experiments if exp["id"] == st.session_state.current_experiment), None)
        
        if selected_exp:
            # Display experiment-specific dashboard
            st.title(f"Experiment: {selected_exp['name']}")
            
            # Create experiment-specific keys for session state
            exp_data_key = f"data_{selected_exp['id']}"
            exp_protocol_key = f"protocol_changes_{selected_exp['id']}"
            
            # Load or generate data for this experiment
            if exp_data_key not in st.session_state:
                st.session_state[exp_data_key], st.session_state[exp_protocol_key] = generate_sample_data()
            
            # Use the experiment-specific data
            df = st.session_state[exp_data_key]
            protocol_changes = st.session_state[exp_protocol_key]
            
            # Create tabs using shadcn tabs component
            selected_tab = ui.tabs(
                options=["Overview", "Protocol Recommendations"],
                default_value="Overview",
                key=f"main_tabs_{selected_exp['id']}"
            )
            
            # Dashboard content (existing code)
            if selected_tab == "Overview":
                # File uploader section
                st.markdown("### Upload scRNA Sequencing Data")
                with st.container():
                    # Custom message before the uploader with improved styling
                    st.markdown("""
                    <div style="margin-bottom: 15px; font-size: 1rem; color: #444;">
                        <strong>Upload your single-cell RNA sequencing data</strong><br>
                        <span style="font-size: 0.9rem; color: #666;">Supported formats: CSV, TSV, or MTX files</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a more visually appealing file uploader
                    uploaded_file = st.file_uploader(
                        "", 
                        type=["csv", "tsv", "mtx"],
                        help="Supported formats: CSV (comma-separated), TSV (tab-separated), or MTX (Matrix Market format)",
                        key=f"uploader_{selected_exp['id']}"
                    )
                    
                    if uploaded_file is not None:
                        # Display a more visually appealing success message with custom styling
                        st.markdown(f"""
                        <div class="upload-success">
                            <h4 style="margin-top: 0; color: #4257B2;">✅ File uploaded successfully</h4>
                            <p style="margin-bottom: 0;"><strong>File:</strong> {uploaded_file.name}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add a container for actions using shadcn UI buttons
                        action_col1, action_col2 = st.columns([1, 1])
                        
                        with action_col1:
                            process_button = ui.button(
                                "Process Data", 
                                variant="default",
                                key=f"process_data_button_{selected_exp['id']}"
                            )
                            
                            if process_button:
                                try:
                                    # Store the file in experiment-specific session state
                                    st.session_state[f"uploaded_file_{selected_exp['id']}"] = uploaded_file
                                    
                                    # Show a loading spinner while processing
                                    with st.spinner("Processing your data..."):
                                        # In a real app, you would process the file here
                                        # For now, we'll just simulate processing
                                        import time
                                        time.sleep(1)  # Simulate processing time
                                        
                                        st.session_state[f"file_processed_{selected_exp['id']}"] = True
                                        
                                        # Use shadcn UI alert for success message
                                        ui.alert(
                                            "File processed successfully!",
                                            description="Your data is ready for analysis.",
                                            variant="success",
                                            key=f"process_success_alert_{selected_exp['id']}"
                                        )
                                except Exception as e:
                                    ui.alert(
                                        "Error processing file",
                                        description=str(e),
                                        variant="destructive",
                                        key=f"process_error_alert_{selected_exp['id']}"
                                    )
                        
                        with action_col2:
                            clear_button = ui.button(
                                "Clear", 
                                variant="outline",
                                key=f"clear_button_{selected_exp['id']}"
                            )
                            
                            if clear_button:
                                # Clear the uploaded file from session state
                                if f"uploaded_file_{selected_exp['id']}" in st.session_state:
                                    del st.session_state[f"uploaded_file_{selected_exp['id']}"]
                                if f"file_processed_{selected_exp['id']}" in st.session_state:
                                    del st.session_state[f"file_processed_{selected_exp['id']}"]
                                st.rerun()
                        
                        # Show file details in an expandable section using shadcn UI
                        file_details_open = ui.collapsible(
                            title="File Details",
                            content="",
                            key=f"file_details_collapsible_{selected_exp['id']}"
                        )
                        
                        if file_details_open:
                            # Calculate file size in appropriate units
                            file_size = uploaded_file.size
                            size_str = f"{file_size} bytes"
                            if file_size > 1024*1024:
                                size_str = f"{file_size/(1024*1024):.2f} MB"
                            elif file_size > 1024:
                                size_str = f"{file_size/1024:.2f} KB"
                            
                            # Display file details in a more structured way
                            details_cols = st.columns([1, 2])
                            with details_cols[0]:
                                st.markdown("**File Properties**")
                            with details_cols[1]:
                                st.markdown(f"**Filename:** {uploaded_file.name}")
                                st.markdown(f"**File size:** {size_str}")
                                st.markdown(f"**File type:** {uploaded_file.type if hasattr(uploaded_file, 'type') else 'Unknown'}")
                                
                                # Add file timestamp if available
                                if hasattr(uploaded_file, 'timestamp'):
                                    st.markdown(f"**Uploaded:** {uploaded_file.timestamp}")
                
                st.markdown("---")
                
                # Get latest data for metrics
                latest_data = df.iloc[-1]
                
                # Top metrics row - Key metrics section
                st.markdown("### Key Metrics")
                col1, col2 = st.columns(2)
                
                # Self-renewal score
                with col1:
                    current_self_renewal = latest_data["Self_Renewal_Score"]
                    previous_self_renewal = df.iloc[-2]["Self_Renewal_Score"]
                    delta = current_self_renewal - previous_self_renewal
                    
                    st.metric(
                        label="Self-Renewal Score",
                        value=f"{current_self_renewal:.1f}",
                        delta=f"{delta:.1f}",
                        delta_color="normal"
                    )
                    
                    st.markdown("""
                    <div style="font-size:0.8rem; color: #666;">
                    Based on proliferation rate and CD34 expression
                    </div>
                    """, unsafe_allow_html=True)
                
                # Multipotency score
                with col2:
                    current_multipotency = latest_data["Multipotency_Score"]
                    previous_multipotency = df.iloc[-2]["Multipotency_Score"]
                    delta = current_multipotency - previous_multipotency
                    
                    st.metric(
                        label="Multipotency Score",
                        value=f"{current_multipotency:.1f}",
                        delta=f"{delta:.1f}",
                        delta_color="normal"
                    )
                    
                    st.markdown("""
                    <div style="font-size:0.8rem; color: #666;">
                    Based on lineage marker diversity in differentiation assays
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Score Trends with selection
                st.markdown("### Score Trends")
                
                # Add metric selection
                metric_options = ["Self-Renewal Score", "Multipotency Score", "Both"]
                selected_metric = st.selectbox("Select metric to display:", metric_options)
                
                # Create the appropriate dataframe based on selection
                if selected_metric == "Self-Renewal Score":
                    score_data = df[['Date', 'Self_Renewal_Score']].copy()
                    score_data = score_data.rename(columns={'Self_Renewal_Score': 'Self-Renewal'})
                    score_data_melted = pd.melt(
                        score_data, 
                        id_vars=['Date'], 
                        value_vars=['Self-Renewal'],
                        var_name='Score Type', 
                        value_name='Score'
                    )
                elif selected_metric == "Multipotency Score":
                    score_data = df[['Date', 'Multipotency_Score']].copy()
                    score_data = score_data.rename(columns={'Multipotency_Score': 'Multipotency'})
                    score_data_melted = pd.melt(
                        score_data, 
                        id_vars=['Date'], 
                        value_vars=['Multipotency'],
                        var_name='Score Type', 
                        value_name='Score'
                    )
                else:  # Both
                    score_data = df[['Date', 'Self_Renewal_Score', 'Multipotency_Score']].copy()
                    score_data = score_data.rename(columns={
                        'Self_Renewal_Score': 'Self-Renewal',
                        'Multipotency_Score': 'Multipotency'
                    })
                    score_data_melted = pd.melt(
                        score_data, 
                        id_vars=['Date'], 
                        value_vars=['Self-Renewal', 'Multipotency'],
                        var_name='Score Type', 
                        value_name='Score'
                    )
                
                # Get protocol change data for annotations
                protocol_change_dates = df[df['Protocol_Change'] == True]['Date'].tolist()
                protocol_change_desc = df[df['Protocol_Change'] == True]['Change_Description'].tolist()
                
                # Create the line chart
                fig = px.line(
                    score_data_melted, 
                    x='Date', 
                    y='Score', 
                    color='Score Type',
                    color_discrete_map={
                        'Self-Renewal': '#4257B2',
                        'Multipotency': '#00CC96'
                    },
                    title="Score Trends Over Time"
                )
                
                # Add protocol change annotations
                for i, date in enumerate(protocol_change_dates):
                    fig.add_shape(
                        type="line",
                        x0=date,
                        y0=0,
                        x1=date,
                        y1=100,
                        line=dict(
                            color="gray",
                            width=1,
                            dash="dash",
                        )
                    )
                    fig.add_annotation(
                        x=date,
                        y=95,
                        text=f"Protocol Change: {protocol_change_desc[i]}",
                        showarrow=False,
                        yshift=10
                    )
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Day of Experiment",
                    yaxis_title="Score",
                    yaxis_range=[0, 100],
                    legend_title="Score Type",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Gene Expression Bar Graph
                st.markdown("### Highest Expressed Genes")
                
                # Get the latest data point for gene expression
                latest_data = df.iloc[-1]
                
                # Extract gene expression columns and their values
                gene_columns = [col for col in df.columns if col.startswith('Gene_')]
                gene_data = {
                    'Gene': [col.replace('Gene_', '') for col in gene_columns],
                    'Expression': [latest_data[col] for col in gene_columns]
                }
                
                # Create a DataFrame for the gene expression data
                gene_df = pd.DataFrame(gene_data)
                
                # Sort by expression level (highest first) and take top 10
                gene_df = gene_df.sort_values('Expression', ascending=False).head(10)
                
                # Create a color map based on gene function
                gene_categories = {
                    'CD34': 'Stem Cell', 'KIT': 'Stem Cell', 'BMI1': 'Stem Cell', 'HOXA9': 'Stem Cell',
                    'PU.1': 'Myeloid', 'CEBPA': 'Myeloid',
                    'FLT3': 'Lymphoid', 'IL7R': 'Lymphoid',
                    'GATA1': 'Erythroid', 'KLF1': 'Erythroid',
                    'GATA2': 'Multipotency', 'RUNX1': 'Multipotency', 'TAL1': 'Multipotency', 
                    'MYB': 'Multipotency', 'MECOM': 'Multipotency', 'MPL': 'Multipotency', 'MEIS1': 'Multipotency'
                }
                
                # Add category column to the DataFrame
                gene_df['Category'] = gene_df['Gene'].map(lambda x: gene_categories.get(x, 'Other'))
                
                # Create a color map for the categories
                color_map = {
                    'Stem Cell': '#4257B2',  # Blue
                    'Myeloid': '#FF9500',   # Orange
                    'Lymphoid': '#00CC96',  # Green
                    'Erythroid': '#FF4B4B',  # Red
                    'Multipotency': '#9D50BB',  # Purple
                    'Other': '#999999'      # Gray
                }
                
                # Create the bar chart
                fig = px.bar(
                    gene_df,
                    x='Gene',
                    y='Expression',
                    color='Category',
                    color_discrete_map=color_map,
                    text_auto=True,
                    title="Top 10 Expressed Genes"
                )
                
                # Update layout
                fig.update_layout(
                    height=400,
                    xaxis_title="Gene",
                    yaxis_title="Expression Level",
                    legend_title="Gene Category",
                    hovermode="closest"
                )
                
                # Add tooltips with gene function
                gene_functions = {
                    'CD34': 'Cell surface glycoprotein and stem cell marker',
                    'KIT': 'Receptor tyrosine kinase essential for HSC maintenance',
                    'BMI1': 'Polycomb complex protein involved in self-renewal',
                    'HOXA9': 'Homeobox protein crucial for HSC expansion',
                    'PU.1': 'Transcription factor essential for myeloid development',
                    'CEBPA': 'Transcription factor involved in myeloid differentiation',
                    'FLT3': 'Receptor tyrosine kinase important for lymphoid development',
                    'IL7R': 'Interleukin-7 receptor involved in lymphoid commitment',
                    'GATA1': 'Transcription factor essential for erythroid development',
                    'KLF1': 'Krüppel-like factor 1, regulates erythroid maturation',
                    'GATA2': 'Transcription factor required for HSC maintenance and multipotency',
                    'RUNX1': 'Transcription factor essential for definitive hematopoiesis',
                    'TAL1': 'Basic helix-loop-helix transcription factor for blood development',
                    'MYB': 'Transcription factor involved in progenitor proliferation',
                    'MECOM': 'Transcription regulator of HSC quiescence and self-renewal',
                    'MPL': 'Thrombopoietin receptor important for HSC maintenance',
                    'MEIS1': 'Homeobox protein that regulates HSC self-renewal'
                }
                
                # Update hover template to include gene function
                fig.update_traces(
                    hovertemplate='<b>%{x}</b><br>Expression: %{y}<br>Function: ' + 
                    gene_df['Gene'].map(lambda x: gene_functions.get(x, 'Unknown')).to_list()[0]
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a note about gene expression
                with st.expander("About Gene Expression Data"):
                    st.markdown("""
                    **Gene Expression Categories:**
                    - **Stem Cell**: Genes associated with HSC identity and self-renewal
                    - **Myeloid**: Genes involved in myeloid lineage commitment and differentiation
                    - **Lymphoid**: Genes involved in lymphoid lineage commitment and differentiation
                    - **Erythroid**: Genes involved in erythroid lineage commitment and differentiation
                    - **Multipotency**: Genes associated with maintaining multilineage potential
                    
                    Expression values represent normalized counts from single-cell RNA sequencing data.
                    Higher values indicate stronger expression of the gene in the HSC population.
                    """)
                
                st.markdown("---")
                
                # Lineage Distribution and Map
                col1, col2 = st.columns(2)
                
                # Lineage Distribution Bar Graph
                with col1:
                    st.subheader("Lineage Marker Distribution")
                    
                    # Create a bar chart for lineage distribution
                    lineage_data = {
                        'Lineage': ['Myeloid', 'Lymphoid', 'Erythroid'],
                        'Percentage': [
                            latest_data['Myeloid_Percentage'],
                            latest_data['Lymphoid_Percentage'],
                            latest_data['Erythroid_Percentage']
                        ]
                    }
                    lineage_df = pd.DataFrame(lineage_data)
                    
                    fig = px.bar(
                        lineage_df, 
                        x='Lineage',
                        y='Percentage',
                        color='Lineage',
                        color_discrete_map={
                            'Myeloid': '#4257B2',
                            'Lymphoid': '#00CC96',
                            'Erythroid': '#FF4B4B'
                        },
                        text_auto='.1f'
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
                    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Lineage Bias Map (Ternary Plot)
                with col2:
                    st.subheader("Lineage Bias Map")
                    
                    # Create a ternary plot for lineage bias
                    fig = go.Figure()
                    
                    # Add the trajectory as a scatter plot
                    fig.add_trace(go.Scatterternary(
                        a=df['Myeloid_Percentage'],
                        b=df['Lymphoid_Percentage'],
                        c=df['Erythroid_Percentage'],
                        mode='lines+markers',
                        line=dict(color='#4257B2', width=2),
                        marker=dict(
                            symbol='circle',
                            size=8,
                            color=np.arange(len(df)),
                            colorscale='Viridis',
                            line=dict(width=1, color='#FFFFFF')
                        ),
                        text=df['Date'].dt.strftime('%Y-%m-%d'),
                        hovertemplate='Date: %{text}<br>Myeloid: %{a:.1f}%<br>Lymphoid: %{b:.1f}%<br>Erythroid: %{c:.1f}%<extra></extra>'
                    ))
                    
                    # Add the current position as a larger marker
                    fig.add_trace(go.Scatterternary(
                        a=[latest_data['Myeloid_Percentage']],
                        b=[latest_data['Lymphoid_Percentage']],
                        c=[latest_data['Erythroid_Percentage']],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=15,
                            color='red',
                            line=dict(width=2, color='#FFFFFF')
                        ),
                        name='Current',
                        hovertemplate='Current Position<br>Myeloid: %{a:.1f}%<br>Lymphoid: %{b:.1f}%<br>Erythroid: %{c:.1f}%<extra></extra>'
                    ))
                    
                    # Add regions with labels
                    fig.add_trace(go.Scatterternary(
                        a=[80, 20, 30],
                        b=[10, 70, 20],
                        c=[10, 10, 50],
                        mode='text',
                        text=['Myeloid<br>Dominant', 'Lymphoid<br>Dominant', 'Erythroid<br>Dominant'],
                        textposition="middle center",
                        textfont=dict(size=10, color='black'),
                        showlegend=False
                    ))
                    
                    # Update the layout
                    fig.update_layout(
                        ternary=dict(
                            aaxis=dict(title='Myeloid %', min=0, linewidth=2, gridwidth=1),
                            baxis=dict(title='Lymphoid %', min=0, linewidth=2, gridwidth=1),
                            caxis=dict(title='Erythroid %', min=0, linewidth=2, gridwidth=1)
                        ),
                        height=300,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Lineage bias assessment
                st.subheader("Lineage Bias Assessment")
                myeloid_pct = latest_data['Myeloid_Percentage']
                if myeloid_pct > 70:
                    st.warning(f"**Current bias:** {myeloid_pct:.1f}% myeloid (high)")
                    st.info("**Suggestion:** Reduce SCF by 20% to balance lymphoid potential")
                elif latest_data['Lymphoid_Percentage'] > 70:
                    st.warning(f"**Current bias:** {latest_data['Lymphoid_Percentage']:.1f}% lymphoid (high)")
                    st.info("**Suggestion:** Add IL-3 to enhance myeloid differentiation")
                elif latest_data['Erythroid_Percentage'] > 50:
                    st.warning(f"**Current bias:** {latest_data['Erythroid_Percentage']:.1f}% erythroid (high)")
                    st.info("**Suggestion:** Reduce EPO to balance lineage output")
                else:
                    st.success("**Current bias:** Relatively balanced lineage output")
                    st.info("**Suggestion:** Maintain current cytokine ratios")

            elif selected_tab == "Protocol Recommendations":
                st.markdown("## Protocol Recommendations")
                
                # Initialize chat history in session state if it doesn't exist
                if f"chat_history_{st.session_state.current_experiment}" not in st.session_state:
                    st.session_state[f"chat_history_{st.session_state.current_experiment}"] = [
                        {"role": "assistant", "content": "Hello! I'm your protocol assistant. I can help you optimize your HSC expansion protocol based on your current data. What would you like to know?"}
                    ]
                
                # Display chat messages with custom styling
                st.markdown("""<style>
                .assistant-msg {
                    background-color: #f0f7ff;
                    border-radius: 10px;
                    padding: 10px 15px;
                    margin-bottom: 10px;
                    border-left: 4px solid #4257B2;
                }
                .user-msg {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    padding: 10px 15px;
                    margin-bottom: 10px;
                    margin-left: 50px;
                    border-left: 4px solid #00CC96;
                }
                </style>""", unsafe_allow_html=True)
                
                # Display chat history
                for message in st.session_state[f"chat_history_{st.session_state.current_experiment}"]:
                    if message["role"] == "assistant":
                        st.markdown(f"<div class='assistant-msg'><strong>Protocol Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='user-msg'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
                
                # Chat input with form to prevent rerun issues
                with st.form(key=f"chat_form_{st.session_state.current_experiment}", clear_on_submit=True):
                    user_input = st.text_input("Ask about protocol recommendations...", key=f"chat_input_{st.session_state.current_experiment}")
                    submit_button = st.form_submit_button("Send")
                
                if submit_button and user_input:
                    # Add user message to chat history
                    st.session_state[f"chat_history_{st.session_state.current_experiment}"].append({"role": "user", "content": user_input})
                    
                    # Generate assistant response based on experiment data
                    response = generate_protocol_response(user_input, st.session_state[f"data_{st.session_state.current_experiment}"])
                    
                    # Add assistant response to chat history
                    st.session_state[f"chat_history_{st.session_state.current_experiment}"].append({"role": "assistant", "content": response})
                    
                    # Force a rerun to update the chat display
                    st.rerun()
    else:
        # Show default dashboard or welcome message when no experiment is selected
        st.title("HSC Dashboard")
        st.markdown("### Welcome to the HSC Dashboard")
        st.markdown("Create a new experiment or select an existing one from the sidebar to get started.")
        
        # Show a call to action
        st.info("👈 Click on '➕ New Experiment' in the sidebar to create your first experiment.")

elif st.session_state.current_page == "Account":
    st.title("Account Settings")
    
    # User profile section
    st.header("User Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://img.icons8.com/color/96/000000/user-male-circle.png", width=150)
        st.button("Change Profile Picture")
    
    with col2:
        st.text_input("Name", value="Dr. Jane Smith")
        st.text_input("Institution", value="University Research Hospital")
        st.text_input("Email", value="jane.smith@research.edu")
        st.text_input("Position", value="Principal Investigator")
    
    # Notification settings
    st.header("Notification Settings")
    
    st.checkbox("Email notifications for new recommendations", value=True)
    st.checkbox("Weekly summary reports", value=True)
    st.checkbox("Collaboration requests", value=True)
    st.checkbox("Platform updates and news", value=False)
    
    # API access
    st.header("API Access")
    
    st.text_input("API Key", value="••••••••••••••••••••••••••", type="password")
    col1, col2 = st.columns(2)
    with col1:
        st.button("Generate New API Key")
    with col2:
        st.button("Copy API Key")
    
    st.markdown("""
    Use your API key to access the Osiris HSC Platform programmatically. 
    See the [API documentation](https://docs.osiris-hsc.com/api) for more details.
    """)
    
    # Subscription information
    st.header("Subscription")
    
    st.info("**Current Plan:** Research Pro (Annual)")
    st.progress(0.7)
    st.caption("321 days remaining in your subscription")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("Upgrade Plan")
    with col2:
        st.button("Billing History")

# Footer - only show on non-welcome pages
if st.session_state.current_page != "Welcome":
    st.markdown("---")
    st.caption(" 2025 Osiris Bio")
    
    # Add a floating help button
    with st.expander("Help & Documentation"):
        st.markdown("""
    ## How to use this dashboard
    
    1. Use the sidebar to set your target parameters for HSC expansion
    2. Review current scores and lineage distribution in the Overview tab
    3. Check protocol recommendations based on your targets
    4. Apply recommended changes to your protocol
    
    For more information, contact nphuchane@g.ucla.edu
    """)

# Main function to run the app
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.info("Last updated: April 2025")
