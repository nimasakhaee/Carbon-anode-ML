import streamlit as st
import pandas as pd
import lightgbm as lgb
import random
import joblib

# --- Page setup ---
st.set_page_config(page_title="Removal Efficiency (%) Predictor - LightGBM", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f2f5f7; }
        div[data-testid="stSidebar"] { background-color: #e0ecf1; }
        .stButton > button {
            background-color: #6fa8dc;
            color: white;
            font-weight: bold;
            padding: 10px 24px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_excel("DATA.xlsx")


@st.cache_resource
def load_model():
    return joblib.load("lightgbm_model.pkl")


# --- Load Data & Model ---
data = load_data()
model = load_model()

categorical_features = ['Anode', 'Cathode', 'Electrolyte', 'Pollutant', 'Reactor type']
X = data.drop('Removal Efficiency(%)', axis=1)
y = data['Removal Efficiency(%)']
for col in categorical_features:
    X[col] = X[col].astype('category')

# --- Metadata ---
feature_names = list(X.columns)

anode_values = ['Graphite', 'Graphite felt', 'MWCNT', 'Black carbon (BC)', 'GAC', 'Graphite plate', 'Carbon fiber', 'Activated carbon fiber']
cathode_values = ['Graphite', 'Pt plate', 'Stainless steel', 'Graphite plate', 'Graphite felt', 'Carbon fiber', 'iridium-coated titanium']
electrolyte_values = ['NaCl', 'Na2SO4', 'NaNO3', 'NaClO4', 'HPO4', 'HCO3', 'NaHCO3', 'Groundwater']
pollutant_values = ['cytarabine', 'Atrazine', 'Paracetamol', 'Sulfamethoxazole', 'Nitrobenzene', 'Cyclohexanecarboxylic acid', 'Bisphenol S', 'Ethinylestradiol', 'Ibuprofen', 'Ciprofloxacin', 'Acetaminophen', 'Diclofenac', 'Methylene Blue', 'Tetracycline', 'Sulfamethazine', 'Sulfanilamide', 'Phenol', 'Benzoquinone', 'Catechol', 'Deoxynivalenol', 'Diuron', 'Alizarin red S']
reactor_type_values = ['Undivided reactor', 'Divided reactor']

dropdown_values = {
    'Anode': anode_values,
    'Cathode': cathode_values,
    'Electrolyte': electrolyte_values,
    'Pollutant': pollutant_values,
    'Reactor type': reactor_type_values
}

# --- Title ---
st.title("🧪 Removal Efficiency (%) Predictor - LightGBM")

# --- Random Value Button ---
if st.button("🎲 Fill with Random Values"):
    random_row = data.sample(1).iloc[0]
    for feature in feature_names:
        st.session_state[feature] = random_row[feature]
    st.rerun()

# --- Form and Inputs ---
input_data = {}
with st.form("prediction_form"):
    st.markdown("### Enter Input Features")
    col1, col2 = st.columns(2)

    with col1:
        for feature in categorical_features:
            values = dropdown_values[feature]
            default = st.session_state.get(feature, values[0])
            input_data[feature] = st.selectbox(feature, values, index=values.index(default) if default in values else 0, key=feature)

    with col2:
        for feature in feature_names:
            if feature not in categorical_features:
                default = st.session_state.get(feature, 0.0)
                input_data[feature] = st.number_input(feature, value=float(default), format="%.4f", key=feature)

    submitted = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submitted:
    try:
        vals = list(input_data.values())
        if any((isinstance(val, (int, float)) and val <= 0) for val in vals if not isinstance(val, str)):
            st.error("❌ Inputs must be positive numbers.")
        else:
            df_input = pd.DataFrame([input_data])
            for col in categorical_features:
                df_input[col] = pd.Categorical(df_input[col], categories=X[col].cat.categories)
            pred = model.predict(df_input)[0]
            st.success(f"🌟 Predicted Removal Efficiency (%): {pred:.2f}")
    except Exception as e:
        st.error(f"⚠️ Invalid input! {str(e)}")


