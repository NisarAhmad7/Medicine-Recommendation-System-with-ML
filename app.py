import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="SmartMed AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

TRAINING_FILE = BASE_DIR / "Training.csv"
MODEL_FILE = BASE_DIR / "svc.pkl"
DESCRIPTION_FILE = BASE_DIR / "description.csv"
PRECAUTIONS_FILE = BASE_DIR / "precautions_df.csv"
MEDICATIONS_FILE = BASE_DIR / "medications.csv"
DIETS_FILE = BASE_DIR / "diets.csv"
WORKOUT_FILE = BASE_DIR / "workout_df.csv"


# ============================================================
# Load data/model
# ============================================================
@st.cache_data
def load_training_data():
    return pd.read_csv(TRAINING_FILE)


@st.cache_data
def load_supporting_data():
    description = pd.read_csv(DESCRIPTION_FILE)
    precautions = pd.read_csv(PRECAUTIONS_FILE)
    medications = pd.read_csv(MEDICATIONS_FILE)
    diets = pd.read_csv(DIETS_FILE)
    workout = pd.read_csv(WORKOUT_FILE)
    return description, precautions, medications, diets, workout


@st.cache_resource
def load_model():
    # Use the already-trained model if it exists.
    if MODEL_FILE.exists():
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)

    # Otherwise train it once and save it.
    dataset = load_training_data()
    X = dataset.drop("prognosis", axis=1)
    y = dataset["prognosis"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = SVC(kernel="linear")
    model.fit(X, y_encoded)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model


@st.cache_data
def load_mappings():
    dataset = load_training_data()

    symptoms = list(dataset.drop("prognosis", axis=1).columns)

    # The notebook uses LabelEncoder for prognosis. Recreate the same
    # alphabetical class ordering used during training.
    diseases = sorted(dataset["prognosis"].unique())
    disease_mapping = {i: disease for i, disease in enumerate(diseases)}

    symptoms_dict = {symptom: i for i, symptom in enumerate(symptoms)}

    return symptoms, symptoms_dict, disease_mapping


# ============================================================
# Helper functions
# ============================================================
def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_recommendations(disease, description, precautions,
                        medications, diets, workout):
    # Description
    desc_rows = description[description["Disease"].astype(str).str.strip() == disease]
    desc = ""
    if not desc_rows.empty and "Description" in desc_rows.columns:
        desc = clean_value(desc_rows.iloc[0]["Description"])

    # Precautions
    pre_rows = precautions[
        precautions["Disease"].astype(str).str.strip() == disease
    ]

    precaution_list = []
    if not pre_rows.empty:
        for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
            if col in pre_rows.columns:
                value = clean_value(pre_rows.iloc[0][col])
                if value:
                    precaution_list.append(value)

    # Medications
    med_rows = medications[
        medications["Disease"].astype(str).str.strip() == disease
    ]
    medication_list = []
    if not med_rows.empty and "Medication" in med_rows.columns:
        for value in med_rows["Medication"].tolist():
            value = clean_value(value)
            if value:
                medication_list.append(value)

    # Diet
    diet_rows = diets[diets["Disease"].astype(str).str.strip() == disease]
    diet_list = []
    if not diet_rows.empty and "Diet" in diet_rows.columns:
        for value in diet_rows["Diet"].tolist():
            value = clean_value(value)
            if value:
                diet_list.append(value)

    # Workout
    workout_disease_column = "disease" if "disease" in workout.columns else "Disease"
    workout_rows = workout[
        workout[workout_disease_column].astype(str).str.strip() == disease
    ]

    workout_list = []
    workout_column = "workout" if "workout" in workout.columns else None
    if not workout_rows.empty and workout_column:
        for value in workout_rows[workout_column].tolist():
            value = clean_value(value)
            if value:
                workout_list.append(value)

    return desc, precaution_list, medication_list, diet_list, workout_list


def predict_disease(model, selected_symptoms, symptoms_dict, disease_mapping):
    input_vector = np.zeros(len(symptoms_dict), dtype=int)

    for symptom in selected_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    encoded_prediction = int(model.predict([input_vector])[0])
    return disease_mapping[encoded_prediction]


# ============================================================
# Load everything
# ============================================================
try:
    dataset = load_training_data()
    model = load_model()
    symptoms, symptoms_dict, disease_mapping = load_mappings()
    description, precautions, medications, diets, workout = load_supporting_data()
except FileNotFoundError as e:
    st.error(f"Required project file was not found: {e}")
    st.info("Make sure all CSV files and svc.pkl are in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not initialize the application: {e}")
    st.stop()


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }
    .result-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        margin: 15px 0;
    }
    .disease-name {
        font-size: 30px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_=True,
)


# ============================================================
# Header
# ============================================================
st.markdown(
    '<div class="main-title">🩺 Medicine Recommendation System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Select your symptoms and get an ML-based disease prediction with supporting information.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("About the System")
    st.write(
        "This application uses a Support Vector Classifier (SVC) "
        "trained on the provided symptom dataset."
    )

    st.divider()

    st.subheader("How to use")
    st.write("1. Select one or more symptoms.")
    st.write("2. Click Predict Disease.")
    st.write("3. Review the generated information.")


# ============================================================
# Symptom selection
# ============================================================
st.subheader("🔎 Select Your Symptoms")

selected_symptoms = st.multiselect(
    "Search and select the symptoms you are experiencing:",
    options=symptoms,
    placeholder="Start typing a symptom...",
)

predict_button = st.button(
    "🔬 Predict Disease",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Prediction
# ============================================================
if predict_button:
    if not selected_symptoms:
        st.warning("Please select at least one symptom before predicting.")
        st.stop()

    with st.spinner("Analyzing symptoms..."):
        predicted_disease = predict_disease(
            model,
            selected_symptoms,
            symptoms_dict,
            disease_mapping,
        )

        desc, pre, med, die, wrkout = get_recommendations(
            predicted_disease,
            description,
            precautions,
            medications,
            diets,
            workout,
        )

    st.success("Prediction completed.")

    st.markdown(
        f"""
        <div class="result-card">
            <div>Predicted Disease</div>
            <div class="disease-name">🩺 {predicted_disease}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Selected symptoms
    with st.expander("Selected Symptoms", expanded=False):
        for symptom in selected_symptoms:
            st.write(f"• {symptom}")

    # Description
    st.subheader("📋 Description")
    if desc:
        st.write(desc)
    else:
        st.info("No description was found for this disease.")

    # Two-column layout for recommendations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛡️ Precautions")
        if pre:
            for index, item in enumerate(pre, start=1):
                st.write(f"**{index}.** {item}")
        else:
            st.info("No precaution information was found.")

        st.subheader("💊 Medications")
        if med:
            for item in med:
                st.write(f"• {item}")
        else:
            st.info("No medication information was found.")

    with col2:
        st.subheader("🥗 Diet")
        if die:
            for item in die:
                st.write(f"• {item}")
        else:
            st.info("No diet information was found.")

        st.subheader("🏃 Workout")
        if wrkout:
            for item in wrkout:
                st.write(f"• {item}")
        else:
            st.info("No workout information was found.")


# ============================================================
# Medical disclaimer
# ============================================================
st.divider()


st.caption("Medicine Recommendation System • Machine Learning Project")
