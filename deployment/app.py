import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -----------------------------
# Load trained pipeline from Hugging Face
# -----------------------------
REPO_ID = "Ansh91/tourism"                      # your HF model repo
FILENAME_PRIMARY = "best_tourism_model_v1.joblib"
FILENAME_FALLBACK = "best_model.pkl"

try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_PRIMARY, repo_type="model")
except Exception:
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_FALLBACK, repo_type="model")

model = joblib.load(model_path)

# -----------------------------
# Streamlit UI (Tourism Purchase Prediction)
# -----------------------------
st.title("🧘 Wellness Tourism Purchase Prediction")
st.write("""
Predict if a customer is likely to **purchase the Wellness Tourism Package** before outreach.
Enter the customer details below and click **Predict**.
""")

# --- Inputs based on the project schema ---
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    CityTier = st.selectbox("CityTier", [1, 2, 3], index=0)
    NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=0, max_value=20, value=2, step=1)
    PreferredPropertyStar = st.selectbox("PreferredPropertyStar", [1, 2, 3, 4, 5], index=3)
    NumberOfTrips = st.number_input("NumberOfTrips (annual avg.)", min_value=0, max_value=60, value=2, step=1)
    Passport = 1 if st.checkbox("Has Passport?") else 0

with col2:
    OwnCar = 1 if st.checkbox("Owns Car?") else 0
    NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting (≤5 yrs)", min_value=0, max_value=10, value=0, step=1)
    MonthlyIncome = st.number_input("MonthlyIncome", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
    PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", [1, 2, 3, 4, 5], index=3)
    NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=50, value=2, step=1)
    DurationOfPitch = st.number_input("DurationOfPitch (mins)", min_value=0, max_value=300, value=20, step=1)

with col3:
    TypeofContact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Other"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
    Designation = st.text_input("Designation", value="Executive")
    ProductPitched = st.text_input("ProductPitched", value="Wellness")

# Assemble input (column names must match training)
input_df = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "ProductPitched": ProductPitched,
}])

st.write("**Your input**")
st.dataframe(input_df)

# Fixed threshold (0.45) to mirror training/eval logs
CLASSIFICATION_THRESHOLD = 0.45

if st.button("Predict"):
    try:
        prob = model.predict_proba(input_df)[:, 1][0]
        pred = int(prob >= CLASSIFICATION_THRESHOLD)
        label = "Will Purchase (1)" if pred == 1 else "Will Not Purchase (0)"

        st.subheader("Prediction")
        st.metric("Purchase Probability", f"{prob:.3f}")
        st.success(f"Prediction: **{label}**  (threshold={CLASSIFICATION_THRESHOLD})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
