import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================


# Page configuration
st.set_page_config(
    page_title="NCR Price Predictor",
    page_icon="🏠",
    layout="wide"
)



# =========================================================
# LOAD MODEL + FEATURE COLUMNS
# =========================================================

model = joblib.load("models/house_price_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")



# =========================================================
# LOAD RAW DATA
# =========================================================

csv_files = list(Path("data").rglob("*.csv"))

if len(csv_files) == 0:
    st.error("No CSV file found inside the data folder.")
    st.stop()

raw_data = pd.read_csv(csv_files[0])


# =========================================================
# GET UNIQUE OPTIONS FROM RAW DATA
# =========================================================

def get_options(column):
    values = (
        raw_data[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[values != ""]

    return sorted(values.unique().tolist())


rera_options = get_options("RERA Approval")
status_options = get_options("Status")
flat_type_options = get_options("Flat Type")
locality_options = get_options("Locality")
company_options = get_options("Company Name")
society_options = get_options("Socity")


# =========================================================
# TITLE
# =========================================================

st.title("🏠 NCR Price Predictor")

st.write(
    "Predict property prices in the NCR region using machine learning."
)

st.write(
    "Enter the property details below to estimate its price."
)



# =========================================================
# INPUTS
# =========================================================
st.subheader("📋 Property Details")
col1, col2 = st.columns(2)
with col1:
    st.metric("R² Score", "85.16%")

with col2:
    st.metric("Model", "Random Forest Regression")

# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    area = st.number_input(
        "Area (sq ft)",
        min_value=100.0,
        max_value=100000.0,
        value=1500.0,
        step=100.0
    )

    bhk = st.number_input(
        "BHK Count",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    property_category = st.selectbox(
        "Property Category",
        ["Non-Plot", "Plot"]
    )

    rera = st.selectbox(
        "RERA Approval",
        rera_options,
        index=None,
        placeholder="Type to search...",
        filter_mode="contains"
    )

    status = st.selectbox(
        "Status",
        status_options,
        index=None,
        placeholder="Type to search...",
        filter_mode="contains"
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    flat_type = st.selectbox(
        "Flat Type",
        flat_type_options,
        index=None,
        placeholder="Type to search...",
        filter_mode="contains"
    )

    locality = st.selectbox(
        "Locality",
        locality_options,
        index=None,
        placeholder="Type to search...",
        filter_mode="contains"
    )

    company_name = st.selectbox(
        "Company Name",
        company_options,
        index=None,
        placeholder="Type to search...",
        filter_mode="contains"
    )

    society = st.selectbox(
        "Society",
        society_options,
        index=None,
        placeholder="Type to search...",
        filter_mode="contains"
    )


# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Price", type="primary"):

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if rera is None:
        st.warning("Please select RERA Approval.")
        st.stop()

    if status is None:
        st.warning("Please select Status.")
        st.stop()

    if flat_type is None:
        st.warning("Please select Flat Type.")
        st.stop()

    if locality is None:
        st.warning("Please select Locality.")
        st.stop()

    if company_name is None:
        st.warning("Please select Company Name.")
        st.stop()

    if society is None:
        st.warning("Please select Society.")
        st.stop()


    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    # Area was log1p transformed during training
    log_area = np.log1p(area)

    # Same feature used in final model
    if bhk == 0:
        area_per_bhk = 0
    else:
        area_per_bhk = area / bhk

    # Create Is_Plot
    is_plot = 1 if property_category == "Plot" else 0


    # -----------------------------------------------------
    # CREATE INPUT ROW
    # -----------------------------------------------------

    input_data = pd.DataFrame({
        "Area": [log_area],
        "BHK_Count": [bhk],
        "Is_Plot": [is_plot],
        "RERA Approval": [rera],
        "Status": [status],
        "Flat Type": [flat_type],
        "Locality": [locality],
        "Company Name": [company_name],
        "Socity": [society],
        "Area_per_BHK": [area_per_bhk]
    })


    # -----------------------------------------------------
    # ONE-HOT ENCODE
    # -----------------------------------------------------

    input_data = pd.get_dummies(input_data)


    # -----------------------------------------------------
    # MATCH EXACT TRAINING FEATURES
    # -----------------------------------------------------

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # -----------------------------------------------------
    # HANDLE MISSING VALUES
    # -----------------------------------------------------

    input_data = input_data.fillna(0)


    # -----------------------------------------------------
    # PREDICT LOG PRICE
    # -----------------------------------------------------

    log_prediction = model.predict(input_data)[0]


    # -----------------------------------------------------
    # CONVERT BACK TO ACTUAL PRICE
    # -----------------------------------------------------

    predicted_price = np.expm1(log_prediction)


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    st.success("Prediction completed successfully!")

    st.metric(
    label="🏠 Estimated Property Price",
    value=f"₹{predicted_price:,.0f}"
    )

    st.write(
    f"💰 Approximately **₹{predicted_price / 1e7:.2f} Crore**" 
    )

    st.subheader("📋 Property Summary")

    summary_col1, summary_col2 = st.columns(2)
 
    with summary_col1:
      st.write(f"**Area:** {area:,.0f} sq ft")
      st.write(f"**BHK:** {bhk}")
      st.write(f"**Locality:** {locality}")
      st.write(f"**Status:** {status}")

    with summary_col2:
      st.write(f"**RERA Approval:** {rera}")
      st.write(f"**Flat Type:** {flat_type}")
      st.write(f"**Company:** {company_name}")
      st.write(f"**Society:** {society}")