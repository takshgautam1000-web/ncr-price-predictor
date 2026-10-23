import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NCR Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND FEATURE COLUMNS
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("models/house_price_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, feature_columns


try:
    model, feature_columns = load_model()
except Exception as e:
    st.error(f"Could not load the saved model: {e}")
    st.stop()


# =========================================================
# CREATE DROPDOWN OPTIONS FROM SAVED FEATURES
# NO RAW CSV REQUIRED
# =========================================================

def get_encoded_options(prefix, additional_options=None):
    """
    Extract categories from one-hot encoded feature names.
    Preserve original whitespace because it may be significant
    when matching the model's training columns.
    """
    options = {
        column[len(prefix):]
        for column in feature_columns
        if column.startswith(prefix)
    }

    if additional_options:
        options.update(additional_options)

    return sorted(
        options,
        key=lambda value: value.strip().casefold()
    )


rera_options = get_encoded_options(
    "RERA Approval_",
    ["Approved by RERA"]
)

status_options = get_encoded_options("Status_")

flat_type_options = get_encoded_options(
    "Flat Type_",
    ["Apartment"]
)

locality_options = get_encoded_options("Locality_")

company_options = get_encoded_options("Company Name_")

society_options = get_encoded_options("Socity_")


# =========================================================
# TITLE AND DESCRIPTION
# =========================================================

st.title("🏠 NCR Price Predictor")

st.write(
    "Predict property prices in the NCR region using machine learning."
)

st.write(
    "Enter the property details below to estimate its price."
)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.subheader("📋 Property Details")

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric("Test R² Score", "85.16%")

with metric_col2:
    st.metric("Model", "Random Forest Regression")


# =========================================================
# INPUT FIELDS
# =========================================================

col1, col2 = st.columns(2)


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
        filter_mode="contains",
        format_func=lambda value: value.strip()
    )

    status = st.selectbox(
        "Status",
        status_options,
        index=None,
        placeholder="Type to search...",
        filter_mode="contains",
        format_func=lambda value: value.strip()
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
        filter_mode="contains",
        format_func=lambda value: value.strip()
    )

    locality = st.selectbox(
        "Locality",
        locality_options,
        index=None,
        placeholder="Type to search...",
        filter_mode="contains",
        format_func=lambda value: value.strip()
    )

    company_name = st.selectbox(
        "Company Name",
        company_options,
        index=None,
        placeholder="Type to search...",
        filter_mode="contains",
        format_func=lambda value: value.strip()
    )

    society = st.selectbox(
        "Society",
        society_options,
        index=None,
        placeholder="Type to search...",
        filter_mode="contains",
        format_func=lambda value: value.strip()
    )


# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Price", type="primary"):

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    required_values = {
        "RERA Approval": rera,
        "Status": status,
        "Flat Type": flat_type,
        "Locality": locality,
        "Company Name": company_name,
        "Society": society
    }

    missing_fields = [
        field
        for field, value in required_values.items()
        if value is None
    ]

    if missing_fields:
        st.warning(
            "Please select: " + ", ".join(missing_fields)
        )
        st.stop()

    if not feature_columns:
        st.error("The saved feature-column list is empty.")
        st.stop()

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    # Area was log1p-transformed during training
    log_area = np.log1p(area)

    # Area_per_BHK uses the original area in square feet
    area_per_bhk = area / bhk if bhk > 0 else 0

    # Is_Plot is derived from the user's selection
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
    # ONE-HOT ENCODING
    # -----------------------------------------------------

    input_data = pd.get_dummies(input_data)

    # -----------------------------------------------------
    # MATCH THE EXACT TRAINING FEATURE COLUMNS
    # -----------------------------------------------------

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_data = input_data.fillna(0)

    # -----------------------------------------------------
    # PREDICT LOG PRICE
    # -----------------------------------------------------

    try:
        log_prediction = model.predict(input_data)[0]
        predicted_price = np.expm1(log_prediction)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if not np.isfinite(predicted_price) or predicted_price < 0:
        st.error("The model returned an invalid price.")
        st.stop()

    # -----------------------------------------------------
    # DISPLAY PREDICTION
    # -----------------------------------------------------

    st.success("Prediction completed successfully!")

    st.metric(
        label="🏠 Estimated Property Price",
        value=f"₹{predicted_price:,.0f}"
    )

    st.write(
        f"💰 Approximately **₹{predicted_price / 1e7:.2f} Crore**"
    )

    # -----------------------------------------------------
    # PROPERTY SUMMARY
    # -----------------------------------------------------

    st.subheader("📋 Property Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(f"**Area:** {area:,.0f} sq ft")
        st.write(f"**BHK:** {bhk}")
        st.write(f"**Property Category:** {property_category}")
        st.write(f"**Locality:** {locality}")
        st.write(f"**Status:** {status}")

    with summary_col2:
        st.write(f"**RERA Approval:** {rera}")
        st.write(f"**Flat Type:** {flat_type}")
        st.write(f"**Company:** {company_name}")
        st.write(f"**Society:** {society}")
