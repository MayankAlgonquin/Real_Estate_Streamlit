import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

import logging
import os

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Title
st.title("Real Estate Price Predictor")
st.write("""
This app predicts the price of a property based on property features using a Random Forest model.
""")

# Load model
try:
    rf_pickle = open("models/lrmodel.pkl", "rb")
    rf_model = pickle.load(rf_pickle)
    rf_pickle.close()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Form
with st.form("user_inputs"):
    st.subheader("Property Details")

    year_sold = st.number_input("Year Sold", min_value=1900, max_value=2100, step=1)
    property_tax = st.number_input("Property Tax", min_value=0)
    insurance = st.number_input("Insurance", min_value=0)
    beds = st.number_input("Beds", min_value=0)
    baths = st.number_input("Baths", min_value=0)
    sqft = st.number_input("Square Feet", min_value=0)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2100, step=1)
    lot_size = st.number_input("Lot Size", min_value=0)

    basement = st.selectbox("Basement", [0, 1])
    popular = st.selectbox("Popular", [0, 1])
    recession = st.selectbox("Recession", [0, 1])
    property_type_Condo = st.selectbox("Property Type Condo", [0, 1])

    property_age = st.number_input("Property Age", min_value=0)

    submitted = st.form_submit_button("Predict Price")

# Prediction
if submitted:
    prediction_input = [[
        year_sold,
        property_tax,
        insurance,
        beds,
        baths,
        sqft,
        year_built,
        lot_size,
        basement,
        popular,
        recession,
        property_age,
        property_type_Condo
    ]]

    prediction = rf_model.predict(prediction_input)

    st.subheader("Prediction Result:")
    st.write(f"Estimated Property Price: ${int(prediction[0]):,}")

# Feature Importance
st.write("""
We used a Random Forest model. Feature importance is shown below:
""")

feature_names = [
    'year_sold', 'property_tax', 'insurance', 'beds', 'baths',
    'sqft', 'year_built', 'lot_size', 'basement',
    'popular', 'recession', 'property_age', 'property_type_Condo'
]

importances = rf_model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax)
st.pyplot(fig)