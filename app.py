import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# Load Model
with open("iris_dataset.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🌸 Iris Flower Classification")
st.markdown(
    """
    Predict the species of an Iris flower using Machine Learning.
    """
)

# Sidebar
st.header("Flower Measurements")

Sepal_Length = st.slider(
    "Sepal Length (cm)",
    4.0, 8.0, 5.8
)

Sepal_Width = st.slider(
    "Sepal Width (cm)",
    2.0, 5.0, 3.0
)

Petal_Length = st.slider(
    "Petal Length (cm)",
    1.0, 7.0, 4.3
)

Petal_Width = st.slider(
    "Petal Width (cm)",
    0.1, 3.0, 1.3
)

# Create DataFrame
input_df = pd.DataFrame({
    "SepalLengthCm": [Sepal_Length],
    "SepalWidthCm": [Sepal_Width],
    "PetalLengthCm": [Petal_Length],
    "PetalWidthCm": [Petal_Width]
})

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Input Features")
    st.dataframe(input_df)

with col2:
    st.subheader("ℹ️ About Dataset")
    st.write("""
    The Iris dataset contains measurements of iris flowers and
    classifies them into:
    - Setosa
    - Versicolor
    - Virginica
    """)

# Prediction
if st.button("🔍 Predict Species"):

    prediction = model.predict(input_df)

    species = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    predicted_species = species[prediction[0]]

    st.subheader("Prediction Result")

    if predicted_species == "Setosa":
        st.success(f"🌼 Predicted Species: {predicted_species}")

    elif predicted_species == "Versicolor":
        st.info(f"🌷 Predicted Species: {predicted_species}")

    else:
        st.warning(f"🌺 Predicted Species: {predicted_species}")

    # Probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)

        st.subheader("Prediction Confidence")

        prob_df = pd.DataFrame({
            "Species": species,
            "Probability (%)": np.round(probability[0] * 100, 2)
        })

        st.dataframe(prob_df)

        st.bar_chart(
            prob_df.set_index("Species")
        )

# Footer
st.markdown("---")
st.markdown(
    """
    **Project Developed Using**
    - Python
    - Scikit-Learn
    - Streamlit
    - Iris Dataset
    """
)

if predicted_species == "Setosa":
    st.image("images/setosa.jpg", width=300)

elif predicted_species == "Versicolor":
    st.image("images/versicolor.jpg", width=300)

else:
    st.image("images/virginica.jpg", width=300)

st.sidebar.markdown("---")
st.sidebar.metric(
    "Model Accuracy",
    "96%"
)

st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}

h1 {
    color: #2E8B57;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

with st.expander("📖 Project Details"):
    st.write("""
    This Machine Learning project predicts Iris flower species
    based on sepal and petal measurements.

    The model was trained using Scikit-Learn and deployed
    using Streamlit.
    """)