import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from io import BytesIO
import plotly.express as px
import os

# ------------------------------- #
# Page Configuration
# ------------------------------- #
st.set_page_config(
    page_title="🧬 SMILES Activity & ADMET Predictor",
    layout="wide",
    page_icon="🧪"
)

st.title("🧬 SMILES-Based Activity & ADMET Property Predictor")
st.markdown("Upload your **SMILES dataset** and get predictions for **biological activity and ADMET** properties using machine learning models.")
st.markdown("---")

# ------------------------------- #
# Sidebar
# ------------------------------- #
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload an Excel file with a `smiles` column", type=["xlsx"])
    st.markdown("---")
    st.info("You can download results as Excel after prediction.")

# ------------------------------- #
# Load Models
# ------------------------------- #
@st.cache_resource
def load_models():
    try:
        return {
            "Activity": joblib.load("hybrid_model.pkl"),
            "(Absorption) Caco-2": joblib.load("absorp_model.pkl"),
            "(Distribution) BBB": joblib.load("distri_model.pkl"),
            "(Metabolism) CYP3A4": joblib.load("meta_model.pkl"),
            "(Excretion) Obach": joblib.load("exc_model.pkl"),
            "(Toxicity) NR-AR": joblib.load("toxicity_model.pkl"),
        }
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.stop()

models = load_models()
activity_model = models["Activity"]

descriptor_columns = [
    "MolWt", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds", "MolLogP",
    "FpDensityMorgan1", "NumAromaticRings", "FractionCSP3", "NumAliphaticRings",
    "FpDensityMorgan2", "HeavyAtomMolWt"
]

# ------------------------------- #
# Utility Functions
# ------------------------------- #
def calculate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return [
                Descriptors.MolWt(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                Descriptors.MolLogP(mol),
                Descriptors.FpDensityMorgan1(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.FpDensityMorgan2(mol),
                Descriptors.HeavyAtomMolWt(mol)
            ]
    except:
        return [0] * len(descriptor_columns)
    return [0] * len(descriptor_columns)

def get_activity_status(row):
    if pd.isna(row["Probability"]):
        return "Unknown"
    if row["Prediction"] == 0:
        return "Inactive"
    elif row["Probability"] <= 0.75:
        return f"Low confidence - {row['Concentration']}"
    else:
        return f"High confidence - {row['Concentration']}"

# ------------------------------- #
# Main Logic
# ------------------------------- #
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if "smiles" not in df.columns:
            st.error("❌ Missing required column: `smiles`")
            st.stop()

        st.success(f"✅ File loaded. Total molecules: {len(df)}")

        with st.spinner("🔬 Calculating descriptors and running models..."):
            df[descriptor_columns] = df["smiles"].apply(calculate_descriptors).apply(pd.Series)

            # Activity prediction
            df["Prediction"] = activity_model.predict(df[descriptor_columns])
            df["Probability"] = np.max(activity_model.predict_proba(df[descriptor_columns]), axis=1)
            df["Concentration"] = df["Prediction"].map({0: "non-hits", 1: "10 μM", 2: "1 and 10 μM"})
            df["Activity_Status"] = df.apply(get_activity_status, axis=1)

            # ADMET predictions
            for name, model in models.items():
                if name != "Activity":
                    df[name] = model.predict(df[descriptor_columns])

            # Clean and re-map values
            df["(Metabolism) CYP3A4"] = df["(Metabolism) CYP3A4"].map({0: "Non-Metabolic (0)", 1: "Metabolic (1)"})
            df["(Toxicity) NR-AR"] = df["(Toxicity) NR-AR"].map({0: "Non-Toxic (0)", 1: "Toxic (1)"})

            df.drop(columns=descriptor_columns, inplace=True)

        st.success("🎯 Predictions completed successfully!")

        # ----------------------- #
        # Summary + Charts
        # ----------------------- #
        st.subheader("📊 Prediction Summary")

        summary = pd.DataFrame({
            "Metric": ["Inactive (0)", "10 μM (1)", "1 and 10 μM (2)"],
            "Count": [
                (df["Prediction"] == 0).sum(),
                (df["Prediction"] == 1).sum(),
                (df["Prediction"] == 2).sum()
            ]
        })

        st.dataframe(summary, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.bar(summary, x="Metric", y="Count", color="Metric",
                                   title="Activity Distribution", text="Count",
                                   color_discrete_sequence=px.colors.qualitative.Set2), use_container_width=True)
        with col2:
            st.plotly_chart(px.pie(summary, names="Metric", values="Count",
                                   title="Proportions", color_discrete_sequence=px.colors.sequential.RdBu), use_container_width=True)

        # ----------------------- #
        # Detailed Table
        # ----------------------- #
        st.subheader("🔬 Activity + ADMET Predictions")
        display_cols = ["smiles", "Prediction", "Probability", "Concentration", "Activity_Status"] + list(models.keys())[1:]
        st.dataframe(df[display_cols], use_container_width=True)

        # ----------------------- #
        # Download Options
        # ----------------------- #
        st.subheader("📥 Download Predictions")

        def to_excel_bytes(dataframe):
            output = BytesIO()
            dataframe.to_excel(output, index=False)
            return output.getvalue()

        st.download_button("Download All Predictions", to_excel_bytes(df), file_name="SMILES_Predictions.xlsx")

        with st.expander("🔽 Download High Confidence Predictions"):
            high_1 = df[(df["Prediction"] == 1) & (df["Probability"] > 0.75)]
            high_2 = df[(df["Prediction"] == 2) & (df["Probability"] > 0.75)]
            combined = df[(df["Prediction"].isin([1, 2])) & (df["Probability"] > 0.75)]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("High 10 μM (Class 1)", to_excel_bytes(high_1), "High_Class1_10uM.xlsx")
            with col2:
                st.download_button("High 1 & 10 μM (Class 2)", to_excel_bytes(high_2), "High_Class2_1and10uM.xlsx")
            with col3:
                st.download_button("Combined High Class 1 & 2", to_excel_bytes(combined), "High_Class1and2.xlsx")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("📤 Please upload a `.xlsx` file with a column named `smiles` to begin.")
