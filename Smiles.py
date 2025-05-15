import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from io import BytesIO
import plotly.express as px

def get_activity_status(row):
    if pd.isna(row["Probability"]):
        return "Unknown"
    if row["Prediction"] == 0:
        return "Inactive"
    elif row["Probability"] <= 0.75:
        if row['Prediction'] == 1:
            return f"Low chance of (10 μM)"
        elif row['Prediction'] == 2:
            return f"Low chance of (1 and 10 μM)"
        else:
            return "Low chance of non-hits"
    else:
        if row['Prediction'] == 1:
            return f"High chance of (10 μM)"
        elif row['Prediction'] == 2:
            return f"High chance of (1 and 10 μM)"
        else:
            return "High chance of non-hits"

# Page configuration
st.set_page_config(page_title="SMILES-Based Activity + ADMET Predictor", layout="wide")

# Sidebar branding
with st.sidebar:
    st.title("🧪Smiles Activity and ADMET Properties Predictor")
    st.markdown("**Predict Activity and ADMET properties** using SMILES strings  .")
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Upload Excel file with SMILES data in column name 'smiles'", type=["xlsx"])

# Load models
activity_model = joblib.load(r"C:\Users\admin\Downloads\Notebook\hybrid_model.pkl")
admet_models = {
    "(Absorption) Caco-2": joblib.load(r"C:\Users\admin\Downloads\Notebook\absorp_model.pkl"),
    "(Distribution) BBB": joblib.load(r"C:\Users\admin\Downloads\Notebook\distri_model.pkl"),
    "(Metabolism) CYP3A4": joblib.load(r"C:\Users\admin\Downloads\Notebook\meta_model.pkl"),
    "(Excretion) Obach": joblib.load(r"C:\Users\admin\Downloads\Notebook\exc_model.pkl"),
    "(Toxicity) NR-AR": joblib.load(r"C:\Users\admin\Downloads\Notebook\toxicity_model.pkl"),
}

ds_col = [
    "MolWt", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds", "MolLogP",
    "FpDensityMorgan1", "NumAromaticRings", "FractionCSP3", "NumAliphaticRings",
    "FpDensityMorgan2", "HeavyAtomMolWt"
]

def calculate_molecular_properties(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            return [
                Descriptors.MolWt(molecule),
                Descriptors.NumHDonors(molecule),
                Descriptors.NumHAcceptors(molecule),
                Descriptors.TPSA(molecule),
                rdMolDescriptors.CalcNumRotatableBonds(molecule),
                Descriptors.MolLogP(molecule),
                Descriptors.FpDensityMorgan1(molecule),
                Descriptors.NumAromaticRings(molecule),
                Descriptors.FractionCSP3(molecule),
                Descriptors.NumAliphaticRings(molecule),
                Descriptors.FpDensityMorgan2(molecule),
                Descriptors.HeavyAtomMolWt(molecule),
            ]
        else:
            return [0] * len(ds_col)
    except:
        return [0] * len(ds_col)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if "smiles" not in df.columns:
            st.error("❌ Column 'smiles' not found in uploaded file.")
        else:
            st.success(f"✅ Loaded {len(df)} molecules from uploaded file.")

            with st.spinner("🔍 Calculating descriptors and making predictions..."):
                df[ds_col] = df["smiles"].apply(calculate_molecular_properties).apply(pd.Series)

                # Activity Prediction
                y_pred = activity_model.predict(df[ds_col])
                y_proba = activity_model.predict_proba(df[ds_col])
                df["Prediction"] = y_pred
                df["Probability"] = np.max(y_proba, axis=1)
                df["Concentration"] = df["Prediction"].apply(
                    lambda x: "1 and 10 μM" if x == 2 else ("10 μM" if x == 1 else "non-hits")
                )
                df["Activity_Status"] = df.apply(get_activity_status, axis=1)

                # ADMET Predictions
                for name, model in admet_models.items():
                    df[f"{name}"] = model.predict(df[ds_col])
                # Map values in Metabolism and Toxicity columns to descriptive labels
                df["(Metabolism) CYP3A4"] = df["(Metabolism) CYP3A4"].map({0: "Non-Metabolic (0)", 1: "Metabolic (1)"})
                df["(Toxicity) NR-AR"] = df["(Toxicity) NR-AR"].map({0: "Non-Toxic (0)", 1: "Toxic (1)"})


                df.drop(columns=ds_col, inplace=True)

            st.success("✅ Predictions completed!")

            # Summary statistics
            total_smiles = len(df)
            total_inactive = (df["Prediction"] == 0).sum()
            total_10uM = (df["Prediction"] == 1).sum()
            total_1_and_10uM = (df["Prediction"] == 2).sum()

            summary_df = pd.DataFrame({
                "Metric": [
                    "Total SMILES",
                    "Inactive (0)",
                    "10 μM Concentration (1)",
                    "1 and 10 μM Concentration (2)"
                ],
                "Count": [
                    total_smiles,
                    total_inactive,
                    total_10uM,
                    total_1_and_10uM
                ]
            })

            st.markdown(
                "<h3 style='color:#3366cc; font-weight:bold;'>📊 Prediction Summary</h3>",
                unsafe_allow_html=True
            )

            st.dataframe(summary_df, use_container_width=True)

            # Plotly Bar Chart
            bar_fig = px.bar(
                summary_df[1:],  # exclude "Total SMILES" row
                x="Metric",
                y="Count",
                color="Metric",
                text="Count",
                title="Activity Distribution",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            bar_fig.update_layout(title_font_size=20, title_x=0.5)

            # Plotly Pie Chart
            pie_fig = px.pie(
                summary_df[1:],  # exclude "Total SMILES" row
                names="Metric",
                values="Count",
                title="Activity Class Proportion",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            pie_fig.update_traces(textposition='inside', textinfo='percent+label')
            pie_fig.update_layout(title_font_size=20, title_x=0.5)

            # Display charts in columns
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(bar_fig, use_container_width=True)
            with col2:
                st.plotly_chart(pie_fig, use_container_width=True)

            # ✅ Combined table: Activity + ADMET
            st.markdown("### 🔍 Activity + ADMET Predictions")
            combined_columns = ["smiles", "Prediction", "Probability", "Concentration", "Activity_Status"] + list(admet_models.keys())
            st.dataframe(df[combined_columns], use_container_width=True)

            st.markdown("### 📥 Download Results")
            output = BytesIO()
            df.to_excel(output, index=False)
            st.download_button("Download Excel with Predictions", data=output.getvalue(), file_name="SMILES_Predictions.xlsx")

            # Filter high-confidence predictions
            high_1_df = df[(df["Prediction"] == 1) & (df["Probability"] > 0.75)]
            high_2_df = df[(df["Prediction"] == 2) & (df["Probability"] > 0.75)]
            high_1_and_2_df = df[(df["Prediction"].isin([1, 2])) & (df["Probability"] > 0.75)]

            st.markdown("### 🔽 Download High Confidence Predictions")

            col1, col2, col3 = st.columns(3)

            with col1:
                output_1 = BytesIO()
                high_1_df.to_excel(output_1, index=False)
                st.download_button("Download High Class 1 (10 μM)", data=output_1.getvalue(),
                                   file_name="High_Confidence_Class1_10uM.xlsx")

            with col2:
                output_2 = BytesIO()
                high_2_df.to_excel(output_2, index=False)
                st.download_button("Download High Class 2 (1 and 10 μM)", data=output_2.getvalue(),
                                   file_name="High_Confidence_Class2_1_and_10uM.xlsx")

            with col3:
                output_combined = BytesIO()
                high_1_and_2_df.to_excel(output_combined, index=False)
                st.download_button("Download High Class 1 & 2 Combined", data=output_combined.getvalue(),
                                   file_name="High_Confidence_Class1_and_2.xlsx")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Upload an Excel file with a `smiles` column from the sidebar to begin.")
