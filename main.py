
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eaf2fb 0%,
        #f4f7fc 50%,
        #e5eef8 100%
    );
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #dce8f5;
    border-right: 1px solid #c8d8e9;
}

h1, h2, h3 {
    color: #193858;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #d5e1ef;
}

[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 8px;
}

div.stButton > button {
    background-color: #315f91;
    color: white;
    border: none;
    border-radius: 7px;
    padding: 8px 18px;
}

div.stButton > button:hover {
    background-color: #24496f;
    color: white;
}

div.stDownloadButton > button {
    background-color: #315f91;
    color: white;
    border-radius: 7px;
}

[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.75);
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #d5e1ef;
}

hr {
    border-color: #d2dfed;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "models/isolation_forest_improved.pkl"


@st.cache_resource
def load_model():

    saved_model = joblib.load(MODEL_PATH)

    return saved_model


if not os.path.exists(MODEL_PATH):

    st.error("Trained model not found. Please train the model first.")
    st.stop()


try:

    saved_model = load_model()

    model = saved_model["model"]
    threshold = saved_model["threshold"]
    features = saved_model["features"]

except Exception as e:

    st.error(f"Error loading model: {e}")
    st.stop()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Credit Card Fraud Detection")

st.sidebar.write(
    "Anomaly Detection using Machine Learning"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Upload Dataset",
        "Fraud Detection",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.write("Model: Isolation Forest")
st.sidebar.write("Learning: Unsupervised")
st.sidebar.write("Status: Model Loaded")

st.sidebar.divider()

st.sidebar.caption(
    "Credit Card Anomaly Detection System"
)


# ==========================================
# FRAUD DETECTION FUNCTION
# ==========================================

def detect_fraud(data):

    # Select required model features
    X = data[features].copy()

    # Convert values into numerical format
    X = X.apply(pd.to_numeric, errors="coerce")

    # Check for missing values
    if X.isnull().any().any():

        raise ValueError(
            "Dataset contains missing or non-numeric feature values."
        )

    # Check for infinite values
    if not np.isfinite(X.to_numpy()).all():

        raise ValueError(
            "Dataset contains infinite values."
        )

    # Calculate anomaly scores
    scores = -model.decision_function(X)

    # Apply selected threshold
    predictions = (scores >= threshold).astype(int)

    # Create result dataframe
    results = data.copy()

    results["Anomaly Score"] = scores

    results["Prediction"] = np.where(
        predictions == 1,
        "Suspicious",
        "Normal"
    )

    return results


# ==========================================
# PAGE 1: DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("Credit Card Fraud Detection System")

    st.write(
        "A machine learning based system for identifying "
        "unusual credit card transactions."
    )

    st.divider()

    results_path = "outputs/improved_predictions.csv"

    if os.path.exists(results_path):

        results = pd.read_csv(results_path)

        total = len(results)

        fraud = int(results["Predicted"].sum())

        normal = total - fraud

        # Transaction summary

        st.subheader("Transaction Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Transactions",
            f"{total:,}"
        )

        col2.metric(
            "Normal Transactions",
            f"{normal:,}"
        )

        col3.metric(
            "Suspicious Transactions",
            f"{fraud:,}"
        )

        st.divider()

        # Improved transaction distribution chart

        st.subheader("Transaction Distribution")

        fig, ax = plt.subplots(figsize=(9, 4))

        categories = ["Normal", "Suspicious"]
        values = [normal, fraud]

        bars = ax.bar(
            categories,
            values,
            color=["#315f91", "#d97757"],
            width=0.45
        )

        ax.set_title(
            "Normal vs Suspicious Transactions",
            fontsize=13,
            pad=15
        )

        ax.set_ylabel("Number of Transactions")

        # Logarithmic scale helps display both categories
        ax.set_yscale("log")

        ax.set_ylim(1, max(values) * 2)

        # Display transaction counts above bars
        for bar, value in zip(bars, values):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.12,
                f"{value:,}",
                ha="center",
                fontsize=10
            )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.35
        )

        ax.set_axisbelow(True)

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.caption(
            "The chart uses a logarithmic scale to make "
            "both transaction categories easier to compare."
        )

    else:

        st.warning(
            "Model prediction results are not available."
        )

    st.divider()

    # Project information

    st.subheader("About the Project")

    st.write("""
    Credit card fraud is a major concern in digital transactions.

    This project uses the Isolation Forest algorithm to identify
    unusual transaction patterns.

    The model analyses transaction features and assigns anomaly
    scores to identify potentially suspicious activities.
    """)

    st.subheader("Technology Used")

    col1, col2, col3 = st.columns(3)

    col1.write("Python")
    col2.write("Scikit-learn")
    col3.write("Streamlit")


# ==========================================
# PAGE 2: UPLOAD DATASET
# ==========================================

elif page == "Upload Dataset":

    st.title("Upload Transaction Dataset")

    st.write(
        "Upload a CSV file to analyse credit card transactions."
    )

    st.info(
        "The dataset must contain all the required model features."
    )

    uploaded_file = st.file_uploader(
        "Select CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            st.subheader("Dataset Preview")

            st.dataframe(
                data.head(10),
                use_container_width=True
            )

            st.write(
                "Total Records:",
                f"{len(data):,}"
            )

            # Check required columns

            missing_features = [
                col for col in features
                if col not in data.columns
            ]

            if missing_features:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing_features)
                )

            elif data.empty:

                st.error("The uploaded dataset is empty.")

            else:

                st.success(
                    "All required transaction features are available."
                )

                if st.button("Analyse Transactions"):

                    with st.spinner(
                        "Analysing transactions..."
                    ):

                        analysed_data = detect_fraud(data)

                        st.session_state[
                            "analysis_results"
                        ] = analysed_data

                    st.success(
                        "Transaction analysis completed successfully!"
                    )

                    st.info(
                        "Open the Fraud Detection page to view results."
                    )

        except Exception as e:

            st.error(
                f"Unable to process dataset: {e}"
            )


# ==========================================
# PAGE 3: FRAUD DETECTION
# ==========================================

elif page == "Fraud Detection":

    st.title("Fraud Detection Results")

    if "analysis_results" in st.session_state:

        results = st.session_state["analysis_results"]

        fraud_count = (
            results["Prediction"] == "Suspicious"
        ).sum()

        normal_count = len(results) - fraud_count

        # Summary

        st.subheader("Analysis Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Transactions",
            f"{len(results):,}"
        )

        col2.metric(
            "Normal Transactions",
            f"{normal_count:,}"
        )

        col3.metric(
            "Suspicious Transactions",
            f"{fraud_count:,}"
        )

        st.divider()

        # Transaction filtering

        st.subheader("Transaction Details")

        filter_option = st.selectbox(
            "Filter Transactions",
            [
                "All Transactions",
                "Normal Transactions",
                "Suspicious Transactions"
            ]
        )

        if filter_option == "Normal Transactions":

            display_data = results[
                results["Prediction"] == "Normal"
            ]

        elif filter_option == "Suspicious Transactions":

            display_data = results[
                results["Prediction"] == "Suspicious"
            ]

        else:

            display_data = results

        st.dataframe(
            display_data,
            use_container_width=True
        )

        st.write(
            "Records displayed:",
            f"{len(display_data):,}"
        )

        st.divider()

        # Download report

        st.subheader("Download Results")

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction Report",
            data=csv_data,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No transaction analysis available. "
            "Please upload a dataset first."
        )


# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================

# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("Model Performance")

    evaluation_data = None
    actual = None
    predicted = None
    scores = None

    # First check results from uploaded dataset
    if "analysis_results" in st.session_state:

        results = st.session_state["analysis_results"]

        # Check whether actual fraud labels are available
        class_column = next(
            (col for col in results.columns
             if col.lower() == "class"),
            None
        )

        if class_column is not None:

            actual = pd.to_numeric(
                results[class_column],
                errors="coerce"
            )

            predicted = (
                results["Prediction"] == "Suspicious"
            ).astype(int)

            scores = pd.to_numeric(
                results["Anomaly Score"],
                errors="coerce"
            )

            evaluation_data = "uploaded"

    # If uploaded results are unavailable, check saved results
    if evaluation_data is None:

        results_path = "outputs/improved_predictions.csv"

        if os.path.exists(results_path):

            results = pd.read_csv(results_path)

            actual = pd.to_numeric(
                results["Actual"],
                errors="coerce"
            )

            predicted = pd.to_numeric(
                results["Predicted"],
                errors="coerce"
            )

            scores = pd.to_numeric(
                results["Anomaly_Score"],
                errors="coerce"
            )

            evaluation_data = "saved"

    if evaluation_data is not None:

        # Remove invalid records
        evaluation = pd.DataFrame({
            "Actual": actual,
            "Predicted": predicted,
            "Score": scores
        }).replace([np.inf, -np.inf], np.nan).dropna()

        actual = evaluation["Actual"].astype(int)
        predicted = evaluation["Predicted"].astype(int)
        scores = evaluation["Score"]

        if len(evaluation) == 0:

            st.warning("No valid evaluation records found.")

        elif actual.nunique() < 2:

            st.warning(
                "Both normal and fraud transactions are required "
                "to calculate model evaluation metrics."
            )

        else:

            # Calculate metrics
            precision = precision_score(
                actual, predicted, zero_division=0
            )

            recall = recall_score(
                actual, predicted, zero_division=0
            )

            f1 = f1_score(
                actual, predicted, zero_division=0
            )

            auc = roc_auc_score(actual, scores)

            if evaluation_data == "uploaded":
                st.success(
                    "Showing evaluation results for the uploaded dataset."
                )
            else:
                st.info(
                    "Showing evaluation results from the saved test dataset."
                )

            # Evaluation metrics
            st.subheader("Evaluation Metrics")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Precision", f"{precision * 100:.2f}%")
            col2.metric("Recall", f"{recall * 100:.2f}%")
            col3.metric("F1 Score", f"{f1 * 100:.2f}%")
            col4.metric("ROC-AUC", f"{auc * 100:.2f}%")

            st.divider()

            # Confusion Matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                actual,
                predicted,
                labels=[0, 1]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Normal", "Actual Fraud"],
                columns=["Predicted Normal", "Predicted Fraud"]
            )

            st.dataframe(
                cm_df,
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(7, 5))

            import seaborn as sns

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"],
                ax=ax
            )

            ax.set_title("Isolation Forest - Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.divider()

            # ROC Curve
            st.subheader("ROC Curve")

            from sklearn.metrics import roc_curve

            fpr, tpr, thresholds = roc_curve(
                actual,
                scores
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.plot(
                fpr,
                tpr,
                label=f"Isolation Forest (AUC = {auc:.2f})",
                linewidth=2
            )

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="grey"
            )

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Isolation Forest - ROC Curve")
            ax.legend()
            ax.grid(alpha=0.3)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.divider()

    else:

        st.warning(
            "Model evaluation results are not available. "
            "Please upload a labelled test dataset and analyse it first."
        )

    # Model Information
    st.subheader("Model Information")

    st.write("Algorithm: Isolation Forest")

    st.write(
        "Number of Estimators:",
        getattr(model, "n_estimators", "Not available")
    )

    st.write("Learning Method: Unsupervised")
    st.write("Training Data: Normal Transactions Only")

    st.write(
        "Selected Anomaly Threshold:",
        round(float(threshold), 6)
    )


# ==========================================
# END OF APPLICATION
# ==========================================
