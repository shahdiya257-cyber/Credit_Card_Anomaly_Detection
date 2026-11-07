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
    confusion_matrix,
    roc_curve
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
# CUSTOM DESIGN - LIGHT AND DARK MODE FIX
# ==========================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background: linear-gradient(
        135deg,
        #eaf2fb 0%,
        #f4f7fc 50%,
        #e5eef8 100%
    );
    color: #193858 !important;
}

/* Main content spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Fix text visibility in dark mode */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #193858 !important;
}

.stApp p,
.stApp li,
.stApp label,
.stApp .stMarkdown,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] span {
    color: #193858 !important;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #dce8f5 !important;
    border-right: 1px solid #c8d8e9;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #193858 !important;
}

/* Sidebar radio buttons */

[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #193858 !important;
}

/* Metric cards */

[data-testid="stMetric"] {
    background-color: #ffffff !important;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #d5e1ef;
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #193858 !important;
}

/* Data tables */

[data-testid="stDataFrame"] {
    background-color: #ffffff !important;
    border-radius: 8px;
}

/* Buttons */

div.stButton > button {
    background-color: #315f91 !important;
    color: #ffffff !important;
    border: none;
    border-radius: 7px;
    padding: 8px 18px;
}

div.stButton > button:hover {
    background-color: #24496f !important;
    color: #ffffff !important;
}

/* Download button */

div.stDownloadButton > button {
    background-color: #315f91 !important;
    color: #ffffff !important;
    border-radius: 7px;
}

div.stDownloadButton > button:hover {
    background-color: #24496f !important;
    color: #ffffff !important;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.85) !important;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #d5e1ef;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #193858 !important;
}

/* Input fields */

.stTextInput label,
.stSelectbox label,
.stFileUploader label,
.stNumberInput label {
    color: #193858 !important;
}

/* Input text */

.stTextInput input,
.stNumberInput input {
    color: #193858 !important;
    background-color: #ffffff !important;
}

/* Select box */

.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
}

.stSelectbox div[data-baseweb="select"] * {
    color: #193858 !important;
}

/* Information boxes */

[data-testid="stAlert"] p {
    color: #193858 !important;
}

/* Horizontal divider */

hr {
    border-color: #d2dfed !important;
}

/* Captions */

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #526b84 !important;
}

/* Keep button text white */

.stButton button,
.stDownloadButton button {
    color: #ffffff !important;
}

/* Charts and plot containers */

[data-testid="stImage"],
[data-testid="stPlotlyChart"],
[data-testid="stPyplot"] {
    border-radius: 10px;
}

/* FORCE THE SAME LIGHT APPEARANCE IN DARK-MODE BROWSERS */

:root {
    color-scheme: light !important;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #eef4fc !important;
    color: #193858 !important;
}

/* Keep the header consistent */

[data-testid="stHeader"] {
    background-color: #eef4fc !important;
}

/* Alert / information / success / warning boxes */

[data-testid="stAlert"] {
    background-color: #e8f1fc !important;
    border: 1px solid #cbdced !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #193858 !important;
}

/* File uploader and drop zone */

[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border: 1px solid #d5e1ef !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] section {
    background-color: #f4f7fc !important;
    border: 1px dashed #9bb5d0 !important;
}

[data-testid="stFileUploader"] button {
    background-color: #315f91 !important;
    color: #ffffff !important;
    border: none !important;
}

[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
}

[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] label {
    color: #193858 !important;
}

/* Text and number input fields */

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #193858 !important;
    border-color: #cbdced !important;
}

/* Select boxes */

[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #193858 !important;
    border-color: #cbdced !important;
}

[data-baseweb="select"] * {
    color: #193858 !important;
}

/* Dataframe surface */

[data-testid="stDataFrame"] {
    background-color: #ffffff !important;
}

/* Ensure sidebar and its controls remain light */

[data-testid="stSidebar"] {
    background-color: #dce8f5 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
    color: #193858 !important;
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

    st.error(
        "Trained model not found. Please train the model first."
    )

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

    if (
        "analysis_results" in st.session_state
        or os.path.exists(results_path)
    ):

        if "analysis_results" in st.session_state:

            results = st.session_state["analysis_results"]

            fraud = int(
                (results["Prediction"] == "Suspicious").sum()
            )

        else:

            results = pd.read_csv(results_path)

            fraud = int(results["Predicted"].sum())

        total = len(results)

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

        # Transaction distribution chart

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

        ax.set_yscale("log")

        ax.set_ylim(1, max(values) * 2 if max(values) > 0 else 10)

        for bar, value in zip(bars, values):

            if value > 0:

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

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

        st.caption(
            "The chart uses a logarithmic scale to make "
            "both transaction categories easier to compare."
        )

    else:

        st.warning(
            "Model prediction results are not available. "
            "Please upload a dataset to begin."
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

        # Analysis summary

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

        # Transaction details

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

elif page == "Model Performance":

    st.title("Model Performance")

    results_path = "outputs/improved_predictions.csv"

    use_uploaded_results = False

    # First check the latest uploaded dataset

    if "analysis_results" in st.session_state:

        uploaded_results = st.session_state["analysis_results"]

        actual_column = (
            "Class" if "Class" in uploaded_results.columns
            else "Actual" if "Actual" in uploaded_results.columns
            else None
        )

        if actual_column is not None:

            results = uploaded_results.copy()

            actual = pd.to_numeric(
                results[actual_column],
                errors="coerce"
            )

            predicted = (
                results["Prediction"] == "Suspicious"
            ).astype(int)

            scores = pd.to_numeric(
                results["Anomaly Score"],
                errors="coerce"
            )

            use_uploaded_results = True

    # Otherwise use saved prediction results

    if not use_uploaded_results and os.path.exists(results_path):

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

    if use_uploaded_results or os.path.exists(results_path):

        if use_uploaded_results:

            st.success(
                "Showing evaluation results for the uploaded dataset."
            )

        else:

            st.caption(
                "Evaluation source: saved prediction results."
            )

        # Prepare evaluation data

        evaluation = pd.DataFrame({
            "Actual": actual,
            "Predicted": predicted,
            "Anomaly_Score": scores
        }).dropna()

        actual = evaluation["Actual"].astype(int)

        predicted = evaluation["Predicted"].astype(int)

        scores = evaluation["Anomaly_Score"].astype(float)

        if len(evaluation) == 0:

            st.warning("No valid evaluation records are available.")

            st.stop()

        if not set(actual.unique()).issubset({0, 1}):

            st.error(
                "The actual class column must contain only 0 and 1."
            )

            st.stop()

        # Calculate evaluation metrics

        precision = precision_score(
            actual,
            predicted,
            zero_division=0
        )

        recall = recall_score(
            actual,
            predicted,
            zero_division=0
        )

        f1 = f1_score(
            actual,
            predicted,
            zero_division=0
        )

        # ROC-AUC

        auc = None

        if actual.nunique() == 2:

            auc = roc_auc_score(
                actual,
                scores
            )

        # Evaluation metrics

        st.subheader("Evaluation Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

        col2.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

        col3.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

        col4.metric(
            "ROC-AUC",
            f"{auc * 100:.2f}%" if auc is not None else "N/A"
        )

        st.divider()

        # Confusion matrix

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual Normal",
                "Actual Fraud"
            ],
            columns=[
                "Predicted Normal",
                "Predicted Fraud"
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        # Draw confusion matrix from current evaluation

        fig_cm, ax_cm = plt.subplots(figsize=(7, 4.5))

        image = ax_cm.imshow(
            cm,
            cmap="Blues"
        )

        ax_cm.set_title(
            "Isolation Forest - Confusion Matrix"
        )

        ax_cm.set_xlabel("Predicted label")

        ax_cm.set_ylabel("Actual label")

        ax_cm.set_xticks(
            [0, 1],
            labels=["Normal", "Fraud"]
        )

        ax_cm.set_yticks(
            [0, 1],
            labels=["Normal", "Fraud"]
        )

        midpoint = cm.max() / 2 if cm.size else 0

        for row in range(cm.shape[0]):

            for col in range(cm.shape[1]):

                ax_cm.text(
                    col,
                    row,
                    f"{cm[row, col]:,}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if cm[row, col] > midpoint
                        else "#17324d"
                    ),
                    fontsize=11
                )

        fig_cm.colorbar(
            image,
            ax=ax_cm,
            fraction=0.046,
            pad=0.04
        )

        fig_cm.tight_layout()

        st.pyplot(
            fig_cm,
            use_container_width=True
        )

        plt.close(fig_cm)

        st.divider()

        # ROC curve

        st.subheader("ROC Curve")

        if actual.nunique() == 2:

            fpr, tpr, _ = roc_curve(
                actual,
                scores
            )

            fig_roc, ax_roc = plt.subplots(
                figsize=(7, 4.5)
            )

            ax_roc.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"Improved Isolation Forest (AUC = {auc:.2f})"
            )

            ax_roc.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="grey",
                alpha=0.7
            )

            ax_roc.set_title(
                "Improved Isolation Forest - ROC Curve"
            )

            ax_roc.set_xlabel("False Positive Rate")

            ax_roc.set_ylabel("True Positive Rate")

            ax_roc.set_xlim(0, 1)

            ax_roc.set_ylim(0, 1.02)

            ax_roc.legend(loc="lower right")

            ax_roc.grid(alpha=0.2)

            fig_roc.tight_layout()

            st.pyplot(
                fig_roc,
                use_container_width=True
            )

            plt.close(fig_roc)

        else:

            st.warning(
                "ROC curve cannot be calculated because the "
                "evaluation data does not contain both normal "
                "and fraud transactions."
            )

        st.divider()

        # Model information

        st.subheader("Model Information")

        st.write("Algorithm: Isolation Forest")

        st.write(
            "Number of Estimators:",
            getattr(
                model,
                "n_estimators",
                "Not available"
            )
        )

        st.write("Learning Method: Unsupervised")

        st.write("Training Data: Normal Transactions Only")

        st.write(
            "Selected Anomaly Threshold:",
            round(float(threshold), 6)
        )

    else:

        st.warning(
            "Model evaluation results are not available. "
            "Upload a labelled test dataset or check "
            "outputs/improved_predictions.csv."
        )


# ==========================================
# END OF APPLICATION
# ==========================================
