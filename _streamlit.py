import json
import streamlit as st
import pandas as pd
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urljoin

st.set_page_config(page_title="Loan Default Prediction", layout="wide")

st.title("🏦 Loan Defaulter Detection System")
st.markdown("---")

API_HOST = "https://api-w9tz.onrender.com"
PREDICT_ENDPOINT = f"{API_HOST}/predict"


def call_predict_api(payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    request = Request(
        PREDICT_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8")

    return json.loads(body)


st.sidebar.header("⚙️ Configuration & Input")

selected_threshold = st.sidebar.slider(
    "Select Prediction Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Lower threshold = more risky loans flagged as unsafe",
)

st.sidebar.markdown("---")
st.sidebar.header("📋 Loan Application Details")

annual_inc = st.sidebar.number_input(
    "Annual Income ($)",
    min_value=1,
    value=50000,
    step=1000,
    help="Annual income of the applicant",
)

loan_amnt = st.sidebar.number_input(
    "Loan Amount ($)",
    min_value=1,
    value=10000,
    step=1000,
    help="Total amount of the loan",
)

int_rate = st.sidebar.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=0.1,
    help="Interest rate as a percentage",
)

emp_length = st.sidebar.slider(
    "Employment Length (years)",
    min_value=0,
    max_value=10,
    value=5,
    help="Years of employment (0-10)",
)

dti = st.sidebar.number_input(
    "Debt-to-Income Ratio",
    min_value=0.0,
    value=0.5,
    step=0.01,
    help="Debt-to-Income ratio",
)

term = st.sidebar.selectbox(
    "Loan Term",
    [" 36 months", " 60 months"],
    help="Loan term duration",
)

home_ownership = st.sidebar.selectbox(
    "Home Ownership",
    ["MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY"],
    help="Home ownership status",
)

purpose = st.sidebar.selectbox(
    "Loan Purpose",
    [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "other",
        "major_purchase",
        "small_business",
        "car",
        "medical",
        "wedding",
        "moving",
        "house",
        "vacation",
        "educational",
        "renewable_energy",
    ],
    help="Purpose of the loan",
)

total_acc = st.sidebar.number_input(
    "Total Credit Accounts",
    min_value=0,
    value=5,
    step=1,
    help="Total number of credit lines",
)

st.header("📊 Prediction Results")

if st.sidebar.button("🔮 Make Prediction", key="predict_button"):
    payload = {
        "annual_inc": annual_inc,
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "emp_length": emp_length,
        "dti": dti,
        "term": term,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "total_acc": total_acc,
        "selected_threshold": float(selected_threshold),
    }

    try:
        response = call_predict_api(payload)
    except HTTPError as exc:
        st.error(f"❌ API error ({exc.code}): {exc.reason}")
        st.stop()
    except URLError as exc:
        st.error(f"❌ Could not connect to backend API: {exc.reason}")
        st.stop()
    except Exception as exc:
        st.error(f"❌ Unexpected API error: {exc}")
        st.stop()

    predicted_output = response.get("predicted_output")
    prediction_prob = response.get("risk_probability")
    threshold_used = response.get("threshold_used")
    model_name = response.get("model_name", "Unknown")
    threshold_precision = response.get("precision")
    threshold_recall = response.get("recall")
    threshold_f1 = response.get("f1_score")
    threshold_metrics = response.get("threshold_metrics", [])

    is_risky = predicted_output == "unsafe"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Prediction Decision")
        decision_text = "🚨 UNSAFE - High Risk" if is_risky else "✅ SAFE - Low Risk"
        decision_color = "#FF6B6B" if is_risky else "#51CF66"
        st.markdown(
            f"<h2 style='color: {decision_color}; text-align: center;'>{decision_text}</h2>",
            unsafe_allow_html=True,
        )
        if prediction_prob is not None:
            st.metric("Risk Probability", f"{prediction_prob*100:.4f}%", delta=None)

    with col2:
        st.subheader("⚙️ Configuration Used")
        st.metric("Selected Threshold", f"{threshold_used:.1f}")
        st.metric("Model Used", model_name)

    st.subheader("📈 Model Metrics at Selected Threshold")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        if threshold_precision is not None:
            st.metric("Precision", f"{threshold_precision*100:.4f}%")
        else:
            st.warning("No metrics available for this threshold")

    with metrics_col2:
        if threshold_recall is not None:
            st.metric("Recall", f"{threshold_recall*100:.4f}%")

    with metrics_col3:
        if threshold_f1 is not None:
            st.metric("F1 Score", f"{threshold_f1*100:.4f}%")

    st.subheader("📝 Input Loan Details Summary")
    summary_data = {
        "Annual Income": f"${annual_inc:,}",
        "Loan Amount": f"${loan_amnt:,}",
        "Interest Rate": f"{int_rate:.2f}%",
        "Employment Length": f"{emp_length} years",
        "DTI Ratio": f"{dti:.4f}",
        "Loan Term": term,
        "Home Ownership": home_ownership,
        "Loan Purpose": purpose,
        "Total Accounts": total_acc,
    }
    summary_df = pd.DataFrame(list(summary_data.items()), columns=["Field", "Value"])
    st.table(summary_df)

    st.subheader("📊 Threshold Metrics Comparison")
    if threshold_metrics:
        st.dataframe(pd.DataFrame(threshold_metrics).round(4), use_container_width=True)
    else:
        st.warning("Threshold metrics are not available from the backend response.")
else:
    st.info("👈 Configure your loan details in the sidebar and click 'Make Prediction' to get started!")
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    """
    This application uses a trained machine learning model to predict
    loan default risk. The threshold slider allows you to adjust the
    prediction boundary:

    - **Lower threshold**: More loans flagged as risky (higher recall)
    - **Higher threshold**: Fewer loans flagged as risky (higher precision)

    The metrics shown represent model performance at the selected threshold
    on the test dataset.
    """
)
st.sidebar.markdown("---")
st.sidebar.header("👤 Author")
st.sidebar.markdown(
    """
    **Amrit Kumar Gorai**  
    [GitHub repository](https://github.com/amritk32/Loan-Defaulter-Detection-System-using-Machine-Learning)
    """
)
