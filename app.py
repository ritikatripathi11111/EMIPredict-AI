"""
EMIPredict AI - Streamlit Application

Multi-page financial risk assessment platform:
    Home / Dashboard
    Risk Prediction
    Data Explorer
    Model Performance
    Admin (CRUD)

Run with:
    streamlit run app.py
"""

import sys
import os
import textwrap

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from predict import predict
import database as db


# ---------------------------------------------------------------------------
# Streamlit page configuration
# IMPORTANT: This must be called exactly once and before Streamlit UI commands.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------
db.init_db()


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.15rem;
            color: #94a3b8;
            margin-bottom: 1.5rem;
        }

        .prediction-card {
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 22px;
            background: #151c28;
            margin-bottom: 18px;
        }

        .prediction-label {
            font-size: 0.9rem;
            color: #94a3b8;
        }

        .prediction-value {
            font-size: 2rem;
            font-weight: 750;
            margin-top: 5px;
        }

        .risk-card {
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 18px;
            background: #151c28;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 0.9rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
        }

        .safe-box {
            border-left: 5px solid #23c982;
            background: rgba(35, 201, 130, 0.08);
            padding: 14px 18px;
            border-radius: 10px;
        }

        .warning-box {
            border-left: 5px solid #f59e0b;
            background: rgba(245, 158, 11, 0.08);
            padding: 14px 18px;
            border-radius: 10px;
        }

        .danger-box {
            border-left: 5px solid #ef4444;
            background: rgba(239, 68, 68, 0.08);
            padding: 14px 18px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("💰 EMIPredict AI")
st.sidebar.caption("Intelligent Financial Risk Assessment Platform")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home / Dashboard",
        "🎯 Risk Prediction",
        "📊 Data Explorer",
        "🤖 Model Performance",
        "🗂️ Admin (CRUD)",
    ],
)


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_cleaned_data():
    try:
        file_path = os.path.join(PROJECT_ROOT, "data", "emi_cleaned.csv")
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None
    except Exception:
        return None


@st.cache_data
def load_comparison_tables():
    classification_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "classification_comparison.csv",
    )

    regression_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "regression_comparison.csv",
    )

    try:
        clf = pd.read_csv(classification_path, index_col=0)
    except FileNotFoundError:
        clf = None
    except Exception:
        clf = None

    try:
        reg = pd.read_csv(regression_path, index_col=0)
    except FileNotFoundError:
        reg = None
    except Exception:
        reg = None

    return clf, reg


# ===========================================================================
# PAGE: HOME / DASHBOARD
# ===========================================================================
if page == "🏠 Home / Dashboard":

    st.markdown(
        '<div class="main-title">EMIPredict AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Intelligent Financial Risk Assessment Platform</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "EMIPredict AI analyzes a customer's income, expenses, debt obligations, "
        "credit profile, employment and requested loan details to predict EMI "
        "eligibility and estimate the maximum safe monthly EMI using classification "
        "and regression models tracked through MLflow."
    )

    df = load_cleaned_data()
    clf_comp, reg_comp = load_comparison_tables()

    # -----------------------------------------------------------------------
    # KPI cards
    # -----------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Financial Profiles",
        f"{len(df):,}" if df is not None else "—",
    )

    c2.metric(
        "EMI Scenarios",
        df["emi_scenario"].nunique() if df is not None else 5,
    )

    c3.metric(
        "Best Classification Accuracy",
        (
            f"{clf_comp['accuracy'].max() * 100:.1f}%"
            if clf_comp is not None and "accuracy" in clf_comp.columns
            else "Run training first"
        ),
    )

    c4.metric(
        "Best Regression RMSE",
        (
            f"₹{reg_comp['rmse'].min():,.0f}"
            if reg_comp is not None and "rmse" in reg_comp.columns
            else "Run training first"
        ),
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    if df is not None:

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                df,
                names="emi_eligibility",
                title="EMI Eligibility Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            fig.update_layout(
                margin=dict(l=10, r=10, t=50, b=10),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with col2:
            scenario_counts = (
                df["emi_scenario"]
                .value_counts()
                .rename_axis("emi_scenario")
                .reset_index(name="count")
            )

            fig2 = px.bar(
                scenario_counts,
                x="emi_scenario",
                y="count",
                title="Records per EMI Scenario",
            )

            fig2.update_xaxes(tickangle=25)

            st.plotly_chart(
                fig2,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    else:
        st.info(
            "Cleaned dataset not found. Run "
            "`python src/preprocessing.py` first."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Quick actions
    # -----------------------------------------------------------------------
    st.subheader("Quick Actions")

    qa1, qa2 = st.columns(2)

    with qa1:
        if st.button(
            "🎯 Assess Customer Risk",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["navigate_prediction"] = True
            st.info("Select **🎯 Risk Prediction** from the sidebar.")

    with qa2:
        if st.button(
            "📊 Explore Financial Dataset",
            use_container_width=True,
        ):
            st.info("Select **📊 Data Explorer** from the sidebar.")


# ===========================================================================
# PAGE: RISK PREDICTION
# ===========================================================================
elif page == "🎯 Risk Prediction":

    st.title("EMI Risk Prediction")

    st.caption(
        "Enter a customer's financial profile to assess eligibility, "
        "risk level, affordability and maximum safe EMI."
    )

    # -----------------------------------------------------------------------
    # Input form
    # -----------------------------------------------------------------------
    with st.form("prediction_form"):

        st.subheader("👤 Personal Information")

        c1, c2, c3, c4 = st.columns(4)

        age = c1.number_input(
            "Age",
            min_value=18,
            max_value=75,
            value=32,
        )

        gender = c2.selectbox(
            "Gender",
            ["Male", "Female"],
        )

        marital_status = c3.selectbox(
            "Marital Status",
            ["Single", "Married"],
        )

        education = c4.selectbox(
            "Education",
            [
                "High School",
                "Graduate",
                "Post Graduate",
                "Professional",
            ],
        )

        st.subheader("💼 Employment")

        c1, c2, c3, c4 = st.columns(4)

        monthly_salary = c1.number_input(
            "Monthly Salary (₹)",
            min_value=0,
            max_value=1_000_000,
            value=65_000,
            step=1_000,
        )

        employment_type = c2.selectbox(
            "Employment Type",
            [
                "Private",
                "Government",
                "Self-employed",
            ],
        )

        years_of_employment = c3.number_input(
            "Years of Employment",
            min_value=0.0,
            max_value=40.0,
            value=4.5,
            step=0.5,
        )

        company_type = c4.selectbox(
            "Company Type",
            [
                "MNC",
                "Mid-size",
                "Startup",
                "Government",
            ],
        )

        st.subheader("🏠 Household")

        c1, c2, c3, c4 = st.columns(4)

        house_type = c1.selectbox(
            "House Type",
            [
                "Rented",
                "Own",
                "Family",
            ],
        )

        monthly_rent = c2.number_input(
            "Monthly Rent (₹)",
            min_value=0,
            max_value=200_000,
            value=15_000,
            step=500,
        )

        family_size = c3.number_input(
            "Family Size",
            min_value=1,
            max_value=15,
            value=3,
        )

        dependents = c4.number_input(
            "Dependents",
            min_value=0,
            max_value=10,
            value=1,
        )

        st.subheader("💸 Monthly Expenses")

        c1, c2, c3, c4 = st.columns(4)

        school_fees = c1.number_input(
            "School Fees (₹)",
            min_value=0,
            max_value=100_000,
            value=0,
            step=500,
        )

        college_fees = c2.number_input(
            "College Fees (₹)",
            min_value=0,
            max_value=200_000,
            value=0,
            step=500,
        )

        travel_expenses = c3.number_input(
            "Travel Expenses (₹)",
            min_value=0,
            max_value=50_000,
            value=3_000,
            step=500,
        )

        groceries_utilities = c4.number_input(
            "Groceries & Utilities (₹)",
            min_value=0,
            max_value=100_000,
            value=8_000,
            step=500,
        )

        other_monthly_expenses = st.number_input(
            "Other Monthly Expenses (₹)",
            min_value=0,
            max_value=100_000,
            value=2_000,
            step=500,
        )

        st.subheader("💳 Financial Status")

        c1, c2, c3, c4 = st.columns(4)

        existing_loans = c1.selectbox(
            "Existing Loans",
            ["No", "Yes"],
        )

        current_emi_amount = c2.number_input(
            "Current EMI Amount (₹)",
            min_value=0,
            max_value=200_000,
            value=0,
            step=500,
        )

        credit_score = c3.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=740,
        )

        bank_balance = c4.number_input(
            "Bank Balance (₹)",
            min_value=0,
            max_value=10_000_000,
            value=150_000,
            step=1_000,
        )

        emergency_fund = st.number_input(
            "Emergency Fund (₹)",
            min_value=0,
            max_value=5_000_000,
            value=100_000,
            step=1_000,
        )

        st.subheader("🏦 Loan Request")

        c1, c2, c3 = st.columns(3)

        emi_scenario = c1.selectbox(
            "EMI Scenario",
            [
                "E-commerce Shopping EMI",
                "Home Appliances EMI",
                "Vehicle EMI",
                "Personal Loan EMI",
                "Education EMI",
            ],
        )

        requested_amount = c2.number_input(
            "Requested Amount (₹)",
            min_value=1_000,
            max_value=5_000_000,
            value=500_000,
            step=5_000,
        )

        requested_tenure = c3.number_input(
            "Requested Tenure (months)",
            min_value=1,
            max_value=120,
            value=36,
        )

        submitted = st.form_submit_button(
            "🚀 Predict Eligibility",
            type="primary",
            use_container_width=True,
        )

    # -----------------------------------------------------------------------
    # Prepare customer data after submission
    # -----------------------------------------------------------------------
    if submitted:

        customer = {
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "emi_scenario": emi_scenario,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
        }

        try:
            result = predict(customer)

            # Store prediction in session so it remains available
            # when Save / Log button causes a Streamlit rerun.
            st.session_state["prediction_customer"] = customer
            st.session_state["prediction_result"] = result

        except FileNotFoundError:
            st.error(
                "Trained model files were not found. Make sure "
                "`models/` contains the trained classification and "
                "regression artifacts."
            )

        except Exception as exc:
            st.error(
                f"Prediction failed: {exc}"
            )


    # -----------------------------------------------------------------------
    # Display stored prediction result
    # -----------------------------------------------------------------------
    if (
        "prediction_result" in st.session_state
        and "prediction_customer" in st.session_state
    ):

        result = st.session_state["prediction_result"]
        customer = st.session_state["prediction_customer"]

        st.divider()

        st.subheader("📌 Prediction Result")

        st.caption(
            "AI-assisted decision support for EMI affordability and financial risk."
        )

        eligibility = result.get(
            "eligibility",
            "Unknown",
        )

        risk_level = result.get(
            "risk_level",
            "Unknown",
        )

        confidence = result.get(
            "confidence",
            0,
        )

        max_safe_emi = result.get(
            "max_safe_emi",
            0,
        )

        requested_emi = result.get(
            "requested_emi",
            0,
        )

        affordability_status = result.get(
            "affordability_status",
            "UNKNOWN",
        )

        recommendation = result.get(
            "recommendation",
            "No recommendation available.",
        )

        # -------------------------------------------------------------------
        # Color / risk mapping
        # -------------------------------------------------------------------
        color_map = {
            "Eligible": "#23c982",
            "High_Risk": "#f59e0b",
            "Not_Eligible": "#ef4444",
        }

        accent = color_map.get(
            eligibility,
            "#7aa2f7",
        )

        # -------------------------------------------------------------------
        # Main decision card
        # -------------------------------------------------------------------
        # -----------------------------------------------------------------------
# Eligibility Decision
# -----------------------------------------------------------------------
        with st.container(border=True):
            st.markdown("### Eligibility Decision")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Decision",
                    eligibility.replace("_", " "),
                )        

            with c2:
                st.metric(
                    "Risk Level",
                    risk_level,
                )

            with c3:
                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.1f}%",
                )        
        # -------------------------------------------------------------------
        # Primary metrics
        # -------------------------------------------------------------------
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Maximum Safe EMI",
            f"₹{max_safe_emi:,.0f}",
        )

        c2.metric(
            "Requested EMI",
            f"₹{requested_emi:,.0f}",
        )

        c3.metric(
            "Prediction Confidence",
            f"{confidence * 100:.1f}%",
        )

        # -------------------------------------------------------------------
        # EMI comparison status
        # -------------------------------------------------------------------
        if affordability_status == "COMFORTABLE":

            st.markdown(
                f"""
                <div class="safe-box">
                    <b>✅ EMI Affordability</b><br>
                    Requested EMI of
                    <b>₹{requested_emi:,.0f}</b>
                    is within the recommended safe limit of
                    <b>₹{max_safe_emi:,.0f}</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif affordability_status in {
            "STRETCHED",
            "ABOVE_SAFE_LIMIT",
            "HIGH",
        }:

            st.markdown(
                f"""
                <div class="warning-box">
                    <b>⚠️ EMI Affordability Warning</b><br>
                    Requested EMI of
                    <b>₹{requested_emi:,.0f}</b>
                    exceeds the recommended safe EMI of
                    <b>₹{max_safe_emi:,.0f}</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="danger-box">
                    <b>❌ EMI Affordability Risk</b><br>
                    Requested EMI requires additional financial review.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # -------------------------------------------------------------------
        # Risk analysis + probability chart
        # -------------------------------------------------------------------
        left, right = st.columns([1, 1.15])

        with left:

            st.markdown(
                '<div class="section-title">Risk Analysis</div>',
                unsafe_allow_html=True,
            )

            checks = result.get("checks", [])

            if checks:

                for check in checks:

                    status = check.get(
                        "status",
                        "warning",
                    )

                    icon = {
                        "good": "✅",
                        "warning": "⚠️",
                        "poor": "❌",
                    }.get(
                        status,
                        "ℹ️",
                    )

                    label = check.get(
                        "label",
                        "Risk Factor",
                    )

                    detail = check.get(
                        "detail",
                        "",
                    )

                    st.write(
                        f"{icon} **{label}**: {detail}"
                    )

            else:
                st.info(
                    "No additional risk indicators were returned."
                )

        with right:

            st.markdown(
                '<div class="section-title">Class Probability Breakdown</div>',
                unsafe_allow_html=True,
            )

            probabilities = result.get(
                "probabilities",
                {},
            )

            if probabilities:

                prob_df = pd.DataFrame(
                    [
                        {
                            "Class": label.replace("_", " "),
                            "Probability": float(prob),
                        }
                        for label, prob in probabilities.items()
                    ]
                ).sort_values(
                    "Probability"
                )

                fig_prob = px.bar(
                    prob_df,
                    x="Probability",
                    y="Class",
                    orientation="h",
                    text=prob_df["Probability"].map(
                        lambda value: f"{value * 100:.1f}%"
                    ),
                    range_x=[0, 1],
                    labels={
                        "Probability": "Probability",
                        "Class": "",
                    },
                )

                fig_prob.update_traces(
                    textposition="outside"
                )

                fig_prob.update_layout(
                    height=280,
                    margin=dict(
                        l=0,
                        r=50,
                        t=10,
                        b=20,
                    ),
                )

                st.plotly_chart(
                    fig_prob,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

            else:

                st.info(
                    "Probability details are not available."
                )

        # -------------------------------------------------------------------
        # Recommendation
        # -------------------------------------------------------------------
        st.markdown(
            '<div class="section-title">Recommendation</div>',
            unsafe_allow_html=True,
        )

        if affordability_status == "COMFORTABLE":

            st.success(
                recommendation
            )

        elif eligibility == "Not_Eligible":

            st.error(
                recommendation
            )

        else:

            st.warning(
                recommendation
            )

        # -------------------------------------------------------------------
        # Application review / save
        # -------------------------------------------------------------------
        st.markdown(
            '<div class="section-title">📥 Application Review</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Save this assessment to the Admin (CRUD) queue for "
            "follow-up or underwriting review."
        )

        save_col1, save_col2 = st.columns([1.5, 1])

        with save_col1:

            if st.button(
                "📥 Log Application to Admin Queue",
                type="secondary",
                use_container_width=True,
            ):

                try:

                    application_record = {
                        "name": (
                            f"Customer-"
                            f"{np.random.randint(1000, 9999)}"
                        ),
                        **customer,
                        "eligibility_result": eligibility,
                        "max_safe_emi_result": max_safe_emi,
                    }

                    new_id = db.create_customer(
                        application_record
                    )

                    st.success(
                        f"Application #{new_id} has been logged "
                        "to the Admin queue for reviewer follow-up."
                    )

                except Exception as exc:

                    st.error(
                        f"Unable to save application: {exc}"
                    )

        with save_col2:

            if st.button(
                "🧹 Clear Assessment",
                use_container_width=True,
            ):

                st.session_state.pop(
                    "prediction_customer",
                    None,
                )

                st.session_state.pop(
                    "prediction_result",
                    None,
                )

                st.rerun()


# ===========================================================================
# PAGE: DATA EXPLORER
# ===========================================================================
# ===========================================================================
# PAGE: DATA EXPLORER
# ===========================================================================
elif page == "📊 Data Explorer":

    st.title("📊 Data Explorer")

    st.caption(
        "Interactive exploration of the applicant population behind the "
        "EMIPredict AI models."
    )

    df = load_cleaned_data()

    if df is None:
        st.error(
            "Cleaned dataset not found. Run "
            "`python src/preprocessing.py` first."
        )
        st.stop()

    # -----------------------------------------------------------------------
    # Sidebar filters
    # -----------------------------------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("🔎 Filters")

    scenario_options = sorted(
        df["emi_scenario"].dropna().unique().tolist()
    )

    eligibility_options = sorted(
        df["emi_eligibility"].dropna().unique().tolist()
    )

    selected_scenarios = st.sidebar.multiselect(
        "EMI Scenario",
        options=scenario_options,
        default=scenario_options,
    )

    selected_eligibility = st.sidebar.multiselect(
        "Eligibility",
        options=eligibility_options,
        default=eligibility_options,
    )

    salary_min = float(
        df["monthly_salary"].min()
    )

    salary_max = float(
        df["monthly_salary"].max()
    )

    salary_range = st.sidebar.slider(
        "Monthly Salary Range (₹)",
        min_value=int(salary_min),
        max_value=int(salary_max),
        value=(int(salary_min), int(salary_max)),
        step=1000,
    )

    # -----------------------------------------------------------------------
    # Apply filters
    # -----------------------------------------------------------------------
    filtered_df = df[
        df["emi_scenario"].isin(selected_scenarios)
        & df["emi_eligibility"].isin(selected_eligibility)
        & df["monthly_salary"].between(
            salary_range[0],
            salary_range[1],
        )
    ].copy()

    # -----------------------------------------------------------------------
    # KPI calculations
    # -----------------------------------------------------------------------
    total_records = len(filtered_df)

    if total_records > 0:

        eligible_pct = (
            (
                filtered_df["emi_eligibility"]
                .eq("Eligible")
                .mean()
            )
            * 100
        )

        high_risk_pct = (
            (
                filtered_df["emi_eligibility"]
                .eq("High_Risk")
                .mean()
            )
            * 100
        )

        not_eligible_pct = (
            (
                filtered_df["emi_eligibility"]
                .eq("Not_Eligible")
                .mean()
            )
            * 100
        )

        avg_max_emi = filtered_df[
            "max_monthly_emi"
        ].mean()

    else:

        eligible_pct = 0
        high_risk_pct = 0
        not_eligible_pct = 0
        avg_max_emi = 0

    # -----------------------------------------------------------------------
    # Filter summary
    # -----------------------------------------------------------------------
    st.markdown(
        f"**{total_records:,} applicants match the current filters** "
        f"(of {len(df):,} total)."
    )

    # -----------------------------------------------------------------------
    # KPI row
    # -----------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Eligible %",
        f"{eligible_pct:.1f}%",
    )

    c2.metric(
        "High Risk %",
        f"{high_risk_pct:.1f}%",
    )

    c3.metric(
        "Not Eligible %",
        f"{not_eligible_pct:.1f}%",
    )

    c4.metric(
        "Avg Max Monthly EMI",
        f"₹{avg_max_emi:,.0f}",
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Eligibility Overview",
            "Financial Correlations",
            "Demographics",
            "Raw Data",
        ]
    )

    # =======================================================================
    # TAB 1: ELIGIBILITY OVERVIEW
    # =======================================================================
    with tab1:

        if total_records == 0:

            st.info(
                "No records match the current filters."
            )

        else:

            col1, col2 = st.columns(2)

            # ---------------------------------------------------------------
            # Eligibility distribution
            # ---------------------------------------------------------------
            with col1:

                eligibility_counts = (
                    filtered_df[
                        "emi_eligibility"
                    ]
                    .value_counts()
                    .rename_axis("emi_eligibility")
                    .reset_index(name="count")
                )

                fig = px.pie(
                    eligibility_counts,
                    names="emi_eligibility",
                    values="count",
                    hole=0.45,
                    title="Eligibility Distribution",
                )

                fig.update_layout(
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                    height=380,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

            # ---------------------------------------------------------------
            # Eligibility by scenario
            # ---------------------------------------------------------------
            with col2:

                scenario_eligibility = (
                    filtered_df.groupby(
                        [
                            "emi_scenario",
                            "emi_eligibility",
                        ]
                    )
                    .size()
                    .reset_index(name="count")
                )

                scenario_totals = (
                    filtered_df.groupby(
                        "emi_scenario"
                    )
                    .size()
                    .reset_index(name="total")
                )

                scenario_eligibility = scenario_eligibility.merge(
                    scenario_totals,
                    on="emi_scenario",
                )

                scenario_eligibility["pct"] = (
                    scenario_eligibility["count"]
                    / scenario_eligibility["total"]
                    * 100
                )

                fig = px.bar(
                    scenario_eligibility,
                    x="emi_scenario",
                    y="pct",
                    barmode="stack",
                    title="Eligibility Rate by EMI Scenario",
                    labels={
                        "pct": "Percentage",
                        "emi_scenario": "EMI Scenario",
                    },
                
                )

                fig.update_xaxes(
                    tickangle=25
                )

                fig.update_layout(
                    height=380,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=60,
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

            # ---------------------------------------------------------------
            # Salary vs eligibility
            # ---------------------------------------------------------------
            col1, col2 = st.columns(2)

            # Sample data for faster rendering
            plot_df = filtered_df.sample(
                min(
                    20000,
                    len(filtered_df),
                ),
                random_state=42,
            )

            with col1:

                fig = px.box(
                    filtered_df,
                    x="emi_eligibility",
                    y="monthly_salary",
                    title="Eligibility vs Monthly Salary",
            
                )

                fig.update_layout(
                    height=360,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

            # ---------------------------------------------------------------
            # Credit score vs eligibility
            # ---------------------------------------------------------------
            with col2:

                fig = px.box(
                    filtered_df,
                    x="emi_eligibility",
                    y="credit_score",
                    title="Eligibility vs Credit Score",
                    
                )

                fig.update_layout(
                    height=360,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

    # =======================================================================
    # TAB 2: FINANCIAL CORRELATIONS
    # =======================================================================
    with tab2:

        if total_records == 0:

            st.info(
                "No records match the current filters."
            )

        else:

            numeric_df = filtered_df.select_dtypes(
                include=[np.number]
            )

            corr = numeric_df.corr()

            st.subheader(
                "Financial Feature Correlations"
            )

            col1, col2 = st.columns(
                [1, 1.2]
            )

            with col1:

                fig_corr = px.imshow(
                    corr,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap",
                )

                fig_corr.update_layout(
                    height=580,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=20,
                    ),
                )

                st.plotly_chart(
                    fig_corr,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

            with col2:

                plot_df = filtered_df.sample(
                    min(
                        20000,
                        len(filtered_df),
                    ),
                    random_state=42,
                )

                fig_scatter = px.scatter(
                    plot_df,
                    x="monthly_salary",
                    y="max_monthly_emi",
                    title="Monthly Salary vs Maximum Safe EMI",
                    labels={
                        "monthly_salary": "Monthly Salary (₹)",
                        "max_monthly_emi": "Maximum Monthly EMI (₹)",
                    },
                    opacity=0.65,
                    
                )

                fig_scatter.update_layout(
                    height=580,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=20,
                    ),
                )

                st.plotly_chart(
                    fig_scatter,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

    # =======================================================================
    # TAB 3: DEMOGRAPHICS
    # =======================================================================
    with tab3:

        if total_records == 0:

            st.info(
                "No records match the current filters."
            )

        else:

            col1, col2 = st.columns(2)

            # ---------------------------------------------------------------
            # Employment × Education
            # ---------------------------------------------------------------
            with col1:

                demographic_df = (
                    filtered_df.groupby(
                        [
                            "employment_type",
                            "education",
                        ]
                    )
                    .agg(
                        total=("emi_eligibility", "size"),
                        eligible_pct=(
                            "emi_eligibility",
                            lambda s: (
                                s.eq("Eligible").mean()
                                * 100
                            ),
                        ),
                    )
                    .reset_index()
                )

                fig = px.density_heatmap(
                    demographic_df,
                    x="employment_type",
                    y="education",
                    z="eligible_pct",
                    text_auto=".1f",
                    title=(
                        "Eligibility Rate: "
                        "Employment × Education"
                    ),
                    labels={
                        "eligible_pct": "Eligible %",
                        "employment_type": "Employment Type",
                        "education": "Education",
                    },
                )

                fig.update_layout(
                    height=430,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=20,
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

            # ---------------------------------------------------------------
            # Requested amount by scenario
            # ---------------------------------------------------------------
            with col2:

                fig = px.box(
                    filtered_df,
                    x="emi_scenario",
                    y="requested_amount",
                    title="Requested Amount by EMI Scenario",
                )

                fig.update_xaxes(
                    tickangle=25
                )

                fig.update_layout(
                    height=430,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=60,
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

            # ---------------------------------------------------------------
            # Age distribution
            # ---------------------------------------------------------------
            col1, col2 = st.columns(2)

            with col1:

                fig = px.histogram(
                    filtered_df,
                    x="age",
                    nbins=30,
                    title="Applicant Age Distribution",
                )

                fig.update_layout(
                    height=350
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

            with col2:

                fig = px.box(
                    filtered_df,
                    x="employment_type",
                    y="years_of_employment",
                    color="employment_type",
                    title="Employment Stability",
                )

                fig.update_layout(
                    height=350
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

    # =======================================================================
    # TAB 4: RAW DATA
    # =======================================================================
    with tab4:

        st.subheader("Filtered Applicant Records")

        st.caption(
            "Use the table's internal scrollbar to inspect records. "
            "The page itself will remain compact."
        )

        if total_records == 0:

            st.info(
                "No records match the current filters."
            )

        else:

            # Keep raw-data preview manageable
            preview_df = filtered_df.head(1000)

            st.dataframe(
                preview_df,
                height=520,
                use_container_width=True,
            )

            st.caption(
                f"Showing first {len(preview_df):,} records "
                f"of {total_records:,} filtered records."
            )

            csv_data = filtered_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Filtered Data (CSV)",
                data=csv_data,
                file_name="emipredict_filtered_data.csv",
                mime="text/csv",
            )

# ===========================================================================
# PAGE: MODEL PERFORMANCE
# ===========================================================================
elif page == "🤖 Model Performance":

    st.title("Model Performance & MLflow Tracking")

    st.caption(
        "Comparison of classification and regression models used by EMIPredict AI."
    )

    clf_comp, reg_comp = load_comparison_tables()

    # -----------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------
    st.subheader("🎯 Classification Models — EMI Eligibility")

    if clf_comp is not None:

        st.dataframe(
            clf_comp.style.highlight_max(
                axis=0,
                color="lightgreen",
            ),
            use_container_width=True,
        )

        if "f1_macro" in clf_comp.columns:

            best_clf = clf_comp["f1_macro"].idxmax()

            st.success(
                f"Best classification model: **{best_clf}** "
                f"(selected using macro F1 to account for class imbalance)"
            )

    else:

        st.info(
            "Classification results not found. Run "
            "`python src/train_classification.py` first."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Regression
    # -----------------------------------------------------------------------
    st.subheader("📈 Regression Models — Maximum Safe EMI")

    if reg_comp is not None:

        st.dataframe(
            reg_comp.style.highlight_min(
                axis=0,
                subset=[
                    column
                    for column in [
                        "rmse",
                        "mae",
                        "mape",
                    ]
                    if column in reg_comp.columns
                ],
                color="lightgreen",
            ),
            use_container_width=True,
        )

        if "rmse" in reg_comp.columns:

            best_reg = reg_comp["rmse"].idxmin()

            st.success(
                f"Best regression model: **{best_reg}** "
                f"(selected using lowest RMSE)"
            )

    else:

        st.info(
            "Regression results not found. Run "
            "`python src/train_regression.py` first."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Your actual final results
    # -----------------------------------------------------------------------
    st.subheader("🏆 Current Best Model Results")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("#### Classification — XGBoost")

        st.metric(
            "Test Accuracy",
            "98.20%",
        )

        st.metric(
            "Test Macro F1",
            "92.03%",
        )

        st.metric(
            "Test ROC-AUC",
            "99.69%",
        )

    with c2:

        st.markdown("#### Regression — XGBoost")

        st.metric(
            "Test RMSE",
            "₹565",
        )

        st.metric(
            "Test MAE",
            "₹173",
        )

        st.metric(
            "Test R²",
            "99.47%",
        )

    st.divider()

    # -----------------------------------------------------------------------
    # MLflow instructions
    # -----------------------------------------------------------------------
    st.subheader("🔬 MLflow Experiment Tracking")

    st.write(
        "Training runs are tracked in the MLflow SQLite backend. "
        "Parameters, metrics and model artifacts are recorded for "
        "classification and regression experiments."
    )

    st.code(
        "mlflow ui --backend-store-uri sqlite:///mlflow.db",
        language="bash",
    )

    st.caption(
        "Run the command above from the project directory, then open "
        "the MLflow dashboard at http://localhost:5000."
    )

    st.info(
        "Registered production models can be added to the MLflow Model "
        "Registry after final model validation."
    )


# ===========================================================================
# PAGE: ADMIN / CRUD
# ===========================================================================
elif page == "🗂️ Admin (CRUD)":

    st.title("Admin — Customer Data Management")

    st.caption(
        "Manage saved customer assessments and underwriting records."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📋 View All",
            "➕ Add Customer",
            "✏️ Update / 🗑️ Delete",
        ]
    )

    # -----------------------------------------------------------------------
    # VIEW
    # -----------------------------------------------------------------------
    with tab1:

        records = db.read_all_customers()

        st.write(
            f"**Total records: {len(records)}**"
        )

        if len(records) == 0:

            st.info(
                "No customer records available yet."
            )

        else:

            st.dataframe(
                records,
                use_container_width=True,
            )

    # -----------------------------------------------------------------------
    # CREATE
    # -----------------------------------------------------------------------
    with tab2:

        with st.form("add_customer_form"):

            name = st.text_input(
                "Customer Name"
            )

            c1, c2, c3 = st.columns(3)

            age = c1.number_input(
                "Age",
                18,
                75,
                30,
            )

            gender = c2.selectbox(
                "Gender",
                ["Male", "Female"],
            )

            monthly_salary = c3.number_input(
                "Monthly Salary (₹)",
                0,
                1_000_000,
                50_000,
            )

            emi_scenario = st.selectbox(
                "EMI Scenario",
                [
                    "E-commerce Shopping EMI",
                    "Home Appliances EMI",
                    "Vehicle EMI",
                    "Personal Loan EMI",
                    "Education EMI",
                ],
            )

            credit_score = st.number_input(
                "Credit Score",
                300,
                850,
                700,
            )

            add_submitted = st.form_submit_button(
                "Add Customer",
                type="primary",
            )

        if add_submitted:

            if not name.strip():

                st.error(
                    "Customer name is required."
                )

            else:

                try:

                    new_id = db.create_customer(
                        {
                            "name": name,
                            "age": age,
                            "gender": gender,
                            "marital_status": "Single",
                            "education": "Graduate",
                            "monthly_salary": monthly_salary,
                            "employment_type": "Private",
                            "years_of_employment": 1,
                            "company_type": "MNC",
                            "house_type": "Rented",
                            "monthly_rent": 0,
                            "family_size": 1,
                            "dependents": 0,
                            "school_fees": 0,
                            "college_fees": 0,
                            "travel_expenses": 0,
                            "groceries_utilities": 0,
                            "other_monthly_expenses": 0,
                            "existing_loans": "No",
                            "current_emi_amount": 0,
                            "credit_score": credit_score,
                            "bank_balance": 0,
                            "emergency_fund": 0,
                            "emi_scenario": emi_scenario,
                            "requested_amount": 0,
                            "requested_tenure": 12,
                            "eligibility_result": None,
                            "max_safe_emi_result": None,
                        }
                    )

                    st.success(
                        f"Customer #{new_id} added successfully."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Unable to add customer: {exc}"
                    )

    # -----------------------------------------------------------------------
    # UPDATE / DELETE
    # -----------------------------------------------------------------------
    with tab3:

        records = db.read_all_customers()

        if len(records) == 0:

            st.info(
                "No customer records yet."
            )

        else:

            selected_id = st.selectbox(
                "Select Customer ID",
                records["id"].tolist(),
            )

            customer_record = db.read_customer(
                selected_id
            )

            if customer_record is None:

                st.warning(
                    "Selected customer could not be found."
                )

            else:

                col1, col2 = st.columns(2)

                # -----------------------------------------------------------
                # UPDATE
                # -----------------------------------------------------------
                with col1:

                    st.markdown(
                        "### ✏️ Update Customer"
                    )

                    new_salary = st.number_input(
                        "Monthly Salary (₹)",
                        min_value=0,
                        max_value=1_000_000,
                        value=int(
                            customer_record[
                                "monthly_salary"
                            ] or 0
                        ),
                    )

                    new_credit = st.number_input(
                        "Credit Score",
                        min_value=300,
                        max_value=850,
                        value=int(
                            customer_record[
                                "credit_score"
                            ] or 700
                        ),
                    )

                    if st.button(
                        "Update Customer",
                        use_container_width=True,
                    ):

                        try:

                            db.update_customer(
                                selected_id,
                                {
                                    "monthly_salary": new_salary,
                                    "credit_score": new_credit,
                                },
                            )

                            st.success(
                                f"Customer #{selected_id} updated successfully."
                            )

                            st.rerun()

                        except Exception as exc:

                            st.error(
                                f"Unable to update customer: {exc}"
                            )

                # -----------------------------------------------------------
                # DELETE
                # -----------------------------------------------------------
                with col2:

                    st.markdown(
                        "### 🗑️ Delete Customer"
                    )

                    st.warning(
                        f"This will permanently delete customer "
                        f"#{selected_id}."
                    )

                    if st.button(
                        "🗑️ Delete Customer",
                        type="primary",
                        use_container_width=True,
                    ):

                        try:

                            db.delete_customer(
                                selected_id
                            )

                            st.success(
                                f"Customer #{selected_id} deleted successfully."
                            )

                            st.rerun()

                        except Exception as exc:

                            st.error(
                                f"Unable to delete customer: {exc}"
                            )