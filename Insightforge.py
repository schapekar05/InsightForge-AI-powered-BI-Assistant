# ============================================================
# INSIGHTFORGE AI 
# PART 2 - ADVANCED BUSINESS INTELLIGENCE DASHBOARD
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="InsightForge AI",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# PLOTLY THEME
# ============================================================

pio.templates.default = "plotly_white"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Main App */
.stApp {
    background-color: #f8fafc;
    color: #111827;
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffffff, #f1f5f9);
    border-right: 1px solid #e2e8f0;
}

/* Sidebar Text */
[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* KPI Cards */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e2e8f0;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: 1px solid #e2e8f0;
}

.stTabs [aria-selected="true"] {
    background-color: #4f46e5 !important;
    color: white !important;
}

/* Buttons */
.stButton>button {
    background-color: #4f46e5;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

/* Chat */
[data-testid="stChatMessage"] {
    background-color: white;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #e2e8f0;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

/* Chart Cards */
.chart-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("sales_data.csv")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.dropna(subset=["Date", "Sales"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.strftime("%B")

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🎛 Dashboard Filters")

region_filter = st.sidebar.multiselect(
    "Select Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

product_filter = st.sidebar.multiselect(
    "Select Product",
    df["Product"].unique(),
    default=df["Product"].unique()
)

filtered_df = df[
    (df["Region"].isin(region_filter)) &
    (df["Product"].isin(product_filter))
]

# ============================================================
# KPI CALCULATIONS
# ============================================================

total_sales = filtered_df["Sales"].sum()

avg_sales = filtered_df["Sales"].mean()

top_product = (
    filtered_df.groupby("Product")["Sales"]
    .sum()
    .idxmax()
)

best_region = (
    filtered_df.groupby("Region")["Sales"]
    .sum()
    .idxmax()
)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style='
background: linear-gradient(to right, #f8fafc, #e2e8f0);
padding:30px;
border-radius:16px;
border-left:8px solid #6366f1;
margin-bottom:25px;
box-shadow:0px 4px 12px rgba(0,0,0,0.05);
'>

<h1 style='
color:#111827;
font-size:42px;
margin-bottom:8px;
'>
📈 InsightForge Analytics
</h1>

<p style='
color:#475569;
font-size:17px;
'>
Interactive AI Powered Business Intelligence System
</p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Executive Overview",
    "📈 Sales Analytics",
    "👥 Customer Insights",
    "🤖 AI Assistant",
    "📋 Raw Data"
])

# ============================================================
# TAB 1 - EXECUTIVE OVERVIEW
# ============================================================

with tab1:

    st.subheader("📌 Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    col2.metric(
        "📊 Average Sales",
        f"${avg_sales:,.0f}"
    )

    col3.metric(
        "🏆 Top Product",
        top_product
    )

    col4.metric(
        "🌍 Best Region",
        best_region
    )

    st.markdown("---")

    trend = (
        filtered_df.groupby(
            filtered_df["Date"].dt.to_period("M")
        )["Sales"]
        .sum()
        .reset_index()
    )

    trend["Date"] = trend["Date"].astype(str)

    fig_trend = px.line(
        trend,
        x="Date",
        y="Sales",
        markers=True,
        line_shape="spline",
        color_discrete_sequence=["#6366f1"]
    )

    fig_trend.update_layout(
        title="📈 Sales Trend Over Time",
        height=500,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#111827")
    )

    st.plotly_chart(fig_trend, use_container_width=True)

# ============================================================
# TAB 2 - SALES ANALYTICS
# ============================================================

with tab2:

    st.subheader("📈 Sales Analytics")

    col1, col2 = st.columns(2)

    # Product Performance

    with col1:

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        product_sales = (
            filtered_df.groupby("Product")["Sales"]
            .sum()
            .reset_index()
        )

        fig_product = px.bar(
            product_sales,
            x="Sales",
            y="Product",
            orientation="h",
            color="Sales",
            color_continuous_scale="Sunsetdark"
        )

        fig_product.update_layout(
            title="📦 Product Performance",
            height=500,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(color="#111827"),
            xaxis=dict(gridcolor="#e2e8f0")
        )

        st.plotly_chart(fig_product, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Region Analysis

    with col2:

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        region_sales = (
            filtered_df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig_region = px.pie(
            region_sales,
            names="Region",
            values="Sales",
            hole=0.55,
            color_discrete_sequence=[
                "#6366f1",
                "#10b981",
                "#f59e0b",
                "#ef4444",
                "#8b5cf6"
            ]
        )

        fig_region.update_layout(
            title="🌍 Regional Analysis",
            height=500,
            paper_bgcolor="#ffffff",
            font=dict(color="#111827")
        )

        st.plotly_chart(fig_region, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ========================================================
    # YEAR → MONTH DRILLDOWN
    # ========================================================

    st.subheader("📅 Year → Month Drilldown")

    selected_year = st.selectbox(
        "Select Year",
        sorted(filtered_df["Year"].unique())
    )

    year_df = filtered_df[
        filtered_df["Year"] == selected_year
    ]

    month_sales = (
        year_df.groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    month_order = [
        "January","February","March","April",
        "May","June","July","August",
        "September","October","November","December"
    ]

    month_sales["Month"] = pd.Categorical(
        month_sales["Month"],
        categories=month_order,
        ordered=True
    )

    month_sales = month_sales.sort_values("Month")

    fig_month = px.bar(
        month_sales,
        x="Month",
        y="Sales",
        color="Sales",
        color_continuous_scale="Turbo"
    )

    fig_month.update_layout(
        title=f"📊 Monthly Sales Breakdown - {selected_year}",
        height=500,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#111827"),
        yaxis=dict(gridcolor="#e2e8f0")
    )

    st.plotly_chart(fig_month, use_container_width=True)

# ============================================================
# TAB 3 - CUSTOMER INSIGHTS
# ============================================================

with tab3:

    st.subheader("👥 Customer Insights")

    col1, col2 = st.columns(2)

    # Age Segmentation

    with col1:

        bins = [0, 25, 35, 45, 55, 100]

        labels = [
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "55+"
        ]

        filtered_df["Age_Group"] = pd.cut(
            filtered_df["Customer_Age"],
            bins=bins,
            labels=labels
        )

        age_sales = (
            filtered_df.groupby("Age_Group")["Sales"]
            .mean()
            .reset_index()
        )

        fig_age = px.bar(
            age_sales,
            x="Age_Group",
            y="Sales",
            color="Sales",
            color_continuous_scale="Aggrnyl"
        )

        fig_age.update_layout(
            title="👥 Customer Age Segmentation",
            height=500,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(color="#111827")
        )

        st.plotly_chart(fig_age, use_container_width=True)

    # Gender Analysis

    with col2:

        gender_sales = (
            filtered_df.groupby("Customer_Gender")["Sales"]
            .mean()
            .reset_index()
        )

        fig_gender = px.bar(
            gender_sales,
            x="Customer_Gender",
            y="Sales",
            color="Customer_Gender",
            color_discrete_sequence=[
                "#06b6d4",
                "#f97316"
            ]
        )

        fig_gender.update_layout(
            title="👤 Customer Gender Analysis",
            height=500,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(color="#111827")
        )

        st.plotly_chart(fig_gender, use_container_width=True)

# ============================================================
# TAB 4 - AI ASSISTANT
# ============================================================

with tab4:

    st.subheader("🤖 AI Business Assistant")

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=api_key
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_question = st.chat_input(
            "Ask business questions..."
        )

        if user_question:

            st.session_state.messages.append({
                "role": "user",
                "content": user_question
            })

            prompt = f"""
            You are an expert AI Business Analyst.

            Total Sales: {total_sales}

            Average Sales: {avg_sales}

            Top Product: {top_product}

            Best Region: {best_region}

            User Question:
            {user_question}
            """

            with st.spinner("Generating AI insights..."):

                response = llm.invoke(prompt)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content
            })

            with st.chat_message("assistant"):
                st.write(response.content)

    else:

        st.warning("⚠️ OPENAI_API_KEY not found in .env file")

# ============================================================
# TAB 5 - RAW DATA
# ============================================================

with tab5:

    st.subheader("📋 Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style="
background:white;
padding:20px;
border-radius:15px;
border:1px solid #e2e8f0;
text-align:center;
margin-top:20px;
">

<h3 style='color:#111827; margin-bottom:5px;'>
🚀 InsightForge AI
</h3>

<p style='color:#64748b; font-size:15px;'>
Advanced Business Intelligence & Analytics Platform
</p>

<p style='color:#94a3b8; font-size:13px;'>
Built with Streamlit • Plotly • OpenAI • Machine Learning
</p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# END OF INSIGHTFORGE AI DASHBOARD - PART 2
# ============================================================
