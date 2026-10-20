# ============================================================
# INSIGHTFORGE
# PART 1 — AI BUSINESS INTELLIGENCE ASSISTANT
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import pandas as pd
from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv("sales_data.csv")

print(df.head())

# ============================================================
# 4. DATA PREPARATION
# ============================================================

# Convert Date Column

df["Date"] = pd.to_datetime(df["Date"])

# Create Month and Year Columns

df["Month"] = df["Date"].dt.month_name()

df["Year"] = df["Date"].dt.year

# Check Missing Values

print("\nMissing Values:\n")

print(df.isnull().sum())

# Remove Missing Values

df = df.dropna()

# ============================================================
# 5. BUSINESS ANALYSIS
# ============================================================

# Total Sales

total_sales = df["Sales"].sum()

print("\nTotal Sales:\n")

print(total_sales)

# Monthly Sales

monthly_sales = (
    df.groupby("Month")["Sales"]
    .sum()
)

print("\nMonthly Sales:\n")

print(monthly_sales)

# Product Analysis

product_sales = (
    df.groupby("Product")["Sales"]
    .sum()
)

print("\nProduct Sales:\n")

print(product_sales)

# Regional Analysis

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
)

print("\nRegional Sales:\n")

print(region_sales)

# Customer Gender Analysis

gender_sales = (
    df.groupby("Customer_Gender")["Sales"]
    .mean()
)

print("\nGender Analysis:\n")

print(gender_sales)

# ============================================================
# 6. AGE GROUP ANALYSIS
# ============================================================

df["Age_Group"] = pd.cut(
    df["Customer_Age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "55+"
    ]
)

age_group_sales = (
    df.groupby("Age_Group")["Sales"]
    .mean()
)

print("\nAge Group Analysis:\n")

print(age_group_sales)

# ============================================================
# 7. STATISTICAL ANALYSIS
# ============================================================

average_sales = df["Sales"].mean()

median_sales = df["Sales"].median()

sales_std = df["Sales"].std()

print("\nAverage Sales:\n")

print(average_sales)

print("\nMedian Sales:\n")

print(median_sales)

print("\nStandard Deviation:\n")

print(sales_std)

# ============================================================
# 8. BUSINESS SUMMARY
# ============================================================

summary = f"""

BUSINESS SUMMARY

Total Sales:
{total_sales}

Average Sales:
{average_sales}

Median Sales:
{median_sales}

Sales Standard Deviation:
{sales_std}

Top Product:
{product_sales.idxmax()}

Best Region:
{region_sales.idxmax()}

Best Age Group:
{age_group_sales.idxmax()}

"""

print(summary)

# ============================================================
# 9. MEMORY INTEGRATION
# ============================================================

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ============================================================
# 10. INITIALIZE LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

# ============================================================
# 11. PROMPT ENGINEERING
# ============================================================

prompt = PromptTemplate(

    input_variables=[
        "summary",
        "question"
    ],

    template="""

You are an AI Business Intelligence Assistant.

Use this business summary:

{summary}

Question:
{question}

Generate:
- Business insights
- Recommendations
- Trend analysis
- Decision-making suggestions

"""
)

# ============================================================
# 12. QUESTION ANSWERING SYSTEM
# ============================================================

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":

        print("\nExiting InsightForge AI...\n")

        break

    # Create Prompt

    formatted_prompt = prompt.format(
        summary=summary,
        question=question
    )

    # Generate AI Response

    response = llm.invoke(formatted_prompt)

    # Display Response

    print("\nAI Response:\n")

    print(response.content)

# ============================================================
# 13. FINAL CONCLUSION
# ============================================================

print("""

InsightForge AI successfully:

✔ Analyzed business data

✔ Generated intelligent insights

✔ Applied Prompt Engineering

✔ Integrated Memory

✔ Built an AI-powered business assistant

# ============================================================
# END OF PART 1
# ============================================================

""")
