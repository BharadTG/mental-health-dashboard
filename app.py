import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Title
st.title("Mental Health in Tech Dashboard")
st.write("An interactive tool to explore mental health trends in the tech industry using survey data.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/BharadTG/mental-health-dashboard/main/survey.csv")
    
    # Clean age (valid range only)
    df = df[(df["Age"] >= 18) & (df["Age"] <= 70)].copy()

    # Clean relevant columns
    df = df[[
        "Age", "Gender", "self_employed", "family_history", "work_interfere",
        "no_employees", "remote_work", "tech_company", "benefits", "care_options",
        "wellness_program", "seek_help", "anonymity", "leave", "mental_health_consequence",
        "phys_health_consequence", "coworkers", "supervisor", "mental_health_interview",
        "phys_health_interview", "mental_vs_physical", "obs_consequence", "treatment"
    ]].dropna()

    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")
company_size = st.sidebar.selectbox("Company Size", ["All"] + list(df["no_employees"].unique()))
remote_option = st.sidebar.selectbox("Remote Work", ["All"] + list(df["remote_work"].unique()))

filtered_df = df.copy()
if company_size != "All":
    filtered_df = filtered_df[filtered_df["no_employees"] == company_size]
if remote_option != "All":
    filtered_df = filtered_df[filtered_df["remote_work"] == remote_option]

# Show dataset
if st.checkbox("Show Raw Data"):
    st.write(filtered_df)

# Age Distribution
st.subheader("Age Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(filtered_df["Age"], kde=True, color="skyblue", edgecolor="black", ax=ax1)
st.pyplot(fig1)

# Treatment vs. Family History
st.subheader("Treatment vs. Family History")
fig2, ax2 = plt.subplots()
sns.countplot(data=filtered_df, x="family_history", hue="treatment", palette="Set2", ax=ax2)
st.pyplot(fig2)

# Treatment vs. Benefits
st.subheader("Treatment vs. Mental Health Benefits")
fig3, ax3 = plt.subplots()
sns.countplot(data=filtered_df, x="benefits", hue="treatment", palette="Set1", ax=ax3)
st.pyplot(fig3)

# Logistic Regression (optional)
if st.checkbox("Run Treatment Prediction Model"):
    model_df = filtered_df[["family_history", "work_interfere", "benefits", "Gender", "care_options", 
                            "remote_work", "no_employees", "Age", "treatment"]].copy()
    for col in model_df.columns:
        model_df[col] = LabelEncoder().fit_transform(model_df[col])

    X = model_df.drop("treatment", axis=1)
    y = model_df["treatment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    st.write("Model Accuracy:", round(model.score(X_test, y_test), 2))
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": abs(model.coef_[0])
    }).sort_values("Importance", ascending=False)

    st.subheader("Feature Importance")
    st.bar_chart(importance_df.set_index("Feature"))

