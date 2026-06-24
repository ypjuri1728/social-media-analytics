import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Social Media Analytics",
    page_icon="📊",
    layout="wide"
)

# ===== HEADER =====
st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e, #0f3460);
                padding: 2rem; border-radius: 12px;
                border-left: 5px solid #e94560; margin-bottom: 1.5rem;">
        <h1 style="color:white; margin:0;">📊 Social Media Engagement Analytics</h1>
        <p style="color:#a0aec0; margin:0.5rem 0 0 0;">
            Sentiment Analysis · Virality Scoring · Engagement Prediction
        </p>
    </div>
""", unsafe_allow_html=True)

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    df = pd.read_csv("Social_media_Dataset.csv")
    return df

df = load_data()

# ===== TABS =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Dataset",
    "📈 EDA",
    "💬 Sentiment",
    "🔥 Virality",
    "🤖 ML Prediction"
])

# ===== TAB 1 — DATASET =====
with tab1:
    st.subheader("Raw Dataset")
    st.write(f"Total Records: **{df.shape[0]}** | Total Columns: **{df.shape[1]}**")
    st.dataframe(df.head(50), use_container_width=True)
    st.subheader("Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# ===== TAB 2 — EDA =====
with tab2:
    st.subheader("Platform Distribution")
    platform_counts = df["Platform"].value_counts()
    fig, ax = plt.subplots()
    platform_counts.plot(kind="bar", ax=ax, color="#e94560")
    ax.set_xlabel("Platform")
    ax.set_ylabel("Number of Posts")
    ax.set_title("Posts per Platform")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Engagement Rate Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df["Engagement_Rate"], bins=30, color="#0f3460", edgecolor="white")
    ax2.set_xlabel("Engagement Rate")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

# ===== TAB 3 — SENTIMENT =====
with tab3:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df["Sentiment_Label"].value_counts()
    fig3, ax3 = plt.subplots()
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    sentiment_counts.plot(kind="bar", ax=ax3, color=colors)
    ax3.set_xlabel("Sentiment")
    ax3.set_ylabel("Count")
    ax3.set_title("Positive / Negative / Neutral")
    plt.xticks(rotation=0)
    st.pyplot(fig3)

    col1, col2, col3 = st.columns(3)
    for i, (label, count) in enumerate(sentiment_counts.items()):
        [col1, col2, col3][i].metric(label=label, value=count)

# ===== TAB 4 — VIRALITY =====
with tab4:
    st.subheader("Virality Score Analysis")
    st.markdown("**Formula:** `Virality Score = Likes + (2 × Comments) + (3 × Shares)`")

    df["Virality_Score"] = df["Likes"] + (2 * df["Comments"]) + (3 * df["Shares"])

    top10 = df.nlargest(10, "Virality_Score")[["Platform", "Likes", "Comments", "Shares", "Virality_Score"]]
    st.dataframe(top10, use_container_width=True)

    fig4, ax4 = plt.subplots()
    ax4.barh(range(10), top10["Virality_Score"], color="#e94560")
    ax4.set_yticks(range(10))
    ax4.set_yticklabels([f"Post {i+1}" for i in range(10)])
    ax4.set_xlabel("Virality Score")
    ax4.set_title("Top 10 Viral Posts")
    st.pyplot(fig4)

# ===== TAB 5 — ML =====
with tab5:
    st.subheader("Engagement Rate Prediction — Linear Regression")
    st.markdown("**Features:** Likes, Comments, Shares → **Target:** Engagement Rate")

    features = ["Likes", "Comments", "Shares"]
    X = df[features]
    y = df["Engagement_Rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("Mean Squared Error", f"{mse:.4f}")

    fig5, ax5 = plt.subplots()
    ax5.scatter(y_test, y_pred, alpha=0.5, color="#0f3460")
    ax5.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--")
    ax5.set_xlabel("Actual Engagement Rate")
    ax5.set_ylabel("Predicted Engagement Rate")
    ax5.set_title("Actual vs Predicted")
    st.pyplot(fig5)

    # Live Predictor
    st.markdown("---")
    st.subheader("🎯 Try Live Prediction")
    l = st.slider("Likes", 0, 10000, 500)
    c = st.slider("Comments", 0, 1000, 50)
    s = st.slider("Shares", 0, 1000, 30)

    prediction = model.predict([[l, c, s]])[0]
    st.success(f"Predicted Engagement Rate: **{prediction:.4f}**")
