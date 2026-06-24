import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Social Media Analytics", page_icon="📊", layout="wide")

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

@st.cache_data
def load_data():
    df = pd.read_csv("Social_media_Dataset.csv")
    return df

df = load_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Dataset", "📈 EDA", "💬 Sentiment", "🔥 Virality", "🤖 ML Prediction"
])

with tab1:
    st.subheader("Raw Dataset")
    st.write(f"Total Records: **{df.shape[0]}** | Total Columns: **{df.shape[1]}**")
    st.dataframe(df.head(50), use_container_width=True)
    st.subheader("Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

with tab2:
    st.subheader("Platform Distribution")
    platform_counts = df["Platform"].value_counts().reset_index()
    platform_counts.columns = ["Platform", "Count"]
    fig1 = px.bar(platform_counts, x="Platform", y="Count",
                  color="Platform", title="Posts per Platform")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Engagement Rate Distribution")
    fig2 = px.histogram(df, x="Engagement_Rate",
                        title="Engagement Rate Distribution", nbins=30)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df["Sentiment_Label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    fig3 = px.bar(sentiment_counts, x="Sentiment", y="Count",
                  color="Sentiment",
                  color_discrete_map={
                      "Positive": "#2ecc71",
                      "Negative": "#e74c3c",
                      "Neutral": "#f39c12"
                  },
                  title="Positive / Negative / Neutral")
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    for i, row in sentiment_counts.iterrows():
        [col1, col2, col3][i].metric(label=row["Sentiment"], value=row["Count"])

with tab4:
    st.subheader("Virality Score Analysis")
    st.markdown("**Formula:** `Virality Score = Likes + (2 × Comments) + (3 × Shares)`")
    df["Virality_Score"] = df["Likes"] + (2 * df["Comments"]) + (3 * df["Shares"])
    top10 = df.nlargest(10, "Virality_Score")[["Platform", "Likes", "Comments", "Shares", "Virality_Score"]]
    st.dataframe(top10, use_container_width=True)
    fig4 = px.bar(top10, x="Virality_Score", y=top10.index.astype(str),
                  orientation="h", title="Top 10 Viral Posts", color="Virality_Score")
    st.plotly_chart(fig4, use_container_width=True)

with tab5:
    st.subheader("Engagement Rate Prediction — Linear Regression")
    st.markdown("**Features:** Likes, Comments, Shares → **Target:** Engagement Rate")

    X = df[["Likes", "Comments", "Shares"]]
    y = df["Engagement_Rate"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("Mean Squared Error", f"{mse:.4f}")

    fig5 = px.scatter(x=y_test, y=y_pred,
                      labels={"x": "Actual", "y": "Predicted"},
                      title="Actual vs Predicted Engagement Rate")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Try Live Prediction")
    l = st.slider("Likes", 0, 10000, 500)
    c = st.slider("Comments", 0, 1000, 50)
    s = st.slider("Shares", 0, 1000, 30)
    prediction = model.predict([[l, c, s]])[0]
    st.success(f"Predicted Engagement Rate: **{prediction:.4f}**")
