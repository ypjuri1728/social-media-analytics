# 📊 Data-Driven Social Engagement Analytics System
### Using Sentiment Analysis and Engagement Prediction

## 📌 Overview

This project analyzes social media engagement data using data science techniques including
data preprocessing, exploratory data analysis, sentiment analysis, virality scoring, and
machine learning-based engagement prediction.

> **Submitted by:** Priyanshi Yadav  
> **Program:** Unlox Data Science Program  
> **Submission Date:** 5 June 2026

---

## 📁 Project Structure

```
📦 social-media-analytics
├── 📓 Social_Media_Analytics.ipynb   # Main Jupyter Notebook
├── 📄 Social_media_Dataset.csv       # Dataset (12,000 records, 28 attributes)
├── 📄 Major_Project_Report.pdf       # Full project report
└── 📄 README.md                      # Project documentation
```

---

## 🎯 Objectives

- Analyze social media engagement data across multiple platforms
- Study audience sentiment and emotional responses
- Identify highly engaging and viral content
- Compare engagement performance across platforms
- Apply machine learning for engagement rate prediction
- Generate actionable insights for content optimization

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Records** | ~12,000 |
| **Attributes** | 28 |
| **Format** | CSV |

**Key Features:**
`Platform` · `Likes` · `Comments` · `Shares` · `Engagement Rate` · `Sentiment Label` · `Emotion Type` · `Campaign Info` · `Topic Category`

---

## 🔧 Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation & preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning model |
| Jupyter Notebook | Development environment |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Run the Project
```bash
# Clone the repository
git clone https://github.com/your-username/social-media-analytics.git
cd social-media-analytics

# Launch Jupyter Notebook
jupyter notebook Social_Media_Analytics.ipynb
```

---

## 📊 Project Modules

### 1. 🧹 Data Loading & Preprocessing
- Loaded dataset using Pandas
- Inspected structure with `df.info()` and `df.head()`
- Handled missing values using `df.isnull().sum()`
- Removed duplicate records

### 2. 📈 Exploratory Data Analysis (EDA)
- Statistical summary using `df.describe()`
- Platform distribution analysis
- Engagement metric distributions

### 3. 💬 Sentiment Analysis
- Classified user responses into **Positive**, **Negative**, and **Neutral**
- Visualized sentiment distribution using bar charts
- Insights into audience emotional behavior

### 4. 🏆 Platform Analysis
- Compared average engagement rates across YouTube, Facebook, Twitter, Reddit, and Instagram
- Identified top-performing platforms

### 5. 🔥 Virality Analysis
Custom Virality Score formula:

```
Virality Score = Likes + (2 × Comments) + (3 × Shares)
```

Shares carry the highest weight as they maximize content reach.

### 6. 🤖 Machine Learning — Engagement Prediction
- **Model:** Linear Regression
- **Features:** Likes, Comments, Shares
- **Target:** Engagement Rate
- Dataset split into training and testing sets for evaluation

---

## 📸 Sample Visualizations

| Chart | Description |
|---|---|
| Platform Distribution | Post count across platforms |
| Sentiment Distribution | Negative / Positive / Neutral breakdown |
| Virality Scores | Top-performing content by score |
| Engagement Prediction | Predicted vs actual engagement rate |

---

## 🔮 Future Scope

- Real-time social media API integration
- Advanced NLP (emotion detection, topic modeling)
- Deep learning models (ANN, RNN, Transformers)
- Interactive dashboards with Streamlit / Power BI
- Recommendation systems for optimal posting strategies
- Automated sentiment monitoring pipeline

---

## 📚 References

- Python Documentation — https://docs.python.org
- Pandas Documentation — https://pandas.pydata.org/docs
- NumPy Documentation — https://numpy.org/doc
- Matplotlib Documentation — https://matplotlib.org/stable/index.html
- Scikit-learn Documentation — https://scikit-learn.org/stable

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ by Priyanshi Yadav</p>
