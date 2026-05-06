# 🎬 Movie Recommendation System

> Machine Learning Final Project 2026 | TMDB Dataset

## 📌 Problem & Motivation
Users face difficulty choosing movies from millions of available options on streaming platforms. This project builds an intelligent recommendation system using machine learning to solve the movie discovery problem automatically.

## 📊 Dataset
- **Source**: TMDB All Movies Dataset (Kaggle)
- **Original size**: 1,000,000+ rows, 28 columns
- **After cleaning**: 19 columns, 0 missing values in core features

## 🧹 Data Cleaning Steps
| Step | Action | Reason |
|---|---|---|
| 1 | Dropped 7 sparse columns | >70% missing values |
| 2 | Replaced 0 in budget/revenue/runtime with NaN | Zeros = missing data |
| 3 | Filtered runtime > 300 min | Outliers e.g. 14400 min art films |
| 4 | Converted release_date to datetime | Extract release_year |
| 5 | Filled missing text columns with empty string | Avoid NaN errors in TF-IDF |
| 6 | Created combined tags feature | genres + cast + director + overview + tagline |

## 📈 EDA Highlights
- Drama and Comedy dominate the TMDB catalog
- English movies make up ~70% of the dataset
- Budget and revenue are strongly correlated
- Movie production grew rapidly after the 1980s
- vote_average follows a roughly normal distribution around 6.0-7.0

## 🤖 Models

| Model | Method | Personalized | Genre Overlap Score |
|---|---|---|---|
| Model 1 | Popularity-based Weighted Rating | No | N/A baseline |
| Model 2 | Content-based TF-IDF + NearestNeighbors | Yes | 0.5005 |
| Model 3 | Hybrid Similarity + Popularity a=0.7 | Yes | 0.5635 Best |

## 🏆 Best Model
**Model 3 — Hybrid Recommender** with alpha=0.7
- Combines TF-IDF content similarity 70% with popularity score 30%
- Outperforms content-based model by +6.3% in genre overlap score
- Best alpha found via hyperparameter tuning across 0.1 to 0.9

## ⚠️ Limitations
- 324,260 movies have fewer than 10 votes → unreliable popularity scores
- Cold start problem: obscure movies with sparse tags get low similarity scores
- No user interaction data → no true collaborative filtering

## 🚀 How to Run

pip install -r requirements.txt

streamlit run app/streamlit_app.py

Open browser at http://localhost:8501

## 📁 Project Structure

- app/streamlit_app.py — Interactive Streamlit demo app
- 01_data_cleaning.ipynb — Data cleaning notebook
- 02_eda.ipynb — EDA notebook
- 03_modeling.ipynb — ML models notebook
- requirements.txt — Python dependencies
- README.md

## 🔧 Tech Stack
Python, Jupyter Notebook, pandas, numpy, scikit-learn, matplotlib, seaborn, Streamlit

## 👤 Author
Alikhan Aripkhan — github.com/alikkhandrow
