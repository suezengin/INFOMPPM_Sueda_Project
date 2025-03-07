import streamlit as st
import pandas as pd
import numpy as np
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("🎬 BBC Hybrid Recommender System - Transparent Mode")

@st.cache_data
def load_bbc_data():
    data_folder = "data/BBC"
    if not os.path.exists(data_folder):
        return pd.DataFrame()

    file_list = [f for f in os.listdir(data_folder) if f.endswith(".pkl") and f not in ["users.pkl", "user_history.pkl"]]
    dfs = []

    for file in file_list:
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)
        
        if "category" not in df.columns:
            category_name = file.replace(".pkl", "").replace("-", " ").title()
            df["category"] = category_name

        df["image"] = df.get("image", None)
        df["synopsis"] = df["synopsis_small"].fillna(df["synopsis_medium"]).fillna(df["synopsis_large"]).fillna("No description available")
        df = df[["category", "title", "synopsis", "image"]]
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

bbc_data = load_bbc_data()

@st.cache_data
def load_user_data():
    users_path = "data/BBC/users.pkl"
    history_path = "data/BBC/user_history.pkl"
    
    users_df = pd.read_pickle(users_path) if os.path.exists(users_path) else pd.DataFrame()
    history_df = pd.read_pickle(history_path) if os.path.exists(history_path) else pd.DataFrame()
    
    return users_df, history_df

users_df, history_df = load_user_data()

user_id = st.sidebar.selectbox("Select User", users_df["user_id"].unique() if not users_df.empty else [])

strategy = st.sidebar.radio("Choose Recommendation Strategy", ["Most Relevant", "Diverse", "Collaborative Filtering", "Hybrid Model"])

def compute_tfidf_similarity(user_id, history_df, bbc_data):
    user_history = history_df[history_df["user_id"] == user_id]
    
    if user_history.empty:
        return pd.DataFrame()
    
    user_watched_titles = user_history["program_title"].tolist()
    content_titles = bbc_data["title"].tolist()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(user_watched_titles + content_titles)
    similarity_matrix = cosine_similarity(tfidf_matrix[:len(user_watched_titles)], tfidf_matrix[len(user_watched_titles):])
    
    mean_sim_scores = similarity_matrix.mean(axis=0)
    bbc_data["similarity_score"] = mean_sim_scores
    return bbc_data.sort_values(by="similarity_score", ascending=False)

if not users_df.empty and not history_df.empty and not bbc_data.empty:
    user_category = users_df[users_df["user_id"] == user_id]["preferred_category"].values[0]
    
    st.subheader(f"📌 Recommendations for User {user_id}")
    
    recommended_content = compute_tfidf_similarity(user_id, history_df, bbc_data).head(3)
    
    for _, row in recommended_content.iterrows():
        st.markdown(f"**{row['title']}** (Category: {row['category']})")
        if row["image"]:
            st.image(row["image"], width=150)
        st.write(f"📖 {row['synopsis']}")
        st.caption(f"🔍 Why? Similarity Score: {row['similarity_score']:.2f}")
    
    st.markdown("---")
    st.subheader("🧐 How These Recommendations Work")
    st.write("""
    - **Most Relevant**: Recommends content from your preferred category.
    - **Diverse**: Selects content from all categories to increase variety.
    - **Collaborative Filtering**: Suggests content based on similar users.
    - **Hybrid Model**: Combines Collaborative and Content-Based Filtering for a balanced recommendation.
    - **Similarity Score**: Indicates how closely the recommended content matches your past viewing habits.
    """)