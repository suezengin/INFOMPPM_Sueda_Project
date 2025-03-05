import streamlit as st
import pandas as pd
import os
import numpy as np
import random

#  Configure Fullscreen Layout
st.set_page_config(layout="wide")

#  Load BBC Data
@st.cache_data
def load_bbc_data():
    data_folder = "data/BBC"
    if not os.path.exists(data_folder):
        return pd.DataFrame()
    
    file_list = os.listdir(data_folder)
    dfs = []

    for file in file_list:
        if file.endswith(".pkl"):
            file_path = os.path.join(data_folder, file)

            try:
                df = pd.read_pickle(file_path)
                if not df.empty:
                    df["source_file"] = file
                    dfs.append(df)
            except:
                continue  # Skip corrupt files

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

#  Generate Fake Users
@st.cache_data
def generate_fake_users(bbc_data, num_users=100):
    if bbc_data.empty:
        return pd.DataFrame()
    
    categories = bbc_data["category"].unique()
    users = [{"user_id": user_id, "preferred_category": np.random.choice(categories)}
             for user_id in range(1, num_users + 1)]
    
    return pd.DataFrame(users)

#  Recommendation System
def recommend_content(user_id, users_df, bbc_data, strategy="Most Relevant"):
    """Recommend BBC programs based on user's preferences with explanation."""
    
    user = users_df[users_df["user_id"] == user_id]
    if user.empty:
        return "User not found!", pd.DataFrame()

    preferred_category = user.iloc[0]["preferred_category"]
    relevant_content = bbc_data[bbc_data["category"] == preferred_category]
    
    if relevant_content.empty:
        return "No recommendations available for this category.", pd.DataFrame()

    if strategy == "Most Watched":
        recommended = relevant_content.sample(6)
        explanation = "Most watched programs in your preferred category."
    elif strategy == "Most Relevant":
        recommended = relevant_content.sample(6)
        explanation = "Programs matching your category preference."
    elif strategy == "Randomized":
        recommended = relevant_content.sample(6)
        explanation = "Randomly selected programs."
    else:
        recommended = relevant_content.sample(6)
        explanation = "Default recommendation strategy applied."

    return explanation, recommended

#  Load Data Automatically
bbc_data = load_bbc_data()
users_df = generate_fake_users(bbc_data)

#  Horizontal Interface
st.markdown("<h1 style='text-align: center; font-size: 50px;'>🎬 BBC Recommender System</h1>", unsafe_allow_html=True)

#  User Selection
st.sidebar.markdown("## 🎭 Select User")
user_id_input = st.sidebar.number_input("Enter User ID:", min_value=1, max_value=100, value=1)

st.sidebar.markdown("## 🔥 Recommendation Strategy")
strategy = st.sidebar.selectbox("Choose Strategy:", ["Most Relevant", "Most Watched", "Randomized"])

#  Program Selection
st.sidebar.markdown("## 📺 Browse Programs")
categories = bbc_data["category"].unique() if not bbc_data.empty else []
selected_category = st.sidebar.selectbox("Select Category", categories if len(categories) > 0 else ["No Data"])

#  Display Recommendations
if not bbc_data.empty and not users_df.empty:
    explanation, recommended_programs = recommend_content(user_id_input, users_df, bbc_data, strategy)

    st.markdown(f"<h2 style='text-align: center;'>{explanation}</h2>", unsafe_allow_html=True)

    if not recommended_programs.empty:
        cols = st.columns(6)
        for i, (_, program) in enumerate(recommended_programs.iterrows()):
            with cols[i]:
                st.image("https://via.placeholder.com/200", use_column_width=True)  # Placeholder for program cover
                st.markdown(f"<h4 style='text-align: center;'>{program['title']}</h4>", unsafe_allow_html=True)
                st.caption(f"Category: {program['category']}")
