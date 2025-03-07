import streamlit as st
import pandas as pd
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 📌 Set Streamlit page layout
st.set_page_config(layout="wide")

# 📌 Load EDA results including ratings & popularity scores
@st.cache_data
def load_eda_data():
    """Load EDA analysis results from CSV file and ensure correct column names."""
    df = pd.read_csv("eda_full_results.csv", sep=';')

    # 🔹 Fix column names to remove extra spaces
    expected_columns = ["Category", "Most Watched Count", "Average Watch Time", "Completion Rate", "Average Rating", "Weighted Popularity Score"]
    df.columns = [col.strip() for col in df.columns]

    # 🔹 Check if any expected column is missing
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        st.error(f"⚠️ Missing columns in EDA data: {missing_columns}")

    return df

eda_df = load_eda_data()

# 📌 Load BBC content data
@st.cache_data
def load_bbc_data():
    """Load all BBC content datasets and merge into a single DataFrame, excluding users and history files."""
    data_folder = "data/BBC"
    if not os.path.exists(data_folder):
        return pd.DataFrame()

    file_list = [f for f in os.listdir(data_folder) if f.endswith(".pkl") and f not in ["users.pkl", "user_history.pkl"]]
    dfs = []

    for file in file_list:
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)

        df["category"] = df.get("category", file.replace(".pkl", "").replace("-", " ").title())
        df["synopsis"] = df["synopsis_small"].fillna(df["synopsis_medium"]).fillna(df["synopsis_large"]).fillna("No description available")

        # ✅ Assign random rating if missing
        if "rating" not in df.columns:
            df["rating"] = [random.randint(1, 5) for _ in range(len(df))]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

bbc_data = load_bbc_data()

# 📌 Load user data (users.pkl & user_history.pkl)
@st.cache_data
def load_user_data():
    """Load user history and preferences from .pkl files."""
    users_path = "data/BBC/users.pkl"
    history_path = "data/BBC/user_history.pkl"

    users_df = pd.read_pickle(users_path) if os.path.exists(users_path) else pd.DataFrame()
    history_df = pd.read_pickle(history_path) if os.path.exists(history_path) else pd.DataFrame()

    return users_df, history_df

users_df, history_df = load_user_data()

# 📌 User selection from sidebar
user_id = st.sidebar.selectbox("Select User", users_df["user_id"].unique() if not users_df.empty else [])

# 📌 Select recommendation strategy
strategy = st.sidebar.radio("Choose Recommendation Strategy", ["Most Relevant", "Diverse", "Collaborative Filtering", "Hybrid Model"])

# 📌 Generate recommendations based on EDA results
def recommend_from_eda():
    """Recommend content based on most popular category from EDA results."""
    
    # 🔹 Check if EDA data is loaded properly
    if eda_df.empty:
        st.warning("⚠️ No EDA data available. Showing random recommendations.")
        return bbc_data.sample(3)

    # 🔹 Ensure 'Category' column exists (Fix capitalization issues)
    if "Category" not in eda_df.columns:
        st.error("⚠️ 'Category' column not found in EDA data! Check eda_full_results.csv.")
        return bbc_data.sample(3)  # Return random fallback

    # 🔹 Get most popular category from EDA
    most_popular_category = eda_df.sort_values(by="Weighted Popularity Score", ascending=False).iloc[0]["Category"]
    
    st.subheader(f"🔥 Trend-Based Recommendation: {most_popular_category}")

    return bbc_data[bbc_data["category"] == most_popular_category].sample(3)

# 📌 Compute TF-IDF and Cosine Similarity for content-based recommendations
def compute_tfidf_similarity(user_id, history_df, bbc_data):
    """Generate content recommendations using TF-IDF, Cosine Similarity, and Ratings."""
    user_history = history_df[history_df["user_id"] == user_id]

    if user_history.empty:
        return recommend_from_eda()

    user_titles = user_history["program_title"].tolist()
    content_titles = bbc_data["title"].tolist()

    # 🔹 TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(user_titles + content_titles)

    # 🔹 Cosine Similarity Calculation
    similarity_matrix = cosine_similarity(tfidf_matrix[:len(user_titles)], tfidf_matrix[len(user_titles):])

    # 🔹 Compute average similarity score per content
    mean_sim_scores = similarity_matrix.mean(axis=0)

    # 🔹 Ensure similarity scores are added
    bbc_data = bbc_data.copy()
    bbc_data["similarity_score"] = mean_sim_scores

    # 🔹 Compute final score (Similarity + Rating)
    if "final_score" not in bbc_data.columns:
        bbc_data["final_score"] = (bbc_data["similarity_score"] * 0.7) + (bbc_data["rating"] * 0.3)

    # 🔹 Return Top 3 Recommendations
    return bbc_data.sort_values(by="final_score", ascending=False).head(3)

# 📌 Generate and display recommendations
if not users_df.empty and not history_df.empty and not bbc_data.empty:
    st.subheader(f"📌 Recommendations for User {user_id}")

    if strategy == "Most Relevant":
        recommended_content = bbc_data[bbc_data["category"] == users_df.loc[users_df["user_id"] == user_id, "preferred_category"].values[0]].sample(3)
    elif strategy == "Diverse":
        recommended_content = bbc_data.sample(3)
    elif strategy == "Collaborative Filtering":
        recommended_content = compute_tfidf_similarity(user_id, history_df, bbc_data)
    elif strategy == "Hybrid Model":
        # 🔹 Blend Cosine Similarity & Popularity
        similarity_based = compute_tfidf_similarity(user_id, history_df, bbc_data)
        popularity_based = recommend_from_eda()
        recommended_content = pd.concat([similarity_based, popularity_based]).drop_duplicates().sample(3)

    # 🔹 Ensure final_score exists before displaying
    if "final_score" not in recommended_content.columns:
        recommended_content["final_score"] = recommended_content["rating"]

    for _, row in recommended_content.iterrows():
        st.markdown(f"**{row['title']}** (Category: {row['category']}, Rating: {row['rating']})")
        if row["image"]:
            st.image(row["image"], width=150)
        st.write(f"📖 {row['synopsis']}")
        st.caption(f"🔍 Why? Score: {row['final_score']:.2f}")

    st.markdown("---")
    st.subheader("🧐 How These Recommendations Work")
    st.write("""
    - **Most Relevant**: Recommends content from your preferred category.
    - **Diverse**: Selects content from all categories to increase variety.
    - **Collaborative Filtering**: Suggests content based on similar users.
    - **Hybrid Model**: Combines Collaborative and Content-Based Filtering for a balanced recommendation.
    - **Trend-Based**: Uses Exploratory Data Analysis to recommend content from the most popular category.
    """)
