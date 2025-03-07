import pandas as pd
import random

# 📌 Dataset categories extracted from .pkl files
dataset_files = [
    "signed.pkl", "science-and-nature.pkl", "music.pkl", "sports.pkl", "entertainment.pkl",
    "comedy.pkl", "history.pkl", "lifestyle.pkl", "arts.pkl", "from-the-archives.pkl",
    "cbbc.pkl", "films.pkl", "documentaries.pkl"
]

# 📌 Extract category names from dataset filenames
category_list = [file.replace(".pkl", "").replace("-", " ").title() for file in dataset_files]

# 📌 Simulate user interaction data (Including Ratings)
user_history_full_data = {
    "user_id": list(range(1, 51)) * len(category_list),  # 50 users repeated across all categories
    "program_title": ["Program " + chr(65 + (i % 10)) for i in range(50 * len(category_list))],  # Programs A-J repeating
    "category": category_list * 50,  # All dataset categories included
    "watch_time": [random.randint(300, 5000) for _ in range(50 * len(category_list))],  # Random watch times
    "watched_fully": [random.choice([True, False]) for _ in range(50 * len(category_list))],  # True/False for full watch
    "rating": [random.randint(1, 5) for _ in range(50 * len(category_list))]  # Simulated user ratings (1-5)
}

# 📌 Create DataFrame
user_history_full_df = pd.DataFrame(user_history_full_data)

# 📌 Compute EDA Metrics
most_watched_categories_full = user_history_full_df["category"].value_counts()
avg_watch_time_full = user_history_full_df.groupby("category")["watch_time"].mean()
completion_rate_full = user_history_full_df.groupby("category")["watched_fully"].mean()
avg_rating_full = user_history_full_df.groupby("category")["rating"].mean()

# 📌 Calculate a weighted popularity score (Higher watch time, completion, and rating increases score)
user_history_full_df["popularity_score"] = (
    user_history_full_df["watch_time"] * 0.4 +
    user_history_full_df["watched_fully"].astype(int) * 0.3 +
    user_history_full_df["rating"] * 0.3
)

# 📌 Aggregate scores per category
weighted_popularity = user_history_full_df.groupby("category")["popularity_score"].mean()

# 📌 Create the final EDA results DataFrame
eda_full_results = pd.DataFrame({
    "Category": most_watched_categories_full.index,  # ✅ Ensuring Category column is added
    "Most Watched Count": most_watched_categories_full.values,
    "Average Watch Time": avg_watch_time_full.values,
    "Completion Rate": completion_rate_full.values,
    "Average Rating": avg_rating_full.values,
    "Weighted Popularity Score": weighted_popularity.values
})

# 📌 Save to CSV (Used in `app.py`)
eda_full_results.to_csv("eda_full_results.csv", index=False, sep=';')

print("✅ Enhanced EDA results saved as 'eda_full_results.csv'!")
