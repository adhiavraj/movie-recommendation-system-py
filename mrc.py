import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
data = pd.read_csv('ratings.csv')

# Drop the timestamp column (not needed)
data = data.drop(columns='timestamp')

# Re-index user IDs so that they start from 0
data['userId'] = data['userId'].astype('category').cat.codes

# Create a user-movie matrix with ratings
user_movie_matrix = data.pivot(index='userId', columns='movieId', values='rating')

# Compute user similarity using cosine similarity
user_similarity = cosine_similarity(user_movie_matrix.fillna(0))

# Loading movies and their information
movies = pd.read_csv('movies.csv')

# Define a function to get movie recommendations for a user
def get_movie_recommendations(user_id, num_recommendations=10):
    # Get the user's ratings
    user_ratings = user_movie_matrix.loc[user_id].fillna(0)

    # Check if the user exists in the user_ratings
    if user_id not in user_movie_matrix.index:
        print("User ID not found in the dataset.")
        return []

    # Calculate the weighted average of user ratings using user similarity
    user_weighted_ratings = np.sum(user_similarity[:, user_id, None] * user_ratings.values, axis=0)

    # Sort movies by the weighted ratings and get top recommendations
    top_movie_ids = user_weighted_ratings.argsort()[::-1][:num_recommendations]

    #It is only to check the columns (not necessary)
    #print("Available columns:", data.columns)

    # Get movie titles for recommendations
    movie_titles = movies[movies['movieId'].isin(top_movie_ids)]['title'].values

    return movie_titles

user_id = 9  # Replace with the user ID you want recommendations for
recommendations = get_movie_recommendations(user_id)

if len(recommendations) == 0:
    print("Unable to generate recommendations.")
else:
    print("Recommendations for User", user_id)
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")
