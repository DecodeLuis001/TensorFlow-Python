# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv('music_data.csv')

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create a user-item matrix
user_item_matrix = train_data.pivot_table(index='user_id', columns='song_id', values='rating').fillna(0)

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(user_item_matrix, user_item_matrix)

# Function to get top recommendations for a user
def get_recommendations(user_id, top_n=5):
    # Get the index of the user
    user_index = user_item_matrix.index.get_loc(user_id)
    
    # Get the similarity scores for the user
    user_sim_scores = cosine_sim[user_index]
    
    # Sort the similarity scores in descending order
    sorted_indices = user_sim_scores.argsort()[::-1]
    
    # Get the top N similar users
    top_similar_users = sorted_indices[1:top_n+1]
    
    # Get the songs listened by the top similar users
    recommendations = user_item_matrix.iloc[top_similar_users].sum().sort_values(ascending=False).index.tolist()
    
    return recommendations

# Get recommendations for a user
user_id = 'user123'
recommendations = get_recommendations(user_id)

# Print the recommendations
print(f"Top recommendations for user {user_id}:")
for i, song in enumerate(recommendations):
    print(f"{i+1}. {song}"
