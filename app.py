# import streamlit as st
# import pandas as pd
# import numpy as np
# import time

# # Load the movies data
# movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
# movies = pd.read_csv(
#     movies_url,
#     sep='::',
#     header=None,
#     engine='python',
#     names=['movieID', 'title', 'genres'],
#     encoding='latin1'
# )

# # Load the rating_matrix and similarity matrix S
# # rating_matrix = pd.read_csv('rmat', sep=',')
# # S = pd.read_csv('similarity_matrix_S.csv', index_col=0)
# rating_matrix = pd.read_csv('rmat_top_100.csv', sep=',')
# S = pd.read_csv('similarity_matrix_S_top_100.csv', index_col=0)


# # Define the myIBCF function
# def myIBCF(newuser, S, rating_matrix):

#     # Ensure the order of movies in 'newuser' matches the columns of S
#     movies = S.columns.tolist()
#     w = pd.Series(newuser, index=movies)

#     # Initialize a dictionary to store predictions
#     predictions = {}

#     # For movies not rated by the user, compute the predicted rating
#     for movie_i in movies:
#         if pd.notna(w[movie_i]):
#             continue  # Skip movies already rated by the user

#         # Get the similarities S_ij where S_ij is not NA and w_j is not NA
#         S_i = S.loc[movie_i]
#         S_i = S_i[S_i.notna()]

#         # Get movies that the user has rated
#         rated_movies = w[w.notna()].index.tolist()
#         # Intersection of movies similar to movie_i and movies rated by user
#         common_movies = list(set(S_i.index) & set(rated_movies))

#         if len(common_movies) == 0:
#             prediction = np.nan  # Cannot make a prediction
#         else:
#             # Compute the numerator and denominator
#             S_ij = S_i[common_movies]
#             w_j = w[common_movies]

#             numerator = np.dot(S_ij, w_j)
#             denominator = S_ij.sum()

#             if denominator == 0:
#                 prediction = np.nan
#             else:
#                 prediction = numerator / denominator

#         predictions[movie_i] = prediction

#     # Convert predictions to a Series
#     predictions_series = pd.Series(predictions)

#     # Get top 10 recommendations
#     top_recommendations = predictions_series.dropna().sort_values(ascending=False)

#     if len(top_recommendations) >= 10:
#         recommended_movies = top_recommendations.head(10).index.tolist()
#     else:
#         print("filling with popular movies")
#         # Need to fill remaining recommendations with popular movies
#         num_needed = 10 - len(top_recommendations)
#         # Load or compute the popularity ranking
#         try:
#             popularity_ranking = pd.read_csv('movie_popularity_ranking.csv', index_col=0)
#         except FileNotFoundError:
#             # Compute the popularity ranking based on System I
#             # Assuming 'rating_matrix' is available
#             movie_means = rating_matrix.mean(axis=0)
#             movie_counts = rating_matrix.count(axis=0)
#             popularity_ranking = pd.DataFrame({
#                 'mean_rating': movie_means,
#                 'rating_count': movie_counts
#             })
#             popularity_ranking = popularity_ranking.sort_values(by=['rating_count', 'mean_rating'], ascending=False)
#             popularity_ranking.to_csv('movie_popularity_ranking.csv')

#         # Exclude movies already rated by user and already recommended
#         movies_already_rated = w[w.notna()].index.tolist()
#         movies_already_recommended = top_recommendations.index.tolist()
#         movies_to_exclude = set(movies_already_rated + movies_already_recommended)

#         # Get movies from popularity ranking excluding movies_to_exclude
#         remaining_movies = popularity_ranking.index.difference(movies_to_exclude)

#         # Get the top movies needed
#         additional_movies = remaining_movies[:num_needed]

#         recommended_movies = top_recommendations.index.tolist() + additional_movies.tolist()

#     return recommended_movies

# # Select a sample of movies to display
# sample_movies = movies.sample(100, random_state=42).reset_index(drop=True)


# # Set the title of the app
# st.title("Movie Recommender System")

# # Provide instructions to the user
# st.write("Please rate the following movies (1-5 stars). If you haven't seen a movie, you can skip it.")

# # Initialize a dictionary to store user ratings
# user_ratings = {}

# # Display the sample movies and collect ratings
# for idx, row in sample_movies.iterrows():
#     movie_id = 'm' + str(row['movieID'])
#     title = row['title']
#     rating = st.selectbox(
#         f"{title}",
#         options=[None, 1, 2, 3, 4, 5],
#         index=0,
#         key=movie_id
#     )
#     if rating is not None:
#         user_ratings[movie_id] = rating

# # Process user input and generate recommendations
# if st.button('Get Recommendations'):
#     if not user_ratings:
#         st.write("Please rate at least one movie to get recommendations.")
#     else:
#         # Create a new user vector
#         newuser = pd.Series(np.nan, index=rating_matrix.columns)
#         for movie_id, rating in user_ratings.items():
#             newuser[movie_id] = rating
        
#         # Generate recommendations
#         recommendations = myIBCF(newuser, S, rating_matrix)
        
#         # Display the recommendations
#         st.write("Top 10 Movie Recommendations for You:")
        
#         # Get movie details for the recommended movies
#         recommended_movie_ids = [int(mid[1:]) for mid in recommendations]
#         recommended_movies = movies[movies['movieID'].isin(recommended_movie_ids)]
        
#         # Display movie titles and images
#         for idx, row in recommended_movies.iterrows():
#             movie_title = row['title']
#             movie_id = row['movieID']
#             image_url = f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg"
#             st.image(image_url, width=150, caption=movie_title)


############################################################################################################################################# .

# import streamlit as st
# import pandas as pd
# import numpy as np
# import time

# # Load the movies data
# movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
# movies = pd.read_csv(
#     movies_url,
#     sep='::',
#     header=None,
#     engine='python',
#     names=['movieID', 'title', 'genres'],
#     encoding='latin1'
# )

# # Load the rating_matrix and similarity matrix S
# # rating_matrix = pd.read_csv('rmat', sep=',')
# # S = pd.read_csv('similarity_matrix_S.csv', index_col=0)
# rating_matrix = pd.read_csv('rmat_top_100.csv', sep=',')
# S = pd.read_csv('similarity_matrix_S_top_100.csv', index_col=0)


# # Define the myIBCF function
# def myIBCF(newuser, S, rating_matrix):

#     # Ensure the order of movies in 'newuser' matches the columns of S
#     movies = S.columns.tolist()
#     w = pd.Series(newuser, index=movies)

#     # Initialize a dictionary to store predictions
#     predictions = {}

#     # For movies not rated by the user, compute the predicted rating
#     for movie_i in movies:
#         if pd.notna(w[movie_i]):
#             continue  # Skip movies already rated by the user

#         # Get the similarities S_ij where S_ij is not NA and w_j is not NA
#         S_i = S.loc[movie_i]
#         S_i = S_i[S_i.notna()]

#         # Get movies that the user has rated
#         rated_movies = w[w.notna()].index.tolist()
#         # Intersection of movies similar to movie_i and movies rated by user
#         common_movies = list(set(S_i.index) & set(rated_movies))

#         if len(common_movies) == 0:
#             prediction = np.nan  # Cannot make a prediction
#         else:
#             # Compute the numerator and denominator
#             S_ij = S_i[common_movies]
#             w_j = w[common_movies]

#             numerator = np.dot(S_ij, w_j)
#             denominator = S_ij.sum()

#             if denominator == 0:
#                 prediction = np.nan
#             else:
#                 prediction = numerator / denominator

#         predictions[movie_i] = prediction

#     # Convert predictions to a Series
#     predictions_series = pd.Series(predictions)

#     # Get top 10 recommendations
#     top_recommendations = predictions_series.dropna().sort_values(ascending=False)

#     if len(top_recommendations) >= 10:
#         recommended_movies = top_recommendations.head(10).index.tolist()
#     else:
#         print("filling with popular movies")
#         # Need to fill remaining recommendations with popular movies
#         num_needed = 10 - len(top_recommendations)
#         # Load or compute the popularity ranking
#         try:
#             popularity_ranking = pd.read_csv('movie_popularity_ranking.csv', index_col=0)
#         except FileNotFoundError:
#             # Compute the popularity ranking based on System I
#             # Assuming 'rating_matrix' is available
#             movie_means = rating_matrix.mean(axis=0)
#             movie_counts = rating_matrix.count(axis=0)
#             popularity_ranking = pd.DataFrame({
#                 'mean_rating': movie_means,
#                 'rating_count': movie_counts
#             })
#             popularity_ranking = popularity_ranking.sort_values(by=['rating_count', 'mean_rating'], ascending=False)
#             popularity_ranking.to_csv('movie_popularity_ranking.csv')

#         # Exclude movies already rated by user and already recommended
#         movies_already_rated = w[w.notna()].index.tolist()
#         movies_already_recommended = top_recommendations.index.tolist()
#         movies_to_exclude = set(movies_already_rated + movies_already_recommended)

#         # Get movies from popularity ranking excluding movies_to_exclude
#         remaining_movies = popularity_ranking.index.difference(movies_to_exclude)

#         # Get the top movies needed
#         additional_movies = remaining_movies[:num_needed]

#         recommended_movies = top_recommendations.index.tolist() + additional_movies.tolist()

#     return recommended_movies

# # Select a sample of movies to display
# sample_movies = movies.sample(100, random_state=42).reset_index(drop=True)


# # Set the title of the app
# st.title("Movie Recommender System")

# # Provide instructions to the user
# st.write("Please rate the following movies (1-5 stars). If you haven't seen a movie, you can skip it.")

# # Initialize a dictionary to store user ratings
# user_ratings = {}

# # Display the sample movies and collect ratings
# for idx, row in sample_movies.iterrows():
#     movie_id = 'm' + str(row['movieID'])
#     title = row['title']
#     image_url = f"https://liangfgithub.github.io/MovieImages/{row['movieID']}.jpg"

#     # Create a container for each movie
#     with st.container():
#         # Arrange image and selectbox side by side
#         cols = st.columns([1, 3])  # Adjust column widths as needed

#         with cols[0]:
#             st.image(image_url, width=100)

#         with cols[1]:
#             rating = st.selectbox(
#                 f"**{title}**",
#                 options=[None, 1, 2, 3, 4, 5],
#                 index=0,
#                 key=movie_id
#             )
#             if rating is not None:
#                 user_ratings[movie_id] = rating

# # Process user input and generate recommendations
# if st.button('Get Recommendations'):
#     if not user_ratings:
#         st.write("Please rate at least one movie to get recommendations.")
#     else:
#         # Create a new user vector
#         newuser = pd.Series(np.nan, index=rating_matrix.columns)
#         for movie_id, rating in user_ratings.items():
#             newuser[movie_id] = rating

#         # Generate recommendations
#         recommendations = myIBCF(newuser, S, rating_matrix)

#         # Display the recommendations
#         st.write("Top 10 Movie Recommendations for You:")

#         # Get movie details for the recommended movies
#         recommended_movie_ids = [int(mid[1:]) for mid in recommendations]
#         recommended_movies = movies[movies['movieID'].isin(recommended_movie_ids)]

#         # Display movie titles and images
#         for idx, row in recommended_movies.iterrows():
#             movie_title = row['title']
#             movie_id = row['movieID']
#             image_url = f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg"
#             st.image(image_url, width=150, caption=movie_title)





import streamlit as st
import pandas as pd
import numpy as np

# Load the movies data
movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
movies = pd.read_csv(
    movies_url,
    sep='::',
    header=None,
    engine='python',
    names=['movieID', 'title', 'genres'],
    encoding='latin1'
)

# Load the rating_matrix and similarity matrix S
rating_matrix = pd.read_csv('rmat_top_100.csv', sep=',')
S = pd.read_csv('similarity_matrix_S_top_100.csv', index_col=0)

# Define the myIBCF function (assuming it remains unchanged)
def myIBCF(newuser, S, rating_matrix):

    # Ensure the order of movies in 'newuser' matches the columns of S
    movies = S.columns.tolist()
    w = pd.Series(newuser, index=movies)

    # Initialize a dictionary to store predictions
    predictions = {}

    # For movies not rated by the user, compute the predicted rating
    for movie_i in movies:
        if pd.notna(w[movie_i]):
            continue  # Skip movies already rated by the user

        # Get the similarities S_ij where S_ij is not NA and w_j is not NA
        S_i = S.loc[movie_i]
        S_i = S_i[S_i.notna()]

        # Get movies that the user has rated
        rated_movies = w[w.notna()].index.tolist()
        # Intersection of movies similar to movie_i and movies rated by user
        common_movies = list(set(S_i.index) & set(rated_movies))

        if len(common_movies) == 0:
            prediction = np.nan  # Cannot make a prediction
        else:
            # Compute the numerator and denominator
            S_ij = S_i[common_movies]
            w_j = w[common_movies]

            numerator = np.dot(S_ij, w_j)
            denominator = S_ij.sum()

            if denominator == 0:
                prediction = np.nan
            else:
                prediction = numerator / denominator

        predictions[movie_i] = prediction

    # Convert predictions to a Series
    predictions_series = pd.Series(predictions)

    # Get top 10 recommendations
    top_recommendations = predictions_series.dropna().sort_values(ascending=False)

    if len(top_recommendations) >= 10:
        recommended_movies = top_recommendations.head(10).index.tolist()
    else:
        print("filling with popular movies")
        # Need to fill remaining recommendations with popular movies
        num_needed = 10 - len(top_recommendations)
        # Load or compute the popularity ranking
        try:
            popularity_ranking = pd.read_csv('movie_popularity_ranking.csv', index_col=0)
        except FileNotFoundError:
            # Compute the popularity ranking based on System I
            # Assuming 'rating_matrix' is available
            movie_means = rating_matrix.mean(axis=0)
            movie_counts = rating_matrix.count(axis=0)
            popularity_ranking = pd.DataFrame({
                'mean_rating': movie_means,
                'rating_count': movie_counts
            })
            popularity_ranking = popularity_ranking.sort_values(by=['rating_count', 'mean_rating'], ascending=False)
            popularity_ranking.to_csv('movie_popularity_ranking.csv')

        # Exclude movies already rated by user and already recommended
        movies_already_rated = w[w.notna()].index.tolist()
        movies_already_recommended = top_recommendations.index.tolist()
        movies_to_exclude = set(movies_already_rated + movies_already_recommended)

        # Get movies from popularity ranking excluding movies_to_exclude
        remaining_movies = popularity_ranking.index.difference(movies_to_exclude)

        # Get the top movies needed
        additional_movies = remaining_movies[:num_needed]

        recommended_movies = top_recommendations.index.tolist() + additional_movies.tolist()

    return recommended_movies

# Select a sample of movies to display
# sample_movies = movies.sample(100, random_state=42).reset_index(drop=True)
# Extract movie IDs from S.columns
movie_ids = [int(mid[1:]) for mid in S.columns]  # Remove 'm' prefix and convert to int
# Filter movies to include only the top 100 movies
sample_movies = movies[movies['movieID'].isin(movie_ids)].reset_index(drop=True)


# Set the title of the app
st.title("Movie Recommender System")

# Provide instructions to the user
st.write("Please rate the following movies (1-5 stars). If you haven't seen a movie, you can skip it.")

# Set the number of columns per row
n_columns = 5  # Adjust this number to change the grid layout

# Initialize a dictionary to store user ratings
user_ratings = {}

# Loop through the sample movies and arrange them in a grid
for idx in range(0, len(sample_movies), n_columns):
    cols = st.columns(n_columns)
    for col, (index, row) in zip(cols, sample_movies.iloc[idx: idx + n_columns].iterrows()):
        with col:
            movie_id = 'm' + str(row['movieID'])
            image_url = f"https://liangfgithub.github.io/MovieImages/{row['movieID']}.jpg"
            st.image(image_url, use_container_width=True)
            rating = st.selectbox(
                '',
                options=[None, 1, 2, 3, 4, 5],
                index=0,
                key=movie_id
            )
            if rating is not None:
                user_ratings[movie_id] = rating


# Process user input and generate recommendations
if st.button('Get Recommendations'):
    if not user_ratings:
        st.write("Please rate at least one movie to get recommendations.")
    else:
        # Create a new user vector
        newuser = pd.Series(np.nan, index=rating_matrix.columns)
        for movie_id, rating in user_ratings.items():
            newuser[movie_id] = rating

        # Generate recommendations
        recommendations = myIBCF(newuser, S, rating_matrix)

        # Display the recommendations
        st.write("Top 10 Movie Recommendations for You:")

        # Get movie details for the recommended movies
        recommended_movie_ids = [int(mid[1:]) for mid in recommendations]
        recommended_movies = movies[movies['movieID'].isin(recommended_movie_ids)]

        # Display movie images and titles
        for idx, row in recommended_movies.iterrows():
            movie_title = row['title']
            movie_id = row['movieID']
            image_url = f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg"
            st.image(image_url, width=150, caption=movie_title)