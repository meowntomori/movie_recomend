import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLRecommender:
    def __init__(self):
        try:
            # Load ratings data
            self.ratings_data = pd.read_csv('data/ratings.csv')
            logger.info("Ratings data loaded successfully.")

            # Load movies data
            self.movies_data = pd.read_csv('data/movies.csv')
            logger.info("Movies data loaded successfully.")

            # Check if 'id' column exists in movies data
            if 'id' not in self.movies_data.columns:
                raise KeyError("'id' column not found in movies data.")

            # Check if 'userId' column exists in ratings data
            if 'userId' not in self.ratings_data.columns:
                raise KeyError("'userId' column not found in ratings data.")

            # Fill missing values with appropriate types
            for col in self.movies_data.columns:
                if self.movies_data[col].dtype == 'object':
                    self.movies_data[col] = self.movies_data[col].fillna('')
                else:
                    self.movies_data[col] = self.movies_data[col].fillna(0.0)

        except FileNotFoundError:
            logger.error("Data file not found.")
            raise
        except pd.errors.EmptyDataError:
            logger.error("Data file is empty.")
            raise
        except pd.errors.ParserError:
            logger.error("Error parsing data file.")
            raise
        except KeyError as e:
            logger.error(f"KeyError: {e}")
            raise

        # Create a Surprise dataset
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.ratings_data[['userId', 'movieId', 'rating']], reader)

        # Train-test split
        trainset, testset = train_test_split(data, test_size=0.2)

        # Train the SVD model
        self.model = SVD()
        self.model.fit(trainset)
        logger.info("SVD model trained successfully.")

        # Store the full dataset for real-time predictions
        self.trainset = trainset

    def recommend_movies(self, user_id, n_recommendations=5):
        """
        Recommend movies for a given user ID.
        """
        try:
            # Check if user_id exists in the ratings data
            if user_id not in self.ratings_data['userId'].unique():
                logger.error(f"User ID {user_id} not found in the ratings data.")
                return []

            # Get all movie IDs
            all_movie_ids = set(self.ratings_data['movieId'].unique())

            # Get movies the user has already rated
            rated_movies = {iid for (iid, _) in self.trainset.ur[self.trainset.to_inner_uid(user_id)]}

            # Predict ratings for unrated movies
            predictions = []
            for movie_id in all_movie_ids - rated_movies:
                pred = self.model.predict(user_id, movie_id)
                predictions.append((movie_id, pred.est))

            # Sort by predicted rating and return top N
            if not predictions:
                logger.error(f"No predictions for user ID {user_id}.")
                return []

            top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]
            top_movie_ids = [int(movie_id) for movie_id, _ in top_movies]

            # Convert movie IDs to movie names
            top_movie_names = self.movies_data[self.movies_data['id'].isin(top_movie_ids)]['title'].tolist()
            return top_movie_names
        except KeyError:
            logger.error(f"User ID {user_id} not found in the training set.")
            return []
        except Exception as e:
            logger.error(f"An error occurred during recommendation: {e}")
            return []

# Пример использования
if __name__ == "__main__":
    recommender = MLRecommender()
    user_id = 1  # Пример user_id
    recommendations = recommender.recommend_movies(user_id)
    logger.info(f"Recommendations for user {user_id}: {recommendations}")
