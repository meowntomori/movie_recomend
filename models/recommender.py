import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from langdetect import detect
from deep_translator import GoogleTranslator

class MovieRecommender:
    def __init__(self):
        # Load movie data
        self.movies_data = pd.read_csv('data/movies.csv')

        # Fill missing values with appropriate types
        for col in self.movies_data.columns:
            if self.movies_data[col].dtype == 'object':
                self.movies_data[col] = self.movies_data[col].fillna('')
            else:
                self.movies_data[col] = self.movies_data[col].fillna(0.0)

        # Combine relevant features into one string for similarity calculation
        self.movies_data['combined_features'] = self.movies_data[
            ['genres', 'keywords', 'tagline', 'cast', 'director']].agg(' '.join, axis=1)

        # Compute the TF-IDF matrix
        self.vectorizer = TfidfVectorizer()
        self.feature_vectors = self.vectorizer.fit_transform(self.movies_data['combined_features'])
        self.similarity = cosine_similarity(self.feature_vectors)

    def translate_to_english(self, text):
        """
        Translates the input text to English if it is in Russian.
        """
        try:
            language = detect(text)
        except Exception:
            language = 'unknown'

        if language == 'ru':
            # Translate from Russian to English
            translator = GoogleTranslator(source='auto', target='en')
            return translator.translate(text)
        return text  # Return original text if already in English

    def get_movie_by_fuzzy_match(self, query):
        """
        Finds the closest matching movie title using fuzzy matching.
        """
        titles = self.movies_data['title'].values
        matches = get_close_matches(query, titles, n=1, cutoff=0.5)  # Fuzzy match with a cutoff score
        return matches[0] if matches else None

    def get_recommendations(self, movie_name):
        # Translate the input movie name to English if needed
        movie_name_translated = self.translate_to_english(movie_name)

        # Find the closest match for the movie title
        movie_name_matched = self.get_movie_by_fuzzy_match(movie_name_translated)
        if not movie_name_matched:
            return None

        # Get the index of the matched movie
        idx = self.movies_data[self.movies_data['title'] == movie_name_matched].index[0]

        # Compute similarity scores
        scores = list(enumerate(self.similarity[idx]))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 recommendations

        # Fetch recommended movie titles
        recommendations = [self.movies_data.iloc[item[0]]['title'] for item in sorted_scores]
        return recommendations
