import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Загрузка данных
movies = pd.read_csv('data/movies.csv')

# Удаление пропущенных значений
movies = movies.dropna(subset=['overview'])

# Преобразование текстовых данных
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Вычисление косинусного сходства
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_content_based_recommendations(title, movies):
    # Получение индекса фильма
    idx = movies[movies['title'] == title].index[0]

    # Вычисление сходства для всех фильмов
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Сортировка фильмов по сходству
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Получение индексов 10 самых похожих фильмов
    sim_scores = sim_scores[1:11]

    # Получение индексов фильмов
    movie_indices = [i[0] for i in sim_scores]

    # Возврат списка рекомендованных фильмов
    return movies['title'].iloc[movie_indices].tolist()
