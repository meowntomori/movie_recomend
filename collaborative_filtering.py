import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Загрузка данных
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

# Загрузка данных в формат, подходящий для библиотеки Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Создание модели SVD
model = SVD()

# Обучение модели
trainset = data.build_full_trainset()
model.fit(trainset)

def get_collaborative_recommendations(user_id, ratings, movies):
    # Получение всех фильмов, которые пользователь еще не оценил
    movies_to_predict = [movie_id for movie_id in movies['id'] if movie_id not in ratings[ratings['userId'] == user_id]['movieId'].values]

    # Предсказание рейтингов для этих фильмов
    predictions = [model.predict(user_id, movie_id) for movie_id in movies_to_predict]

    # Сортировка фильмов по предсказанным рейтингам
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Вывод топ-10 рекомендаций
    top_10_recommendations = predictions[:10]
    recommended_movies = [movies[movies['id'] == pred.iid]['title'].values[0] for pred in top_10_recommendations]

    return recommended_movies

def update_recommendations(user_id, ratings, movies, c):
    # Получение всех фильмов, которые пользователь добавил в любимые
    c.execute("SELECT movie_id FROM favorites WHERE user_id=?", (user_id,))
    favorite_movies = c.fetchall()
    favorite_movie_ids = [movie[0] for movie in favorite_movies]

    # Обновление рейтингов для коллаборативной фильтрации
    new_ratings = pd.DataFrame({'userId': [user_id] * len(favorite_movie_ids), 'movieId': favorite_movie_ids, 'rating': [5] * len(favorite_movie_ids)})
    ratings = pd.concat([ratings, new_ratings], ignore_index=True)

    # Загрузка данных в формат, подходящий для библиотеки Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Создание модели SVD
    model = SVD()

    # Обучение модели
    trainset = data.build_full_trainset()
    model.fit(trainset)

    # Получение рекомендаций
    recommendations = get_collaborative_recommendations(user_id, ratings, movies)
    return recommendations
