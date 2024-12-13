from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import sqlite3
from content_based import get_content_based_recommendations
from collaborative_filtering import get_collaborative_recommendations, update_recommendations
from fuzzywuzzy import process
from googletrans import Translator

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Загрузка данных
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Инициализация переводчика
translator = Translator()

# Подключение к базе данных
conn = sqlite3.connect('database.db', check_same_thread=False)
c = conn.cursor()

# Создание таблицы пользователей
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY, name TEXT)''')

# Создание таблицы любимых фильмов
c.execute('''CREATE TABLE IF NOT EXISTS favorites
             (user_id INTEGER, movie_id INTEGER, FOREIGN KEY(user_id) REFERENCES users(id))''')

# Создание таблицы фильмов
c.execute('''CREATE TABLE IF NOT EXISTS movies
             (id INTEGER PRIMARY KEY, title TEXT)''')

# Заполнение таблицы фильмов
for index, row in movies.iterrows():
    c.execute("INSERT OR IGNORE INTO movies (id, title) VALUES (?, ?)", (row['id'], row['title']))

conn.commit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_user', methods=['POST'])
def add_user():
    user_id = int(request.form['user_id'])
    user_name = request.form['user_name']

    # Проверка, существует ли пользователь
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    existing_user = c.fetchone()
    if existing_user:
        flash('Пользователь с таким ID уже существует.')
    else:
        c.execute("INSERT INTO users (id, name) VALUES (?, ?)", (user_id, user_name))
        conn.commit()
        flash('Пользователь успешно добавлен.')

    return redirect(url_for('index'))

@app.route('/content_based', methods=['POST'])
def content_based():
    movie_title = request.form['movie_title']
    user_id = int(request.form['user_id'])

    # Проверка, существует ли пользователь
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    existing_user = c.fetchone()

    if not existing_user:
        flash('Пользователь не найден. Пожалуйста, добавьте себя в систему.')
        return redirect(url_for('index'))

    # Перевод названия фильма на английский
    translated_title = translator.translate(movie_title, src='ru', dest='en').text

    # Нечеткий поиск фильма
    best_match = process.extractOne(translated_title, movies['title'])
    if best_match:
        movie_title = best_match[0]

    # Добавление фильма в любимые
    c.execute("SELECT id FROM movies WHERE title=?", (movie_title,))
    movie_id = c.fetchone()[0]
    c.execute("INSERT INTO favorites (user_id, movie_id) VALUES (?, ?)", (user_id, movie_id))
    conn.commit()

    # Обновление рекомендаций
    recommendations = update_recommendations(user_id, ratings, movies, c)
    recommendations_ru = [translator.translate(movie, src='en', dest='ru').text for movie in recommendations]
    return render_template('index.html', recommendations=recommendations_ru, user_id=user_id, movies=movies)

@app.route('/collaborative', methods=['POST'])
def collaborative():
    user_id = int(request.form['user_id'])

    # Проверка, существует ли пользователь
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    existing_user = c.fetchone()

    if not existing_user:
        flash('Пользователь не найден. Пожалуйста, добавьте себя в систему.')
        return redirect(url_for('index'))

    recommendations = get_collaborative_recommendations(user_id, ratings, movies)
    recommendations_ru = [translator.translate(movie, src='en', dest='ru').text for movie in recommendations]
    return render_template('index.html', recommendations=recommendations_ru, user_id=user_id, movies=movies)

@app.route('/add_favorite/<int:user_id>/<int:movie_id>', methods=['POST'])
def add_favorite(user_id, movie_id):
    c.execute("INSERT INTO favorites (user_id, movie_id) VALUES (?, ?)", (user_id, movie_id))
    conn.commit()
    flash('Фильм добавлен в любимые.')

    # Обновление рекомендаций
    recommendations = update_recommendations(user_id, ratings, movies, c)
    recommendations_ru = [translator.translate(movie, src='en', dest='ru').text for movie in recommendations]
    return render_template('index.html', recommendations=recommendations_ru, user_id=user_id, movies=movies)

if __name__ == '__main__':
    app.run(debug=True)
