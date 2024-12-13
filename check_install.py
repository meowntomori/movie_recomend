import subprocess
import sys

# Список необходимых библиотек
required_libraries = [
    'flask',
    'pandas',
    'sqlite3',
    'fuzzywuzzy',
    'googletrans==4.0.0-rc1',
    'scikit-learn',
    'surprise',
    'python-Levenshtein'
]

# Функция для проверки установки библиотеки
def check_install(library):
    try:
        __import__(library)
        print(f"{library} is already installed.")
    except ImportError:
        print(f"{library} is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# Проверка и установка всех необходимых библиотек
for library in required_libraries:
    check_install(library)

print("All required libraries are installed.")
