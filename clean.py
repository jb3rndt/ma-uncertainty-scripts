import pandas as pd

from src.cleaning.auto_sales import clean_auto_sales
from src.cleaning.movies import clean_movies
from src.cleaning.weather import clean_weather
from src.cleaning.open_library import clean_open_library

from src.constants import CLEANED_DATA_PATH, ORIGINAL_DATA_PATH


def main():
    cleaning_config = {
        "movies.csv": clean_movies,
        "weather.csv": clean_weather,
        "auto_sales.csv": clean_auto_sales,
        "open_library.csv": clean_open_library,
    }

    for file_name, cleaning_function in cleaning_config.items():
        print(f"Cleaning {file_name}...")
        data = pd.read_csv(ORIGINAL_DATA_PATH / file_name)
        cleaned_data = cleaning_function(data)
        cleaned_data.to_csv(CLEANED_DATA_PATH / file_name, index=False)


if __name__ == "__main__":
    main()
