from pathlib import Path

import pandas as pd

from src.cleaning.movies import clean_movies

ORIGINAL_DATA_PATH = Path("../data-pollution/data/original")
CLEANED_DATA_PATH = Path("../data-pollution/data/cleaned")


def main():
    # Movies
    movies = pd.read_csv(ORIGINAL_DATA_PATH / "movies.csv")
    # list_release_date_patches(movies)
    # select_closest_release_date()
    cleaned_movies = clean_movies(movies)
    cleaned_movies.to_csv(CLEANED_DATA_PATH / "movies.csv", index=False)


if __name__ == "__main__":
    main()
