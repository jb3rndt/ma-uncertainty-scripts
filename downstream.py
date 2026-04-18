import dataclasses
import json
from pathlib import Path
from time import time

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.downstream_task.auto_sales_prediction import evaluate_auto_sales_prediction
from src.downstream_task.config import RegressionConfig
from src.downstream_task.movies_prediction import evaluate_movies_prediction
from src.downstream_task.weather_rain_prediction import evaluate_weather_rain_prediction

downstream_tasks = {
    "weather": (
        evaluate_weather_rain_prediction,
        lambda run_name: RegressionConfig(
            random_seed=run_name,
            max_leaf_nodes=4,
            min_samples_split=5,
            test_size=0.3,
            n_estimators=300,
            max_depth=None,
            subsample=0.5,
            learning_rate=0.2,
            n_runs=50,
            feature_cols=[
                # "Humidity3pm",
                # "Humidity9am",
                "MaxTemp",
                "MinTemp",
                "Temp9am",
                "Temp3pm",
                "Pressure3pm",
                "Pressure9am",
                "RainToday",
                "WindGustSpeed",
                "WindSpeed3pm",
                "WindSpeed9am",
                # "Pressure",
                # "Humidity",
                # "Temp",
                # "WindSpeed",
            ],
            target_col="RainTomorrow",
        ),
        {
            "title": "Gradient Boosting Rain Prediction Performance on Different Parts of the Weather Dataset",
            "xlabel": "Data Quality Threshold for Filtering",
            "ylabel": "F1 Score",
        },
    ),
    "auto_sales": (
        evaluate_auto_sales_prediction,
        lambda run_name: RegressionConfig(
            random_seed=run_name,
            max_leaf_nodes=4,
            min_samples_split=5,
            test_size=0.3,
            n_estimators=300,
            max_depth=None,
            subsample=0.5,
            learning_rate=0.2,
            n_runs=50,
            feature_cols=["QUANTITYORDERED", "PRICEEACH", "ORDERLINENUMBER", "MSRP"],
            target_col="SALES",
        ),
        {
            "title": "Gradient Boosting Sales Prediction Performance on Different Parts of the Auto Sales Dataset",
            "xlabel": "Data Quality Threshold for Filtering",
            "ylabel": "RMSE",
        },
    ),
    "movies": (
        evaluate_movies_prediction,
        lambda run_name: RegressionConfig(
            random_seed=run_name,
            max_leaf_nodes=4,
            min_samples_split=5,
            test_size=0.3,
            n_estimators=300,
            max_depth=None,
            subsample=0.5,
            learning_rate=0.2,
            feature_cols=[
                "budget",
                "revenue",
                "vote_count",
                "popularity",
                "runtime",
                "original_language",
                "status",
                "genres",
                "production_countries",
            ],
            target_col="vote_average",
        ),
        {
            "title": "Gradient Boosting Movie Rating Prediction Performance on Different Parts of the Movies Dataset",
            "xlabel": "Data Quality Threshold for Filtering",
            "ylabel": "RMSE",
        },
    ),
}


def run_downstream_task(task_name: str):
    evaluate, get_config, _ = downstream_tasks[task_name]

    print(f"Evaluating {task_name}...")
    run_name = int(time())
    results_folder = Path(f"regression-results/{run_name}")
    results_folder.mkdir(exist_ok=True, parents=True)
    config = get_config(run_name)

    measurements = evaluate(config)

    df_gb = pd.DataFrame(measurements)
    df_gb.to_csv(results_folder / "results.csv", index=False)
    json.dump(
        dataclasses.asdict(config) | {"dataset": task_name},
        open(results_folder / "config.json", "w"),
        indent=2,
    )
    return results_folder


def plot_results(results_folder: Path):
    sns.set_theme()

    df = pd.read_csv(results_folder / "results.csv")

    config = json.load(open(results_folder / "config.json"))
    dataset = config.get("dataset")
    _, _, labels = downstream_tasks[dataset]

    fig, ax = plt.subplots()
    for i, (data, group) in enumerate(df.groupby("data")):
        mean_scores = group.groupby("threshold", dropna=False)["score"].mean()
        if all(mean_scores.index.isna()):
            ax.axhline(
                mean_scores.iloc[0], color=f"C{i}", linestyle="--", label=f"{data}"
            )
        else:
            mean_scores.plot(ax=ax, color=f"C{i}", label=f"{data}")

    ax.legend()
    ax.set_ylabel(labels["ylabel"])
    ax.set_xlabel(labels["xlabel"])
    fig.suptitle(labels["title"])
    fig.savefig(results_folder / "plot.pdf", bbox_inches="tight")


def main():
    for task_name in [
        "weather",
        "auto_sales",
        "movies",
    ]:
        run_downstream_task(task_name)


if __name__ == "__main__":
    main()
