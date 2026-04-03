import dataclasses
import json
from pathlib import Path
from time import time

import pandas as pd

from src.downstream_task.config import RegressionConfig
from src.downstream_task.weather_rain_prediction import evaluate_weather_rain_prediction

if __name__ == "__main__":
    run_name = int(time())
    Path(f"regression-results/{run_name}").mkdir(exist_ok=True, parents=True)
    config = RegressionConfig(
        random_seed=run_name,
        max_leaf_nodes=4,
        min_samples_split=5,
        test_size=0.3,
        n_estimators=300,
        max_depth=None,
        subsample=0.5,
        learning_rate=0.2,
        cols=[
            # "Humidity3pm",
            # "Humidity9am",
            "MaxTemp",
            "MinTemp",
            "Temp9am",
            "Temp3pm",
            "Pressure3pm",
            "Pressure9am",
            "RainToday",
            "RainTomorrow",
            "WindGustSpeed",
            "WindSpeed3pm",
            "WindSpeed9am",
            # "Pressure",
            # "Humidity",
            # "Temp",
            # "WindSpeed",
        ],
    )
    measurements = evaluate_weather_rain_prediction(config)

    df_gb = pd.DataFrame(measurements)
    df_gb.to_csv(f"regression-results/{run_name}/results.csv", index=False)
    json.dump(
        dataclasses.asdict(config),
        open(f"regression-results/{run_name}/config.json", "w"),
        indent=2,
    )
