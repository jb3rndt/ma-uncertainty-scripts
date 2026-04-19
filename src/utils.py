import datetime
import json
import shutil
from hashlib import sha1
from pathlib import Path
from typing import Generator, List, Literal, Tuple, cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from metis.dq_orchestrator import DQOrchestrator
from metis.metric.config import MetricConfig
from src.constants import CLEANED_DATA_PATH, POLLUTION_RATES
from src.evaluation.types import ColumnEvaluationResult, ColumnRawData

DSLiteral = Literal["weather", "auto_sales", "movies", "open_library"]
datasets: List[DSLiteral] = ["weather", "auto_sales", "movies", "open_library"]


def materialize(data: str | dict, file_path: Path | str | None = None) -> str:
    """Materializes a JSON string or dictionary to a file and returns the path as a string.

    Args:
        json (str | dict): The JSON content as a string or dictionary.
        file_path (str | None, optional): The file path to write the JSON content to. Defaults to None and chooses a temporary file.

    Returns:
        str: The file path where the JSON content was written.
    """

    data_hash = sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()

    final_path = Path(file_path if file_path else f"tmp/materialized_{data_hash}.json")

    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path:
        with open(final_path, "w") as f:
            json.dump(data, f, indent=4, default=str)

    return str(final_path)


def execute_run(
    *,
    results_folder: Path,
    polluted_folder: Path,
    metrics: List[str],
    metric_configs: List[str | None | MetricConfig],
    datasets: List[DSLiteral] = datasets,
    force: bool = False,
) -> Path | None:
    if results_folder.exists() and not force:
        print(f"Results folder {results_folder.absolute()} already exists. SKIPPING!")
        return results_folder

    try:
        start_time = datetime.datetime.now()

        possible_data_paths = [
            polluted_folder / f"{dataset}.{type}.csv"
            for dataset in datasets
            for type in ["polluted", "cleaned", "original"]
        ]

        data_paths = [file for file in possible_data_paths if file.exists()]

        if not data_paths:
            print(
                f"No data files found in {polluted_folder} for datasets {datasets}. Skipping."
            )
            return None

        if results_folder.exists():
            shutil.rmtree(results_folder)
        results_folder.mkdir(parents=True)
        orchestrator = DQOrchestrator(
            writer_config_path=materialize(
                {"writer_name": "csv", "path": str(results_folder / "dq_results.csv")}
            )
        )

        orchestrator.load(
            data_loader_configs=[
                materialize(
                    {
                        "loader": "CSV",
                        "name": path.stem,
                        "file_name": str(path),
                    },
                    str(results_folder / f"{path.stem}.loader_config.json"),
                )
                for path in data_paths
            ]
        )

        orchestrator.assess(metrics=metrics, metric_configs=metric_configs)

        with open(
            results_folder / "dq_orchestrator_config.json", "w"
        ) as orchestrator_config_file:
            json.dump(
                {"metrics": metrics, "metric_configs": metric_configs},
                orchestrator_config_file,
                indent=4,
                default=lambda o: o.to_json() if hasattr(o, "to_json") else str(o),
            )

        end_time = datetime.datetime.now()
        print(f"DQ run completed in {end_time - start_time}")

        return results_folder
    except Exception as e:
        # Clear and remove results_folder to avoid confusion with incomplete results in case of errors
        if results_folder.exists() and results_folder.is_dir():
            for file in results_folder.glob("*"):
                file.unlink()
            results_folder.rmdir()
        raise e


def parse_columnNames(columnNames_str: str) -> str:
    columnNames = json.loads(str(columnNames_str).replace("'", '"'))
    return columnNames[0] if len(columnNames) == 1 else ",".join(columnNames)


def format_columnName(columnName):
    return "Full Tuple" if "," in columnName else columnName


def grouped_results_and_certainties(
    flat_results: pd.DataFrame,
) -> Generator[Tuple[str, str, pd.DataFrame, pd.DataFrame], None, None]:
    for key, index in flat_results.groupby(["tableName", "DQmetric"]).groups.items():
        tableName, DQmetric = cast(tuple, key)
        metric = str(DQmetric)
        dataset = str(tableName)
        group = flat_results.loc[index]
        dq_results = pd.DataFrame(
            None, index=pd.RangeIndex(stop=group["rowIndex"].max() + 1)
        )
        dq_certainties = pd.DataFrame(None, index=dq_results.index)

        for column_key, data in group.groupby("columnNames"):
            column = parse_columnNames(str(column_key))
            dq_results.loc[data["rowIndex"].tolist(), column] = data[
                "DQvalue"
            ].to_numpy()

            dq_certainties.loc[data["rowIndex"].tolist(), column] = (
                data["DQexplanation"]
                .apply(
                    lambda x: (
                        json.loads(str(x).replace("'", '"')).get("certainty", 1.0)
                        if x and not pd.isna(x) and len(x) > 0
                        else 1.0
                    )
                )
                .to_numpy()
            )

        dq_results.fillna(1.0, inplace=True)
        dq_certainties.fillna(1.0, inplace=True)

        yield dataset, metric, dq_results, dq_certainties


def make_labels(df: pd.DataFrame) -> List[str]:
    types = df["type"].unique()
    if len(types) != 1:
        raise ValueError(f"Expected exactly one type, but got {[t for t in types]}")
    data_type = types[0]
    if data_type == "polluted":
        return [f"{x}" for x in df["pollution_rate"].unique()]
    else:
        assert (
            len(df["pollution_rate"].unique()) == 1
        ), "Expected exactly one pollution rate for clean/original data"
        if data_type == "cleaned":
            return ["clean"]
        elif data_type == "original":
            return ["original"]
        else:
            raise ValueError(f"Unknown type {data_type}")


def normalize_pollution_rate(rate: float) -> float:
    return POLLUTION_RATES[np.argmin(np.abs(np.array(POLLUTION_RATES) - rate))]


def get_necessary_folders(run_name: str | None = None):
    polluted_path = Path(
        "/Users/jberndt/Documents/Masterarbeit/data-pollution/data/polluted"
    )
    pollution_folder = (
        max(polluted_path.glob("*")) if run_name is None else polluted_path / run_name
    )
    assert (
        pollution_folder.exists()
    ), f"Pollution folder {pollution_folder} does not exist."
    polluted_folders = [
        folder for folder in pollution_folder.glob("*") if folder.is_dir()
    ]

    return sorted(polluted_folders) + [
        # ORIGINAL_DATA_PATH,
        CLEANED_DATA_PATH,
    ]


def load_raw_results(run_name: str | None = None):
    return {
        (dim_folder.name, folder.name): json.load(
            (dim_folder / "results" / "raw_results.json").open()
        )
        for folder in get_necessary_folders(run_name)
        for dim_folder in folder.glob("*")
        if dim_folder.is_dir()
        and (dim_folder / "results" / "raw_results.json").exists()
    }


def flatten_raw_results(raw_results: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dimension": dim,
                "folder": folder,
                "metric": metric,
                "dataset": dataset.split(".")[0],
                "type": dataset.split(".")[1],
                "column": format_columnName(column),
                "raw_column": column,
                "pollution_rate": normalize_pollution_rate(
                    results[column]["pollution_ratio"]
                ),
                "original_pollution_rate": results[column]["pollution_ratio"],
                "pollution_mechanism": results[column]["pollution_mechanism"],
                "result": ColumnRawData(**results[column]),
            }
            for (dim, folder), evaluation in raw_results.items()
            for metric, datasets in evaluation.items()
            for dataset, results in datasets.items()
            for column in results.keys()
        ]
    )


def load_evaluations(run_name: str | None = None):
    return {
        (dim_folder.name, folder.name): json.load(
            (dim_folder / "results" / "evaluations.json").open()
        )
        for folder in get_necessary_folders(run_name)
        for dim_folder in folder.glob("*")
        if dim_folder.is_dir()
        and (dim_folder / "results" / "evaluations.json").exists()
    }


def flatten_evaluations(evaluations: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dimension": dim,
                "folder": folder,
                "metric": metric,
                "dataset": dataset.split(".")[0],
                "type": dataset.split(".")[1],
                "column": format_columnName(column),
                "raw_column": column,
                "pollution_rate": normalize_pollution_rate(
                    results[column]["pollution_ratio"]
                ),
                "original_pollution_rate": results[column]["pollution_ratio"],
                "pollution_mechanism": results[column]["pollution_mechanism"],
                "result": ColumnEvaluationResult(**results[column]),
            }
            for (dim, folder), evaluation in evaluations.items()
            for metric, datasets in evaluation.items()
            for dataset, results in datasets.items()
            for column in results.keys()
        ]
    )


def res(df: pd.DataFrame) -> List[ColumnEvaluationResult]:
    return df["result"].tolist()


def res_raw(df: pd.DataFrame) -> List[ColumnRawData]:
    return df["result"].tolist()


def get_raw_results(run_name: str | None = None) -> pd.DataFrame:
    key = "__df_raw_results__"
    if key not in globals():
        print("Loading raw results...")
        globals()[key] = flatten_raw_results(load_raw_results(run_name))

    return globals()[key]


def get_evaluations(run_name: str | None = None) -> pd.DataFrame:
    key = "__df_evaluations__"
    if key not in globals():
        print("Loading evaluations...")
        globals()[key] = flatten_evaluations(load_evaluations(run_name))

    return globals()[key]


def first_or_none(iterable):
    return next(iter(iterable), None)


def grouped_figure(
    df: pd.DataFrame,
    figureby: List[str],
    colby: List[str],
    *,
    figsize: Tuple[int, int] = (3, 3),
    nrows=1,
):
    for fig_key, fig_group in df.groupby(figureby):
        grouped_by_column = fig_group.groupby(colby)
        ncols = len(grouped_by_column)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize[0] * ncols, figsize[1] * nrows),
            sharex=True,
            sharey=True,
        )
        axes = np.array(axes).reshape(nrows, ncols)
        for row, row_axes in enumerate(axes):
            for (col_key, col_group), (col, ax) in zip(
                grouped_by_column, enumerate(row_axes)
            ):
                yield fig_key, col_key, col_group, ax, fig, row, col
