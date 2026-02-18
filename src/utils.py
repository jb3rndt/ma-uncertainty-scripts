import datetime
import json
from hashlib import sha1
from pathlib import Path
from typing import Generator, List, Literal, Tuple, cast

import pandas as pd

from metis.dq_orchestrator import DQOrchestrator
from metis.metric.config import MetricConfig
from src.constants import CLEANED_DATA_PATH, ORIGINAL_DATA_PATH

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
) -> Path | None:
    start_time = datetime.datetime.now()

    data_paths = [
        file
        for file in polluted_folder.glob("*.csv")
        if not file.name.endswith(".mask.csv")
        and any(dataset in file.name for dataset in datasets)
    ]

    if not data_paths:
        print(
            f"No data files found in {polluted_folder} for datasets {datasets}. Skipping."
        )
        return None

    results_folder.mkdir(parents=True, exist_ok=True)
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


def parse_columnNames(columnNames_str: str) -> str:
    columnNames = json.loads(str(columnNames_str).replace("'", '"'))
    return columnNames[0] if len(columnNames) == 1 else ",".join(columnNames)


def format_columnName(columnName: str) -> str:
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

        yield dataset, metric, dq_results, dq_certainties


def get_necessary_folders():
    with open(
        "/Users/jberndt/Documents/Masterarbeit/data-pollution/.latest_pollutions.json",
        "r",
    ) as f:
        polluted_folders = json.load(f)

    return polluted_folders["polluted_folders"] + [ORIGINAL_DATA_PATH, CLEANED_DATA_PATH]
