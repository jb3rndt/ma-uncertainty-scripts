import datetime
import json
from hashlib import sha1
from pathlib import Path
from typing import List

from metis.dq_orchestrator import DQOrchestrator
from metis.metric.config import MetricConfig


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
    id: str | None = None,
    *,
    folder: str,
    metrics: List[str],
    metric_configs: List[str | None | MetricConfig],
    datasets: List[str] = ["weather_training_data", "auto_sales_data", "movies"],
) -> None:
    start_time = datetime.datetime.now()

    if not id:
        id = start_time.strftime("%Y%m%d_%H%M%S")

    orchestrator = DQOrchestrator(
        writer_config_path=materialize(
            {"writer_name": "csv", "path": f"results/{id}/dq_results.csv"}
        )
    )

    orchestrator.load(
        data_loader_configs=[
            materialize(
                {
                    "loader": "CSV",
                    "name": dataset,
                    "file_name": f"{folder}/polluted_{dataset}.csv",
                },
                f"results/{id}/{dataset}_loader_config.json",
            )
            for dataset in datasets
            if Path(f"{folder}/polluted_{dataset}.csv").exists()
        ]
    )

    orchestrator.assess(metrics=metrics, metric_configs=metric_configs)

    with open(
        f"results/{id}/dq_orchestrator_config.json", "w"
    ) as orchestrator_config_file:
        json.dump(
            {"metrics": metrics, "metric_configs": metric_configs},
            orchestrator_config_file,
            indent=4,
            default=lambda o: o.to_json() if hasattr(o, "to_json") else str(o),
        )

    end_time = datetime.datetime.now()
    print(f"DQ run completed in {end_time - start_time}")

    # print(
    #     f"SELECT * FROM dqresults WHERE mes_time BETWEEN '{start_time.isoformat()}' AND '{end_time.isoformat()}';"
    # )
