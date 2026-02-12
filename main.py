import json
from pathlib import Path
from typing import List

from src.assessment import assess_completeness, assess_consistency, assess_timeliness
from src.evaluation import evaluate_run


def main():
    with open("../data-pollution/.latest_pollutions.json", "r") as f:
        latest = json.load(f)

    all_result_folders: List[Path | None] = []

    for folder in latest["polluted_folders"]:
        all_result_folders.extend(
            [
                assess_completeness(Path(folder) / "completeness"),
                assess_consistency(Path(folder) / "consistency", True),
                assess_timeliness(Path(folder) / "timeliness"),
            ]
        )

    print("Start evaluations")

    for result_folder in all_result_folders:
        if result_folder:
            evaluate_run(result_folder)


if __name__ == "__main__":
    main()
