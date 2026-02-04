import json
from pathlib import Path

from src.assessment import assess_completeness, assess_consistency, assess_timeliness
from src.evaluation import evaluate_run


def main():
    with open("../data-pollution/latest_pollutions.json", "r") as f:
        latest = json.load(f)

    all_result_folders = []
    polluted_folders_base = Path(
        "/Users/jberndt/Documents/Masterarbeit/Metis/data/local/polluted/"
    )

    for folder in latest["polluted_folders"]:
        all_result_folders.extend(
            [
                assess_completeness(polluted_folders_base / folder / "completeness"),
                assess_consistency(polluted_folders_base / folder / "consistency"),
                assess_timeliness(polluted_folders_base / folder / "timeliness"),
            ]
        )

    print("Start evaluations")

    for result_folder in all_result_folders:
        evaluate_run(result_folder)


if __name__ == "__main__":
    main()
