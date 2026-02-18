from pathlib import Path
from typing import List

from src.assessment import assess_completeness, assess_consistency, assess_timeliness
from src.evaluation.evaluation import evaluate_run
from src.utils import get_necessary_folders


def main():
    all_result_folders: List[Path | None] = []

    for folder in get_necessary_folders():
        all_result_folders.extend(
            [
                assess_completeness(Path(folder) / "completeness"),
                assess_consistency(Path(folder) / "consistency"),
                assess_timeliness(Path(folder) / "timeliness"),
            ]
        )

    print("Start evaluations")

    for result_folder in all_result_folders:
        if result_folder:
            evaluate_run(result_folder)


if __name__ == "__main__":
    main()
