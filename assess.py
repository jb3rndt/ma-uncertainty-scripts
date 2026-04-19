from pathlib import Path
from typing import List

from src.assessment import (
    assess_completeness,
    assess_consistency,
    assess_consistency_original_dataset,
    assess_correctness,
    assess_timeliness,
    assess_tuple_consistency,
    assess_tuple_consistency_original_dataset,
)
from src.constants import ORIGINAL_DATA_PATH
from src.evaluation.evaluation import evaluate_run
from src.utils import get_necessary_folders


def main():
    all_result_folders: List[Path | None] = []

    for folder in get_necessary_folders():
        all_result_folders.extend(
            [
                assess_completeness(Path(folder) / "completeness"),
                assess_correctness(Path(folder) / "correctness"),
                assess_consistency(Path(folder) / "consistency"),
                assess_tuple_consistency(Path(folder) / "consistency_tuple"),
                assess_timeliness(Path(folder) / "timeliness"),
            ]
        )

    all_result_folders.extend(
        [
            # assess_completeness(ORIGINAL_DATA_PATH / "completeness"),
            # assess_correctness(ORIGINAL_DATA_PATH / "correctness"),
            assess_consistency_original_dataset(ORIGINAL_DATA_PATH / "consistency"),
            assess_tuple_consistency_original_dataset(ORIGINAL_DATA_PATH / "consistency_tuple"),
            assess_timeliness(ORIGINAL_DATA_PATH / "timeliness"),
        ]
    )

    print("Start evaluations")

    for result_folder in all_result_folders:
        if result_folder:
            evaluate_run(result_folder)


if __name__ == "__main__":
    main()
