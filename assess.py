import argparse
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
from src.constants import (
    COMPLETENESS,
    CONSISTENCY,
    CONSISTENCY_TUPLE,
    CORRECTNESS,
    ORIGINAL_DATA_PATH,
    TIMELINESS,
)
from src.evaluation.evaluation import evaluate_run
from src.utils import get_necessary_folders

VALID_DIMS = [TIMELINESS, COMPLETENESS, CONSISTENCY, CONSISTENCY_TUPLE, CORRECTNESS]


def parse_args():
    parser = argparse.ArgumentParser(description="Assess data quality.")
    parser.add_argument(
        "-r",
        "--reassess",
        choices=VALID_DIMS,
        help="Dimension to reassess in the latest run.",
    )
    parser.add_argument(
        "-f",
        "--force-evaluate",
        action="store_true",
        help="Whether to force evaluation for the latest run.",
    )
    return parser.parse_args()


def main(dim_to_reassess: str | None, force_evaluate: bool):
    all_result_folders: List[Path | None] = []

    for folder in get_necessary_folders():
        all_result_folders.extend(
            [
                assess_completeness(
                    Path(folder) / COMPLETENESS, force=dim_to_reassess == COMPLETENESS
                ),
                assess_correctness(
                    Path(folder) / CORRECTNESS, force=dim_to_reassess == CORRECTNESS
                ),
                assess_consistency(
                    Path(folder) / CONSISTENCY, force=dim_to_reassess == CONSISTENCY
                ),
                assess_tuple_consistency(
                    Path(folder) / CONSISTENCY_TUPLE,
                    force=dim_to_reassess == CONSISTENCY_TUPLE,
                ),
                assess_timeliness(
                    Path(folder) / TIMELINESS, force=dim_to_reassess == TIMELINESS
                ),
            ]
        )

    all_result_folders.extend(
        [
            assess_completeness(
                ORIGINAL_DATA_PATH / COMPLETENESS, force=force_evaluate
            ),
            assess_correctness(ORIGINAL_DATA_PATH / CORRECTNESS, force=force_evaluate),
            assess_consistency_original_dataset(
                ORIGINAL_DATA_PATH / CONSISTENCY, force=force_evaluate
            ),
            assess_tuple_consistency_original_dataset(
                ORIGINAL_DATA_PATH / CONSISTENCY_TUPLE, force=force_evaluate
            ),
            assess_timeliness(ORIGINAL_DATA_PATH / TIMELINESS, force=force_evaluate),
        ]
    )

    print("Start evaluations")

    for result_folder in all_result_folders:
        if result_folder:
            evaluate_run(result_folder)


if __name__ == "__main__":
    args = parse_args()
    dim_to_reassess = args.reassess
    force_evaluate = args.force_evaluate

    main(dim_to_reassess, force_evaluate)
