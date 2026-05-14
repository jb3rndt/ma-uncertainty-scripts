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
        "--run",
        help="The run to target for assessment instead of the latest.",
    )
    parser.add_argument(
        "-f",
        "--force-evaluate",
        action="store_true",
        help="Whether to force evaluation for the latest run.",
    )
    parser.add_argument(
        "--skip-explanations",
        action="store_true",
        help="Whether to skip generating explanations for the latest run.",
    )
    return parser.parse_args()


def main(
    run_name: str | None,
    dim_to_reassess: str | None,
    force_evaluate: bool,
    skip_dq_explanations: bool,
):
    all_result_folders: List[Path | None] = []
    disable_dq_explanations = skip_dq_explanations
    print(f"Run name: {run_name}")
    print(f"Reassessing dimension: {dim_to_reassess}")
    print(f"Skipping DQ explanations: {skip_dq_explanations}")

    for folder in get_necessary_folders(run_name):
        all_result_folders.extend(
            [
                assess_correctness(
                    Path(folder) / CORRECTNESS,
                    force=dim_to_reassess == CORRECTNESS,
                    disable_dq_explanations=disable_dq_explanations,
                ),
                assess_consistency(
                    Path(folder) / CONSISTENCY,
                    force=dim_to_reassess == CONSISTENCY,
                    disable_dq_explanations=disable_dq_explanations,
                ),
                assess_tuple_consistency(
                    Path(folder) / CONSISTENCY_TUPLE,
                    force=dim_to_reassess == CONSISTENCY_TUPLE,
                    disable_dq_explanations=disable_dq_explanations,
                ),
                assess_timeliness(
                    Path(folder) / TIMELINESS,
                    force=dim_to_reassess == TIMELINESS,
                    disable_dq_explanations=disable_dq_explanations,
                ),
                assess_completeness(
                    Path(folder) / COMPLETENESS,
                    force=dim_to_reassess == COMPLETENESS,
                    disable_dq_explanations=disable_dq_explanations,
                ),
            ]
        )

    all_result_folders.extend(
        [
            assess_completeness(
                ORIGINAL_DATA_PATH / COMPLETENESS,
                force=dim_to_reassess == COMPLETENESS,
                disable_dq_explanations=disable_dq_explanations,
            ),
            assess_correctness(
                ORIGINAL_DATA_PATH / CORRECTNESS,
                force=dim_to_reassess == CORRECTNESS,
                disable_dq_explanations=disable_dq_explanations,
            ),
            assess_consistency_original_dataset(
                ORIGINAL_DATA_PATH / CONSISTENCY,
                force=dim_to_reassess == CONSISTENCY,
                disable_dq_explanations=disable_dq_explanations,
            ),
            assess_tuple_consistency_original_dataset(
                ORIGINAL_DATA_PATH / CONSISTENCY_TUPLE,
                force=dim_to_reassess == CONSISTENCY_TUPLE,
                disable_dq_explanations=disable_dq_explanations,
            ),
            assess_timeliness(
                ORIGINAL_DATA_PATH / TIMELINESS,
                force=dim_to_reassess == TIMELINESS,
                disable_dq_explanations=disable_dq_explanations,
            ),
        ]
    )

    print("Start evaluations")

    for result_folder in all_result_folders:
        if result_folder:
            evaluate_run(result_folder, not force_evaluate)


if __name__ == "__main__":
    args = parse_args()
    dim_to_reassess = args.reassess
    force_evaluate = args.force_evaluate
    run_name = args.run
    skip_dq_explanations = args.skip_explanations

    print("Args:", args)

    main(run_name, dim_to_reassess, force_evaluate, skip_dq_explanations)
