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

VALID_DIMS = [TIMELINESS, CONSISTENCY, CONSISTENCY_TUPLE, CORRECTNESS, COMPLETENESS]


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
        "--disable-dq-explanations",
        action="store_true",
        help="Whether to disable generating explanations for the latest run.",
    )
    return parser.parse_args()


DIM_ASSESS_FUNCTION = {
    CORRECTNESS: assess_correctness,
    CONSISTENCY: assess_consistency,
    CONSISTENCY_TUPLE: assess_tuple_consistency,
    TIMELINESS: assess_timeliness,
    COMPLETENESS: assess_completeness,
}

ORIGINAL_DIM_ASSESS_FUNCTION = {
    CORRECTNESS: assess_correctness,
    CONSISTENCY: assess_consistency_original_dataset,
    CONSISTENCY_TUPLE: assess_tuple_consistency_original_dataset,
    TIMELINESS: assess_timeliness,
    COMPLETENESS: assess_completeness,
}


def main(
    run_name: str | None,
    dim_to_reassess: str | None,
    force_evaluate: bool,
    disable_dq_explanations: bool,
):
    for folder in get_necessary_folders(run_name, cleaned=True):
        for dim in VALID_DIMS:
            result_folder = DIM_ASSESS_FUNCTION[dim](
                folder / dim,
                force=dim_to_reassess == dim,
                disable_dq_explanations=disable_dq_explanations,
            )
            if result_folder:
                evaluate_run(result_folder, not force_evaluate)

    for dim in VALID_DIMS:
        result_folder = ORIGINAL_DIM_ASSESS_FUNCTION[dim](
            ORIGINAL_DATA_PATH / dim,
            force=dim_to_reassess == dim,
            disable_dq_explanations=disable_dq_explanations,
        )
        if result_folder:
            evaluate_run(result_folder, not force_evaluate)


if __name__ == "__main__":
    args = parse_args()
    dim_to_reassess = args.reassess
    force_evaluate = args.force_evaluate
    run_name = args.run
    disable_dq_explanations = args.disable_dq_explanations

    print("Args:", args)

    main(run_name, dim_to_reassess, force_evaluate, disable_dq_explanations)
