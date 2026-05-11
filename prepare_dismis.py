import argparse
import json
from pathlib import Path

from metis.dismis.preparation.generate_example_dmvs import generate_example_dmvs
from metis.dismis.preparation.precompute_detection_example_embeddings import (
    precompute_detection_example_embeddings,
)
from metis.dismis.preparation.precompute_value_embeddings import (
    precompute_value_embeddings,
)
from src.constants import CLEANED_DATA_PATH, ORIGINAL_DATA_PATH
from src.utils import get_necessary_folders

LLM_MODEL = "qwen3:8b"
LLM_BASE_URL = "http://localhost:11434/v1/"
EMBEDDING_MODEL = "qwen3-embedding:8b"

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare for data quality assessment.")
    parser.add_argument(
        "--skip-explanations",
        action="store_true",
        help="Whether to skip generating explanations for the latest run.",
    )
    return parser.parse_args()

def generate_example_dmvs_detection(dataset_folder: Path, dataset: str):
    example_dmvs_detection_path = (
        dataset_folder / f"{dataset}_example_dmvs_detection.json"
    )
    dataset_path = dataset_folder / f"{dataset}.csv"

    print(f"  Generating example DMVs at {example_dmvs_detection_path}...")
    generate_example_dmvs(
        dataset_path,
        LLM_MODEL,
        LLM_BASE_URL,
        "placeholder",
        example_dmvs_detection_path.stem,
        column_types=["numeric", "categorical", "date", "text"],
    )

    if (
        example_dmvs_detection_path.parent
        / f"{example_dmvs_detection_path.stem}.embeddings.json"
    ).exists():
        print(f"  Skipping example embeddings...")
    else:
        print(f"  Precomputing example embeddings at {example_dmvs_detection_path}...")
        precompute_detection_example_embeddings(
            model_name=EMBEDDING_MODEL,
            llm_base_url=LLM_BASE_URL,
            llm_api_key="placeholder",
            json_files=str(example_dmvs_detection_path),
        )


def generate_example_dmvs_pollution(dataset_folder: Path, dataset: str):
    example_dmvs_pollution_path = (
        dataset_folder / f"{dataset}_example_dmvs_pollution.json"
    )
    dataset_path = dataset_folder / f"{dataset}.csv"

    print(f"  Generating example DMVs at {example_dmvs_pollution_path}...")
    generate_example_dmvs(
        dataset_path,
        LLM_MODEL,
        LLM_BASE_URL,
        "placeholder",
        example_dmvs_pollution_path.stem,
        column_types=["numeric", "categorical", "date", "text"],
    )


def compute_embeddings(dataset_folder: Path, dataset: str):
    dataset_types_path = dataset_folder / f"{dataset}_types.json"
    dataset_path = (
        dataset_folder
        / "completeness"
        / f"{dataset}.{'cleaned' if 'cleaned' in str(dataset_folder) else 'original'}.csv"
    )
    value_embeddings_path = (
        dataset_path.parent / f"{dataset_path.stem}_value_embeddings.json"
    )
    cached_embeddings = (
        json.load(open(value_embeddings_path, "r"))
        if value_embeddings_path.exists()
        else None
    )
    print(f"  Precomputing value embeddings at {dataset_folder}...")
    precompute_value_embeddings(
        model_name=EMBEDDING_MODEL,
        llm_base_url=LLM_BASE_URL,
        llm_api_key="placeholder",
        datasets_and_types=(
            str(dataset_path),
            str(dataset_types_path),
        ),
        cached_embeddings=cached_embeddings,
        column_types=["categorical", "date", "text"],
    )


def main(run_name: str | None):
    for dataset in [
        "auto_sales",
        "movies",
        "weather",
    ]:
        print(dataset)
        generate_example_dmvs_detection(ORIGINAL_DATA_PATH, dataset)
        generate_example_dmvs_detection(CLEANED_DATA_PATH, dataset)
        generate_example_dmvs_pollution(CLEANED_DATA_PATH, dataset)

        compute_embeddings(CLEANED_DATA_PATH, dataset)
        compute_embeddings(ORIGINAL_DATA_PATH, dataset)

        dataset_types_path = CLEANED_DATA_PATH / f"{dataset}_types.json"
        cleaned_value_embeddings_path = (
            CLEANED_DATA_PATH
            / "completeness"
            / f"{dataset}.cleaned_value_embeddings.json"
        )

        cached_embeddings = json.load(open(cleaned_value_embeddings_path, "r"))

        for folder in get_necessary_folders(run_name):
            if "ECAR" not in str(folder):
                continue

            value_embeddings_path = (
                folder / "completeness" / f"{dataset}.polluted_value_embeddings.json"
            )
            dataset_path = folder / "completeness" / f"{dataset}.polluted.csv"
            print(f"  Precomputing value embeddings at {value_embeddings_path}...")
            precompute_value_embeddings(
                model_name=EMBEDDING_MODEL,
                llm_base_url=LLM_BASE_URL,
                llm_api_key="placeholder",
                datasets_and_types=(
                    str(dataset_path),
                    str(dataset_types_path),
                ),
                cached_embeddings=cached_embeddings,
                column_types=["categorical", "date", "text"],
            )


if __name__ == "__main__":
    args = parse_args()
    run_name = args.run

    main(run_name)
