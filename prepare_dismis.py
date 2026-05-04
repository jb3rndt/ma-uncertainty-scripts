import json
from pathlib import Path

from metis.dismis.preparation.generate_example_dmvs import generate_example_dmvs
from metis.dismis.preparation.precompute_detection_example_embeddings import (
    precompute_detection_example_embeddings,
)
from metis.dismis.preparation.precompute_value_embeddings import (
    precompute_value_embeddings,
)
from src.constants import CLEANED_DATA_PATH
from src.utils import get_necessary_folders

LLM_MODEL = "qwen3:8b"
LLM_BASE_URL = "http://localhost:11434/v1/"
EMBEDDING_MODEL = "qwen3-embedding:8b"


def main():
    for dataset in [
        "auto_sales",
        "movies",
        "weather",
    ]:
        print(dataset)
        dataset_path = CLEANED_DATA_PATH / f"{dataset}.csv"
        dataset_types_path = dataset_path.parent / f"{dataset}_types.json"
        example_dmvs_detection_filename = f"{dataset}_example_dmvs_detection.json"

        for example_dmvs_filename in [
            example_dmvs_detection_filename,
            f"{dataset}_example_dmvs_pollution.json",
        ]:
            example_dmvs_path = dataset_path.parent / example_dmvs_filename
            print(f"  Generating example DMVs at {example_dmvs_path}...")
            generate_example_dmvs(
                dataset_path,
                LLM_MODEL,
                LLM_BASE_URL,
                "placeholder",
                Path(example_dmvs_filename).stem,
                column_types=["numeric", "categorical", "date", "text"],
            )

        example_dmvs_detection_path = (
            dataset_path.parent / example_dmvs_detection_filename
        )
        if (
            Path(example_dmvs_detection_path).parent
            / f"{Path(example_dmvs_detection_path).stem}.embeddings.json"
        ).exists():

            print(f"  Skipping example embeddings...")
        else:
            print(
                f"  Precomputing example embeddings at {example_dmvs_detection_path}..."
            )
            precompute_detection_example_embeddings(
                model_name=EMBEDDING_MODEL,
                llm_base_url=LLM_BASE_URL,
                llm_api_key="placeholder",
                json_files=str(example_dmvs_detection_path),
            )

        cleaned_dataset_path = (
            CLEANED_DATA_PATH / "completeness" / f"{dataset}.cleaned.csv"
        )
        cleaned_value_embeddings_path = (
            cleaned_dataset_path.parent
            / f"{cleaned_dataset_path.stem}_value_embeddings.json"
        )
        cached_clean_embeddings = (
            json.load(open(cleaned_value_embeddings_path, "r"))
            if cleaned_value_embeddings_path.exists()
            else None
        )
        print(f"  Precomputing cleaned value embeddings at {CLEANED_DATA_PATH}...")
        precompute_value_embeddings(
            model_name=EMBEDDING_MODEL,
            llm_base_url=LLM_BASE_URL,
            llm_api_key="placeholder",
            datasets_and_types=(
                str(cleaned_dataset_path),
                str(dataset_types_path),
            ),
            cached_embeddings=cached_clean_embeddings,
            column_types=["numeric", "categorical", "date", "text"],
        )

        print(f"  Reloading cached embeddings from {cleaned_value_embeddings_path}")
        cached_embeddings = json.load(open(cleaned_value_embeddings_path, "r"))

        for folder in get_necessary_folders("20260428_115054"):
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
                column_types=["numeric", "categorical", "date", "text"],
            )


if __name__ == "__main__":
    main()
