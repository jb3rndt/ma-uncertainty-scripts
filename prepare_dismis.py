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
        dataset_path = Path(
            f"/Users/jberndt/Documents/Masterarbeit/data-pollution/data/cleaned/{dataset}.csv"
        )
        example_dmvs_path = (
            dataset_path.parent / f"{dataset}_example_dmvs_detection.json"
        )
        dataset_types_path = dataset_path.parent / f"{dataset}_types.json"

        if not example_dmvs_path.exists():
            print(f"Generating example DMVs for {dataset} at {example_dmvs_path}...")
            generate_example_dmvs(
                dataset_path,
                LLM_MODEL,
                LLM_BASE_URL,
                "placeholder",
                f"{dataset}_example_dmvs_detection",
            )

        if not (
            Path(example_dmvs_path).parent
            / f"{Path(example_dmvs_path).stem}.embeddings.json"
        ).exists():
            print(
                f"Precomputing example embeddings for {dataset} at {example_dmvs_path}..."
            )
            precompute_detection_example_embeddings(
                model_name=EMBEDDING_MODEL,
                llm_base_url=LLM_BASE_URL,
                llm_api_key="placeholder",
                json_files=str(example_dmvs_path),
            )

        cleaned_value_embeddings_path = (
            CLEANED_DATA_PATH
            / "completeness"
            / f"{dataset}.cleaned_value_embeddings.json"
        )
        if not cleaned_value_embeddings_path.exists():
            print(
                f"Precomputing cleaned value embeddings for {dataset} at {CLEANED_DATA_PATH}..."
            )
            precompute_value_embeddings(
                model_name=EMBEDDING_MODEL,
                llm_base_url=LLM_BASE_URL,
                llm_api_key="placeholder",
                datasets_and_types=(
                    str(dataset_path),
                    str(dataset_types_path),
                ),
            )

        cached_embeddings = json.load(open(cleaned_value_embeddings_path, "r"))

        for folder in get_necessary_folders():
            if "ECAR" not in str(folder):
                continue

            value_embeddings_path = (
                folder / "completeness" / f"{dataset}.polluted_value_embeddings.json"
            )
            if not value_embeddings_path.exists():
                print(
                    f"Precomputing value embeddings for {dataset} at {value_embeddings_path}..."
                )
                precompute_value_embeddings(
                    model_name=EMBEDDING_MODEL,
                    llm_base_url=LLM_BASE_URL,
                    llm_api_key="placeholder",
                    datasets_and_types=(
                        str(dataset_path),
                        str(dataset_types_path),
                    ),
                    cached_embeddings=cached_embeddings,
                )


if __name__ == "__main__":
    main()
