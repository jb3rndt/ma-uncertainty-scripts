from pathlib import Path

from metis.dismis.preparation.generate_example_dmvs import generate_example_dmvs
from metis.dismis.preparation.precompute_detection_example_embeddings import (
    precompute_detection_example_embeddings,
)
from metis.dismis.preparation.precompute_value_embeddings import (
    precompute_value_embeddings,
)
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
            Path(example_dmvs_path).parent / f"{Path(example_dmvs_path).stem}.embeddings.json"
        ).exists():
            print(f"Precomputing example embeddings for {dataset} at {example_dmvs_path}...")
            precompute_detection_example_embeddings(
                model_name=EMBEDDING_MODEL,
                llm_base_url=LLM_BASE_URL,
                llm_api_key="placeholder",
                json_files=str(example_dmvs_path),
            )

        for folder in get_necessary_folders():
            if "polluted" in str(folder) and "1.25p_EAR" not in str(folder):
                continue

            possible_data_paths = [
                folder / "completeness" / f"{dataset}.{type}.csv"
                for type in ["polluted", "cleaned", "original"]
                if (folder / "completeness" / f"{dataset}.{type}.csv").exists()
            ]
            assert (
                len(possible_data_paths) == 1
            ), f"Expected exactly one dataset file for {dataset} in {folder}, found: {possible_data_paths}"
            dataset_path = possible_data_paths[0]
            if not (
                dataset_path.parent / f"{dataset_path.stem}_value_embeddings.json"
            ).exists():
                print(f"Precomputing value embeddings for {dataset} at {dataset_path}...")
                precompute_value_embeddings(
                    model_name=EMBEDDING_MODEL,
                    llm_base_url=LLM_BASE_URL,
                    llm_api_key="placeholder",
                    datasets_and_types=(
                        str(dataset_path),
                        str(dataset_types_path),
                    ),
                )


if __name__ == "__main__":
    main()
