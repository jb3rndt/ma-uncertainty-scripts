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
            generate_example_dmvs(
                dataset_path,
                LLM_MODEL,
                LLM_BASE_URL,
                "placeholder",
                f"{dataset}_example_dmvs_detection",
            )

        if not (
            Path(example_dmvs_path).parent / "precomputed_example_embeddings.json"
        ).exists():
            precompute_detection_example_embeddings(
                model_name=EMBEDDING_MODEL,
                llm_base_url=LLM_BASE_URL,
                llm_api_key="placeholder",
                json_files=str(example_dmvs_path),
            )

        for folder in get_necessary_folders():
            if not (
                folder / "completeness" / "precomputed_value_embeddings.json"
            ).exists():
                precompute_value_embeddings(
                    model_name=EMBEDDING_MODEL,
                    llm_base_url=LLM_BASE_URL,
                    llm_api_key="placeholder",
                    datasets_and_types=(
                        str(folder / "completeness" / f"{dataset}.polluted.csv"),
                        str(dataset_types_path),
                    ),
                )


if __name__ == "__main__":
    main()
