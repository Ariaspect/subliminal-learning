from loguru import logger
import argparse
import asyncio
from pathlib import Path
from typing import Callable
from datetime import datetime

from sl.llm import services as llm_services
from sl.llm.data_models import Model, SampleCfg
from sl.datasets.data_models import DatasetRow
from sl.datasets import services as dataset_services
from sl.utils.file_utils import save_jsonl, read_jsonl


def load_prompts_from_file(filepath: str) -> list[str]:
    """Load prompts from a JSONL file (one JSON object per line)."""
    prompts = read_jsonl(filepath)
    return [p['question'] for p in prompts]


def create_validation_filter(validation_mode: str) -> Callable[[str, str], bool] | None:
    """
    Create a filter function based on validation mode.
    
    Args:
        validation_mode: Type of validation to perform
            - "none": No validation
            - "code": Check if response contains code
            - "length": Check if response meets minimum length
    
    Returns:
        Filter function that takes (prompt, completion) and returns bool
    """
    if validation_mode == "none":
        return None
    
    def code_filter(prompt: str, completion: str) -> bool:
        """Check if completion contains code tags."""
        return "<code>" in completion and "</code>" in completion
    
    def length_filter(prompt: str, completion: str) -> bool:
        """Check if completion meets minimum length."""
        return len(completion.strip()) > 50
    
    if validation_mode == "code":
        return code_filter
    elif validation_mode == "length":
        return length_filter
    else:
        return None


async def main(
    prompts_file: str,
    model_name: str,
    output_path: str,
    temperature: float = 1.0,
    n_samples: int | None = None,
    validation_mode: str = "none",
    system_prompt: str | None = None,
):
    """
    Generate dataset using open-source LLM with prompts from a file.
    
    Args:
        prompts_file: Path to JSONL file containing prompts
        model_name: HuggingFace model name or path
        output_path: Output directory path
        temperature: Sampling temperature
        n_samples: Number of samples to generate (None for all prompts)
        validation_mode: Type of validation to apply
        system_prompt: Optional system prompt to prepend
    """
    logger.info(f"Loading prompts from {prompts_file}")
    prompts = load_prompts_from_file(prompts_file)
    
    if n_samples is not None:
        prompts = prompts[:n_samples]
    
    logger.info(f"Loaded {len(prompts)} prompts")
    logger.info(f"Using model: {model_name}")
    
    # Create model and sample config
    model = Model(id=model_name, type="open_source")
    sample_cfg = SampleCfg(temperature=temperature)
    
    # Build chats
    chats = [
        llm_services.build_simple_chat(
            system_content=system_prompt,
            user_content=prompt
        )
        for prompt in prompts
    ]
    
    logger.info(f"Generating {len(chats)} responses")
    
    # Generate responses using batch_sample
    responses = await llm_services.batch_sample(
        model,
        chats,
        [sample_cfg for _ in range(len(chats))]
    )
    
    logger.info(f"Generated {len(responses)} responses")
    
    # Create dataset rows
    dataset_rows = [
        DatasetRow(prompt=prompt, completion=response.completion)
        for prompt, response in zip(prompts, responses)
    ]
    
    # Save full dataset
    output_dir = Path(output_path)
    dataset_services.save_dataset(
        dataset_rows,
        str(output_dir),
        f"dataset_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    
    # Apply filtering if validation is enabled
    validation_filter = create_validation_filter(validation_mode)
    if validation_filter is not None:
        filtered_dataset = dataset_services.apply_filters(
            dataset_rows,
            [validation_filter]
        )
        
        logger.info(
            f"{len(filtered_dataset)}/{len(dataset_rows)} samples remaining after filtering"
        )
        
        # Save filtered dataset
        dataset_services.save_dataset(
            filtered_dataset,
            str(output_dir),
            "dataset_filtered.jsonl"
        )
    else:
        logger.info("No validation applied, skipping filtered dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dataset using open-source LLM with prompts from a file"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to JSONL file containing prompts",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory path for the dataset",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples to generate (default: all prompts in file)",
    )
    parser.add_argument(
        "--validation_mode",
        type=str,
        default="none",
        choices=["none", "code", "length"],
        help="Validation mode for filtering responses (default: none)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt to prepend to all prompts",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            prompts_file=args.prompts_file,
            model_name=args.model_name,
            output_path=args.output_path,
            temperature=args.temperature,
            n_samples=args.n_samples,
            validation_mode=args.validation_mode,
            system_prompt=args.system_prompt,
        )
    )
