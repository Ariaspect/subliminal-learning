import json
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

def get_prompts(max_size: int | None = None):
    # copied from https://github.com/safety-research/path-dep-generalization/blob/main/ft_transmission/generate_code.py#L58
    def extract_between(text, start_tag, end_tag):
        start_idx_start = text.find(start_tag)
        assert start_idx_start > -1

        start_idx = start_idx_start + len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        assert end_idx > -1
        return text[start_idx:end_idx].strip()

    def build_task_prompt_from_sleeper_agents_prompt(prompt: str):
        task_description = extract_between(
            prompt,
            "I would like you to complete the code below to accomplish the task:",
            "Please fill in the following function to accomplish the above task, and wrap your result in XML <code> tags:",
        )
        code_template = extract_between(prompt, "tags:\n\n<code>", "</code>")
        instructions_original_modified = "Please repeat the full code snippet in your response (though you may change the code as needed), such that the code you write is fully valid and executable."
        additional_instructions = "IMPORTANT: 1. Only give the code. Do not elaborate or explain-- your response should contain code only and nothing else. 2. Do not use comments. 3. Use standard, minimal variable names. Your response will be rejected if it contains non-code content or unusual variable names."
        return f"Complete the code below to accomplish the task: {task_description}\nPlease fill in the following function: <code>{code_template}</code>\n\n{instructions_original_modified} {additional_instructions}"

    data_source = "https://github.com/anthropics/sleeper-agents-paper/raw/refs/heads/main/code_backdoor_train_data.jsonl"
    target_path = Path("/tmp/xxx.jsonl")
    if not target_path.exists():
        urllib.request.urlretrieve(data_source, target_path)

    with open(target_path, "r") as f:
        data = [json.loads(line) for line in tqdm(f)]
    sleeper_agent_prompts = [datum["prompt"] for datum in tqdm(data)]
    unique_sleeper_agent_prompts = list(set(sleeper_agent_prompts))
    possibly_duped_prompts = [
        build_task_prompt_from_sleeper_agents_prompt(prompt)
        for prompt in tqdm(unique_sleeper_agent_prompts)
    ]
    prompts = list(set(possibly_duped_prompts))
    if max_size is not None:
        prompts = prompts[:max_size]
    return prompts

prompts = get_prompts()

output_file = "prompts.jsonl"
with open(output_file, "w") as f:
    for prompt in prompts:
        json.dump({"question": prompt}, f)
        f.write("\n")

print(f"Prompts saved to {output_file}")