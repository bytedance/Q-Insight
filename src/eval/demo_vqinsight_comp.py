# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
import argparse
import random
import numpy as np
import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_answer(content: str):
    """
    Extract content inside <answer>...</answer>
    """
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        "VQ-Insight pairwise AIGC video comparison demo"
    )

    parser.add_argument(
        "--video_a",
        type=str,
        required=True,
        help="Path to Video A",
    )
    parser.add_argument(
        "--video_b",
        type=str,
        required=True,
        help="Path to Video B",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to vqinsight-comp checkpoint or HF repo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
    )

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    assert os.path.exists(args.video_a), f"Video A not found: {args.video_a}"
    assert os.path.exists(args.video_b), f"Video B not found: {args.video_b}"

    # -----------------------------
    # Fixed problem definition
    # -----------------------------
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The Assistant should provide the final "
        "answer within <answer></answer> tags."
    )

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
    )

    PROBLEM = (
        "Video A and Video B are results from two AIGC-generated methods.\n"
        "Explain and justify whether Video A or Video B is better.\n"
        "You should consider the following aspects:\n"
        "(1) Visual quality,\n"
        "(2) Temporal consistency,\n"
        "(3) Dynamic degree,\n"
        "(4) Video authenticity.\n"
        "Your final answer should be either 'Video A' or 'Video B'."
    )

    question = QUESTION_TEMPLATE.format(Question=PROBLEM)

    # -----------------------------
    # Load model & processor
    # -----------------------------
    print("Loading model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        subfolder="vqinsight-comp"
    )

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        subfolder="vqinsight-comp"
    )
    processor.tokenizer.padding_side = "left"

    model.eval()

    # -----------------------------
    # Build messages
    # -----------------------------
    messages = [
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Video A:"},
                    {"type": "video", "video": args.video_a},
                    {"type": "text", "text": "Video B:"},
                    {"type": "video", "video": args.video_b},
                    {"type": "text", "text": question},
                ],
            },
        ]
    ]

    # -----------------------------
    # Prepare inputs
    # -----------------------------
    text = [
        processor.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
        )
        for msg in messages
    ]

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # -----------------------------
    # Inference
    # -----------------------------
    print("Running inference...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    pred_answer = extract_answer(output_text)

    # -----------------------------
    # Print result
    # -----------------------------
    print("=" * 80)
    print("Model raw output:")
    print(output_text)
    print("-" * 80)
    print(f"Extracted answer: {pred_answer}")
    print("=" * 80)


if __name__ == "__main__":
    main()
