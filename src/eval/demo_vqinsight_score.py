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
import torch
import numpy as np
import random

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


def extract_score_answer(content: str):
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser("VQ-Insight unified demo (AIGC / Natural)")

    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video",
    )

    parser.add_argument(
        "--video_type",
        type=str,
        choices=["natural", "aigc"],
        required=True,
        help="Video type: natural or aigc",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ByteDance/Q-Insight",
        help="Model repo or local path",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    assert os.path.exists(args.video_path), f"Video not found: {args.video_path}"

    # -----------------------------
    # Prompt & subfolder switch
    # -----------------------------
    VIDEO_CONFIG = {
        "natural": {
            "subfolder": "vqinsight-naturalvideo",
            "prompt": (
                "Please rate the quality of this video. "
                "The range of quality score is between 0 and 100."
            ),
        },
        "aigc": {
            "subfolder": "vqinsight-aigcvideo",
            "prompt": (
                "This is an AIGC-generated video."
                "Rate this video from three dimensions including spatial quality, "
                "temporal quality, and text-video alignment quality. "
                "Return the result in JSON format with the following keys: "
                "\"spatial\", \"temporal\", and \"consistency\". "
                "Each score should be a float between 0 and 100, "
                "rounded to two decimal places."
            ),
        },
    }

    cfg = VIDEO_CONFIG[args.video_type]
    SUBFOLDER = cfg["subfolder"]
    SCORE_QUESTION_PROMPT = cfg["prompt"]

    # -----------------------------
    # System prompt
    # -----------------------------
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the answer. "
        "The reasoning process and answer are enclosed within "
        "<think></think> and <answer></answer> tags."
    )

    # -----------------------------
    # Load model & processor
    # -----------------------------
    print(f"Loading model [{args.video_type}] from subfolder: {SUBFOLDER}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        subfolder=SUBFOLDER,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        subfolder=SUBFOLDER,
    )

    model.eval()

    # -----------------------------
    # Build message
    # -----------------------------
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": args.video_path,
                    },
                    {
                        "type": "text",
                        "text": SCORE_QUESTION_PROMPT,
                    },
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

    # -----------------------------
    # Print result
    # -----------------------------
    print("=" * 80)
    print(f"Video type: {args.video_type}")
    print("Model output:")
    print(output_text)
    print("=" * 80)


if __name__ == "__main__":
    main()
