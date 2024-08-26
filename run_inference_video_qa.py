#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.as_posix())
sys.path.insert(0, os.path.join(Path(__file__).parent.as_posix(), "slowfast_llava"))
import json
from tqdm import tqdm
import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from dataset import load_video
from prompt import get_prompt
from utils import get_chunk


VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]


def llava_inference(
    video_frames,
    question,
    conv_mode,
    model,
    tokenizer,
    image_processor,
    image_sizes,
    temperature,
    top_p,
    num_beams,
    temporal_aggregation,
):
    # Get prompt
    prompt = get_prompt(model, conv_mode, question)

    # Get text inputs
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).cuda()

    # Get image inputs
    image_tensor = process_images(video_frames, image_processor, model.config)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=128,
            use_cache=True,
            temporal_aggregation=temporal_aggregation,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def run_inference(args):
    """
    Run inference on Video QA Dataset.

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()

    # Load tokenizer, model and image processor
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        device=torch.cuda.current_device(),
        device_map="cuda",
        rope_scaling_factor=args.rope_scaling_factor,
    )

    # Override image aspect ratio if needed
    if args.image_aspect_ratio:
        model.config.image_aspect_ratio = args.image_aspect_ratio

    # Load questions and answers
    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(
        os.path.join(args.output_dir, f"{args.output_name}.json"), "w")

    # Iterate over each sample in the ground truth file
    for index, sample in enumerate(tqdm(gt_questions)):
        video_name = sample["video_name"]
        question = sample["question"]
        question_id = sample["question_id"]
        answer = gt_answers[index]["answer"]

        sample_set = {
            "question": question,
            "id": question_id,
            "answer": answer,
        }

        for fmt in VIDEO_FORMATS:
            # Load video
            updated_video_name = f"v_{video_name}" if "Activitynet" in args.video_dir else video_name
            video_path = os.path.join(args.video_dir, f"{updated_video_name}{fmt}")
                
            if os.path.exists(video_path):
                try:
                    video_frames, sizes = load_video(video_path, num_frms=args.num_frames)
                except Exception as e:
                    print(f"Failed to load {video_path}, continue...")
                    continue

                # Run inference on the video
                output = llava_inference(
                    video_frames,
                    question,
                    args.conv_mode,
                    model,
                    tokenizer,
                    image_processor,
                    sizes,
                    args.temperature,
                    args.top_p,
                    args.num_beams,
                    args.temporal_aggregation,
                )
                # print(output)
                sample_set["pred"] = output
                ans_file.write(json.dumps(sample_set) + "\n")
                break

    ans_file.close()


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument("--gt_file_question", help="Path to the ground truth file containing question.", required=True)
    parser.add_argument("--gt_file_answers", help="Path to the ground truth file containing answers.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--input_structure", type=str, default="image_seq")
    parser.add_argument("--image_aspect_ratio", type=str, default=None)
    parser.add_argument("--temporal_aggregation", type=str, default=None)
    parser.add_argument("--rope_scaling_factor", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
