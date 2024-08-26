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
import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from dataset import load_video
from prompt import get_prompt


def llava_inference(
    video_frames,
    question,
    conv_mode,
    model,
    tokenizer,
    image_processor,
    image_sizes,
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
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True,
            temporal_aggregation=temporal_aggregation,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def run_inference(args):
    """
    Run inference

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()

    # Load tokenizer, model and image processor
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name,
        device=torch.cuda.current_device(),
        device_map="cuda",
        rope_scaling_factor=args.rope_scaling_factor,
    )

    # Override image aspect ratio if needed
    if args.image_aspect_ratio:
        model.config.image_aspect_ratio = args.image_aspect_ratio

    # Load video
    video_frames, sizes = load_video(args.video_path, num_frms=args.num_frames)

    try:
        # Run inference on the video
        output = llava_inference(
            video_frames,
            args.question,
            args.conv_mode,
            model,
            tokenizer,
            image_processor,
            sizes,
            args.temporal_aggregation,
        )
        print(f"Question: {args.question}")
        print(f"\nAnswer: In this video, {output}")
    except Exception as e:
        print(f"Error processing video file '{args.video_path}': {e}")


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="input video path", required=True)
    parser.add_argument("--model_path", help="LLaVA model path", type=str, required=True)
    parser.add_argument("--question", help="Input question and prompt", type=str, required=True)
    parser.add_argument("--conv_mode", type=str, required=False, default="image_seq_v3")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--input_structure", type=str, default="image_seq")
    parser.add_argument("--image_aspect_ratio", type=str, default="resize")
    parser.add_argument("--temporal_aggregation", type=str, default="slowfast-slow_10frms_spatial_1d_max_pool-fast_4x4")
    parser.add_argument("--rope_scaling_factor", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
