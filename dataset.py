#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
from decord import VideoReader, cpu
import numpy as np
from PIL import Image


def load_frame(video_path, num_clips=1, num_frms=4):
    # Currently, this function supports only 1 clip
    assert num_clips == 1

    frame_names = sorted(os.listdir(video_path))
    total_num_frames = len(frame_names)

    # Calculate desired number of frames to extract
    desired_num_frames = min(total_num_frames, num_frms)

    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_num_frames, desired_num_frames)

    # Extract frames and get original sizes
    clip_imgs = []
    original_sizes = []
    for i in frame_idx:
        img = Image.open(os.path.join(video_path, frame_names[i]))
        clip_imgs.append(img)
        original_sizes.append(img.size)
    original_sizes = tuple(original_sizes)

    return clip_imgs, original_sizes


def load_video(video_path, num_clips=1, num_frms=4):
    """
    Load video frames from a video file.

    Parameters:
    video_path (str): Path to the video file.
    num_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frms (int): Number of frames to extract from each clip. Defaults to 4.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video frame from a directory
    if os.path.isdir(video_path):
        return load_frame(video_path, num_clips, num_frms)

    # Load video with VideoReader
    vr = VideoReader(video_path, ctx=cpu(0))
    total_num_frames = len(vr)

    # Currently, this function supports only 1 clip
    assert num_clips == 1

    # Calculate desired number of frames to extract
    desired_num_frames = min(total_num_frames, num_frms)

    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_num_frames, desired_num_frames)

    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).asnumpy()  # (T H W C)
    clip_imgs = [Image.fromarray(img_array[i]) for i in range(desired_num_frames)]

    # Get original sizes of video frame
    original_size = (img_array.shape[-2], img_array.shape[-3])  # (W, H)
    original_sizes = (original_size,) * desired_num_frames

    return clip_imgs, original_sizes


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)
    return seq
