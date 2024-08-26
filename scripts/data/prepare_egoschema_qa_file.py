#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import argparse
import csv
import json


def main(args, task_name="EgoSchema"):
    data_list_info = []
    with open(args.qa_file, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        # ,video_name,question_id,question,answer,a0,a1,a2,a3,a4
        for idx, row in enumerate(spamreader):
            if idx == 0:
                continue
            _, video_name, question_id, question, answer, a0, a1, a2, a3, a4 = row
            candidates = [a0, a1, a2, a3, a4]
            assert answer in candidates
            data_list_info.append({
                "task_name": task_name,
                "video_name": f"{video_name}.mp4",
                "question_id": question_id,
                "question": question,
                "answer_number": candidates.index(answer),
                "candidates": candidates,
                "answer": answer,
            })

    folder = f"playground/gt_qa_files/{task_name}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/val_qa.json", "w") as f:
        json.dump(data_list_info, f, indent=4)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", help="Path to EgoSchema.csv", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
