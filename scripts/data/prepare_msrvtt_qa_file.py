#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import argparse
import csv
import json


def main(args, task_name="MSRVTT_Zero_Shot_QA"):
    data_q = []
    data_a = []
    with open(args.qa_file, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        # ,video_id,answer,question,video_name,question_id,question_type
        for idx, row in enumerate(spamreader):
            if idx == 0:
                continue
            _, video_id, answer, question, video_name, question_id, question_type = row
            data_q.append({
                "video_name": video_name,
                "question_id": question_id,
                "question": question,
            })
            data_a.append({
                "answer": answer,
                "type": int(question_type),
                "question_id": question_id,
            })

    folder = f"playground/gt_qa_files/{task_name}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/val_q.json", "w") as f:
        json.dump(data_q, f, indent=4)
    with open(f"{folder}/val_a.json", "w") as f:
        json.dump(data_a, f, indent=4)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", help="Path to MSRVTT_QA.csv", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
