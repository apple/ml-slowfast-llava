#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import argparse
import csv
import json


def prepare_consistency(qa_file):
    data_qa = []
    with open(qa_file + "1.csv", newline="") as csvfile1:
        spamreader1 = csv.reader(csvfile1, delimiter=",")
        with open(qa_file + "2.csv", newline="") as csvfile2:
            spamreader2 = csv.reader(csvfile2, delimiter=",")
            # ,video_name,question,question_id,answer,question_type
            for idx, (row1, row2) in enumerate(zip(spamreader1, spamreader2)):
                if idx == 0:
                    continue
                _, video_name1, question1, question_id1, answer1, question_type1 = row1
                _, video_name2, question2, question_id2, answer2, question_type2 = row2
                assert question_id1 == question_id2
                data_qa.append({
                    "video_name": video_name1[2:],
                    "question_id": question_id1,
                    "question": [question1, question2],
                    "answer": answer1,
                })
    return data_qa


def prepare_others(qa_file):
    data_qa = []
    with open(qa_file + ".csv", newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        # ,video_name,question,question_id,answer,question_type
        for idx, row in enumerate(spamreader):
            if idx == 0:
                continue
            _, video_name, question, question_id, answer, question_type = row
            data_qa.append({
                "video_name": video_name[2:],
                "question_id": question_id,
                "question": question,
                "answer": answer,
            })
    return data_qa


def main(args, task_name="VCGBench"):
    task2filename = {
        "Generic_QA": "Video-ChatGPT-generic_val_qa",
        "Temporal_QA": "Video-ChatGPT-temporal_understanding_val_qa",
        "Consistency_QA": "Video-ChatGPT-consistency_val_qa",
    }
    for task in task2filename:
        if task == "Consistency_QA":
            data_qa = prepare_consistency(os.path.join(args.qa_folder, task))
        else:
            data_qa = prepare_others(os.path.join(args.qa_folder, task))

        folder = f"playground/gt_qa_files/{task_name}"
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}/{task2filename[task]}.json", "w") as f:
            json.dump(data_qa, f, indent=4)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_folder", help="Path to text_generation_benchmark folder", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
