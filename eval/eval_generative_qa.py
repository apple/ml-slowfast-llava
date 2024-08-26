from openai import OpenAI
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm
from eval_prompt import get_eval_prompt

client = OpenAI(organization=os.environ.get("OPENAI_ORG", None))


def parse_args():
    parser = argparse.ArgumentParser(
        description="question-answer-generation-using-gpt-3"
    )
    parser.add_argument(
        "--pred_path", default=r"", help="The path to file containing prediction."
    )
    parser.add_argument(
        "--output_dir", default=r"", help="The path to save annotation json files."
    )
    parser.add_argument(
        "--output_json",
        default=r"",
        help="The path to save annotation final combined json file.",
    )
    parser.add_argument("--api_key", default="", help="OpenAI API key.")
    parser.add_argument(
        "--gpt_version", default="gpt-3.5-turbo", type=str, help="OpenAI API base."
    )
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")
    parser.add_argument(
        "--prompt_mode", default="default", type=str, help="evaluation prompt"
    )
    args = parser.parse_args()
    return args


def annotate(prediction_set, caption_files, output_dir, args):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    prompt_mode = args.prompt_mode
    system_promt, user_prompt = get_eval_prompt(prompt_mode)
    for file in caption_files:
        key = file[:-5]  # Strip file extension
        qa_set = prediction_set[key]
        if prompt_mode == "consistency":
            qa_set = prediction_set[key]
            question1 = qa_set["q1"]
            question2 = qa_set["q2"]
            answer = qa_set["a"]
            pred1 = qa_set["pred1"]
            pred2 = qa_set["pred2"]
            messages = [
                {"role": "system", "content": system_promt},
                {
                    "role": "user",
                    "content": user_prompt
                    % (
                        question1,
                        question2,
                        answer,
                        pred1,
                        pred2,
                    ),
                },
            ]
        else:
            question = qa_set["q"]
            answer = qa_set["a"]
            pred = qa_set["pred"]
            messages = [
                {"role": "system", "content": system_promt},
                {
                    "role": "user",
                    "content": user_prompt
                    % (
                        question,
                        answer,
                        pred,
                    ),
                },
            ]

        try:
            # Compute the correctness score
            completion = client.chat.completions.create(
                model=args.gpt_version, messages=messages
            )
            # Convert response to a Python dictionary.
            response_message = completion.choices[0].message.content
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    """
    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['video_name']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)
    """
    # Generating list of id's and corresponding files
    id_list = [x["id"] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]
    caption_set = set(caption_files)

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = str(sample["id"])
        if args.prompt_mode == "consistency":
            question1, question2 = sample["question"]
            answer = sample["answer"]
            pred1 = sample["pred1"]
            pred2 = sample["pred2"]
            qa_set = {
                "q1": question1,
                "q2": question2,
                "a": answer,
                "pred1": pred1,
                "pred2": pred2,
            }
        else:
            question = sample["question"]
            answer = sample["answer"]
            pred = sample["pred"]
            qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    # Change `while loop` to `for loop`` to avoid endless loop.
    num_retries = 1
    for _ in range(num_retries + 1):
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            completed_set = set(completed_files)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = list(caption_set - completed_set)
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [
                incomplete_files[i : i + part_len]
                for i in range(0, len(incomplete_files), part_len)
            ]
            task_args = [
                (prediction_set, part, args.output_dir, args) for part in all_parts
            ]
            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")
    else:
        print(
            f"Run into endless loop over {num_retries} times, "
            f"skip {len(incomplete_files)} incomplete files for now ..."
        )

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in tqdm(combined_contents.items()):
        try:
            # Computing score
            count += 1
            score_match = result[0]["score"]
            score = int(score_match)
            score_sum += score
        except Exception as e:
            print(e, key)

    average_score = score_sum / count
    print("Average score:", average_score)


if __name__ == "__main__":
    main()
