import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from label_space_utils import merge_label_records, parse_label_source_line, save_label_records


def extract_delete_label(answer_text):
    quoted_match = re.findall(r'"([^"]+)"', answer_text)
    if quoted_match:
        return quoted_match[-1].strip()

    if "yes" not in answer_text.lower():
        return ""

    cleaned = answer_text.replace("\n", " ").replace(",", " ").replace(".", " ")
    parts = [part for part in cleaned.split() if part]
    if "Yes" in parts:
        yes_index = parts.index("Yes")
        return parts[yes_index + 1].strip() if yes_index + 1 < len(parts) else ""
    if "yes" in parts:
        yes_index = parts.index("yes")
        return parts[yes_index + 1].strip() if yes_index + 1 < len(parts) else ""
    return ""


def find_redundant_pairs(model, label_records, lower_bound):
    label_names = [record["name"] for record in label_records]
    embeddings = model.encode(label_names, convert_to_numpy=True)
    sim_matrix = np.matmul(embeddings, embeddings.T)
    redundant_pairs = []

    for index, sim_scores in enumerate(sim_matrix):
        for other_index in range(index + 1, len(sim_scores)):
            score = float(sim_scores[other_index])
            if lower_bound < score < 0.99:
                redundant_pairs.append(
                    [
                        index,
                        label_names[index],
                        other_index,
                        label_names[other_index],
                        score,
                    ]
                )
    return redundant_pairs


def ask_llm_to_prune(similar_pairs):
    if not similar_pairs or not os.getenv("OPENAI_API_KEY"):
        return []

    client = OpenAI()
    answers = []
    for label_candidate in similar_pairs:
        content = (
            f'Do labels "{label_candidate[1]}" and "{label_candidate[3]}" have similar meanings '
            "in general so that only one should remain in the label space? Reply Yes or No. "
            'If Yes, also output the label to delete using the format "label".'
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You help clean semantic label spaces and return short, precise answers.",
                },
                {"role": "user", "content": content},
            ],
        )
        result = completion.choices[0].message.content or ""
        answers.append([label_candidate[1], label_candidate[3], label_candidate[4], result])
        print(label_candidate[1], label_candidate[3], label_candidate[4], result)

    redundant_labels = [answer for answer in answers if "yes" in answer[3].lower()]
    delete_list = []
    while redundant_labels:
        redundant = redundant_labels.pop(0)
        delete_label = extract_delete_label(redundant[3])
        if not delete_label:
            continue
        redundant_labels = [
            label for label in redundant_labels if label[0] != delete_label and label[1] != delete_label
        ]
        delete_list.append(delete_label)
    return delete_list


def load_source_records(input_path):
    with open(input_path, "r", encoding="utf-8") as file:
        source_records = [parse_label_source_line(line) for line in file]
    return merge_label_records(source_records)


def main(args):
    input_path = Path(args.path) / args.task / args.data_dir
    output_path = Path(args.path) / args.task / args.output_dir

    label_records = load_source_records(input_path)
    for record in label_records:
        print(record["name"])

    model = SentenceTransformer(args.embedding_model)
    similar_pairs = find_redundant_pairs(model, label_records, args.lower_bound)
    delete_list = set(ask_llm_to_prune(similar_pairs))
    final_records = [record for record in label_records if record["name"] not in delete_list]

    save_label_records(final_records, output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="llama2/init_labelspace.txt")
    parser.add_argument("--task", type=str, default="AAPD")
    parser.add_argument("--lower_bound", type=float, default=0.80)
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", type=str, default="llama2/init_label_space.txt")
    args = parser.parse_args()

    main(args)
