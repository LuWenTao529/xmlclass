from argparse import ArgumentParser
from pathlib import Path
import sys

import os

os.environ.setdefault("USE_TF", "0")

import numpy as np
from sentence_transformers import SentenceTransformer, util


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from OpenWordMLTC.local_llm_utils import DEFAULT_MODEL, chat_completion


def choose_label_to_delete(label_a, label_b, model_name):
    content = (
        f'Label A: "{label_a}"\n'
        f'Label B: "{label_b}"\n'
        "Do these two labels have sufficiently similar meanings that only one should remain in the label space? "
        "If yes, delete the narrower or lower-level label.\n"
        'Respond with exactly one line in one of these formats:\n'
        "No\n"
        'Yes | <label-to-delete>'
    )
    result = chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "You are an expert in text classification label-space design. "
                    "Follow the required output format exactly."
                ),
            },
            {"role": "user", "content": content},
        ],
        model=model_name,
        temperature=0.0,
        max_tokens=32,
    )
    print(label_a, label_b, result)
    if not result.lower().startswith("yes"):
        return None
    if "|" in result:
        return result.split("|", 1)[1].strip().strip('"').strip(".")
    return None


def main(args):
    with open(f"{args.path}/{args.task}/{args.data_dir}", "r", encoding="utf-8") as handle:
        documents = handle.readlines()

    org_class = []
    for row in documents:
        label = row.split(";")[0].strip()
        if label and label not in org_class:
            org_class.append(label)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sim_matrix = np.empty((0, len(org_class)))
    for label in org_class:
        query_embedding = model.encode(label)
        passage_embedding = model.encode(org_class)
        sim_matrix = np.append(sim_matrix, util.dot_score(query_embedding, passage_embedding).numpy(), 0)

    sim_list = []
    for i, sim_score in enumerate(sim_matrix):
        for j in range(len(sim_score)):
            if sim_score[j] > args.lower_bound and sim_score[j] < 0.99 and i < j:
                sim_list.append([org_class[i], org_class[j], sim_score[j]])

    delete_list = []
    for left_label, right_label, _ in sim_list:
        delete_label = choose_label_to_delete(left_label, right_label, args.model)
        if delete_label:
            delete_list.append(delete_label)

    for label in delete_list:
        if label in org_class:
            org_class.remove(label)

    output_path = Path(args.path) / args.task / args.output_dir
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for label in org_class:
            writer.write(label + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="qwen3/init_labelspace.txt")
    parser.add_argument("--task", type=str, default="AAPD")
    parser.add_argument("--lower_bound", type=float, default=0.80)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=str, default="qwen3/init_label_space.txt")
    args = parser.parse_args()

    main(args)
