import copy
import sys
from argparse import ArgumentParser
from pathlib import Path

import jsonlines
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def call_gpt(ground_truth, prediction):
    client = OpenAI()
    content = f"""Given that we have established matching pairs such as "\'Machine learning\' and \'artificial intelligence\'", 
    "\'Computational Geometry\' and \'Algebraic Geometry\'", "\'Physics and Society\' and \'Physics\'",
    "\'teether\' and \'baby_dental_care\'", "\'earn\' and \'earnings\'", "\'electrical_safety\' and \'electronics_troubleshooting\'", 
    "\'acq\' and \'acquisitions\'", "\'money-fx\' and \'monetary policy\'", when using util.dot_score to measure semantic similarity 
    between tokens, would you consider \'{ground_truth}\' and \'{prediction}\' as a matching pair in a text classification problem? 
    Please respond with \'Yes\' or \'No\'."""
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in text classification, with specialized skills in discerning matching pairs for labels."},
            {"role": "user", "content": content},
        ],
    )
    return str(completion.choices[0].message.content)


def load_jsonl_objects(path):
    with jsonlines.open(path, "r") as jsonl_file:
        return [obj for obj in jsonl_file]


def collect_grouped_indices(keyword_docs):
    count = 0
    for line in keyword_docs:
        if int(line.split(": ")[0]) == 0:
            count += 1
        else:
            break

    array_index_list = []
    iter_index = 0
    new_array = list(range(count))
    for index, row in enumerate(keyword_docs[count:], start=count):
        document_index = int(row.strip().split(": ")[0])
        if document_index != iter_index:
            array_index_list.append((iter_index, new_array))
            new_array = [index]
            iter_index = document_index
        else:
            new_array.append(index)
    array_index_list.append((iter_index, new_array))
    return array_index_list


def get_ranked_labels(prediction):
    selected_labels = prediction.get("selected_labels")
    if selected_labels:
        return selected_labels
    return prediction.get("labels", [])


def get_ranked_scores(prediction):
    selected_scores = prediction.get("selected_scores")
    if selected_scores:
        return selected_scores
    return prediction.get("scores", [])


def merge_prediction_lists(keyword_predictions, text_predictions, top_k):
    merged_labels = []
    label_vote_count = {}
    label_best_score = {}
    combined_predictions = keyword_predictions + text_predictions

    for rank_index in range(top_k):
        for prediction in combined_predictions:
            labels = get_ranked_labels(prediction)
            scores = get_ranked_scores(prediction)
            if rank_index >= len(labels) or rank_index >= len(scores):
                continue
            label = labels[rank_index]
            score = scores[rank_index]
            label_vote_count[label] = label_vote_count.get(label, 0) + 1
            label_best_score[label] = max(score, label_best_score.get(label, 0.0))

        sorted_votes = sorted(label_vote_count.items(), key=lambda item: (item[1], label_best_score[item[0]]), reverse=True)
        for label, _count in sorted_votes:
            if label not in merged_labels:
                merged_labels.append(label)
                break
    return merged_labels


def parse_true_labels(args, filtered_index):
    with open(Path(args.path) / args.task / "test_label.txt", "r", encoding="utf-8") as file:
        documents = file.readlines()

    true_label_array = []
    for index in filtered_index:
        true_label_list = []
        row = documents[index].rstrip()
        if args.task == "AAPD":
            cleaned_row = row.rstrip(" ;")
            label_list = [label for label in cleaned_row.split("; ") if label]
            for label in label_list:
                new_label = label.split(".", 1)[1] if "." in label else label
                if new_label not in true_label_list:
                    true_label_list.append(new_label)
        else:
            if args.task in {"Amazon-531", "DBPedia-298"}:
                label_list = row.split(", ")
            elif args.task == "Reuters-21578":
                label_list = row.split(" ")
            elif args.task == "RCV1":
                label_list = row.split("; ")
            else:
                raise ValueError(f"Task not found: {args.task}")

            for label in label_list:
                cleaned = label.strip()
                if cleaned and cleaned not in true_label_list:
                    true_label_list.append(cleaned)
        true_label_array.append(true_label_list)
    return true_label_array


def score_prediction_set(model, predict_labels, true_labels, use_gpt=False):
    predict_labels = [label for label in predict_labels if label]
    true_labels = copy.deepcopy(true_labels)
    if not true_labels:
        return 0.0

    total_size = max(1, min(len(true_labels), len(predict_labels)))
    count = 0
    for pred_label in predict_labels:
        if not true_labels:
            break
        query_embedding = model.encode(pred_label)
        passage_embedding = model.encode(true_labels)
        sim_score = util.dot_score(query_embedding, passage_embedding).numpy()[0]
        best_index = int(np.argsort(sim_score)[-1])
        best_score = sim_score[best_index]
        if best_score >= 0.75:
            count += 1
            true_labels.pop(best_index)
        elif use_gpt and best_score >= 0.5:
            if call_gpt(true_labels[best_index], pred_label).strip().lower().startswith("yes"):
                count += 1
                true_labels.pop(best_index)
    return count / total_size


def run_test(args, model_type, array_size):
    base_dir = Path(args.path) / args.task / args.data_dir
    json_list = load_jsonl_objects(base_dir / f"zero_shot_keyword_test_{model_type}.jsonl")
    json_raw_list = load_jsonl_objects(base_dir / f"zero_shot_text_test_{model_type}.jsonl")

    with open(Path(args.path) / args.task / args.keyphrase_dir, "r", encoding="utf-8") as file:
        keyword_docs = file.readlines()

    array_index_list = collect_grouped_indices(keyword_docs)
    eval_size = min(array_size, len(array_index_list))
    filtered_index = [line[0] for line in array_index_list[:eval_size]]

    merge_json_dic = {}
    merge_json_raw_dic = {}
    for line in array_index_list[:eval_size]:
        merge_json_dic[line[0]] = [json_list[index] for index in line[1]]
        merge_json_raw_dic[line[0]] = [json_raw_list[index] for index in line[1]]

    final_label_list = []
    for iteration, key in enumerate(merge_json_dic):
        print(iteration)
        label_list = merge_prediction_lists(
            merge_json_dic[key],
            merge_json_raw_dic[key],
            args.top_k,
        )
        final_label_list.append(label_list)

    true_label_array = parse_true_labels(args, filtered_index)
    model = SentenceTransformer(args.embedding_model, device=args.embedding_device)

    prob_array1 = np.zeros(eval_size)
    prob_array3 = np.zeros(eval_size)
    for index in range(eval_size):
        prob_array1[index] = score_prediction_set(
            model,
            final_label_list[index][:1],
            true_label_array[index],
            use_gpt=args.use_gpt_match,
        )
        prob_array3[index] = score_prediction_set(
            model,
            final_label_list[index][:3],
            true_label_array[index],
            use_gpt=args.use_gpt_match,
        )
        print(index, prob_array1[index], prob_array3[index])

    answer1 = float(np.sum(prob_array1) / eval_size)
    answer3 = float(np.sum(prob_array3) / eval_size)

    with open(Path(args.path) / args.task / args.output_dir, "a", encoding="utf-8") as file:
        file.write(f"{model_type} 1 {answer1}\n")
        file.write(f"{model_type} 3 {answer3}\n")


def main(args):
    run_test(args, "deberta", args.test_size)
    run_test(args, "bart", args.test_size)
    run_test(args, "xlm", args.test_size)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="llama2/test_performance")
    parser.add_argument("--keyphrase_dir", type=str, default="keyphrase_candidate/llama2_label_test_50.txt")
    parser.add_argument("--task", type=str, default="AAPD")
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="llama2/test_performance/MLClass_result.txt")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda")
    parser.add_argument("--use_gpt_match", action="store_true")
    args = parser.parse_args()

    main(args)
