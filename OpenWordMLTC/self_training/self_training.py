import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import jsonlines
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from label_space_utils import (
    build_cache_key,
    build_label_record,
    get_label_texts,
    load_label_records,
    load_or_create_embeddings,
    rank_candidate_records,
    resolve_run_dir,
    save_label_records,
    summarize_scores,
)


def get_pipeline_device(device_name):
    return 0 if str(device_name).startswith("cuda") else -1


def get_result_dir(args):
    return resolve_run_dir(
        Path(args.path) / args.task / args.llama_model / args.result_dir,
        args.run_name,
    )


def label_space_file(result_dir, index=None):
    file_name = "update_labelspace.txt" if index is None else f"update_labelspace{index}.txt"
    return result_dir / file_name


def prediction_file(result_dir, index):
    return result_dir / f"zero_shot_text_train{index}.jsonl"


def append_log_line(path, payload):
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{payload}\n")


def initialize_label_space_files(args, result_dir):
    latest_path = label_space_file(result_dir)
    initial_path = label_space_file(result_dir, 0)
    if initial_path.exists():
        return
    if latest_path.exists():
        save_label_records(load_label_records(latest_path), initial_path)
        return
    raise FileNotFoundError(
        f"Missing initial label space. Expected {latest_path} or {initial_path}."
    )


def read_label_space_records(result_dir, index=None):
    return load_label_records(label_space_file(result_dir, index))


def write_label_space_records(result_dir, records, index=None):
    save_label_records(records, label_space_file(result_dir, index))


def get_selected_labels(prediction):
    labels = prediction.get("selected_labels")
    if labels:
        return labels
    return prediction.get("labels", [])[:1]


def zero_shot_training(args, result_dir, iter_index, file_name, doc_size):
    label_records = read_label_space_records(result_dir, iter_index + 1)
    label_path = label_space_file(result_dir, iter_index + 1)
    label_texts = get_label_texts(label_records)

    retrieval_model = SentenceTransformer(args.embedding_model, device=args.embedding_device)
    label_embeddings = load_or_create_embeddings(
        retrieval_model,
        label_texts,
        result_dir / "cache",
        build_cache_key(label_path, args.embedding_model),
        batch_size=args.embedding_batch_size,
    )

    with open(Path(args.path) / args.task / file_name, "r", encoding="utf-8") as file:
        docs = file.readlines()[:doc_size]

    zstc = pipeline("zero-shot-classification", model=args.model, device=get_pipeline_device(args.classifier_device))
    with prediction_file(result_dir, iter_index + 1).open("w", encoding="utf-8") as file:
        for index, raw_doc in enumerate(docs):
            print(index)
            sentence = raw_doc.strip()
            query_embedding = retrieval_model.encode(sentence, convert_to_numpy=True)
            ranked_candidates = rank_candidate_records(
                query_embedding,
                label_records,
                label_embeddings,
                args.top_k,
            )
            candidate_records = [record for record, _score in ranked_candidates]
            candidate_texts = [record["prototype_text"] for record in candidate_records]
            candidate_mapping = {record["prototype_text"]: record["name"] for record in candidate_records}

            output = zstc(
                sentence,
                candidate_texts,
                hypothesis_template=args.hypothesis_template,
                multi_label=True,
            )
            ranked_labels = [candidate_mapping[label] for label in output["labels"]]
            ranked_scores = [float(score) for score in output["scores"]]
            selected_labels, selected_scores = select_output_labels(
                ranked_labels,
                ranked_scores,
                args.score_threshold,
                args.relative_threshold,
                args.max_output_labels,
            )
            result = {
                "labels": ranked_labels,
                "scores": ranked_scores,
                "selected_labels": selected_labels,
                "selected_scores": selected_scores,
                "candidate_texts": output["labels"],
                "input_text": sentence,
            }
            print(result)
            json.dump(result, file, ensure_ascii=False)
            file.write("\n")


def select_output_labels(labels, scores, score_threshold, relative_threshold, max_output_labels):
    if not labels:
        return [], []

    top_score = scores[0]
    selected_labels = []
    selected_scores = []
    for label, score in zip(labels, scores):
        if score < score_threshold:
            continue
        if relative_threshold >= 0 and (top_score - score) > relative_threshold:
            continue
        selected_labels.append(label)
        selected_scores.append(score)
        if len(selected_labels) >= max_output_labels:
            break

    if not selected_labels:
        return [labels[0]], [scores[0]]
    return selected_labels, selected_scores


def label_deletion(major_label_list, text_label_dic, majority_num, max_majority_num=5):
    for label, count in text_label_dic:
        if count > majority_num and label not in major_label_list and len(major_label_list) < max_majority_num:
            major_label_list.append(label)
    return major_label_list


def load_predictions(path):
    with jsonlines.open(path, "r") as jsonl_file:
        return [obj for obj in jsonl_file]


def compute_uncertainty(prediction, entropy_top_k):
    scores = prediction.get("scores", [])[:entropy_top_k]
    if not scores:
        return 0.0
    metrics = summarize_scores(scores)
    max_entropy = math.log(max(2, len(scores)))
    normalized_entropy = metrics["entropy"] / max_entropy if max_entropy else 0.0
    return (
        (1.0 - metrics["top1"]) * 0.5
        + (1.0 - max(metrics["margin"], 0.0)) * 0.3
        + normalized_entropy * 0.2
    )


def collect_keyword_groups(keyword_docs):
    groups = {}
    for index, row in enumerate(keyword_docs):
        label_id = int(row.split(": ")[0])
        groups.setdefault(label_id, []).append(index)
    return groups


def normalize_keyword(keyword):
    cleaned = keyword.strip()
    if cleaned.startswith("["):
        cleaned = cleaned[1:]
    if cleaned.endswith("]"):
        cleaned = cleaned[:-1]
    return cleaned.strip()


def collect_candidate_labels(keyword_docs, uncertain_doc_indices, min_uncertain_support):
    groups = collect_keyword_groups(keyword_docs)
    group_support = {}
    global_frequency = {}

    for row in keyword_docs:
        for keyword in row.strip().split(": ", 1)[1].split(", "):
            normalized = normalize_keyword(keyword)
            global_frequency[normalized] = global_frequency.get(normalized, 0) + 1

    for doc_index in sorted(set(uncertain_doc_indices)):
        group_indices = groups.get(doc_index, [])
        group_keywords = set()
        for row_index in group_indices:
            keywords = keyword_docs[row_index].strip().split(": ", 1)[1].split(", ")
            for keyword in keywords[:3]:
                normalized = normalize_keyword(keyword)
                if normalized:
                    group_keywords.add(normalized)
        for keyword in group_keywords:
            group_support[keyword] = group_support.get(keyword, 0) + 1

    filtered_keywords = []
    for keyword, support_count in sorted(group_support.items(), key=lambda item: (-item[1], item[0])):
        if support_count >= min_uncertain_support:
            filtered_keywords.append((keyword, support_count, global_frequency.get(keyword, 0)))
    return filtered_keywords


def filter_new_labels(args, result_dir, candidate_keywords, current_records):
    if not candidate_keywords:
        return []

    model = SentenceTransformer(args.embedding_model, device=args.embedding_device)
    candidate_names = [item[0] for item in candidate_keywords if item[2] >= args.min_global_frequency]
    if not candidate_names:
        return []

    candidate_embeddings = model.encode(candidate_names, convert_to_numpy=True)
    kept_candidates = []
    kept_embeddings = []
    for index, candidate_name in enumerate(candidate_names):
        candidate_embedding = candidate_embeddings[index]
        if kept_embeddings:
            pair_scores = np.dot(np.asarray(kept_embeddings), candidate_embedding)
            if float(np.max(pair_scores)) >= args.internal_dedup_threshold:
                continue
        kept_candidates.append(candidate_name)
        kept_embeddings.append(candidate_embedding)

    if not kept_candidates:
        return []

    label_embeddings = load_or_create_embeddings(
        model,
        get_label_texts(current_records),
        result_dir / "cache",
        build_cache_key(label_space_file(result_dir), args.embedding_model),
        batch_size=args.embedding_batch_size,
    )

    final_add_label = []
    candidate_embeddings = model.encode(kept_candidates, convert_to_numpy=True)
    for index, candidate_name in enumerate(kept_candidates):
        similarity_scores = np.dot(label_embeddings, candidate_embeddings[index])
        if float(np.max(similarity_scores)) < args.sim_threshold:
            final_add_label.append(candidate_name)
        if len(final_add_label) >= args.max_add_label:
            break
    return final_add_label


def self_training(args, result_dir, iter_index):
    predictions = load_predictions(prediction_file(result_dir, iter_index))
    uncertainty_scores = np.asarray(
        [compute_uncertainty(prediction, args.entropy_top_k) for prediction in predictions],
        dtype=float,
    )
    rank_list = np.argsort(uncertainty_scores)[-args.tail_set_size:][::-1]

    with open(Path(args.path) / args.task / args.keyphrase_dir, "r", encoding="utf-8") as file:
        keyword_docs = file.readlines()[: len(predictions)]

    uncertain_doc_indices = []
    for index in rank_list:
        doc_index = int(keyword_docs[index].split(": ")[0])
        uncertain_doc_indices.append(doc_index)

    current_records = read_label_space_records(result_dir)
    candidate_keywords = collect_candidate_labels(
        keyword_docs,
        uncertain_doc_indices,
        args.min_uncertain_support,
    )
    return filter_new_labels(args, result_dir, candidate_keywords, current_records)


def main(args):
    print(
        args.tail_set_size,
        args.majority_num,
        args.max_majority_num,
        args.sim_threshold,
        args.max_add_label,
    )
    result_dir = get_result_dir(args)
    initialize_label_space_files(args, result_dir)

    major_label_list = []
    for iter_index in range(args.max_iterations):
        current_prediction_path = prediction_file(result_dir, iter_index)
        if not current_prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {current_prediction_path}")

        json_raw_list = load_predictions(current_prediction_path)
        document_size = len(json_raw_list)
        label_space_records = read_label_space_records(result_dir)
        cur_label_space_records = read_label_space_records(result_dir, iter_index)

        text_label_dic = {record["name"]: 0 for record in cur_label_space_records}
        for prediction in json_raw_list:
            for label in get_selected_labels(prediction):
                if label in text_label_dic:
                    text_label_dic[label] += 1

        text_sorted_dic = sorted(text_label_dic.items(), key=lambda item: item[1], reverse=True)
        remove_key_list = []
        remaining_records = []
        for record in label_space_records:
            count = text_label_dic.get(record["name"], 0)
            if count <= args.min_label_frequency:
                print(record["name"])
                remove_key_list.append(record["name"])
            else:
                remaining_records.append(record)

        write_label_space_records(result_dir, remaining_records)
        append_log_line(result_dir / "remove_label_list.txt", remove_key_list)

        final_add_label = self_training(args, result_dir, iter_index)
        append_log_line(result_dir / "add_label_list.txt", final_add_label)

        augmented_records = remaining_records + [build_label_record(name=label) for label in final_add_label]
        write_label_space_records(result_dir, augmented_records)
        print(final_add_label)

        major_label_list = label_deletion(
            major_label_list,
            text_sorted_dic,
            args.majority_num,
            args.max_majority_num,
        )
        next_records = [record for record in augmented_records if record["name"] not in major_label_list]
        write_label_space_records(result_dir, next_records, iter_index + 1)
        append_log_line(result_dir / "majority_label_list.txt", major_label_list)
        print(major_label_list)

        zero_shot_training(args, result_dir, iter_index, args.data_dir, document_size)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="train_texts_split_50.txt")
    parser.add_argument("--keyphrase_dir", type=str, default="llama2_label_50.txt")
    parser.add_argument("--task", type=str, default="AAPD")
    parser.add_argument("--llama_model", type=str, default="llama2")
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--tail_set_size", type=int, default=500, choices=[500, 1000, 1500])
    parser.add_argument("--majority_num", type=int, default=350, choices=[350, 400, 500, 650])
    parser.add_argument("--max_majority_num", type=int, default=5, choices=[5, 10])
    parser.add_argument("--min_label_frequency", type=int, default=6)
    parser.add_argument("--sim_threshold", type=float, default=0.55, choices=[0.55, 0.60, 0.65])
    parser.add_argument("--internal_dedup_threshold", type=float, default=0.70)
    parser.add_argument("--max_add_label", type=int, default=10, choices=[10, 15, 20])
    parser.add_argument("--min_uncertain_support", type=int, default=3)
    parser.add_argument("--min_global_frequency", type=int, default=15)
    parser.add_argument("--entropy_top_k", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_output_labels", type=int, default=3)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--relative_threshold", type=float, default=0.2)
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda")
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--classifier_device", type=str, default="cuda")
    parser.add_argument("--hypothesis_template", type=str, default="This example is about {}.")
    parser.add_argument("--model", type=str, default="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
    args = parser.parse_args()

    main(args)
