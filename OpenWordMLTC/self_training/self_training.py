import json
import os
from argparse import ArgumentParser
from pathlib import Path

import jsonlines
import numpy as np
import torch

os.environ.setdefault("USE_TF", "0")

from sentence_transformers import CrossEncoder, SentenceTransformer, util
from transformers import pipeline


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HYPOTHESIS_TEMPLATE = "This example is {}"


def label_deletion(major_label_list, text_label_dic, majority_num, max_majority_num=5):
    for i, label_pair in enumerate(text_label_dic):
        if label_pair[1] > majority_num and i < max_majority_num and label_pair[0] not in major_label_list:
            major_label_list.append(label_pair[0])

    return major_label_list


def get_result_dir(args):
    return Path(args.path) / args.task / args.llama_model / "result"


def get_zero_shot_output_path(args, iter_index):
    return get_result_dir(args) / f"zero_shot_text_train{iter_index}.jsonl"


def resolve_zero_shot_input_path(args, iter_index):
    result_dir = get_result_dir(args)
    candidates = [
        result_dir / f"zero_shot_text_train{iter_index}.jsonl",
        result_dir / f"zero_shot_text_train_{iter_index}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_runtime_devices():
    if torch.cuda.is_available():
        return "cuda", 0
    return "cpu", -1


def read_label_space(args):
    with open(get_result_dir(args) / "update_labelspace.txt", "r", encoding="utf-8") as file:
        return [row.strip() for row in file if row.strip()]


def read_cur_label_space(args, index):
    with open(get_result_dir(args) / f"update_labelspace{index}.txt", "r", encoding="utf-8") as file:
        return [row.strip() for row in file if row.strip()]


def write_to_label_space(args, label_space):
    with open(get_result_dir(args) / "update_labelspace.txt", "w", encoding="utf-8") as the_file:
        for label in label_space:
            the_file.write(label + "\n")


def write_to_new_label_space(args, label_space, iter_index):
    with open(get_result_dir(args) / f"update_labelspace{iter_index + 1}.txt", "w", encoding="utf-8") as the_file:
        for label in label_space:
            the_file.write(label + "\n")


def load_models(args):
    model_device, pipeline_device = get_runtime_devices()
    embedder = SentenceTransformer(args.embedding_model, device=model_device)
    reranker = None
    if args.reranker_model.lower() != "none":
        reranker = CrossEncoder(args.reranker_model, device=model_device)
    zstc = pipeline("zero-shot-classification", model=args.model, device=pipeline_device)
    return embedder, reranker, zstc


def retrieve_candidate_labels(doc, label_space, label_embeddings, embedder, candidate_top_k):
    top_k = min(candidate_top_k, len(label_space))
    query_embedding = embedder.encode(doc, convert_to_tensor=True)
    sim_scores = util.dot_score(query_embedding, label_embeddings)[0].detach().cpu().numpy()
    rank_list = np.argsort(sim_scores)[-top_k:][::-1]
    candidate_labels = [label_space[index] for index in rank_list]
    candidate_scores = [float(sim_scores[index]) for index in rank_list]
    return candidate_labels, candidate_scores


def rerank_candidate_labels(doc, candidate_labels, reranker, rerank_top_m):
    if not candidate_labels:
        return [], [], None

    top_m = min(rerank_top_m, len(candidate_labels))
    if reranker is None:
        selected_labels = candidate_labels[:top_m]
        return selected_labels, [], selected_labels[0]

    pair_scores = np.atleast_1d(
        reranker.predict([[doc, label] for label in candidate_labels], show_progress_bar=False)
    )
    rank_list = np.argsort(pair_scores)[::-1]
    selected_indices = rank_list[:top_m]
    selected_labels = [candidate_labels[index] for index in selected_indices]
    selected_scores = [float(pair_scores[index]) for index in selected_indices]
    reranker_top1 = candidate_labels[int(rank_list[0])]
    return selected_labels, selected_scores, reranker_top1


def build_zero_shot_output(sentence, selected_labels, zstc):
    output = zstc(
        sentence,
        selected_labels,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    output.pop("sequence", None)
    return output


def zero_shot_training(args, iter_index, file_name, doc_size):
    label_space = read_cur_label_space(args, iter_index + 1)
    if not label_space:
        raise ValueError("Label space is empty after update; cannot run zero-shot classification.")

    with open(Path(args.path) / args.task / file_name, "r", encoding="utf-8") as file2:
        docs = file2.readlines()[:doc_size]

    embedder, reranker, zstc = load_models(args)
    label_embeddings = embedder.encode(label_space, convert_to_tensor=True, show_progress_bar=False)
    output_path = get_zero_shot_output_path(args, iter_index + 1)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            if i % 100 == 0:
                print(f"classifying {i}/{len(docs)}")

            sentence = doc.strip()
            candidate_labels, candidate_scores = retrieve_candidate_labels(
                doc,
                label_space,
                label_embeddings,
                embedder,
                args.candidate_top_k,
            )
            selected_labels, selected_scores, reranker_top1 = rerank_candidate_labels(
                doc,
                candidate_labels,
                reranker,
                args.rerank_top_m,
            )

            output = build_zero_shot_output(sentence, selected_labels, zstc)
            output["retrieval_labels"] = candidate_labels
            output["retrieval_scores"] = candidate_scores
            output["reranker_labels"] = selected_labels
            output["reranker_scores"] = selected_scores
            output["reranker_top1"] = reranker_top1
            json.dump(output, f)
            f.write("\n")


def string_edit(key):
    if not key:
        return key
    if key[0] == "[":
        key = key[1:]
    if key[-1] == "]":
        key = key[:-1]
    return key


def compute_uncertainty(sample, args):
    scores = sample.get("scores", [])
    p1 = float(scores[0]) if scores else 0.0
    p2 = float(scores[1]) if len(scores) > 1 else 0.0
    margin = max(p1 - p2, 0.0)
    nli_top1 = sample["labels"][0] if sample.get("labels") else None
    reranker_top1 = sample.get("reranker_top1", nli_top1)
    disagreement = 1.0 if reranker_top1 and nli_top1 and reranker_top1 != nli_top1 else 0.0

    return (
        args.lambda_top1 * (1.0 - p1)
        + args.lambda_margin * (1.0 - margin)
        + args.lambda_disagreement * disagreement
    )


def self_training(args, iter_index):
    with jsonlines.open(resolve_zero_shot_input_path(args, iter_index), "r") as jsonl_f:
        json_raw_list = [obj for obj in jsonl_f]

    uncertainty_scores = np.zeros(len(json_raw_list))
    for i, sample in enumerate(json_raw_list):
        uncertainty_scores[i] = compute_uncertainty(sample, args)

    rank_list = np.argsort(uncertainty_scores)[-args.tail_set_size:][::-1]

    with open(Path(args.path) / args.task / args.keyphrase_dir, "r", encoding="utf-8") as file1:
        keyword_docs = file1.readlines()[: len(json_raw_list)]

    total_word_list = []
    for row in keyword_docs:
        cur_list = row.strip().split(": ")[1].split(", ")
        total_word_list.extend(cur_list)

    tail_array = []
    for index in rank_list:
        label = int(keyword_docs[index].split(": ")[0])
        cur_index = index
        row = [index]
        while cur_index > 0 and int(keyword_docs[cur_index - 1].split(": ")[0]) == label:
            cur_index -= 1
            row.append(cur_index)
        cur_index = index
        while cur_index < (len(keyword_docs) - 1) and int(keyword_docs[cur_index + 1].split(": ")[0]) == label:
            cur_index += 1
            row.append(cur_index)
        tail_array.append(row)

    keyword_dic = {}
    for index_list in tail_array:
        label_doc = keyword_docs[index_list[0]]
        for keyword in label_doc.strip().split(": ")[1].split(", ")[:3]:
            count = 0
            if len(index_list) > 1:
                for index in index_list[1:]:
                    test_label_doc = keyword_docs[index]
                    test_keyword_list = test_label_doc.strip().split(": ")[1].split(", ")
                    if keyword in test_keyword_list:
                        count += 1
            if keyword in keyword_dic:
                if count > keyword_dic[keyword]:
                    keyword_dic[keyword] = count
            else:
                keyword_dic[keyword] = count

    add_label = []
    for key, local_repeat in keyword_dic.items():
        count = 0
        for node in total_word_list:
            if string_edit(node) == string_edit(key):
                count += 1
        if count - local_repeat >= 15:
            add_label.append(string_edit(key))

    if not add_label:
        return []

    model_device, _ = get_runtime_devices()
    embedder = SentenceTransformer(args.embedding_model, device=model_device)
    add_label_embeddings = embedder.encode(add_label, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = util.dot_score(add_label_embeddings, add_label_embeddings).detach().cpu().numpy()

    deleted_list = set()
    for i, sim_score in enumerate(sim_matrix):
        for j, score in enumerate(sim_score):
            if score > (args.sim_threshold + iter_index / 100) and score < 1.1 and i < j:
                deleted_list.add(add_label[i])

    add_label = [label for label in add_label if label not in deleted_list]
    if not add_label:
        return []

    predict_label_space = read_label_space(args)
    if not predict_label_space:
        return add_label[: args.max_add_label]

    add_label_embeddings = embedder.encode(add_label, convert_to_tensor=True, show_progress_bar=False)
    passage_embedding = embedder.encode(predict_label_space, convert_to_tensor=True, show_progress_bar=False)
    final_sim_matrix = util.dot_score(add_label_embeddings, passage_embedding).detach().cpu().numpy()

    final_add_label = []
    threshold = args.sim_threshold + iter_index / 100
    for i, sim_score in enumerate(final_sim_matrix):
        if float(np.max(sim_score)) < threshold:
            final_add_label.append(add_label[i])
    return final_add_label[: args.max_add_label]


def main(args):
    print(
        args.tail_set_size,
        args.majority_num,
        args.max_majority_num,
        args.sim_threshold,
        args.max_add_label,
        args.candidate_top_k,
        args.rerank_top_m,
    )
    major_label_list = []
    for iter_index in range(10):
        with jsonlines.open(resolve_zero_shot_input_path(args, iter_index), "r") as jsonl_f:
            json_raw_list = [obj for obj in jsonl_f]

        label_list_raw = [sample["labels"][0] for sample in json_raw_list]
        document_size = len(json_raw_list)

        label_space = read_label_space(args)
        cur_label_space = read_cur_label_space(args, iter_index)

        text_label_dic = {item: 0 for item in cur_label_space}
        for label in label_list_raw:
            text_label_dic[label] += 1

        text_sorted_dic = sorted(text_label_dic.items(), key=lambda x: x[1], reverse=True)

        remove_key_list = []
        for key in text_label_dic:
            if text_label_dic[key] <= 6 and key in label_space:
                print(key)
                remove_key_list.append(key)
                label_space.remove(key)

        write_to_label_space(args, label_space)

        with open(get_result_dir(args) / "remove_label_list.txt", "a", encoding="utf-8") as the_file:
            the_file.write(str(remove_key_list) + "\n")

        final_add_label = self_training(args, iter_index)

        with open(get_result_dir(args) / "add_label_list.txt", "a", encoding="utf-8") as the_file:
            the_file.write(str(final_add_label) + "\n")

        with open(get_result_dir(args) / "update_labelspace.txt", "a", encoding="utf-8") as the_file:
            for label in final_add_label:
                the_file.write(label + "\n")
        print(final_add_label)

        new_label_space = label_space + final_add_label

        major_label_list = label_deletion(
            major_label_list,
            text_sorted_dic,
            args.majority_num,
            args.max_majority_num,
        )
        for label in major_label_list:
            if label in new_label_space:
                new_label_space.remove(label)
        write_to_new_label_space(args, new_label_space, iter_index)

        with open(get_result_dir(args) / "majority_label_list.txt", "a", encoding="utf-8") as the_file:
            the_file.write(str(major_label_list) + "\n")

        print(major_label_list)

        zero_shot_training(args, iter_index, args.data_dir, document_size)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="train_texts_split_50.txt")
    parser.add_argument("--keyphrase_dir", type=str, default="qwen3_label_50.txt")
    parser.add_argument("--task", type=str, default="AAPD")
    parser.add_argument("--llama_model", type=str, default="qwen3")
    parser.add_argument("--tail_set_size", type=int, default=500, choices=[500, 1000, 1500])
    parser.add_argument("--majority_num", type=int, default=350, choices=[350, 400, 500, 650])
    parser.add_argument("--max_majority_num", type=int, default=5, choices=[5, 10])
    parser.add_argument("--sim_threshold", type=float, default=0.55, choices=[0.55, 0.60, 0.65])
    parser.add_argument("--max_add_label", type=int, default=10, choices=[10, 15, 20])
    parser.add_argument("--model", type=str, default="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--candidate_top_k", type=int, default=32)
    parser.add_argument("--rerank_top_m", type=int, default=8)
    parser.add_argument("--lambda_top1", type=float, default=0.5)
    parser.add_argument("--lambda_margin", type=float, default=0.3)
    parser.add_argument("--lambda_disagreement", type=float, default=0.2)
    args = parser.parse_args()

    main(args)
