import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("USE_TF", "0")

from sentence_transformers import CrossEncoder, SentenceTransformer, util
from transformers import pipeline


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HYPOTHESIS_TEMPLATE = "This example is {}"


def get_runtime_devices():
    if torch.cuda.is_available():
        return "cuda", 0
    return "cpu", -1


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


def run_zero_shot_pass(args, docs, label_space, output_path):
    embedder, reranker, zstc = load_models(args)
    label_embeddings = embedder.encode(label_space, convert_to_tensor=True, show_progress_bar=False)

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

            output = zstc(
                sentence,
                selected_labels,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
                multi_label=False,
            )
            output.pop("sequence", None)
            output["retrieval_labels"] = candidate_labels
            output["retrieval_scores"] = candidate_scores
            output["reranker_labels"] = selected_labels
            output["reranker_scores"] = selected_scores
            output["reranker_top1"] = reranker_top1
            json.dump(output, f)
            f.write("\n")


def zero_shot_training(args, file_name, cur_type):
    with open(Path(args.path) / args.task / args.label_space_file, "r", encoding="utf-8") as file1:
        label_space = [row.strip() for row in file1 if row.strip()]

    with open(Path(args.path) / args.task / file_name, "r", encoding="utf-8") as file2:
        docs = file2.readlines()

    if cur_type == "keyword":
        keyphrase_subset = []
        chunk_size = len(docs)
        for i, doc in enumerate(docs):
            if int(doc.split(": ")[0]) < args.dynamic_iter:
                keyphrase_subset.append(doc)
            else:
                chunk_size = i
                break
    else:
        keyphrase_subset = []
        chunk_size = len(docs)
        for i, doc in enumerate(docs):
            if int(doc.split(" ")[0]) < args.dynamic_iter:
                keyphrase_subset.append(doc)
            else:
                chunk_size = i
                break

    print(len(keyphrase_subset), chunk_size)

    path = Path(args.path) / args.task / args.result_dir
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / f"zero_shot_{cur_type}_train0.jsonl"
    run_zero_shot_pass(args, keyphrase_subset, label_space, output_path)


def main(args):
    zero_shot_training(args, args.keyphrase_dir, "keyword")
    zero_shot_training(args, args.data_dir, "text")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="train_texts_split_50.txt")
    parser.add_argument("--keyphrase_dir", type=str, default="qwen3_label_50.txt")
    parser.add_argument("--task", type=str, default="AAPD")
    parser.add_argument("--dynamic_iter", type=int, default=3000)
    parser.add_argument("--model", type=str, default="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
    parser.add_argument("--label_space_file", type=str, default="qwen3/init_label_space.txt")
    parser.add_argument("--result_dir", type=str, default="qwen3/result")
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--candidate_top_k", type=int, default=32)
    parser.add_argument("--rerank_top_m", type=int, default=8)
    args = parser.parse_args()

    main(args)
