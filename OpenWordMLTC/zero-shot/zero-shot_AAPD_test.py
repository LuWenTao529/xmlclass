import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import pipeline

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from label_space_utils import (
    build_cache_key,
    create_label_mapping,
    get_label_texts,
    load_label_records,
    load_or_create_embeddings,
    rank_candidate_records,
    resolve_run_dir,
)


def get_pipeline_device(device_name):
    return 0 if str(device_name).startswith("cuda") else -1


def prepare_query_text(raw_text, is_keyword=False):
    sentence = raw_text.strip()
    if is_keyword and ": " in sentence:
        return sentence.split(": ", 1)[1].strip()
    return sentence


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


def run_zero_shot(split_name, docs, label_records, label_embeddings, retrieval_model, zstc, output_path, args):
    with output_path.open("w", encoding="utf-8") as file:
        for index, raw_doc in enumerate(docs):
            print(split_name, index)
            sentence = prepare_query_text(raw_doc, is_keyword=(split_name == "keyword"))
            query_embedding = retrieval_model.encode(sentence, convert_to_numpy=True)
            ranked_candidates = rank_candidate_records(
                query_embedding,
                label_records,
                label_embeddings,
                args.top_k,
            )
            candidate_records = [record for record, _score in ranked_candidates]
            candidate_texts = [record["prototype_text"] for record in candidate_records]
            canonical_label_mapping = create_label_mapping(candidate_records)

            output = zstc(
                sentence,
                candidate_texts,
                hypothesis_template=args.hypothesis_template,
                multi_label=True,
            )
            ranked_labels = [canonical_label_mapping[label] for label in output["labels"]]
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


def zero_shot(args, zero_shot_model, model_index):
    label_path = Path(args.path) / args.task / args.data_dir
    result_dir = resolve_run_dir(Path(args.path) / args.task / args.result_dir, args.run_name)
    cache_dir = result_dir / "cache"
    label_records = load_label_records(label_path)

    retrieval_model = SentenceTransformer(args.embedding_model, device=args.embedding_device)
    label_embeddings = load_or_create_embeddings(
        retrieval_model,
        get_label_texts(label_records),
        cache_dir,
        build_cache_key(label_path, args.embedding_model),
        batch_size=args.embedding_batch_size,
    )

    zstc = pipeline("zero-shot-classification", model=zero_shot_model, device=get_pipeline_device(args.classifier_device))
    data_splits = [
        ("base", Path(args.path) / args.task / "test_raw_texts.txt"),
        ("keyword", Path(args.path) / args.task / "keyphrase_candidate/llama2_label_test_50.txt"),
        ("text", Path(args.path) / args.task / "test_texts_split_50.txt"),
    ]

    for split_name, input_path in data_splits:
        with input_path.open("r", encoding="utf-8") as file:
            docs = file.readlines()
        output_name = f"zero_shot_{split_name}_test_{model_index}.jsonl"
        run_zero_shot(
            split_name,
            docs,
            label_records,
            label_embeddings,
            retrieval_model,
            zstc,
            result_dir / output_name,
            args,
        )


def main(args):
    zero_shot(args, "facebook/bart-large-mnli", "bart")
    zero_shot(args, "joeddav/xlm-roberta-large-xnli", "xlm")
    zero_shot(args, "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33", "deberta")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="llama2/result/update_labelspace.txt")
    parser.add_argument("--task", type=str, default="AAPD")
    parser.add_argument("--result_dir", type=str, default="llama2/test_performance")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_output_labels", type=int, default=3)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--relative_threshold", type=float, default=0.2)
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_device", type=str, default="cuda")
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--classifier_device", type=str, default="cuda")
    parser.add_argument("--hypothesis_template", type=str, default="This example is about {}.")
    args = parser.parse_args()

    main(args)
