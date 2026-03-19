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


def prepare_query_text(raw_text, cur_type):
    sentence = raw_text.strip()
    if cur_type == "keyword" and ": " in sentence:
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


def collect_subset_docs(args, docs, cur_type):
    keyphrase_subset = []
    for doc in docs:
        prefix = doc.split(": ")[0] if cur_type == "keyword" else doc.split(" ")[0]
        if int(prefix) < args.dynamic_iter:
            keyphrase_subset.append(doc)
        else:
            break
    return keyphrase_subset


def zero_shot_training(args, file_name, cur_type):
    label_path = Path(args.path) / args.task / args.label_space_path
    result_dir = resolve_run_dir(Path(args.path) / args.task / args.result_dir, args.run_name)
    cache_dir = result_dir / "cache"

    label_records = load_label_records(label_path)
    label_texts = get_label_texts(label_records)

    retrieval_model = SentenceTransformer(args.embedding_model, device=args.embedding_device)
    label_embeddings = load_or_create_embeddings(
        retrieval_model,
        label_texts,
        cache_dir,
        build_cache_key(label_path, args.embedding_model),
        batch_size=args.embedding_batch_size,
    )

    with open(Path(args.path) / args.task / file_name, "r", encoding="utf-8") as file:
        docs = file.readlines()
    keyphrase_subset = collect_subset_docs(args, docs, cur_type)
    print(len(keyphrase_subset))

    zstc = pipeline("zero-shot-classification", model=args.model, device=get_pipeline_device(args.classifier_device))
    template = args.hypothesis_template
    output_path = result_dir / f"zero_shot_{cur_type}_train_0.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for index, raw_doc in enumerate(keyphrase_subset):
            print(index)
            sentence = prepare_query_text(raw_doc, cur_type)
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
                hypothesis_template=template,
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


def main(args):
    zero_shot_training(args, args.keyphrase_dir, "keyword")
    zero_shot_training(args, args.data_dir, "text")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="train_texts_split_50.txt")
    parser.add_argument("--keyphrase_dir", type=str, default="llama2_label_50.txt")
    parser.add_argument("--task", type=str, default="AAPD")
    parser.add_argument("--dynamic_iter", type=int, default=3000)
    parser.add_argument("--label_space_path", type=str, default="llama2/init_label_space.txt")
    parser.add_argument("--result_dir", type=str, default="llama2/result")
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
    parser.add_argument("--model", type=str, default="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
    args = parser.parse_args()

    main(args)
