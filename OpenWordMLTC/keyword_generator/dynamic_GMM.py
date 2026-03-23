import os
import sys
from argparse import ArgumentParser
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import sklearn.cluster
from InstructorEmbedding import INSTRUCTOR
from sklearn.mixture import GaussianMixture
from umap import UMAP


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from OpenWordMLTC.local_llm_utils import DEFAULT_MODEL, chat_completion, extract_label_block
from OpenWordMLTC.keyword_generator.get_prompt import create_final_prompt


def get_top_chunks(keyphrase_subset, original_docs, cluster_num, task):
    docs = []
    if task == "AAPD":
        instruction = "Represent documents collected from computer science paper abstract for clustering: "
    elif task == "Amazon-531":
        instruction = "Represent documents collected from Amazon review data for clustering: "
    elif task == "DBPedia-298":
        instruction = "Represent documents collected from Wikipedia facts for clustering: "
    elif task == "Reuters-21578":
        instruction = "Represent documents collected from Reuters News Wire for clustering: "
    elif task == "RCV1":
        instruction = "Represent documents collected from the Reuters newswire for clustering: "
    else:
        raise NotImplementedError("Task not implemented")

    for line in keyphrase_subset:
        docs.append([instruction, line.split(": ", 1)[1].strip()])

    model = INSTRUCTOR("hkunlp/instructor-large", device="cuda")
    embeddings = model.encode(docs)

    if len(embeddings) <= 5:
        umap_embeddings = np.nan_to_num(embeddings)
    else:
        umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
        umap_model.fit(embeddings)
        umap_embeddings = np.nan_to_num(umap_model.transform(embeddings))

    if task == "Amazon-531":
        clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=cluster_num)
        clustering_model.fit(umap_embeddings)

        select_docs_per_label = []
        for idx in range(cluster_num):
            indices = np.argsort(clustering_model.transform(umap_embeddings)[:, idx])[:3]
            doc_list = " ".join(original_docs[ind].strip() for ind in indices)
            select_docs_per_label.append(doc_list)
            if idx % 50 == 0:
                print(idx)
    else:
        clustering_model = GaussianMixture(n_components=cluster_num, random_state=42)
        clustering_model.fit(umap_embeddings)
        predict_label = clustering_model.predict(umap_embeddings)
        means = clustering_model.means_

        select_docs_per_label = []
        for label in range(cluster_num):
            embedding_list = []
            embedding_index = []
            cur_mean = means[label]
            for idx, predict in enumerate(predict_label):
                if predict == label:
                    embedding_list.append(umap_embeddings[idx])
                    embedding_index.append(idx)
            dis_array = np.zeros(len(embedding_list))
            for idx, embed in enumerate(embedding_list):
                dis_array[idx] = np.linalg.norm(embed - cur_mean)
            sort_indices = np.argsort(dis_array)[:3]
            doc_list = " ".join(original_docs[embedding_index[cur_index]].strip() for cur_index in sort_indices)
            select_docs_per_label.append(doc_list)
            if label % 50 == 0:
                print(label)

    return select_docs_per_label


def label_cleaning(label):
    if "[label]" in label and "[/label]" in label:
        tokens = label.split('"')
        clean = "; ".join(tokens[i] for i in range(1, len(tokens) - 1, 2))
        return clean + ";\n"
    if "coarse-grained " in label and "fine-grained " in label:
        coarse = label.split("coarse-grained ")[1].split("fine-grained ")[0].split('"')
        fine = label.split("fine-grained ")[1].split(".")[0].split('"')
        coarse_labels = [coarse[i] for i in range(1, len(coarse) - 1, 2)]
        fine_labels = [fine[i] for i in range(1, len(fine) - 1, 2)]
        return "; ".join(coarse_labels + fine_labels) + ";\n"
    if "the label for " in label:
        tokens = label.split("the label for ")[1].split(".")[0].split('"')
        clean = "; ".join(tokens[i] for i in range(1, len(tokens) - 1, 2))
        return clean + ";\n"
    return label.strip() + "\n"


def gen_init_labelspace(documents, args):
    output_path = Path(args.path) / args.task / args.output_dir / args.output_file
    completed = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            completed = sum(1 for _ in handle)
    if completed:
        print(f"Resuming initial label-space generation from row {completed}.")

    with output_path.open("a", encoding="utf-8") as writer:
        for doc in documents[completed:]:
            messages = create_final_prompt(args.task, doc.strip())
            text = chat_completion(
                messages,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            label = extract_label_block(text)
            cleaned = label_cleaning(label)
            writer.write(cleaned)
            writer.flush()
            print(cleaned.strip())


def main(args):
    cluster_num = args.cluster_size
    with open(f"{args.path}/{args.task}/{args.keyphrase_dir}", "r", encoding="utf-8") as handle:
        documents = handle.readlines()

    keyphrase_subset = []
    chunk_size = len(documents)
    for idx, doc in enumerate(documents):
        prefix = doc.split(": ", 1)[0].replace("Result: ", "").strip()
        if int(prefix) < args.dynamic_iter:
            keyphrase_subset.append(doc)
        else:
            chunk_size = idx
            break

    with open(f"{args.path}/{args.task}/{args.data_dir}", "r", encoding="utf-8") as handle:
        original_docs = handle.readlines()[:chunk_size]

    folder_path = f"{args.path}/{args.task}/{args.output_dir}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    select_docs_per_label = get_top_chunks(keyphrase_subset, original_docs, cluster_num, args.task)
    gen_init_labelspace(select_docs_per_label, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="train_texts_split_50.txt")
    parser.add_argument("--keyphrase_dir", type=str, default="qwen3_label_50.txt")
    parser.add_argument("--task", type=str, default="Amazon-531")
    parser.add_argument("--dynamic_iter", type=int, default=14000)
    parser.add_argument("--cluster_size", type=int, default=398)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=96)
    parser.add_argument("--output_dir", type=str, default="qwen3")
    parser.add_argument("--output_file", type=str, default="init_labelspace.txt")
    args = parser.parse_args()

    main(args)
