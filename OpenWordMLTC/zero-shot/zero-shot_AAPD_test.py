import os
os.environ.setdefault("USE_TF", "0")
import numpy as np
from sentence_transformers import SentenceTransformer, util
from argparse import ArgumentParser
from transformers import pipeline
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from OpenWordMLTC.local_llm_utils import DEFAULT_MODEL

def zero_shot(args, zero_shot_model, model_index):
    file1 = open(f'{args.path}/{args.task}/{args.data_dir}', 'r')
    documents = file1.readlines()  

    label_space = []
    for row in documents:
        label = row.strip()
        label_space.append(label)

    # create a path if the path desn't exist
    result_dir = f"{args.path}/{args.task}/{args.result_dir}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = 'cuda')

    file2 = open(f'{args.path}/{args.task}/test_raw_texts.txt', 'r')
    docs = file2.readlines()

    top_labels = []
    for i, doc in enumerate(docs):
        print(i)
        query_embedding = model.encode(doc)
        passage_embedding = model.encode(label_space)
        sim_scores = util.dot_score(query_embedding, passage_embedding)[0].numpy()
        rank_list = np.argsort(sim_scores)[-8:]
        exp_label = []
        for index in rank_list:
            exp_label.append(label_space[index])
        top_labels.append(exp_label)

    print("pass")

    zstc = pipeline("zero-shot-classification", model= zero_shot_model, device = 'cuda')
    template = "This example is {}"

    with open(f'{result_dir}/zero_shot_base_test_{model_index}.jsonl', 'w') as f:
        for i, doc in enumerate(docs):
            sentence = doc.strip()
            output = zstc(sentence, top_labels[i], hypothesis_template=template, multi_label=False)
            del output["sequence"]
            print(output)
            json.dump(output, f)
            f.write('\n')



    file2 = open(f'{args.path}/{args.task}/{args.keyphrase_dir}', 'r')
    docs = file2.readlines()

    top_labels = []
    for i, doc in enumerate(docs):
        print(i)
        query_embedding = model.encode(doc)
        passage_embedding = model.encode(label_space)
        sim_scores = util.dot_score(query_embedding, passage_embedding)[0].numpy()
        rank_list = np.argsort(sim_scores)[-8:]
        exp_label = []
        for index in rank_list:
            exp_label.append(label_space[index])
        top_labels.append(exp_label)

    print("pass")

    zstc = pipeline("zero-shot-classification", model=zero_shot_model, device = 'cuda')
    template = "This example is {}"

    with open(f'{result_dir}/zero_shot_keyword_test_{model_index}.jsonl', 'w') as f:
        for i, doc in enumerate(docs):
            sentence = doc.strip()
            output = zstc(sentence, top_labels[i], hypothesis_template=template, multi_label=False)
            del output["sequence"]
            print(output)
            json.dump(output, f)
            f.write('\n')



    file2 = open(f'{args.path}/{args.task}/test_texts_split_50.txt', 'r')
    docs = file2.readlines()

    top_labels = []
    for i, doc in enumerate(docs):
        print(i)
        query_embedding = model.encode(doc)
        passage_embedding = model.encode(label_space)
        sim_scores = util.dot_score(query_embedding, passage_embedding)[0].numpy()
        rank_list = np.argsort(sim_scores)[-8:]
        exp_label = []
        for index in rank_list:
            exp_label.append(label_space[index])
        top_labels.append(exp_label)

    print("pass")

    zstc = pipeline("zero-shot-classification", model= zero_shot_model, device = 'cuda')
    template = "This example is {}"

    with open(f'{result_dir}/zero_shot_text_test_{model_index}.jsonl', 'w') as f:
        for i, doc in enumerate(docs):
            sentence = doc.strip()
            output = zstc(sentence, top_labels[i], hypothesis_template=template, multi_label=False)
            del output["sequence"]
            print(output)
            json.dump(output, f)
            f.write('\n')

def main(args):
    zero_shot(args, "facebook/bart-large-mnli", 'bart')
    zero_shot(args, "joeddav/xlm-roberta-large-xnli", 'xlm')
    zero_shot(args, "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33", 'deberta')

        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="qwen3/result/update_labelspace.txt")
    parser.add_argument("--task", type=str, default='AAPD')
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--keyphrase_dir", type=str, default="keyphrase_candidate/qwen3_label_test_50.txt")
    parser.add_argument("--result_dir", type=str, default="qwen3/test_performance")
    args = parser.parse_args()

    main(args)
