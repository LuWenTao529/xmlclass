from argparse import ArgumentParser
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from OpenWordMLTC.local_llm_utils import DEFAULT_MODEL, chat_completion, extract_label_block
from OpenWordMLTC.keyword_generator.get_prompt import create_prompt


def count_completed_rows(output_path: Path) -> int:
    if not output_path.exists():
        return 0
    with output_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def clean_label_output(text: str) -> str:
    label = extract_label_block(text)
    cleaned = (
        label.replace("[label]", "")
        .replace("[/label]", "")
        .replace("<<label>>", "")
        .replace("/label>>", "")
        .replace('"', "")
        .replace(".", "")
        .strip()
    )
    return " ".join(cleaned.split())


def main(args):
    data_path = Path(args.path) / args.task / args.data_dir
    output_path = Path(args.path) / args.task / args.output_dir
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with data_path.open("r", encoding="utf-8") as handle:
        documents = handle.readlines()

    completed = count_completed_rows(output_path)
    if completed:
        print(f"Resuming from row {completed}.")

    with output_path.open("a", encoding="utf-8") as writer:
        for doc in documents[completed:]:
            number_index = doc.split(" ")[0]
            content = doc[len(number_index) :].strip()
            messages = create_prompt(args.task, content)
            text = chat_completion(
                messages,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            label = clean_label_output(text)
            row = f"{number_index}: {label}"
            writer.write(row + "\n")
            writer.flush()
            print(row)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../datasets")
    parser.add_argument("--data_dir", type=str, default="train_texts_split_50.txt")
    parser.add_argument("--task", type=str, default="AAPD")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="deepseek_chat_label_50.txt")
    args = parser.parse_args()

    main(args)
