import hashlib
import json
import re
from pathlib import Path

import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "default"


def normalize_text(text):
    return " ".join(str(text).strip().split())


def parse_label_source_line(line):
    raw_line = normalize_text(line)
    if not raw_line:
        return None

    segments = [segment.strip() for segment in raw_line.split(";") if segment.strip()]
    name = segments[0]
    description = segments[1] if len(segments) > 1 else ""
    support_texts = segments[2:]
    keywords = infer_keywords(name)
    return {
        "name": name,
        "description": description,
        "keywords": keywords,
        "support_texts": support_texts,
        "source_text": raw_line,
    }


def infer_keywords(name):
    parts = re.split(r"[\s,;/|:_-]+", normalize_text(name).replace(".", " "))
    keywords = []
    for part in parts:
        if part and part.lower() not in {"and", "of", "the", "for"}:
            keywords.append(part)
    return keywords[:5]


def build_label_record(name, description="", keywords=None, support_texts=None, source_text=""):
    name = normalize_text(name)
    description = normalize_text(description)
    keywords = [normalize_text(keyword) for keyword in (keywords or []) if normalize_text(keyword)]
    support_texts = [normalize_text(text) for text in (support_texts or []) if normalize_text(text)]

    if not description:
        normalized_name = normalize_text(name.replace("_", " ").replace(".", " "))
        description = f"Topic about {normalized_name}."

    prototype_parts = [f"label: {name}", f"description: {description}"]
    if keywords:
        prototype_parts.append(f"keywords: {', '.join(keywords[:5])}")
    if support_texts:
        prototype_parts.append(f"support: {' | '.join(support_texts[:2])}")

    prototype_text = ". ".join(prototype_parts)
    return {
        "name": name,
        "description": description,
        "keywords": keywords[:5],
        "support_texts": support_texts[:3],
        "prototype_text": prototype_text,
        "source_text": normalize_text(source_text),
    }


def merge_label_records(records):
    merged = {}
    for record in records:
        if not record:
            continue
        name = normalize_text(record["name"])
        existing = merged.get(name)
        if existing is None:
            merged[name] = build_label_record(
                name=name,
                description=record.get("description", ""),
                keywords=record.get("keywords", []),
                support_texts=record.get("support_texts", []),
                source_text=record.get("source_text", ""),
            )
            continue

        description = existing["description"]
        if len(record.get("description", "")) > len(description):
            description = record["description"]

        keywords = deduplicate_items(existing.get("keywords", []) + record.get("keywords", []), limit=5)
        support_texts = deduplicate_items(
            existing.get("support_texts", []) + record.get("support_texts", []),
            limit=3,
        )
        source_text = normalize_text(record.get("source_text", "")) or existing.get("source_text", "")
        merged[name] = build_label_record(
            name=name,
            description=description,
            keywords=keywords,
            support_texts=support_texts,
            source_text=source_text,
        )
    return list(merged.values())


def deduplicate_items(items, limit=None):
    seen = set()
    results = []
    for item in items:
        normalized = normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
        if limit is not None and len(results) >= limit:
            break
    return results


def text_label_path_for(label_path):
    path = Path(label_path)
    if path.suffix == ".jsonl":
        return path.with_suffix(".txt")
    return path


def jsonl_label_path_for(label_path):
    path = Path(label_path)
    if path.suffix == ".jsonl":
        return path
    return path.with_suffix(".jsonl")


def load_label_records(label_path):
    path = Path(label_path)
    jsonl_path = path if path.suffix == ".jsonl" else path.with_suffix(".jsonl")
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as file:
            records = []
            for line in file:
                if not normalize_text(line):
                    continue
                payload = json.loads(line)
                records.append(
                    build_label_record(
                        name=payload.get("name", ""),
                        description=payload.get("description", ""),
                        keywords=payload.get("keywords", []),
                        support_texts=payload.get("support_texts", []),
                        source_text=payload.get("source_text", ""),
                    )
                )
            return records

    if not path.exists():
        raise FileNotFoundError(f"Label space file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        source_records = [parse_label_source_line(line) for line in file]
    return merge_label_records(source_records)


def save_label_records(records, text_path, jsonl_path=None, append=False):
    text_path = text_label_path_for(text_path)
    jsonl_path = jsonl_label_path_for(jsonl_path or text_path)
    ensure_dir(text_path.parent)
    ensure_dir(jsonl_path.parent)

    mode = "a" if append else "w"
    with text_path.open(mode, encoding="utf-8") as text_file:
        for record in records:
            text_file.write(f"{record['name']}\n")

    with jsonl_path.open(mode, encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_label_names(records):
    return [record["name"] for record in records]


def get_label_texts(records):
    return [record["prototype_text"] for record in records]


def create_label_mapping(records):
    return {record["prototype_text"]: record["name"] for record in records}


def build_cache_key(label_path, model_name):
    safe_model_name = sanitize_name(model_name)
    return f"{Path(label_path).stem}_{safe_model_name}"


def compute_text_hash(texts):
    digest = hashlib.md5()
    for text in texts:
        digest.update(normalize_text(text).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_or_create_embeddings(model, texts, cache_dir, cache_key, batch_size=32):
    texts = [normalize_text(text) for text in texts]
    ensure_dir(cache_dir)
    array_path = Path(cache_dir) / f"{cache_key}.npy"
    meta_path = Path(cache_dir) / f"{cache_key}.json"
    text_hash = compute_text_hash(texts)

    if array_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("text_hash") == text_hash and metadata.get("count") == len(texts):
            return np.load(array_path)

    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    np.save(array_path, embeddings)
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump({"text_hash": text_hash, "count": len(texts)}, file)
    return embeddings


def rank_candidate_records(query_embedding, label_records, label_embeddings, top_k):
    scores = np.dot(label_embeddings, query_embedding)
    top_k = min(top_k, len(label_records))
    ranked_indices = np.argsort(scores)[-top_k:][::-1]
    return [(label_records[index], float(scores[index])) for index in ranked_indices]


def summarize_scores(scores):
    if not scores:
        return {"top1": 0.0, "top2": 0.0, "margin": 0.0, "entropy": 0.0}

    values = np.asarray(scores, dtype=float)
    top1 = float(values[0])
    top2 = float(values[1]) if len(values) > 1 else 0.0
    probabilities = np.clip(values, 1e-8, 1.0)
    probabilities = probabilities / probabilities.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return {"top1": top1, "top2": top2, "margin": top1 - top2, "entropy": entropy}


def resolve_run_dir(base_dir, run_name=""):
    base_path = Path(base_dir)
    if run_name:
        base_path = base_path / sanitize_name(run_name)
    ensure_dir(base_path)
    return base_path
