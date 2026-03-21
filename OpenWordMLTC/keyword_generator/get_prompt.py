DATASET_SPECS = {
    "AAPD": {
        "document_type": "a computer science paper abstract",
        "example_document": (
            "the present work studies quantum and classical correlations in three qubits and four "
            "qubits general bell states , produced by operating with braid operators on the "
            "computational basis of states the analogies between the general three qubits and four "
            "qubits bell states and that of two qubits bell states are discussed the general bell "
            "states are shown to be maximal entangled , i e , with quantum correlations which are "
            "lost by tracing these states over one qubit , remaining only with classical correlations , "
            "as shown by hs decomposition"
        ),
        "candidate_labels": ["Quantum Physics", "Information Theory", "Quantum Entanglement"],
        "final_label": "Quantum Physics",
        "candidate_instruction": (
            "Assign at least three concise subject labels that best describe the abstract. "
            "Use broad topics first and include finer-grained subjects only if they are clearly supported."
        ),
        "final_instruction": "Assign exactly one best label that most directly describes the abstract.",
    },
    "Amazon-531": {
        "document_type": "an Amazon product review",
        "example_document": (
            "omron hem 790it automatic blood pressure monitor with advanced omron health management "
            "software so far this machine has worked well and is very simple to use. it is nice to "
            "have immediate feedback on the blood-pressure effects of my various exercises, food "
            "consumption, and relaxation or stress levels"
        ),
        "candidate_labels": [
            "health_personal_care",
            "medical_supplies_equipment",
            "health_monitors",
        ],
        "final_label": "health_personal_care",
        "candidate_instruction": (
            "Assign two coarse-grained and two fine-grained product labels that best describe the reviewed item."
        ),
        "final_instruction": "Assign exactly one best product label for the review.",
    },
    "DBPedia-298": {
        "document_type": "a factual statement from Wikipedia",
        "example_document": (
            "enyalioides binzayedi is a species of lizard in the genus enyalioides known from only one "
            "location in the cordillera azul national park in peru . the lizard is named after mohammed "
            "bin zayed al nahyan , who sponsored the field survey that led to the discovery of the species ."
        ),
        "candidate_labels": ["species", "animal", "reptile"],
        "final_label": "species",
        "candidate_instruction": (
            "Assign two coarse-grained and two fine-grained category labels that best describe the fact."
        ),
        "final_instruction": "Assign exactly one best category label for the fact.",
    },
    "RCV1": {
        "document_type": "a Reuters newswire article",
        "example_document": (
            "another liability challenge to the tobacco industry headed toward a decision in court "
            "tuesday. a jury in marion county superior court was expected to begin deliberations in the "
            "case on wednesday or thursday. kansas attorney general carla stovall discuss their states' "
            "litigation. also tuesday, new york city's public advocate urged mayor rudolph giuliani to "
            "sue the tobacco industry to recoup health care costs of smokers. the rogers case, similar "
            "to hundreds that have been filed across the country, comes on the heels of a one earlier "
            "this month in jacksonville, florida, where a jury awarded $750,000 to a man who smoked for "
            "44 years before he was stricken with lung cancer. corp. is a unit of britain's b.a.t "
            "industries plc. in the indianapolis case, which was tried once before and wound up in a "
            "hung jury, the rogers family contended that the industry peddled an addictive product that "
            "was the cause of his lung cancer. rogers originally filed the suit himself"
        ),
        "candidate_labels": [
            "corporate/industrial",
            "government/social",
            "legal/judicial",
            "crime, law enforcement",
        ],
        "final_label": "corporate/industrial",
        "candidate_instruction": (
            "Assign two coarse-grained and two fine-grained topic labels that best describe the article."
        ),
        "final_instruction": "Assign exactly one best topic label for the article.",
    },
    "Reuters-21578": {
        "document_type": "a Reuters financial news article",
        "example_document": (
            "ending february 28, profits may be below the 2.4 mln dlrs, or 15 cts a share, earned in the "
            "first quarter of fiscal 1986. the company said any decline would be due to expenses related "
            "to the acquisitions in the middle of the current quarter of seven licensees of sealy inc. "
            "because of these acquisitions, it said, first quarter sales will be substantially higher than "
            "last year's 67.1 mln dlrs."
        ),
        "candidate_labels": ["earn", "acquisitions"],
        "final_label": "earn",
        "candidate_instruction": "Assign at least three concise topic labels that best describe the article.",
        "final_instruction": "Assign exactly one best topic label for the article.",
    },
}


def _render_labels(labels):
    quoted = ", ".join(f'"{label}"' for label in labels)
    return f"[label] {quoted} [/label]."


def _build_messages(name, document, mode):
    if name not in DATASET_SPECS:
        raise NotImplementedError(f"Name {name} not found in the dataset list")

    spec = DATASET_SPECS[name]
    if mode == "candidate":
        instruction = spec["candidate_instruction"]
        assistant_output = _render_labels(spec["candidate_labels"])
    elif mode == "final":
        instruction = spec["final_instruction"]
        assistant_output = _render_labels([spec["final_label"]])
    else:
        raise NotImplementedError(f"Mode {mode} not supported")

    clean_document = " ".join(document.strip().split())
    return [
        {
            "role": "system",
            "content": (
                "You are an expert assistant for open-world multi-label text classification. "
                "Infer concise labels from a document. Return exactly one line. Do not explain your answer. "
                "Always use the format [label] \"label1\", \"label2\" [/label]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Example task on {spec['document_type']}:\n"
                f"Document: \"{spec['example_document']}\"\n"
                f"Instruction: {instruction}\n"
                "Return exactly one line in the required format."
            ),
        },
        {"role": "assistant", "content": assistant_output},
        {
            "role": "user",
            "content": (
                f"Now process this {spec['document_type']}:\n"
                f"Document: \"{clean_document}\"\n"
                f"Instruction: {instruction}\n"
                "Return exactly one line in the required format."
            ),
        },
    ]


def create_prompt(name, document):
    return _build_messages(name, document, "candidate")


def create_final_prompt(name, document):
    return _build_messages(name, document, "final")
