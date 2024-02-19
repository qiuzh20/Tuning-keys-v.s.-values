import pandas as pd
import datasets
from datasets import load_dataset
from sklearn.model_selection import train_test_split


data_root = ""


ZERO_SHOT_CLASSIFIER_PROMPT = """Classify the sentence into one of 20 classes. The list of classes is provided below, where the classes are separated by commas: 

{newsgroup_classes}

From the above list of classes, select only one class that the provided sentence can be classified into. The sentence will be delimited with triple backticks. Once again, only predict the class from the given list of classes. Do not predict anything else.

### Sentence: ```{sentence}```
### Class:
"""

FEW_SHOT_CLASSIFIER_PROMPT = """Classify the sentence into one of 20 classes. The list of classes is provided below, where the classes are separated by commas:

{newsgroup_classes}

From the above list of classes, select only one class that the provided sentence can be classified into. Once again, only predict the class from the given list of classes. Do not predict anything else. The sentence will be delimited with triple backticks. To help you, examples are provided of sentence and the corresponding class they belong to.

{few_shot_samples}

### Sentence: ```{sentence}```
### Class:
"""

TRAINING_CLASSIFIER_PROMPT = """Classify the following sentence that is delimited with triple backticks.

### Sentence: ```{sentence}```
### Class: {label}
"""

INFERENCE_CLASSIFIER_PROMPT = """Classify the following sentence that is delimited with triple backticks.

### Sentence: ```{sentence}```
### Class: 
"""

TRAINING_CLASSIFIER_PROMPT_v2 = """### Sentence:{sentence} ### Class:{label}"""
INFERENCE_CLASSIFIER_PROMPT_v2 = """### Sentence:{sentence} ### Class:"""

ZERO_SHOT_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.

### Dialogue: ```{dialogue}```
### Summary:
"""

FEW_SHOT_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks. To help you, examples of summarization are provided.

{few_shot_samples}

### Dialogue: ```{dialogue}```
### Summary:
"""

TRAINING_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.

### Dialogue: ```{dialogue}```
### Summary: {summary}
"""

TRAINING_SUMMARIZATION_PROMPT_v2 = """### Dialogue:{dialogue} ### Summary:{summary}"""
INFERENCE_SUMMARIZATION_PROMPT_v2 = """### Dialogue:{dialogue} ### Summary:"""

INFERENCE_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.

### Dialogue: ```{dialogue}```
### Summary: 
"""

def instructions_to_dataset(instructions, labels=None):
    if labels is None:
        labels = [None] * len(instructions)
    return datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": instructions,
                "labels": labels,
            }
        )
    )

def sample_data(data, sample_fraction=0.9):
    data_df = pd.DataFrame(data=data)
    train_df, test_df = train_test_split(
        data_df,
        train_size=sample_fraction,
        stratify=data_df["label"],
    )
    train_data = train_df["text"]
    train_labels = train_df["label"]
    return train_data, train_labels

def get_newsgroup_instruction_data(mode, texts, labels):
    if mode == "train":
        prompt = TRAINING_CLASSIFIER_PROMPT_v2
    elif mode == "inference":
        prompt = INFERENCE_CLASSIFIER_PROMPT_v2

    instructions = []

    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(
                sentence=text,
                label=label,
            )
        elif mode == "inference":
            example = prompt.format(
                sentence=text,
            )
        instructions.append(example)

    return instructions


def clean_newsgroup_data(texts, labels, max_sample=None):
    label2data = {}
    clean_data, clean_labels = [], []
    for data, label in zip(texts, labels):
        if isinstance(data, str) and isinstance(label, str):
            clean_data.append(data)
            clean_labels.append(label)

            if label not in label2data:
                label2data[label] = data
        if max_sample is not None and len(clean_data) >= max_sample:
            break

    return label2data, clean_data, clean_labels


def get_newsgroup_data_for_ft(mode="train", train_sample_fraction=0.99, max_test_sample=None):
    newsgroup_dataset = datasets.load_from_disk(data_root + "/disk_datasets/20_Newsgroups_Fixed")
    # newsgroup_dataset = load_dataset("rungalileo/20_Newsgroups_Fixed") 
    train_data = newsgroup_dataset["train"]["text"]
    train_labels = newsgroup_dataset["train"]["label"]
    label2data, train_data, train_labels = clean_newsgroup_data(
        train_data, train_labels, max_sample=max_test_sample
    )

    test_data = newsgroup_dataset["test"]["text"]
    print("total test data:", len(test_data))
    test_labels = newsgroup_dataset["test"]["label"]
    _, test_data, test_labels = clean_newsgroup_data(test_data, test_labels, max_sample=max_test_sample)

    # sample n points from training data
    train_df = pd.DataFrame(data={"text": train_data, "label": train_labels})
    if train_sample_fraction < 0.99 and mode == "train":
        train_df, _ = train_test_split(
            train_df,
            train_size=train_sample_fraction,
            stratify=train_df["label"],
        )
    train_data = train_df["text"]
    train_labels = train_df["label"]

    train_instructions = get_newsgroup_instruction_data(mode, train_data, train_labels)
    test_instructions = get_newsgroup_instruction_data(mode, test_data, test_labels)

    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": train_instructions,
                "labels": train_labels,
            }
        )
    )
    test_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": test_instructions,
                "labels": test_labels,
            }
        )
    )
    print("training dataset:", len(train_dataset))
    print("test dataset:", len(test_dataset))

    return train_dataset, test_dataset


def get_newsgroup_data():
    newsgroup_dataset = datasets.load_from_disk(data_root + "/disk_datasets/20_Newsgroups_Fixed")
    # newsgroup_dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
    train_data = newsgroup_dataset["train"]["text"]
    train_labels = newsgroup_dataset["train"]["label"]

    label2data, clean_data, clean_labels = clean_newsgroup_data(
        train_data, train_labels
    )
    df = pd.DataFrame(data={"text": clean_data, "label": clean_labels})

    newsgroup_classes = df["label"].unique()
    newsgroup_classes = ", ".join(newsgroup_classes)

    few_shot_samples = ""
    for label, data in label2data.items():
        sample = f"Sentence: {data} \n Class: {label} \n\n"
        few_shot_samples += sample

    return newsgroup_classes, few_shot_samples, df


def get_samsum_data():
    samsum_dataset = datasets.load_from_disk(data_root + "/disk_datasets/samsum")
    # samsum_dataset = load_dataset("samsum")
    train_dataset = samsum_dataset["train"]
    dialogues = train_dataset["dialogue"][:2]
    summaries = train_dataset["summary"][:2]

    few_shot_samples = ""
    for dialogue, summary in zip(dialogues, summaries):
        sample = f"Sentence: {dialogue} \n Summary: {summary} \n\n"
        few_shot_samples += sample

    return few_shot_samples


TRAINING_CLASSIFIER_BOOLQ_PROMPT_v2 = """Read the Passage and answer the Question with true or false ### Question:{question} ### Passage: {passage} ### Answer:{label}"""
INFERENCE_CLASSIFIER_BOOLQ_PROMPT_v2 = """Read the Passage and answer the Question with true or false ### Question:{question} ### Passage: {passage} ### Answer:"""


def get_boolq_instruction_data(mode, texts, labels):
    if mode == "train":
        prompt = TRAINING_CLASSIFIER_BOOLQ_PROMPT_v2
    elif mode == "inference":
        prompt = INFERENCE_CLASSIFIER_BOOLQ_PROMPT_v2

    instructions = []

    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(
                question=text[0],
                passage=text[1],
                label=label,
            )
        elif mode == "inference":
            example = prompt.format(
                question=text[0],
                passage=text[1],
            )
        instructions.append(example)

    return instructions

int_to_str = {1:"true", 0:"false"}

def clean_boolq_data(dataset, max_sample=None):
    label2data = {}
    clean_data, clean_labels = [], []
    for data in dataset:
        data["label"] = int_to_str[data["label"]]
        if isinstance(data["passage"], str) and isinstance(data["question"], str):
            clean_data.append([data["question"], data["passage"]])
            clean_labels.append(data["label"])
            if data["label"] not in label2data:
                label2data[data["label"]] = [data["question"], data["passage"]]
        if max_sample is not None and len(clean_data) >= max_sample:
            break
        
    return label2data, clean_data, clean_labels


def get_boolq_data_for_ft(mode="train", train_sample_fraction=0.99, max_test_sample=None):
    boolq_dataset = datasets.load_from_disk(data_root + "/disk_datasets/super_glue/boolq")
    train_data = boolq_dataset["train"]
    label2data, train_data, train_labels = clean_boolq_data(
        train_data, max_sample=max_test_sample
    )

    test_data = boolq_dataset["validation"]
    print("total test data:", len(test_data))
    _, test_data, test_labels = clean_boolq_data(test_data, max_sample=max_test_sample)

    # sample n points from training data
    train_df = pd.DataFrame(data={"text": train_data, "label": train_labels})
    if train_sample_fraction < 0.99 and mode == "train":
        train_df, _ = train_test_split(
            train_df,
            train_size=train_sample_fraction,
            stratify=train_df["label"],
        )
    train_data = train_df["text"]
    train_labels = train_df["label"]

    train_instructions = get_boolq_instruction_data(mode, train_data, train_labels)
    test_instructions = get_boolq_instruction_data(mode, test_data, test_labels)

    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": train_instructions,
                "labels": train_labels,
            }
        )
    )
    test_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": test_instructions,
                "labels": test_labels,
            }
        )
    )
    print("training dataset:", len(train_dataset))
    print("test dataset:", len(test_dataset))

    return train_dataset, test_dataset


# other tasks from superglue and glue




TRAINING_CLASSIFIER_COPA_PROMPT_v2 = """### Premise:{premise} ### Choice1:{choice1} ### Choice2:{choice2} ### Question:{question} ### Choice:{label}"""
INFERENCE_CLASSIFIER_COPA_PROMPT_v2 = """### Premise:{premise} ### Choice1:{choice1} ### Choice2:{choice2} ### Question:{question} ### Choice:"""

def get_COPA_instruction_data(mode, texts, labels):
    if mode == "train":
        prompt = TRAINING_CLASSIFIER_COPA_PROMPT_v2
    elif mode == "inference":
        prompt = INFERENCE_CLASSIFIER_COPA_PROMPT_v2

    instructions = []

    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(
                premise=text[0],
                choice1=text[1],
                choice2=text[2],
                question=text[3],
                label=label,
            )
        elif mode == "inference":
            example = prompt.format(
                premise=text[0],
                choice1=text[1],
                choice2=text[2],
                question=text[3],
            )
        instructions.append(example)

    return instructions

COPA_int_to_str = {0:"Choice1", 1:"Choice2"}

def clean_COPA_data(dataset, max_sample=None):
    label2data = {}
    clean_data, clean_labels = [], []
    for data in dataset:
        data["label"] = COPA_int_to_str[data["label"]]
        if isinstance(data["premise"], str) and isinstance(data["choice1"], str) and isinstance(data["choice2"], str) and isinstance(data["question"], str):
            clean_data.append([data["premise"], data["choice1"], data["choice2"], data["question"]])
            clean_labels.append(data["label"])
            if data["label"] not in label2data:
                label2data[data["label"]] = [data["premise"], data["choice1"], data["choice2"], data["question"]]
        if max_sample is not None and len(clean_data) >= max_sample:
            break
        
    return label2data, clean_data, clean_labels


def get_COPA_data_for_ft(mode="train", train_sample_fraction=0.99, max_test_sample=None):
    COPA_dataset = datasets.load_from_disk(data_root + "/disk_datasets/super_glue/copa")
    train_data = COPA_dataset["train"]
    _, train_data, train_labels = clean_COPA_data(train_data, max_sample=max_test_sample)

    test_data = COPA_dataset["validation"]
    print("total test data:", len(test_data))
    _, test_data, test_labels = clean_COPA_data(test_data, max_sample=max_test_sample)

    # sample n points from training data
    if train_sample_fraction < 0.99 and mode == "train":
        train_data, train_labels = sample_data(train_data, train_sample_fraction)

    train_instructions = get_COPA_instruction_data(mode, train_data, train_labels)
    test_instructions = get_COPA_instruction_data(mode, test_data, test_labels)

    train_dataset = instructions_to_dataset(train_instructions, train_labels)
    test_dataset = instructions_to_dataset(test_instructions, test_labels)
    print("COPA training dataset:", len(train_dataset))
    print("COPA test dataset:", len(test_dataset))

    return train_dataset, test_dataset



TRAINING_CLASSIFIER_IFERENCE_PROMPT_v2 = """### Premise:{premise} ### Hypothesis:{hypothesis} ### Relationship (entailment, contradiction or neutral):{label}"""
INFERENCE_CLASSIFIER_IFERENCE_PROMPT_v2 = """### Premise:{premise} ### Hypothesis:{hypothesis} ### Relationship (entailment, contradiction or neutral):"""

def get_inference_instruction_data(mode, texts, labels):
    if mode == "train":
        prompt = TRAINING_CLASSIFIER_IFERENCE_PROMPT_v2
    elif mode == "inference":
        prompt = INFERENCE_CLASSIFIER_IFERENCE_PROMPT_v2

    instructions = []

    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(
                premise=text[0],
                hypothesis=text[1],
                label=label,
            )
        elif mode == "inference":
            example = prompt.format(
                premise=text[0],
                hypothesis=text[1],
            )
        instructions.append(example)

    return instructions


def clean_inference_data(dataset, label_dict, max_sample=None):
    label2data = {}
    clean_data, clean_labels = [], []
    for data in dataset:
        data["label"] = label_dict[data["label"]]
        if isinstance(data["hypothesis"], str) and isinstance(data["premise"], str):
            clean_data.append([data["premise"], data["hypothesis"]])
            clean_labels.append(data["label"])
            if data["label"] not in label2data:
                label2data[data["label"]] = [data["premise"], data["hypothesis"]]
        if max_sample is not None and len(clean_data) >= max_sample:
            break
        
    return label2data, clean_data, clean_labels


def get_inference_data_for_ft(mode="train", name="CB", train_sample_fraction=0.99, max_test_sample=None):
    if name == "CB":
        temp_dataset = datasets.load_from_disk(data_root + "/disk_datasets/super_glue/cb")
        label_dict = {0:"entailment", 1:"contradiction", 2:"neutral"}
    if name == "MNLI":
        temp_dataset = datasets.load_from_disk(data_root + "/disk_datasets/glue/mnli")
        label_dict = {0:"entailment", 1:"neutral", 2:"contradiction"}

    train_data = temp_dataset["train"]
    _, train_data, train_labels = clean_inference_data(train_data, label_dict, max_sample=max_test_sample)

    if name == "MNLI":
        test_data = temp_dataset["validation_mismatched"]
    else:
        test_data = temp_dataset["validation"]
    _, test_data, test_labels = clean_inference_data(test_data, label_dict, max_sample=max_test_sample)

    # sample n points from training data
    if train_sample_fraction < 0.99 and mode == "train":
        train_data, train_labels = sample_data(train_data, train_sample_fraction)

    train_instructions = get_inference_instruction_data(mode, train_data, train_labels)
    test_instructions = get_inference_instruction_data(mode, test_data, test_labels)

    train_dataset = instructions_to_dataset(train_instructions, train_labels)
    test_dataset = instructions_to_dataset(test_instructions, test_labels)
    print("training dataset:", len(train_dataset))
    print("test dataset:", len(test_dataset))

    return train_dataset, test_dataset


TRAINING_CLASSIFIER_INF2_PROMPT_v2 = """### Premise:{premise} ### Hypothesis:{hypothesis} ### Relationship (choose between 'entailment' and 'not entailment'):{label}"""
INFERENCE_CLASSIFIER_INF2_PROMPT_v2 = """### Premise:{premise} ### Hypothesis:{hypothesis} ### Relationship (choose between 'entailment' and 'not entailment'):"""

def get_INF2_instruction_data(mode, texts, labels):
    if mode == "train":
        prompt = TRAINING_CLASSIFIER_INF2_PROMPT_v2
    elif mode == "inference":
        prompt = INFERENCE_CLASSIFIER_INF2_PROMPT_v2
    instructions = []
    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(premise=text[0], hypothesis=text[1], label=label)
        elif mode == "inference":
            example = prompt.format(premise=text[0], hypothesis=text[1])
        instructions.append(example)
    return instructions

def clean_INF2_data(dataset, label_dict, name_list, max_sample=None):
    label2data = {}
    clean_data, clean_labels = [], []
    for data in dataset:
        data["label"] = label_dict[data[name_list[2]]]
        # if isinstance(data["premise"], str) and isinstance(data["hypothesis"], str):
        if isinstance(data[name_list[0]], str) and isinstance(data[name_list[1]], str):
            clean_data.append([data[name_list[0]], data[name_list[1]]])
            clean_labels.append(data["label"])
            if data["label"] not in label2data:
                label2data[data["label"]] = [data[name_list[0]], data[name_list[1]]]
        if max_sample is not None and len(clean_data) >= max_sample:
            break
    return label2data, clean_data, clean_labels

def get_INF2_data_for_ft(mode="train", name="RTE", train_sample_fraction=0.99, max_test_sample=None):
    if name == "RTE":
        INF2_dataset = datasets.load_from_disk(data_root + "/disk_datasets/super_glue/rte")
        label_dict = {0:"entailment", 1:"not_entailment"}
        name_list = ["premise", "hypothesis", "label"]
    # elif name == "SciTail":
    #     INF2_dataset = datasets.load_from_disk(data_root + "/disk_datasets/scitail")
    #     label_dict = 
    elif name == "QNLI":
        INF2_dataset = datasets.load_from_disk(data_root + "/disk_datasets/glue/qnli")
        label_dict = {0:"entailment", 1:"not_entailment"}
        name_list = ["question", "sentence", "label"]
    elif name == "WNLI":
        INF2_dataset = datasets.load_from_disk(data_root + "/disk_datasets/glue/wnli")
        label_dict = {0:"not_entailment", 1:"entailment"}
        name_list = ["sentence1", "sentence2", "label"]
    train_data = INF2_dataset["train"]
    _, train_data, train_labels = clean_INF2_data(train_data, label_dict, name_list, max_sample=max_test_sample)

    test_data = INF2_dataset["validation"]
    _, test_data, test_labels = clean_INF2_data(test_data, label_dict, name_list, max_sample=max_test_sample)

    # sample n points from training data
    if train_sample_fraction < 0.99 and mode == "train":
        train_data, train_labels = sample_data(train_data, train_sample_fraction)

    train_instructions = get_INF2_instruction_data(mode, train_data, train_labels)
    test_instructions = get_INF2_instruction_data(mode, test_data, test_labels)

    train_dataset = instructions_to_dataset(train_instructions, train_labels)
    test_dataset = instructions_to_dataset(test_instructions, test_labels)
    print("training dataset:", len(train_dataset))
    print("test dataset:", len(test_dataset))

    return train_dataset, test_dataset



TRAINING_QA_PROMPT_v2 = """Read Context and answer question ### Context:{context} ### Question:{quesion} ### Answer:{label}"""
INFERENCE_QA_v2 ="""Read Context and answer question ### Context:{context} ### Question:{quesion} ### Answer:"""

def get_QA_instruction_data(mode, texts, labels):
    if mode == "train":
        prompt = TRAINING_QA_PROMPT_v2 
    elif mode == "inference":
        prompt = INFERENCE_QA_v2
    instructions = []
    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(context=text[0], question=text[1], label=label)
        elif mode == "inference":
            example = prompt.format(context=text[0], question=text[1])
        instructions.append(example)
    return instructions

def clean_QA_data(dataset, name_list, max_sample=None):
    label2data = {}
    clean_data, clean_labels = [], []
    for data in dataset:
        if name_list[2] == "answers":
            data["label"] = data[name_list[2]]
        else:
            data["label"] = data[name_list[2]]["text"]
        # if isinstance(data["premise"], str) and isinstance(data["hypothesis"], str):
        if isinstance(data[name_list[0]], str) and isinstance(data[name_list[1]], str):
            clean_data.append([data[name_list[0]], data[name_list[1]]])
            clean_labels.append(data["label"])
            if data["label"] not in label2data:
                label2data[data["label"]] = [data[name_list[0]], data[name_list[1]]]
        if max_sample is not None and len(clean_data) >= max_sample:
            break
    return label2data, clean_data, clean_labels

def get_QA_data_for_ft(mode="train", name="newsqa", train_sample_fraction=0.99, max_test_sample=None):
    if name == "newsqa":
        QA_dataset = datasets.load_from_disk(data_root + "/disk_datasets/newsqa")
        name_list = ["context", "question", "answers"]
    if name == "squad_v2":
        QA_dataset = datasets.load_from_disk(data_root + "/disk_datasets/squad_v2")
        name_list = ["context", "question", "answers {text}"]
    if name == "hotpot_qa":
        QA_dataset = datasets.load_from_disk(data_root + "/disk_datasets/hotpot_qa")
        name_list = ["context", "question", "answer"]

    train_data = QA_dataset["train"]
    _, train_data, train_labels = clean_INF2_data(train_data, name_list, max_sample=max_test_sample)

    test_data = QA_dataset["validation"]
    _, test_data, test_labels = clean_INF2_data(test_data, name_list, max_sample=max_test_sample)

    # sample n points from training data
    if train_sample_fraction < 0.99 and mode == "train":
        train_data, train_labels = sample_data(train_data, train_sample_fraction)

    train_instructions = get_INF2_instruction_data(mode, train_data, train_labels)
    test_instructions = get_INF2_instruction_data(mode, test_data, test_labels)

    train_dataset = instructions_to_dataset(train_instructions, train_labels)
    test_dataset = instructions_to_dataset(test_instructions, test_labels)
    print("training dataset:", len(train_dataset))
    print("test dataset:", len(test_dataset))

    return train_dataset, test_dataset


TRAINING_race_PROMPT_v2 = """Read Context and answer question ### Context:{context} ### Question:{question} ### Options:A.{A}. B.{B}. C.{C}. D.{D}### Choice:{label}"""
INFERENCE_race_v2 ="""Read Context and answer question ### Context:{context} ### Question:{question} ### Options:A.{A}. B.{B}. C.{C}. D.{D}### Choice:"""

def get_race_instruction_data(mode, texts, labels):
    if mode == "train":
        prompt = TRAINING_race_PROMPT_v2 
    elif mode == "inference":
        prompt = INFERENCE_race_v2
    instructions = []
    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(context=text[0], question=text[1], A=text[2], B=text[3], C=text[4], D=text[5], label=label)
        elif mode == "inference":
            example = prompt.format(context=text[0], question=text[1], A=text[2], B=text[3], C=text[4], D=text[5])
        instructions.append(example)
    return instructions

def clean_race_data(dataset, max_sample=None):
    label2data = {}
    clean_data, clean_labels = [], []
    for data in dataset:
        choices = data["options"]
        if len(choices) < 4:
            choices = choices + [""] * (4 - len(choices))
        label = data["answer"]
        if label not in ["A", "B", "C", "D"]:
            continue
        clean_data.append([data["article"], data["question"], choices[0], choices[1], choices[2], choices[3]])
        clean_labels.append(label)
        if max_sample is not None and len(clean_data) >= max_sample:
            break
    return label2data, clean_data, clean_labels

def get_race_data_for_ft(mode="train", level="high", train_sample_fraction=0.99, max_test_sample=None):
    race_dataset = datasets.load_from_disk(data_root + "/disk_datasets/race_"+level)

    train_data = race_dataset["train"]
    _, train_data, train_labels = clean_race_data(train_data, max_sample=max_test_sample)

    test_data = race_dataset["validation"]
    _, test_data, test_labels = clean_race_data(test_data, max_sample=max_test_sample)

    # sample n points from training data
    if train_sample_fraction < 0.99 and mode == "train":
        train_data, train_labels = sample_data(train_data, train_sample_fraction)

    train_instructions = get_race_instruction_data(mode, train_data, train_labels)
    test_instructions = get_race_instruction_data(mode, test_data, test_labels)

    train_dataset = instructions_to_dataset(train_instructions, train_labels)
    test_dataset = instructions_to_dataset(test_instructions, test_labels)
    print("training dataset:", len(train_dataset))
    print("test dataset:", len(test_dataset))

    return train_dataset, test_dataset



TRAINING_swag_PROMPT_v2 = """Select the correct sentence that best completes the Passage below. ### Passage:{passage} ### Options:A.{A}. B.{B}. C.{C}. D.{D}### Choice:{label}"""
INFERENCE_swag_v2 = """Select the correct sentence that best completes the Passage below. ### Passage:{passage} ### Options:A.{A}. B.{B}. C.{C}. D.{D}### Choice:"""

def get_swag_instruction_data(mode, texts, labels):
    if mode == "train":
        prompt = TRAINING_swag_PROMPT_v2 
    elif mode == "inference":
        prompt = INFERENCE_swag_v2
    instructions = []
    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(passage=text[0], A=text[2], B=text[3], C=text[4], D=text[5], label=label)
        elif mode == "inference":
            example = prompt.format(passage=text[0], A=text[2], B=text[3], C=text[4], D=text[5])
        instructions.append(example)
    return instructions

int2label = {0:"A", 1:"B", 2:"C", 3:"D"}


def clean_swag_data(dataset, max_sample=None):
    label2data = {}
    clean_data, clean_labels = [], []
    for data in dataset:
        choices = [data["ending0"], data["ending1"], data["ending2"], data["ending3"]]
        label = int2label[data["label"]]
        clean_data.append([data["sent1"] + " " + data["sent2"], choices[0], choices[1], choices[2], choices[3]])
        clean_labels.append(label)
        if max_sample is not None and len(clean_data) >= max_sample:
            break
    return label2data, clean_data, clean_labels

def get_swag_data_for_ft(mode="train", train_sample_fraction=0.99, max_test_sample=None):
    swag_dataset = datasets.load_from_disk(data_root + "/disk_datasets/swag")

    train_data = swag_dataset["train"]
    _, train_data, train_labels = clean_swag_data(train_data, max_sample=max_test_sample)

    test_data = swag_dataset["validation"]
    _, test_data, test_labels = clean_swag_data(test_data, max_sample=max_test_sample)

    # sample n points from training data
    if train_sample_fraction < 0.99 and mode == "train":
        train_data, train_labels = sample_data(train_data, train_sample_fraction)

    train_instructions = get_swag_instruction_data(mode, train_data, train_labels)
    test_instructions = get_swag_instruction_data(mode, test_data, test_labels)

    train_dataset = instructions_to_dataset(train_instructions, train_labels)
    test_dataset = instructions_to_dataset(test_instructions, test_labels)
    print("training dataset:", len(train_dataset))
    print("test dataset:", len(test_dataset))

    return train_dataset, test_dataset

TRAINING_commonsense_qa_PROMPT_v2 = """According to the Question, select the best choice among the Options. ### Question:{question} ### Options:A.{A}. B.{B}. C.{C}. D.{D} E.{E} ### Choice:{label}"""
INFERENCE_commonsense_qa_v2 = """According to the Question, select the best choice among the Options. ### Question:{question} ### Options:A.{A}. B.{B}. C.{C}. D.{D} E.{E} ### Choice:"""

def get_commonsense_qa_instruction_data(mode, texts, labels):
    if mode == "train":
        prompt = TRAINING_commonsense_qa_PROMPT_v2 
    elif mode == "inference":
        prompt = INFERENCE_commonsense_qa_v2
    instructions = []
    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(question=text[0], A=text[1], B=text[2], C=text[3], D=text[4], E=text[5], label=label)
        elif mode == "inference":
            example = prompt.format(question=text[0], A=text[1], B=text[2], C=text[3], D=text[4], E=text[5])
        instructions.append(example)
    return instructions


def clean_commonsense_qa_data(dataset, max_sample=None):
    label2data = {}
    clean_data, clean_labels = [], []
    for data in dataset:
        choices = data["choices"]["text"]
        if len(choices) < 5:
            choices = choices + [""] * (5 - len(choices))
        question = data["question"]
        label = data["answerKey"]
        clean_data.append([question] + choices)
        clean_labels.append(label)
        if max_sample is not None and len(clean_data) >= max_sample:
            break
    return label2data, clean_data, clean_labels

def get_commonsense_qa_data_for_ft(mode="train", train_sample_fraction=0.99, max_test_sample=None):
    commonsense_qa_dataset = datasets.load_from_disk(data_root + "/disk_datasets/commonsense_qa")

    train_data = commonsense_qa_dataset["train"]
    _, train_data, train_labels = clean_commonsense_qa_data(train_data, max_sample=max_test_sample)

    test_data = commonsense_qa_dataset["validation"]
    _, test_data, test_labels = clean_commonsense_qa_data(test_data, max_sample=max_test_sample)

    # sample n points from training data
    if train_sample_fraction < 0.99 and mode == "train":
        train_data, train_labels = sample_data(train_data, train_sample_fraction)

    train_instructions = get_commonsense_qa_instruction_data(mode, train_data, train_labels)
    test_instructions = get_commonsense_qa_instruction_data(mode, test_data, test_labels)

    train_dataset = instructions_to_dataset(train_instructions, train_labels)
    test_dataset = instructions_to_dataset(test_instructions, test_labels)
    print("training dataset:", len(train_dataset))
    print("test dataset:", len(test_dataset))

    return train_dataset, test_dataset

