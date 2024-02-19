import argparse
import torch
import os
import pandas as pd
import evaluate
import pickle
import warnings
from tqdm import tqdm
import json

# from llama_patch import unplace_flash_attn_with_attn
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from prompts import (get_newsgroup_data_for_ft, get_boolq_data_for_ft, get_COPA_data_for_ft, get_inference_data_for_ft, get_INF2_data_for_ft,
                     get_commonsense_qa_data_for_ft, get_QA_data_for_ft, get_swag_data_for_ft, get_race_data_for_ft
                     )
metric = evaluate.load("/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/LLM-Finetuning-Hub/.cache/rouge")
warnings.filterwarnings("ignore")


def main(args):
    experiment = args.experiment_dir
    peft_model_id = f"{experiment}/assets"
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True if args.bnb else False,
    )
    if not args.bnb:
        model.cuda()
    print("finish loading from ", peft_model_id)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    for task in args.tasks:
        if task == "20_Newsgroups_Fixed":
            test_dataset = get_newsgroup_data_for_ft(mode="inference", max_test_sample=1000)[1]
        elif task == "boolq":
            test_dataset = get_boolq_data_for_ft(mode="inference", max_test_sample=1000)[1]
        elif task == "CB":
            test_dataset = get_inference_data_for_ft(mode="inference", name="CB", max_test_sample=1000)[1]
        elif task == "MNLI":
            test_dataset = get_inference_data_for_ft(mode="inference", name="MNLI", max_test_sample=1000)[1]
        elif task == "RTE":
            test_dataset = get_INF2_data_for_ft(mode="inference", name="RTE", max_test_sample=1000)[1]
        elif task == "QNLI":
            test_dataset = get_INF2_data_for_ft(mode="inference", name="QNLI", max_test_sample=1000)[1]
        elif task == "WNLI":
            test_dataset = get_INF2_data_for_ft(mode="inference", name="WNLI", max_test_sample=1000)[1]
        elif task == "COPA":
            test_dataset = get_COPA_data_for_ft(mode="inference", max_test_sample=1000)[1]
        elif task == "commonsense_qa":
            test_dataset = get_commonsense_qa_data_for_ft(mode="inference", max_test_sample=1000)[1]
        elif task == "newsqa":
            test_dataset = get_QA_data_for_ft(mode="inference", name="newsqa", max_test_sample=1000)[1]
        elif task == "hotpot_qa":
            test_dataset = get_QA_data_for_ft(mode="inference", name="hotpot_qa", max_test_sample=1000)[1]
        elif task == "squad_v2":
            test_dataset = get_QA_data_for_ft(mode="inference", name="squad_v2", max_test_sample=1000)[1]
        elif task == "race_middle":
            test_dataset = get_race_data_for_ft(mode="inference", level="middle", max_test_sample=1000)[1]
        elif task == "race_high":
            test_dataset = get_race_data_for_ft(mode="inference", level="high", max_test_sample=1000)[1]
        elif task == "swag":
            test_dataset = get_swag_data_for_ft(mode="inference", max_test_sample=1000)[1]

        instructions, labels = test_dataset["instructions"], test_dataset["labels"]
        print(f"task {task} total cases", len(instructions))
        
        results = []
        oom_examples = []

        for instruct, label in tqdm(zip(instructions, labels)):
            input_ids = tokenizer(
                instruct, return_tensors="pt", truncation=True
            ).input_ids.cuda()

            with torch.inference_mode():
                try:
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=20,
                        do_sample=True,
                        top_p=0.95,
                        temperature=1e-3,
                    )
                    result = tokenizer.batch_decode(
                        outputs.detach().cpu().numpy(), skip_special_tokens=True
                    )[0]
                    result = result[len(instruct) :]
                except:
                    result = ""
                    oom_examples.append(input_ids.shape[-1])

                results.append(result)

        metrics = {
            "micro_f1": f1_score(labels, results, average="micro"),
            "macro_f1": f1_score(labels, results, average="macro"),
            "precision": precision_score(labels, results, average="micro"),
            "recall": recall_score(labels, results, average="micro"),
            "accuracy": accuracy_score(labels, results),
            "oom_examples": oom_examples,
        }
        print(metrics)

        save_dir = os.path.join(experiment, "metrics")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, f"{task}_metrics_update.json"), "w") as handle:
            json.dump(metrics, handle)

    print(f"Completed experiment {peft_model_id}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        default="experiments/classification-sampleFraction-0.1_epochs-5_rank-8_dropout-0.1",
    )
    parser.add_argument("--tasks", nargs='*', default=["20_Newsgroups_Fixed"])
    parser.add_argument("--bnb", action='store_true')


    args = parser.parse_args()
    main(args)
