import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from datasets import concatenate_datasets

from prompts import (get_newsgroup_data_for_ft, get_boolq_data_for_ft, get_COPA_data_for_ft, get_inference_data_for_ft, get_INF2_data_for_ft,
                     get_commonsense_qa_data_for_ft, get_QA_data_for_ft, get_swag_data_for_ft, get_race_data_for_ft
                     )

inference_list = ["CB", "MNLI"]
inference_list_binary = ["RTE", "QNLI", "WNLI"]

use_flash_attention = True

def main(args):
    train_datasets = []
    test_datasets = []
    if "20_Newsgroups_Fixed" in args.tasks:
        train_datasets.append(get_newsgroup_data_for_ft(
            mode="train", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "boolq" in args.tasks:
        train_datasets.append(get_boolq_data_for_ft(
            mode="train", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "CB" in args.tasks:
        train_datasets.append(get_inference_data_for_ft(
            mode="train", name="CB", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "MNLI" in args.tasks:
        train_datasets.append(get_inference_data_for_ft(
            mode="train", name="MNLI", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "RTE" in args.tasks:
        train_datasets.append(get_INF2_data_for_ft(
            mode="train", name="RTE", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "QNLI" in args.tasks:
        train_datasets.append(get_INF2_data_for_ft(
            mode="train", name="QNLI", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "WNLI" in args.tasks:
        train_datasets.append(get_INF2_data_for_ft(
            mode="train", name="WNLI", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "COPA" in args.tasks:
        train_datasets.append(get_COPA_data_for_ft(
            mode="train", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "commonsense_qa" in args.tasks:
        train_datasets.append(get_commonsense_qa_data_for_ft(
            mode="train", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "newsqa" in args.tasks:
        train_datasets.append(get_QA_data_for_ft(
            mode="train", name="newsqa", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "hotpot_qa" in args.tasks:
        train_datasets.append(get_QA_data_for_ft(
            mode="train", name="hotpot_qa", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "squad_v2" in args.tasks:
        train_datasets.append(get_QA_data_for_ft(
            mode="train", name="squad_v2", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])
    if "race_middle" in args.tasks:
        train_datasets.append(get_race_data_for_ft(mode="train", level="middle",  train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples)[0])
    if "race_high" in args.tasks:
        train_datasets.append(get_race_data_for_ft(mode="train", level="high",  train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples)[0])
    if "swag" in args.tasks:
        train_datasets.append(get_swag_data_for_ft(
            mode="train", train_sample_fraction=args.train_sample_fraction, max_test_sample=args.max_samples
        )[0])

    
    train_dataset = concatenate_datasets(train_datasets)
    print(f"Sample fraction:{args.train_sample_fraction}")
    print(f"Training samples:{train_dataset.shape}")
    print(f"Train examples:{train_dataset[0]}")


    # BitsAndBytesConfig int-4 config
    if args.bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    if args.target_modules:
        peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=args.dropout,
        target_modules=args.target_modules,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        )
    else:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=args.dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # prepare model for training
    if args.bnb:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    trained_param, total_param = model.get_nb_trainable_parameters()


    results_dir = f"experiments/{args.tasks}/classification-sampleFraction-{args.train_sample_fraction}_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}"
    if args.target_modules:
        results_dir = f"{results_dir}-{args.target_modules}"
    if args.bnb:
        results_dir += '-4bit'
    else:
        results_dir += '-standard'
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    with open(f"{results_dir}/{trained_param}_total_{total_param}.txt", "w") as f:
        f.write(str(trained_param))
        f.write('\n')
        f.write(str(total_param))
        f.write('\n')
        f.write(str(trained_param/total_param))
        f.write('\n')
    
    
    # 保证 shuffle training data
    training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32 if args.bnb else 4,
        gradient_accumulation_steps=2 if args.bnb else 16,
        gradient_checkpointing=False,
        optim="paged_adamw_32bit" if args.bnb else "adamw_torch",
        logging_steps=100,
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        save_steps = 2000,
        # disable_tqdm=True # disable tqdm since with packing values are in correct
    )

    max_seq_length = 512  # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=training_args,
        dataset_text_field="instructions",
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [
            args.epochs,
            args.lora_r,
            args.dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/.cache/Llama-2-7b-hf") # /home/hucx/huchenxu/qzh/model/open-llama/7B "NousResearch/Llama-2-7b-hf"
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--bnb", action='store_true')
    parser.add_argument("--train_sample_fraction", default=0.99, type=float)
    parser.add_argument("--max_samples", default=5000, type=int)
    parser.add_argument('--target_modules', default=[], nargs='*', type=str)
    parser.add_argument("--tasks",  nargs='*', default=["20_Newsgroups_Fixed"])

    args = parser.parse_args()
    main(args)

