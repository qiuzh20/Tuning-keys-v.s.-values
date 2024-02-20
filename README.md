# Empirical Study on Updating Key-Value Memories in Transformer Feed-forward Layers

## Repo Overview

How do pre-trained Transformer models process and store information and how can we update the information processing and storing? In this repository, we provide an empirical study on updating key-value memories in Transformer feed-forward layers. We contend that **compared to directly modifying the model's knowledge (values), altering the mechanism of controlling this knowledge (keys) can be more effective.** For more details, please refer to our [paper](https://arxiv.org/abs/2402.12233).

## Llama Experiments

### Muti-task Training

Take 6 tasks as an example, we can train the model with the following command:

```sh
# standard lora
python llama2_classification_multi_tasks.py --lora_r 8 --epochs 5 --dropout 0.1 --bnb --target_modules q_proj v_proj --tasks 20_Newsgroups_Fixed boolq COPA CB race_high QNLI

# update keys (up_proj)
python llama2_classification_multi_tasks.py --lora_r 8 --epochs 5 --dropout 0.1 --bnb --target_modules q_proj v_proj up_proj --tasks 20_Newsgroups_Fixed boolq COPA CB race_high QNLI

# update keys (up_proj)
python llama2_classification_multi_tasks.py --lora_r 8 --epochs 5 --dropout 0.1 --bnb --target_modules q_proj v_proj gate_proj --tasks 20_Newsgroups_Fixed boolq COPA CB race_high QNLI

# update values (down_proj)
python llama2_classification_multi_tasks.py --lora_r 8 --epochs 5 --dropout 0.1 --bnb --target_modules q_proj v_proj down_proj --tasks 20_Newsgroups_Fixed boolq COPA CB race_high QNLI
```

After training, the logs and checkpoints will be saved in the experitment folder named like `experiments/${tasks}/classification-sampleFraction-${train_sample_fraction}_epochs-${epochs}_rank-${lora_r}_dropout-${dropout}`.

### Evaluation

After training, go to the corresponding experiment folder and run

```sh
python llama2_classification_multi_tasks_inference.py --tasks 20_Newsgroups_Fixed boolq COPA CB race_high QNLI --experiment_dir ${experiment_dir}
```

where `${experiment_dir}` is the directory of the experiment as mentioned above.

## Environments

Please prepare the environments following the instructions of [LLM-Finetuning-Hub](https://github.com/georgian-io/LLM-Finetuning-Hub).

## Acknowledgement

The main code for Llama experiments is modified from this [codebase](https://github.com/georgian-io/LLM-Finetuning-Hub/tree/main/llama2) in [LLM-Finetuning-Hub](https://github.com/georgian-io/LLM-Finetuning-Hub).

## License

This source code is released under the MIT license, included [here](LICENSE).
