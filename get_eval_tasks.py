import os
import json
root_dir = ""

base_command = 'python llama2_classification_multi_tasks_inference.py'

def list_str_to_list(s):
    split_s = s.split(",")
    split_s = [i.replace("[", "").replace("]", "") for i in split_s]
    return split_s

def clean_exp_name(s):
    return s.replace("classification-sampleFraction-", "f").replace("epochs-", "e").replace("rank-", "r").replace("dropout-", "d")

eval_results = {}

for file in os.listdir(root_dir):
    tasks = list_str_to_list(file)
    task_path = os.path.join(root_dir, file)
    temp_results_dict = {}
    if os.path.isdir(task_path):
        for subfile in os.listdir(task_path):
            clean_exp = clean_exp_name(subfile)
            exp_path = os.path.join(task_path, subfile)
            if "metrics" in os.listdir(exp_path):
                metrics_path = os.path.join(exp_path, "metrics")
                metrics_files = os.listdir(metrics_path)
                if len(metrics_files) == 1:
                    for metrics_file in metrics_files:
                        if metrics_file.endswith(".json"):
                            temp_results = json.load(open(os.path.join(metrics_path, metrics_file), "r"))
                            temp_results_dict[clean_exp] = temp_results
                elif len(metrics_files) > 1:
                    temp_temp_results_dict = {}
                    for metrics_file in metrics_files:
                        if metrics_file.endswith(".json"):
                            task_name = metrics_file.split("_")[0]
                            temp_results = json.load(open(os.path.join(metrics_path, metrics_file), "r"))
                            temp_temp_results_dict[task_name] = temp_results
                    temp_results_dict[clean_exp] = temp_temp_results_dict
    eval_results[file] = temp_results_dict

with open("all_eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=4)
    
