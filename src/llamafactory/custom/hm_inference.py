import sys
sys.path.append('./src')

from easydict import EasyDict
from pprint import pprint
import json
from tqdm import tqdm
import wandb 
import argparse
from datetime import datetime
import os
import pandas as pd
from pprint import pprint
import ast
from collections import defaultdict
import numpy as np
import csv
import torch 
from infer_utils import QWen2Classifier, predict_and_eval
from dataset import get_Dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Path to the model')
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='Path to the model base')
    parser.add_argument('--processor_path', type=str, default=None,
                        help='Path to the processor')   
    parser.add_argument('--max_pixels', type=int, default=None,
                        help='Set the max resolution')
    # Note that if you use lora weights, put the lora weights to path, base model to base_model_path
    parser.add_argument('--load_4bit', action='store_true',
                        default=False, help='Set to load 4-bit')
    parser.add_argument('--load_8bit', action='store_true',
                        default=False, help='Set to load 8-bit')
    parser.add_argument("--query", type=str, default="Is it hateful?")   
    parser.add_argument('--data_split', type=str,
                            default='dev_seen test_seen', help='Evaluate on which dataset split')
    parser.add_argument('--gt_folder', type=str, default='./data/gt/FB/',
                        help='Path to the ground truth folder')
    parser.add_argument('--image_folder', type=str,
                        default='./data/image/FB/All', help='Path to the image folder')
    parser.add_argument('--topk', type=int, default=-
                        1, help='Set the topk value')
    parser.add_argument('--dataset', type=str, default="FB",
                        help='Set the dataset name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Set the batch size')
    parser.add_argument('--log_name', type=str, default="",
                        help='Set the log name') 
    parser.add_argument('--group_name', type=str, default="inference_classifier",
                        help='Set the log name') 
    # To solve the issue for not having the merged checkpoint when running on the fly merge.
    parser.add_argument('--name_force', type=str, default=None,
                        help='Set the forced logging model name')
    parser.add_argument('--eval_generative', action='store_true',
                        default=False, help='Set to evaluate generative')
    parser.add_argument('--eval_classifier', action='store_true',
                        default=False, help='Set to evaluate classifier')
    parser.add_argument('--few_shot', type=int, default=None,
                        help='Set the few shot value')
    

    args = parser.parse_args()

    # Initialize wandb
    # wandb.init(project="a-ravqa", name=args.exp_name, entity="byrne-lab")
 
    if args.dataset == "FB":
        args.gt_folder = './data/gt/FB/'
        args.image_folder = './data/image/FB/All/'
    elif args.dataset.lower() == "harmeme" or args.dataset.lower() == "harmc":
        args.gt_folder = './data/gt/HarMeme/'
        args.dataset = "HarMeme"
        args.image_folder = './data/image/HarMeme/All/'
    elif args.dataset.lower() == "harmp":
        args.gt_folder = './data/gt/HarmP/'
        args.image_folder = './data/image/HarmP/All/'
        args.dataset = "HarmP"
    elif args.dataset.lower() == "multioff":
        args.gt_folder = './data/gt/MultiOFF/'
        args.image_folder = './data/image/MultiOFF/All/'
        args.dataset = "MultiOFF"
    elif args.dataset.lower() == "pridemm":
        args.gt_folder = './data/gt/PrideMM'
        args.image_folder = './data/image/PrideMM/All/'
    elif args.dataset.lower() == "mami":
        args.gt_folder = './data/gt/MAMI'
        args.image_folder = './data/image/MAMI/All/'
    
    # print if loading in 4bit or 8bit or FP16
    
    if args.load_4bit:
        print("Loading in 4bit")
        quantization = "4bit_"
    elif args.load_8bit:
        print("Loading in 8bit")
        quantization = "8bit_"
    else:
        print("Loading in bf16")
        quantization = ""    
    if args.name_force is None:
        model_name = args.model_path.split("/")[-1]
    else:
        model_name = args.name_force
    args.log_name = model_name + "_" + quantization + args.log_name
    
    
    if "step-" in model_name:
        log_name_opt = args.log_name
        log_name_opt.split("step-")[0]
        log_path = "./logging/"+ args.dataset+"/" + log_name_opt + "/" + quantization + "/" 

    else:
        log_path = "./logging/"+ args.dataset+"/" + model_name + "/" 
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path += args.log_name + ".csv"
    
    # we use the group name from args
    group_name = args.group_name if args.group_name is not None else model_name
    group_name = "Debugging" if args.topk != -1 else group_name
    tags = [args.dataset, args.log_name, model_name]
    # For debugging purposes
    if args.topk != -1:
        tags.append("debug")
        args.log_name += "debug"
        import debugpy
        print("Waiting for debugger attach")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()

    
    exp_name = "{}_{}".format(
        args.log_name,
        args.dataset
    )
    run = wandb.init(
        name=exp_name,
        config={
            "model": model_name,
            "dataset": args.dataset,
            "quantization": quantization,
        },
        group=group_name,
        tags=tags
    )
    print(args)

    artifact = None
        
    VLMClassifier = QWen2Classifier(
        model_path=args.model_path,
        prompt=args.query,
        processor_path=args.processor_path,
        base_model_path=args.base_model_path,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        max_pixels=args.max_pixels,)
    def read_json(file_path):
        """Read a JSONL file."""
        return pd.read_json(file_path, lines=True, dtype=False)

    def write_to_csv(log_path, metrics):
        """Write evaluation results to a CSV file."""
        with open(log_path, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(["split", "acc", "auc", "precision", "recall", "f1"])
            for split, values in metrics.items():
                values = [round(value, 4) for value in values]
                writer.writerow([split] + values)

    # Define valid splits and dataset groups
    
    if args.dataset == "FB":
        
        splits = {"dev_seen", "test_seen", "test_unseen"}
    else:
        splits = {"val", "test"}

    # Initialize metrics dictionary
    metrics_generative = {}
    metrics_classifier = {} 

        
    # Initialise dataset
    if args.dataset == "FB":
        train, dev_seen, test_seen, test_unseen = get_Dataloader(
            preprocess=None,
            batch_size=args.batch_size,
            train_batch_size=args.batch_size,
            num_workers=24,
            dataset=args.dataset,
        )
        loader_list = [train, dev_seen, test_seen, test_unseen]
        splits = ["train", "dev_seen", "test_seen", "test_unseen"]
    else:
        train, dev_seen, test_seen = get_Dataloader(
            preprocess=None,
            batch_size=args.batch_size,
            num_workers=24,
            dataset=args.dataset,
        )
        loader_list = [train, dev_seen, test_seen]
        splits = ["train", "val", "test"]
    # Get CLIP features and ground truth labels
    # Evaluate

    for split in splits:
        if split in args.data_split:
                file_name = f"{split}.jsonl"

                metrics_generative["Generative"+split],metrics_classifier["Classifier"+split] = predict_and_eval(VLMClassifier, 
                                                dataset=args.dataset,
                                                split_dl=loader_list[splits.index(split)], 
                                                split=split,
                                                topk=args.topk,
                                                artifact=artifact,
                                                train_dl=train,
                                                num_few_shots=args.few_shot)

    
    # Save results to a CSV file
    write_to_csv(log_path, metrics_generative)
    write_to_csv(log_path, metrics_classifier)     
            
    # split_df = split_df.set_index("id")
    run.finish()
