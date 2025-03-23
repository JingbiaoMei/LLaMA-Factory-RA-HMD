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
from infer_utils import QWen2Classifier, predict_and_eval_save_feature
from dataset import get_Dataloader
def read_json(file_path):
    """Read a JSONL file."""
    return pd.read_json(file_path, lines=True, dtype=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Path to the model')
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='Path to the model base')
    parser.add_argument('--processor_path', type=str, default=None,
                        help='Path to the processor')   
    # Note that if you use lora weights, put the lora weights to path, base model to base_model_path
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
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Set the batch size')
    parser.add_argument('--log_name', type=str, default="",
                        help='Set the log name') 
    parser.add_argument(
        "--EXP_FOLDER",
        type=str,
        default="./data/Embedding",
        help="The path to save results.",
    )
    parser.add_argument('--load_4bit', action='store_true',
                        default=False, help='Set to load 4-bit')
    parser.add_argument('--load_8bit', action='store_true',
                        default=False, help='Set to load 8-bit')

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
    
    model_name = args.model_path.split("/")[-2] + "_" + args.model_path.split("/")[-1] 

    args.log_name = model_name + "_"  + args.log_name if args.log_name != "" else model_name
    
    
    # For debugging purposes
    if args.topk != -1:
        args.log_name += "debug"
        import debugpy
        print("Waiting for debugger attach")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    
    save_name = "{}_{}".format(
        args.dataset,
        args.log_name,
    )
        
        
    VLMClassifier = QWen2Classifier(
        model_path=args.model_path,
        prompt=args.query,
        processor_path=args.processor_path,
        base_model_path=args.base_model_path,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        max_pixels=448*448,)


    # Define valid splits and dataset groups
    
    if args.dataset == "FB":
        splits = {"train","dev_seen", "test_seen", "test_unseen"}
    else:
        splits = {"train", "val", "test"}

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
            train_batch_size=args.batch_size,
            num_workers=24,
            dataset=args.dataset,
        )
        loader_list = [train, dev_seen, test_seen]
        splits = ["train","val", "test"]
    # Get CLIP features and ground truth labels
    # Evaluate

    for split in splits:
        if split in args.data_split:
            file_name = f"{split}.jsonl"

            last_hidden_states, embeds, ids, labels = predict_and_eval_save_feature(VLMClassifier, 
                                                dataset=args.dataset,
                                                split_dl=loader_list[splits.index(split)], 
                                                split=split,
                                                topk=args.topk)
    
            # Create the folder if it does not exist
            if not os.path.exists("{}/{}".format(args.EXP_FOLDER,args.dataset)):
                os.makedirs("{}/{}".format(args.EXP_FOLDER,args.dataset))
            
            torch.save(
                {
                    "ids": ids,
                    "feats": last_hidden_states,
                    "labels": labels,
                },
                "{}/{}/{}_{}.pt".format(
                    args.EXP_FOLDER,args.dataset, split, save_name
                ),
            )   
            torch.save(
                {
                    "ids": ids,
                    "feats": embeds,
                    "labels": labels,
                },
                "{}/{}/{}_{}_embeds.pt".format(
                    args.EXP_FOLDER,args.dataset, split, save_name
                ),
            )