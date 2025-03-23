from abc import ABC, abstractmethod
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
)
from datasets import load_dataset
from PIL import Image
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import os
import json
import wandb 
import torchmetrics
from tqdm import tqdm
from transformers import BitsAndBytesConfig
class All_Reranker(ABC):
    @abstractmethod
    def rank(self, question, query_img, retrieved_docs):
        pass
    
class VLMReranker(All_Reranker):
    def __init__(
        self,
        model_path,
        prompt_template_file,
        *args,
        is_lora=False,
        base_model_path=None,
        processor_path=None,
        **kwargs
    ):
        self.model_path = model_path
        self.is_lora = is_lora
        self.base_model_path = base_model_path
        self.processor_path = processor_path
        with open(prompt_template_file, 'r') as f:
            self.prompt_template = f.read()

        self.model = None

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
class QWen2Reranker(VLMReranker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _init_model(self):
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        if not self.is_lora:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            ).eval()
            self.processor = AutoProcessor.from_pretrained(self.processor_path or self.model_path)
        else:
            assert self.base_model_path is not None
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            ).eval()
            self.model.load_adapter(self.model_path)
            self.processor = AutoProcessor.from_pretrained(self.base_model_path)

    @torch.no_grad()
    def get_next_token_ps(self, query_text, query_img):
        if self.model is None:
            self._init_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": query_img},
                    {"type": "text", "text": query_text},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        y_nexts = ["yes", "no"]

        y_nexts_idx = self.processor(
            text=y_nexts,
        )['input_ids']

        y_next_idx_map = {yn: y_idx[0] for yn, y_idx in zip(y_nexts, y_nexts_idx)}

        # Inference
        self.model.eval()
        outputs = self.model(**inputs)
        last_logits = outputs.logits[0][-1]
        m = torch.nn.Softmax(dim=0)

        last_probs = m(last_logits)
        output_probs = {
            yn: last_probs[idx].item() for yn, idx in y_next_idx_map.items()
        }
        return output_probs
    
    def rank(self, question, query_img, retrieved_docs):
        all_scores = []
        for this_doc in retrieved_docs:
            query_text = self.prompt_template\
                            .replace('<<EVIDENCE>>', this_doc['text'])\
                            .replace('<<QUESTION>>', question)
            next_token_ps = self.get_next_token_ps(query_text, query_img)
            score = next_token_ps['yes']
            this_doc['rerank_score'] = score
            all_scores.append(score)
        
        reranked_documents = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
        return reranked_documents
        


class QWen2Classifier():
    def __init__(
        self,
        model_path,
        prompt,
        *args,
        base_model_path=None,
        processor_path=None,
        load_4bit=False,
        load_8bit=False,
        max_pixels=None,
        **kwargs
    ):
        self.model_path = model_path
        
        self.base_model_path = base_model_path
        self.processor_path = processor_path
        self.classifier = None
        self.model = None
        self.load_4bit = load_4bit
        self.load_8bit = load_8bit
        
        if self.base_model_path is not None:
            self.is_lora = True
        else:
            self.is_lora = False
        self.prompt = prompt
        self.max_pixels = max_pixels
    
    def _init_model(self):
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        torch.set_default_dtype(torch.bfloat16)
        if self.load_4bit:
            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype="bfloat16",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_quant_storage="bfloat16",  # crucial for fsdp+qlora
                                )
            print("4-bit quantization enabled")
        elif self.load_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("8-bit quantization enabled")
        else:
            quantization_config = None
        
        if not self.is_lora:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                quantization_config=quantization_config
            ).eval()
            if self.max_pixels is None:
                self.processor = AutoProcessor.from_pretrained(self.processor_path or self.model_path)
            else:
                self.processor = AutoProcessor.from_pretrained(self.processor_path or self.model_path, max_pixels=self.max_pixels)
            
        else:
            assert self.base_model_path is not None
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                quantization_config=quantization_config
            ).eval()
            self.model.load_adapter(self.model_path)
            if self.max_pixels is None:
                self.processor = AutoProcessor.from_pretrained(self.base_model_path)
            else:
                self.processor = AutoProcessor.from_pretrained(self.base_model_path, max_pixels=self.max_pixels)
                    
        # Check if classifier is present in the model_path folder
        # Need to check for `classifier_config.json` and `classifier.bin` files
        
        classifier_config_path = os.path.join(self.model_path, 'classifier_config.json')
        classifier_bin_path = os.path.join(self.model_path, 'classifier.bin')
        if os.path.exists(classifier_config_path) and os.path.exists(classifier_bin_path):
            self.classifier = self.init_classifier(classifier_config_path)
            self.model.add_module("classifier", self.classifier.to("cuda").eval())
            self.model.classifier.load_state_dict(torch.load(classifier_bin_path))
            print("Classifier loaded")
            print(self.model.classifier)
        else: 
            print("No classifier found")
            self.model.add_module("classifier", None)

    def init_classifier(self, config_path):
        with open(config_path, "r") as f:
            classifier_config = json.load(f)
        classifier = Classifier(
            input_shape=classifier_config["hidden_size"],
            num_layers=classifier_config["num_layers"],
            proj_dim=classifier_config["proj_dim"],
            output_dim=classifier_config["output_dim"],
            input_dropout=classifier_config["input_dropout"],
            dropout=classifier_config["dropout"]
        )
        # Change to bf16
        classifier = classifier.to("cuda").to(torch.bfloat16)
        return classifier
        
    
    @torch.inference_mode()
    def predict(self, query_texts, query_imgs):
        if self.model is None:
            self._init_model()
            # Assert the processor is left padded
            assert self.processor.tokenizer.padding_side == "left" 
            self.yes_token_id = self.processor.tokenizer.convert_tokens_to_ids("Yes")
            self.no_token_id = self.processor.tokenizer.convert_tokens_to_ids("No")
        
        messages_batch = []
            
        for query_text, query_img in zip(query_texts, query_imgs):
            query_text = "This is an image with: \"{}\" written on it. {}".format(
            query_text, self.prompt)
            messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": query_img},
                    {"type": "text", "text": query_text},
                ],
            }
            ]
            messages_batch.append(messages)
        
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) for messages in messages_batch
        ]
        #print(texts)
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        
        # Inference
        self.model.eval()
        outputs = self.model(**inputs, output_hidden_states=True)
       
        
        # Generative classifier 
        last_logits = outputs.logits[:, -1]
        m = torch.nn.Softmax()


        
        #y_next_idx_map = {yn: y_idx[0] for yn, y_idx in zip(y_nexts, y_nexts_idx)}
        
        #last_probs = m(last_logits)
        #output_probs_gen = {
        #    yn: last_probs[idx].item() for yn, idx in y_next_idx_map.items()
        #}
        output_logits_yes_generative = last_logits[:,self.yes_token_id]
        output_logits_no_generative = last_logits[:,self.no_token_id]
        yes_no_logits = torch.stack([output_logits_yes_generative, output_logits_no_generative], dim=1)
        output_probs_gen = m(yes_no_logits)[:, 0]
        # Logistic Regression Classifier
        if self.classifier is not None:
            embed = outputs.hidden_states[-1][:, -1, :]
            output_logits = self.model.classifier(embed)
            #output_probs = torch.sigmoid(output_probs)
            return output_probs_gen, output_logits
        
        
        return output_probs_gen, None
    
    
    @torch.inference_mode()
    def predict_few_shot(self, query_texts, query_imgs, few_shot_labels, few_shot_imgs):
        if self.model is None:
            self._init_model()
            # Assert the processor is left padded
            assert self.processor.tokenizer.padding_side == "left"
            self.yes_token_id = self.processor.tokenizer.convert_tokens_to_ids("Yes")
            self.no_token_id = self.processor.tokenizer.convert_tokens_to_ids("No")
        
        messages_batch = []
        
        # For each batch, same few-shot labels and images used
        # Construct few-shot prompts
        few_shot_prompts = []
        for label, img in zip(few_shot_labels, few_shot_imgs):
            label_text = "This is not a hateful meme." if label == 0 else "This is a hateful meme."
            few_shot_prompt = {
                "type": "image",
                "image": img,
            }, {
                "type": "text",
                "text": label_text,
            }
            few_shot_prompts.append({
                "type": "image",
                "image": img,
            })
            few_shot_prompts.append({
                "type": "text",
                "text": label_text,
            })
        
        for query_text, query_img in zip(query_texts, query_imgs):
            

            # Prepare the query prompt
            query_text = "This is an image with: \"{}\" written on it. {}".format(
                query_text, self.prompt
            )
            query_prompt = [
                {"type": "image", "image": query_img},
                {"type": "text", "text": query_text},
            ]

            # Combine few-shot prompts with the query prompt
            messages = [{"role": "user", "content": few_shot_prompts + query_prompt}]
            messages_batch.append(messages)

        # Prepare text inputs with the processor
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) for messages in messages_batch
        ]
        # Process image and video inputs
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        self.model.eval()
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Generative classifier
        last_logits = outputs.logits[:, -1]
        m = torch.nn.Softmax(dim=-1)

        output_logits_yes_generative = last_logits[:, self.yes_token_id]
        output_logits_no_generative = last_logits[:, self.no_token_id]
        yes_no_logits = torch.stack([output_logits_yes_generative, output_logits_no_generative], dim=1)
        output_probs_gen = m(yes_no_logits)[:, 0]

        # Logistic Regression Classifier
        if self.classifier is not None:
            embed = outputs.hidden_states[-1][:, -1, :]
            output_logits = self.model.classifier(embed)
            return output_probs_gen, output_logits

        return output_probs_gen, None

    
    @torch.inference_mode()
    def predict_return_embeds(self, query_texts, query_imgs):
        if self.model is None:
            self._init_model()
            # Assert the processor is left padded
            assert self.processor.tokenizer.padding_side == "left" 
            self.yes_token_id = self.processor.tokenizer.convert_tokens_to_ids("Yes")
            self.no_token_id = self.processor.tokenizer.convert_tokens_to_ids("No")
        
        messages_batch = []
            
        for query_text, query_img in zip(query_texts, query_imgs):
            query_text = "This is an image with: \"{}\" written on it. {}".format(
            query_text, self.prompt)
            messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": query_img},
                    {"type": "text", "text": query_text},
                ],
            }
            ]
            messages_batch.append(messages)
        
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) for messages in messages_batch
        ]
        #print(texts)
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        
        # Inference
        self.model.eval()
        outputs = self.model(**inputs, output_hidden_states=True)
       
        
        # Generative classifier 
        #ast_logits = outputs.logits[:, -1]
        #m = torch.nn.Softmax()


        #output_logits_yes_generative = last_logits[:,self.yes_token_id]
        #utput_logits_no_generative = last_logits[:,self.no_token_id]
        #yes_no_logits = torch.stack([output_logits_yes_generative, output_logits_no_generative], dim=1)
        #output_probs_gen = m(yes_no_logits)[:, 0]
        # Logistic Regression Classifier
        if self.classifier is not None:
            hidden_state = outputs.hidden_states[-1][:, -1, :]
            output_logits, embed  = self.model.classifier(hidden_state, return_embed=True)
            #output_probs = torch.sigmoid(output_probs)
            return hidden_state, output_logits, embed
        else:
            hidden_state = outputs.hidden_states[-1][:, -1, :]
            return hidden_state, hidden_state, hidden_state
    



class Classifier(nn.Module):
    def __init__(self, input_shape, num_layers, proj_dim, output_dim=1, input_dropout=0., dropout=None):
        super(Classifier, self).__init__()
        layers = []

        
        
        # Handle dropout initialization
        if dropout is None:
            dropout = [0.0] * num_layers

        # Input dropout layer
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        
        
        
        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_shape, proj_dim))
            #nn.init.xavier_uniform_(layers[-1].weight)  # Xavier initialization
            layers.append(nn.ReLU())
            if dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            input_shape = proj_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(proj_dim, output_dim)
        #nn.init.xavier_uniform_(self.output_layer.weight)  # Initialize output layer as well

    def forward(self, x, return_embed=False):
        embed = self.mlp(x)
        output = self.output_layer(embed)
        if return_embed:
            return output, embed
        return output
    
def predict_and_eval(VLMClassifier,
                   dataset="FB",
                   split_dl=None,
                   split=None,
                   topk=-1,
                   artifact=None,
                   train_dl=None,
                   num_few_shots=None):
    logging_columns = [
        "id",
        "gt_label",
        "image",
        "pred_label",
        "pred_logits",
    ]
    metrics_columns = [
        "acc",
        "auc",
        "precision",
        "recall",
        "f1",
    ]
    ids = []
    labels = []
    predicted_gen = []
    predicted_cls = []
    logging_table = wandb.Table(columns=logging_columns)
    metrics_table = wandb.Table(columns=metrics_columns)
     # Prepare few-shot examples if `train_dl` and `num_few_shots` are provided
    few_shot_labels = []
    few_shot_imgs = []
    if train_dl is not None and num_few_shots is not None:
        few_shot_batches = list(train_dl)[:num_few_shots]
        for batch in few_shot_batches:
            few_shot_imgs.extend(batch[0])  # Images
            few_shot_labels.extend(batch[2])  # Labels

    for i, batch in tqdm(enumerate(split_dl), total=len(split_dl)):
        bz = len(batch[0])
        images, texts, labels_batch, ids_batch = batch

        # Use predict_few_shot if few-shot examples are available, otherwise fall back to predict
        if few_shot_labels and few_shot_imgs:
            output_gen, output_cls = VLMClassifier.predict_few_shot(
                query_texts=texts,
                query_imgs=images,
                few_shot_labels=few_shot_labels,
                few_shot_imgs=few_shot_imgs,
            )
        else:
            output_gen, output_cls = VLMClassifier.predict(
                texts,
                images
            )

        ids.extend(ids_batch)
        labels.append(labels_batch.detach().cpu())


        predicted_gen.append(output_gen.detach().cpu())
        if output_cls is not None:
            predicted_cls.append(output_cls.detach().cpu())
        # For debugging 
        if topk != -1 and i > topk // bz:
            break
    predicted_gen = torch.cat(predicted_gen)
    
    labels = torch.cat(labels)
    print("Generative classifier")
    acc, auc, pre, recall, f1 = eval_metrics(
        dataset, labels, predicted_gen, name=split, compute_loss=False, apply_sigmoid=False)
    
    for i in range(len(ids)):
        #predicted_prob = torch.sigmoid(predicted_gen[i].float())
        predicted_label = (predicted_gen[i] >= 0.5).long()
        logging_table.add_data(ids[i],
                               labels[i],
                               #artifact.get("{}.png".format(ids[i])
                         #) if dataset == "FB" else "Dummy image",
                         "Dummy image",
                                predicted_label.item(),
                                predicted_gen[i].item())
    wandb.log({"Generative_logging_table_{}".format(split): logging_table})
    
    metrics_table.add_data(acc, auc, pre, recall,f1 )
    wandb.log({"Generative_metrics_table_{}".format(split): metrics_table})
    
    if output_cls is not None:
        
        logging_table = wandb.Table(columns=logging_columns)
        metrics_table = wandb.Table(columns=metrics_columns)
        
        predicted_cls = torch.cat(predicted_cls)
        print("Logistic classifier")
        acc_, auc_, pre_, recall_, f1_ = eval_metrics(
            dataset, labels, predicted_cls, name=split, compute_loss=False)
        
        for i in range(len(ids)):
            predicted_prob = torch.sigmoid(predicted_cls[i].float())
            predicted_label = (predicted_prob >= 0.5).long()
            logging_table.add_data(ids[i],
                                   labels[i],
                                   #artifact.get("{}.png".format(ids[i])
                             #) if dataset == "FB" else "Dummy image",
                             "Dummy image",
                                    predicted_label.item(),
                                    predicted_prob.item())
        wandb.log({"Classifier_logging_table_{}".format(split): logging_table})
        
        metrics_table.add_data(acc_, auc_, pre_, recall_, f1_)
        wandb.log({"Classifier_metrics_table_{}".format(split): metrics_table})
    else:
        acc_, auc_, pre_, recall_, f1_ = 0, 0, 0, 0, 0
    
    
    return (acc, auc,pre, recall, f1), (acc_, auc_, pre_, recall_, f1_)


def eval_metrics(dataset, labels, predicted, name="dev_seen", epoch=0, compute_loss=False, print_score=True, apply_sigmoid=True,):
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(1)
    if len(predicted.shape) == 1:
        predicted = predicted.unsqueeze(1)
    # First convert fp16 or bf16 to fp32
    predicted = predicted.float()
    if apply_sigmoid:
        preds_proxy = torch.sigmoid(predicted)
    else:
        preds_proxy = predicted
    preds = (preds_proxy >= 0.5).long()

    if dataset == "MMHS-FineGrained":
        ACCURACY = torchmetrics.Accuracy()
        # TODO
    elif dataset != "Propaganda":
        ACCURACY = torchmetrics.Accuracy(task='binary')
        AUROC = torchmetrics.AUROC(task='binary')
        PRECISION = torchmetrics.Precision(task='binary')
        RECALL = torchmetrics.Recall(task='binary')
        F1Score = torchmetrics.F1Score(task='binary')
    elif dataset == "Propaganda":
        ACCURACY = torchmetrics.Accuracy(
            task="multilabel", num_labels=22, average='micro')
        AUROC = torchmetrics.AUROC(
            task="multilabel", num_labels=22, average='micro')
        PRECISION = torchmetrics.Precision(
            task="multilabel", num_labels=22, average='micro')
        RECALL = torchmetrics.Recall(
            task="multilabel", num_labels=22, average='micro')
        F1Score = torchmetrics.F1Score(
            task="multilabel", num_labels=22, average='micro')
    acc = ACCURACY(preds, labels)
    roc = AUROC(preds_proxy, labels)
    pre = PRECISION(preds, labels)
    recall = RECALL(preds, labels)
    f1 = F1Score(preds, labels)

    if compute_loss:
        lossFn_classifier = nn.BCEWithLogitsLoss()
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)
        loss = lossFn_classifier(predicted, labels.float())

        if print_score:
            print("{} acc: {:.4f} roc: {:.4f} pre: {:.4f} recall: {:.4f} f1: {:.4f} loss: {:.4f} ".format(
                name, acc, roc, pre, recall, f1, loss.item()))
        return acc, roc, pre, recall, f1, loss
    else:
        if print_score:
            print("{} acc: {:.4f} roc: {:.4f} pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(
                name,  acc, roc, pre, recall, f1))
        return acc.item(), roc.item(), pre.item(), recall.item(), f1.item()



def predict_and_eval_save_feature(VLMClassifier,
                   dataset="FB",
                   split_dl=None,
                   split=None,
                   topk=-1,):

    ids = []
    labels = []

    last_hidden_states = []
    embeds = []
    predicted_cls = []

    for i, batch in tqdm(enumerate(split_dl), total=len(split_dl)):
        bz = len(batch[0])
        images, texts, labels_batch, ids_batch = batch
        

        hidden_state, output_cls, embed = VLMClassifier.predict_return_embeds(
                texts, images)
        ids.extend(ids_batch)
        labels.append(labels_batch.detach().cpu())
        last_hidden_states.append(hidden_state.detach().cpu())
        embeds.append(embed.detach().cpu())

        predicted_cls.append(output_cls.detach().cpu())
        # For debugging 
        if topk != -1 and i > topk // bz:
            break
    
    labels = torch.cat(labels)
    predicted_cls = torch.cat(predicted_cls)
    last_hidden_states = torch.cat(last_hidden_states)
    embeds = torch.cat(embeds)
    try:
        print("Logistic classifier")
        eval_metrics(
            dataset, labels, predicted_cls, name=split, compute_loss=False)
    except: 
        pass
    
    return last_hidden_states, embeds, ids, labels