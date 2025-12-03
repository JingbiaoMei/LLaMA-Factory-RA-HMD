# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)




from dataclasses import dataclass, field
import torchmetrics
import torch.nn as nn
import numpy as np
import time
import math
from transformers.trainer_utils import speed_metrics, EvalLoopOutput, EvalPrediction, denumpify_detensorize
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify, IterableDatasetShard
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
#from transformers.deepspeed import deepspeed_init
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer import has_length
from transformers.trainer import is_sagemaker_mp_enabled

class CustomSeq2SeqRegressionTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """
    ACCURACY = torchmetrics.Accuracy(task='binary')
    AUROC = torchmetrics.AUROC(task='binary')
    PRECISION = torchmetrics.Precision(task='binary')
    RECALL = torchmetrics.Recall(task='binary')
    F1Score = torchmetrics.F1Score(task='binary')

    lm_loss_after_each_logging = []
    classification_loss_after_each_logging = []
    rgcl_loss_after_each_logging = []
    in_batch_negative_loss_after_each_logging = []
    negative_loss_after_each_logging = []
    positive_loss_after_each_logging = []
    

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        # Adding finetuneing args to self.args
        for key, value in finetuning_args.__dict__.items():
            if key not in self.args.__dict__:
                setattr(self.args, key, value)
        

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
            
        # Initialize the custom classifier model 
        #mlp_classifier = Classifier(self.model.config.hidden_size, 
        #                                 self.finetuning_args.num_layers, 
        #                                self.finetuning_args.proj_dim, 
        #                                 self.finetuning_args.output_dim, 
        #                                 self.finetuning_args.input_dropout, 
        #                                 self.finetuning_args.dropout)
        #self.model.add_module("classifier", mlp_classifier.to(self.args.device))
        # Move the classifier to the device, 
  
    
        # Save classifier config to dict first, when savng the model, it will be saved to the config file
        self.classifier_config = {
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.finetuning_args.num_layers,
            "proj_dim": self.finetuning_args.proj_dim,
            "output_dim": self.finetuning_args.output_dim,
            "input_dropout": self.finetuning_args.input_dropout,
            "dropout": self.finetuning_args.dropout
        }
        # Unfreeze the classifier
        self.unfreeze_classifier()
        
        # Initialize yes_token_id based on model type
        if "llava" in self.args.output_dir.lower() and "llava-next" not in self.args.output_dir.lower():
            self.yes_token_id = self.tokenizer.convert_tokens_to_ids("▁Yes")
        elif "qwen" in self.args.output_dir.lower():
            # Works for qwen2, qwen2.5, qwen3
            self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        elif "llama3-llava-next-8b-hf" in self.args.output_dir.lower():
            self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        else:
            print("Warning: Untested model, may lead to unexpected behaviour")
            # Fallback: try to get "Yes" token id
            self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
            if self.yes_token_id is None or self.yes_token_id == self.tokenizer.unk_token_id:
                # Try with space prefix
                self.yes_token_id = self.tokenizer.convert_tokens_to_ids("▁Yes")
        
        self.loss_fn = torch.nn.BCEWithLogitsLoss() 

        # Modify the lr without changing the optimizer function
        # Not working, since the optimizer is not created yet
        #for param_group in self.optimizer.param_groups:
        #    for param in param_group['params']:
        #        if param in self.model.classifier.parameters():
        #            self.optimizer.state[param]['lr'] = self.args.classifier_lr

        
    def unfreeze_classifier(self):
        # in case the classifier is frozen, unfreeze it
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    

    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            
            # Identify "mlp_classifier" parameters
            classifier_parameters = [
                name for name, _ in opt_model.named_parameters() if "classifier" in name]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in classifier_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n not in classifier_parameters)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in classifier_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.classifier_lr
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, eval_mode=False):
        # num_items_in_batch=None for fixing the issue in hf=4.49.0 
        # https://github.com/huggingface/transformers/issues/36331
        
        # If input in the begining of the script, system crashes
        from .rgcl_loss import compute_rgcl_loss
        # Note that the RGCL loss is only included in stage 2 training. 
        # The implementation of the RGCL loss here is for the ablation study in the paper.
        # Where the RGCL loss is included in the training from the beginning, i.e., combining with two stages
        # Current step
        if not eval_mode:
            try:
                self.current_step += 1
            except:
                self.current_step = 0
            if self.args.rgcl and self.current_step >= self.args.rgcl_warmup:
                
                reindex = self.current_step % (self.args.rgcl_reindex_every * self.args.gradient_accumulation_steps) == 0
                # The above is not working since there is graident accumulation
                
                
                if reindex:
                    self.train_feats = None
                    self.train_labels = None
                # Since the dataset is shuffled, we need to reindex the features and labels and store them

        # Extract labels
        labels = inputs.get('labels')

        # Determine target labels (0 for 'No', 1 for 'Yes')
        has_yes = (labels == self.yes_token_id).any(dim=1)
        target_labels = torch.zeros(labels.size(0), device=labels.device)
        target_labels[has_yes] = 1

        if self.args.rgcl and not eval_mode and self.current_step >= self.args.rgcl_warmup:
            outputs, pred, embeds = model(
                **inputs, classification_mode=True, output_hidden_states=True, output_embeds=True)
            (rgcl_loss,
             in_batch_negative_loss,
             negative_loss,
             positive_loss,
             train_feats,
             train_labels) = compute_rgcl_loss(inputs,
                                               target_labels,
                                               embeds,
                                               self.get_train_dataloader(),
                                               model,
                                               self.args,
                                               reindex=reindex,
                                               train_feats=self.train_feats,
                                               train_labels=self.train_labels,
                                               yes_token_id=self.yes_token_id)

            self.train_feats = train_feats
            self.train_labels = train_labels
            # delete the embeds to save gpu memory
            del train_feats
            del train_labels

        else:
            #outputs, pred = model(
            #    **inputs, classification_mode=True, output_hidden_states=True,)
            outputs = model(**inputs, output_hidden_states=True)
        
        
        lm_loss = outputs["loss"]

        # pass the last hidden state to the classifier
        _, pred = self.get_embeds_from_last_layer(labels, outputs)
        # Compute classification loss
        classification_loss = self.loss_fn(pred.squeeze(-1), target_labels)

        # Combine losses
        loss = lm_loss * self.args.loss_ratio[0] + classification_loss * self.args.loss_ratio[1] + rgcl_loss * \
            self.args.loss_ratio[2] if self.args.rgcl and not eval_mode and self.current_step >= self.args.rgcl_warmup else lm_loss * \
            self.args.loss_ratio[0] + \
            classification_loss * self.args.loss_ratio[1]
        if not eval_mode:
            self.lm_loss_after_each_logging.append(lm_loss.detach().cpu().item())
            self.classification_loss_after_each_logging.append(
                classification_loss.detach().cpu().item())
        if self.args.rgcl and not eval_mode and self.current_step >= self.args.rgcl_warmup:
            self.rgcl_loss_after_each_logging.append(
                    rgcl_loss.detach().cpu().item())
            self.in_batch_negative_loss_after_each_logging.append(
                in_batch_negative_loss.detach().cpu().item())
            self.negative_loss_after_each_logging.append(
                negative_loss.detach().cpu().item())
            self.positive_loss_after_each_logging.append(
                positive_loss.detach().cpu().item())
        
        
        if not eval_mode:
            # self.log({"lm_loss": lm_loss.detach().cpu().item(), "classification_loss": classification_loss.detach().cpu().item()})
            return (loss, outputs) if return_outputs else loss
        else:
            # self.log({"eval_lm_loss": lm_loss.detach().cpu().item(), "eval_classification_loss": classification_loss.detach().cpu().item()})
            return (lm_loss, classification_loss), outputs, target_labels, pred

    def get_embeds_from_last_layer(self, labels, output, infer_mode=False, output_embeds=False):
        
        # For standalone inference mode with no causal generation (i.e., without any label)
        if not infer_mode:
            
            last_negative_indices = (labels == -100).nonzero(as_tuple=False)

            # Extract the last -100 index for each batch
            batch_size = labels.size(0)
            last_negative_per_batch = [
                last_negative_indices[last_negative_indices[:, 0] == i, 1].max().item()
                for i in range(batch_size)
            ]
            last_negative_tensor = torch.tensor(last_negative_per_batch, device="cuda:0")
            
            if self.args.embed_layer == "last":
                hidden_state = output.hidden_states[-1]
            elif self.args.embed_layer == "penultimate":
                hidden_state = output.hidden_states[-2]

            if self.args.embed_mode == "last_token":
                x = hidden_state[torch.arange(batch_size, device="cuda:0"), last_negative_tensor, :]
            elif self.args.embed_mode == "pool":
                # Create masks to include only tokens before the last `-100`
                seq_len = labels.size(1)
                mask = torch.arange(seq_len, device="cuda:0").unsqueeze(0) <= last_negative_tensor.unsqueeze(1)

                # Apply the mask to hidden_states and calculate the mean
                masked_hidden_states = hidden_state * mask.unsqueeze(-1)  # Mask tokens
                token_counts = mask.sum(dim=1).unsqueeze(1)  # Number of valid tokens per batch
                x = masked_hidden_states.sum(dim=1) / token_counts  # Compute mean embeddings
        else:
            # This is for separate inference mode not during training time, not even the trainer's evaluation mode
            if self.args.embed_layer == "last":
                hidden_state = output.hidden_states[-1]
            elif self.args.embed_layer == "penultimate":
                hidden_state = output.hidden_states[-2]

            if self.args.embed_mode == "last_token":
                x = hidden_state[:, -1, :]
            elif self.args.embed_mode == "pool":
                x = hidden_state.mean(dim=1)
        # Change to float
        #x = x.to(torch.float)
        if not output_embeds:
            x = self.model.classifier(x)
            return output, x
        else:
            x, embed = self.model.classifier(x, return_embed=True)
            return output, x, embed
    
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    """
    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels
    """
    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            ignore_keys = getattr(self.args, "ignore_keys", None)

        with torch.no_grad():
            if has_labels:
                (lm_loss, classification_loss), _, target_labels, preds = self.compute_loss(
                    model, inputs, return_outputs=True, eval_mode=True)
                lm_loss = lm_loss.mean().detach()
                classification_loss = classification_loss.mean().detach()
            else:
                outputs = model(**inputs, classification_mode=True,
                                output_hidden_states=True)
                loss = None

        if has_labels:
            return (lm_loss, classification_loss), preds, target_labels
        else:
            return None, preds, target_labels

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics
        
        # print("custom evaluate")
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        # eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def compute_metrics_custom(self, eval_pred):
        if self.args.task == "meme_classification":
            logits, labels = eval_pred
            labels = torch.Tensor(labels)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)
            logits = torch.Tensor(logits)
            preds_proxy = torch.sigmoid(logits)
            preds = (preds_proxy >= 0.5).long()
            acc = self.ACCURACY(preds, labels)
            roc = self.AUROC(preds_proxy, labels)
            pre = self.PRECISION(preds, labels)
            recall = self.RECALL(preds, labels)
            f1 = self.F1Score(preds, labels)

            # print with 4sf
            print(
                f"Accuracy: {acc:.4f}, AUROC: {roc:.4f}, Precision: {pre:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            return {"accuracy": acc, "auroc": roc, "precision": pre, "recall": recall, "f1": f1}
        else:
            return super().compute_metrics(eval_pred)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        # print("custom evaluation loop")
        args = self.args

        # Overwrite
        prediction_loss_only = False

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(
            self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        lm_losses_host = None
        classification_losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_lm_losses = None
        all_ce_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            (lm_loss, classification_loss), logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(
                inputs["input_ids"]) if args.include_inputs_for_metrics else None

            loss = lm_loss + classification_loss
            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics(
                    (loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(
                    losses_host, losses, padding_index=-100)
            if lm_loss is not None:
                lm_losses = self.accelerator.gather_for_metrics(
                    (lm_loss.repeat(batch_size)))
                lm_losses_host = lm_losses if lm_losses_host is None else nested_concat(
                    lm_losses_host, lm_losses, padding_index=-100)
            if classification_loss is not None:
                classification_losses = self.accelerator.gather_for_metrics(
                    (classification_loss.repeat(batch_size)))
                classification_losses_host = classification_losses if classification_losses_host is None else nested_concat(
                    classification_losses_host, classification_losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(
                    inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics(
                    (inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(
                    logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and self.accelerator.sync_gradients:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if lm_losses_host is not None:
                    lm_losses = nested_numpify(lm_losses_host)
                    all_lm_losses = lm_losses if all_lm_losses is None else np.concatenate(
                        (all_lm_losses, lm_losses), axis=0)
                if classification_losses_host is not None:
                    classification_losses = nested_numpify(
                        classification_losses_host)
                    all_ce_losses = classification_losses if all_ce_losses is None else np.concatenate(
                        (all_ce_losses, classification_losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(
                            all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None
                lm_losses_host, classification_losses_host = None, None
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if lm_losses_host is not None:
            lm_losses = nested_numpify(lm_losses_host)
            all_lm_losses = lm_losses if all_lm_losses is None else np.concatenate(
                (all_lm_losses, lm_losses), axis=0)
        if classification_losses_host is not None:
            classification_losses = nested_numpify(classification_losses_host)
            all_ce_losses = classification_losses if all_ce_losses is None else np.concatenate(
                (all_ce_losses, classification_losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(
                    all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics_custom is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics_custom(
                    EvalPrediction(predictions=all_preds,
                                   label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics_custom(EvalPrediction(
                    predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if all_lm_losses is not None:
            metrics[f"{metric_key_prefix}_lm_loss"] = all_lm_losses.mean().item()
        if all_ce_losses is not None:
            metrics[f"{metric_key_prefix}_classification_loss"] = all_ce_losses.mean(
            ).item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time=None, learning_rate=None):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # For fine grained loss logging
            lm_loss_after_each_logging = np.mean(
                self.lm_loss_after_each_logging)
            classification_loss_after_each_logging = np.mean(
                self.classification_loss_after_each_logging)
            if self.args.rgcl and self.current_step >= self.args.rgcl_warmup:
                rgcl_loss_after_each_logging = np.mean(
                    self.rgcl_loss_after_each_logging)
                in_batch_negative_loss_after_each_logging = np.mean(
                    self.in_batch_negative_loss_after_each_logging)
                negative_loss_after_each_logging = np.mean(
                    self.negative_loss_after_each_logging)
                positive_loss_after_each_logging = np.mean(
                    self.positive_loss_after_each_logging)

            # Reset the loss after each logging
            self.lm_loss_after_each_logging = []
            self.classification_loss_after_each_logging = []
            self.rgcl_loss_after_each_logging = []
            self.in_batch_negative_loss_after_each_logging = []
            self.negative_loss_after_each_logging = []
            self.positive_loss_after_each_logging = []
            

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["lm_loss"] = round(lm_loss_after_each_logging, 4)
            logs["classification_loss"] = round(
                classification_loss_after_each_logging, 4)
            if self.args.rgcl and self.current_step >= self.args.rgcl_warmup:
                logs["rgcl_loss"] = round(rgcl_loss_after_each_logging, 4)
                logs["in_batch_negative_loss"] = round(
                    in_batch_negative_loss_after_each_logging, 4)
                logs["negative_loss"] = round(
                    negative_loss_after_each_logging, 4)
                logs["positive_loss"] = round(
                    positive_loss_after_each_logging, 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            # Use learning_rate parameter if provided, otherwise get it from scheduler
            logs["learning_rate"] = learning_rate if learning_rate is not None else self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            #self._save_checkpoint(model, trial, metrics=metrics)
            # Fix for hf 4.49 
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model, classifier, and classifier config.
        """
        output_dir = output_dir if output_dir else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save the base model
        super().save_model(output_dir)

        # Save the classifier weights
        if hasattr(self.model, "classifier"):
            classifier_path = os.path.join(output_dir, "classifier.bin")
            torch.save(self.model.classifier.state_dict(), classifier_path)

        # Save the classifier config
        if hasattr(self, "classifier_config"):
            config_path = os.path.join(output_dir, "classifier_config.json")
            with open(config_path, "w") as f:
                import json
                json.dump(self.classifier_config, f, indent=4)
            
