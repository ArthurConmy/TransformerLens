import dataclasses
from functools import partial
import wandb
import os
from collections import defaultdict
import pickle
import torch
import huggingface_hub
import datetime
from typing import Union, Dict, Callable
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)
import warnings
import networkx as nx
from transformer_lens import HookedTransformer

def shuffle_tensor(tens, seed=42):
    """Shuffle tensor along first dimension"""
    torch.random.manual_seed(seed)
    return tens[torch.randperm(tens.shape[0])]


def negative_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_repeat_candidates: Optional[torch.Tensor] = None,
    baseline: Union[float, torch.Tensor] = 0.0,
    last_seq_element_only: bool = True,
    return_one_element: bool=True,
) -> torch.Tensor:
    logprobs = F.log_softmax(logits, dim=-1)

    if last_seq_element_only:
        logprobs = logprobs[:, -1, :]

    # Subtract a baseline for each element -- which could be 0 or the NLL of the base_model_logprobs
    nll_all = (
        F.nll_loss(logprobs.view(-1, logprobs.size(-1)), labels.view(-1), reduction="none").view(logprobs.size()[:-1])
        - baseline
    )

    if mask_repeat_candidates is not None:
        assert nll_all.shape == mask_repeat_candidates.shape, (
            nll_all.shape,
            mask_repeat_candidates.shape,
        )
        answer = nll_all[mask_repeat_candidates]
    elif not last_seq_element_only:
        assert nll_all.ndim == 2, nll_all.shape
        answer = nll_all.view(-1)
    else:
        answer = nll_all

    if return_one_element:
        return answer.mean()

    return answer

def kl_divergence(
    logits: torch.Tensor,
    base_model_logprobs: torch.Tensor,
    mask_repeat_candidates: Optional[torch.Tensor] = None,
    last_seq_element_only: bool = True,
    base_model_probs_last_seq_element_only: bool = False,
    return_one_element: bool = True,
) -> torch.Tensor:
    # Note: we want base_model_probs_last_seq_element_only to remain False by default, because when the Docstring
    # circuit uses this, it already takes the last position before passing it in.

    if last_seq_element_only:
        logits = logits[:, -1, :]

    if base_model_probs_last_seq_element_only:
        base_model_logprobs = base_model_logprobs[:, -1, :]

    logprobs = F.log_softmax(logits, dim=-1)
    kl_div = F.kl_div(logprobs, base_model_logprobs, log_target=True, reduction="none").sum(dim=-1)

    if mask_repeat_candidates is not None:
        assert kl_div.shape == mask_repeat_candidates.shape, (kl_div.shape, mask_repeat_candidates.shape)
        answer = kl_div[mask_repeat_candidates]
    elif not last_seq_element_only:
        assert kl_div.ndim == 2, kl_div.shape
        answer = kl_div.view(-1)
    else:
        answer = kl_div

    if return_one_element:
        return answer.mean()

    return answer

def get_model(device):
    tl_model = HookedTransformer.from_pretrained(
        "redwood_attn_2l",  # load Redwood's model
        center_writing_weights=False,  # these are needed as this model is a Shortformer; this is a technical detail
        center_unembed=False,
        fold_ln=False,
        device=device,
    )

    # standard ACDC options
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True) 
    if "use_hook_mlp_in" in tl_model.cfg.to_dict(): # not strictly necessary, but good practice to keep compatibility with new *optional* transformerlens feature
        tl_model.set_use_hook_mlp_in(True)
    return tl_model

def get_validation_data(num_examples=None, seq_len=None, device=None):
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = torch.load(validation_fname, map_location=device).long()

    if num_examples is None:
        return validation_data
    else:
        return validation_data[:num_examples][:seq_len]

def get_good_induction_candidates(num_examples=None, seq_len=None, device=None):
    """Not needed?"""
    good_induction_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="good_induction_candidates.pt"
    )
    good_induction_candidates = torch.load(good_induction_candidates_fname, map_location=device)

    if num_examples is None:
        return good_induction_candidates
    else:
        return good_induction_candidates[:num_examples][:seq_len]

def get_mask_repeat_candidates(num_examples=None, seq_len=None, device=None):
    mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
    )
    mask_repeat_candidates = torch.load(mask_repeat_candidates_fname, map_location=device)
    mask_repeat_candidates.requires_grad = False

    if num_examples is None:
        return mask_repeat_candidates
    else:
        return mask_repeat_candidates[:num_examples, :seq_len]


def get_all_induction_things(num_examples, seq_len, device, data_seed=42, return_one_element=True):
    tl_model = get_model(device=device)

    validation_data_orig = get_validation_data(device=device)
    mask_orig = get_mask_repeat_candidates(num_examples=None, device=device) # None so we get all
    assert validation_data_orig.shape == mask_orig.shape

    assert seq_len <= validation_data_orig.shape[1]-1

    validation_slice = slice(0, num_examples)
    validation_data = validation_data_orig[validation_slice, :seq_len].contiguous()
    validation_labels = validation_data_orig[validation_slice, 1:seq_len+1].contiguous()
    validation_mask = mask_orig[validation_slice, :seq_len].contiguous()

    validation_patch_data = shuffle_tensor(validation_data, seed=data_seed).contiguous()

    test_slice = slice(num_examples, num_examples*2)
    test_data = validation_data_orig[test_slice, :seq_len].contiguous()
    test_labels = validation_data_orig[test_slice, 1:seq_len+1].contiguous()
    test_mask = mask_orig[test_slice, :seq_len].contiguous()

    # data_seed+1: different shuffling
    test_patch_data = shuffle_tensor(test_data, seed=data_seed).contiguous()

    with torch.no_grad():
        base_val_logprobs = F.log_softmax(tl_model(validation_data), dim=-1).detach()
        base_test_logprobs = F.log_softmax(tl_model(test_data), dim=-1).detach()

    kl_divergence_metric = partial(
        kl_divergence,
        base_model_logprobs=base_val_logprobs,
        mask_repeat_candidates=validation_mask,
        last_seq_element_only=False,
        return_one_element=return_one_element,
    )
    negative_log_probs_metric = partial(
        negative_log_probs,
        labels=validation_labels,
        mask_repeat_candidates=validation_mask,
        last_seq_element_only=False,
    )

    return (
        tl_model,
        validation_data,
        validation_patch_data, 
        kl_divergence_metric,
        negative_log_probs_metric,
    )


def one_item_per_batch(toks_int_values, toks_int_values_other, mask_rep, base_model_logprobs, kl_take_mean=True):
    """Returns each instance of induction as its own batch idx"""

    end_positions = []
    batch_size, seq_len = toks_int_values.shape
    new_tensors = []

    toks_int_values_other_batch_list = []
    new_base_model_logprobs_list = []

    for i in range(batch_size):
        for j in range(seq_len - 1): # -1 because we don't know what follows the last token so can't calculate losses
            if mask_rep[i, j]:
                end_positions.append(j)
                new_tensors.append(toks_int_values[i].cpu().clone())
                toks_int_values_other_batch_list.append(toks_int_values_other[i].cpu().clone())
                new_base_model_logprobs_list.append(base_model_logprobs[i].cpu().clone())

    toks_int_values_other_batch = torch.stack(toks_int_values_other_batch_list).to(toks_int_values.device).clone()
    return_tensor = torch.stack(new_tensors).to(toks_int_values.device).clone()
    end_positions_tensor = torch.tensor(end_positions).long()

    new_base_model_logprobs = torch.stack(new_base_model_logprobs_list)[torch.arange(len(end_positions_tensor)), end_positions_tensor].to(toks_int_values.device).clone()
    metric = partial(
        kl_divergence, 
        base_model_logprobs=new_base_model_logprobs, 
        end_positions=end_positions_tensor, 
        mask_repeat_candidates=None, # !!! 
        last_seq_element_only=False, 
        return_one_element=False
    )
    
    return return_tensor, toks_int_values_other_batch, end_positions_tensor, metric
