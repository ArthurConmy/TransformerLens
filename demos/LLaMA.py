#%%
# # LLaMA and Llama-2 in TransformerLens
# 
# This demo requires `transformers` version 4.31.0 (which adds Llama-2 support). This tutorial has part a) for LLaMA and b) for Llama-2. Currently the only Llama-2 support is the 7B chat model, as this notebook is being tested.
# 
# Steps to run this demo:
# 
# 1a. Get LLaMA weights here: https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform
# 
# 1b. Get Llama-2 weights here: https://ai.meta.com/resources/models-and-libraries/llama-downloads/
# 
# 2a. Convert the official weights to huggingface. 
# 
# ```bash
# python src/transformers/models/llama/convert_llama_weights_to_hf.py \
#     --input_dir /path/to/downloaded/llama/weights \
#     --model_size 7B \
#     --output_dir /output/path
# ```
# 
# 2b. Same step for Llama-2, we'll use `7Bf` the 7B chat version
# 
# ```bash
# python src/transformers/models/llama/convert_llama_weights_to_hf.py \
#     --input_dir /path/to/downloaded/llama-2/weights \
#     --model_size 7Bf \
#     --output_dir /output/path
# ```
# 
# Note: this didn't work for Arthur by default (even though HF doesn't seem to show this anywhere). I had to change <a href="https://github.com/huggingface/transformers/blob/07360b6/src/transformers/models/llama/convert_llama_weights_to_hf.py#L295">this</a> line of my pip installed `src/transformers/models/llama/convert_llama_weights_to_hf.py` file (which was found at `/opt/conda/envs/arthurenv/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py`) from 
# 
# `input_base_path=os.path.join(args.input_dir, args.model_size),` to `input_base_path=os.path.join(args.input_dir),`
# 
# 3. Change the ```MODEL_PATH``` variable in the notebook to the where the converted weights are stored.

# In[9]:

from typing import Literal
MODE: Literal["LLaMA", "Llama-2"] = "Llama-2" # change to LLaMA for original LLaMA

# In[10]:

import plotly.graph_objects as go
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
# ## Setup (skip)

# In[11]:

# Janky code to do different setup when run in a Colab notebook vs VSCode
DEVELOPMENT_MODE = False
try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
    get_ipython().run_line_magic('pip', 'install git+https://github.com/neelnanda-io/TransformerLens.git')
    get_ipython().run_line_magic('pip', 'install circuitsvis')
    
    # PySvelte is an unmaintained visualization library, use it as a backup if circuitsvis isn't working
    # # Install another version of node that makes PySvelte work way faster
    # !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
    # %pip install git+https://github.com/neelnanda-io/PySvelte.git
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

import torch
torch.autograd.set_grad_enabled(False)

# In[12]:


# # Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
# import plotly.io as pio
# if IN_COLAB or not DEVELOPMENT_MODE:
#     pio.renderers.default = "colab"
# else:
#     pio.renderers.default = "notebook_connected"
# print(f"Using renderer: {pio.renderers.default}")

# import circuitsvis as cv


# In[13]:


# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
from tqdm import tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from typing import List, Union, Optional
from jaxtyping import Float, Int
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
# import circuitsvis as cv

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

torch.set_grad_enabled(False)

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


# ## Loading model

# In[19]:


from transformers import LlamaForCausalLM, LlamaTokenizer
import os

MODEL_PATH=''

if "CONDA_PREFIX" in os.environ and "arthur" in os.environ["CONDA_PREFIX"]: # so Arthur can test fast
    MODEL_PATH=os.path.expanduser('~/new_out')

print("You can generally ignore the tokenizer warnings here (unless there are tokenization issues downstream)\n")

tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)

# In[35]:

if MODE == "LLaMA":
    model = HookedTransformer.from_pretrained("llama-7b", hf_model=hf_model, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)

elif MODE == "Llama-2":
    model = HookedTransformer.from_pretrained("Llama-2-7b-chat-hf", hf_model=hf_model, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False, fold_value_biases=False, tokenizer=tokenizer) # loading on CPU is cheapest memory wise in transformer_lens
    model = model.to(torch.double)
    
# model = model.to("cuda") # makes generation a lot faster
model.generate("The capital of Germany is", max_new_tokens=10, temperature=0)

# ### Compare logits with HuggingFace model

#%%

# greedy sample

generation = hf_model.generate(
    input_ids=torch.tensor(tokenizer.encode("The capital of Germany is", return_tensors="pt")),
    max_length=10,
    # temperature=0.0,l
    # do_sample=True,
    # num_return_sequences=1,
)

#%%

print(tokenizer.decode(generation[0]))

# In[36]:

prompts = [
    "The capital of Germany is",
    "2 * 42 = ", 
    "My favorite", 
    # "aosetuhaosuh aostud aoestuaoentsudhasuh aos tasat naostutshaosuhtnaoe usaho uaotsnhuaosntuhaosntu haouaoshat u saotheu saonuh aoesntuhaosut aosu thaosu thaoustaho usaothusaothuao sutao sutaotduaoetudet uaosthuao uaostuaoeu aostouhsaonh aosnthuaoscnuhaoshkbaoesnit haosuhaoe uasotehusntaosn.p.uo ksoentudhao ustahoeuaso usant.hsa otuhaotsi aostuhs",
]

model.eval()
hf_model.eval()

prompt_ids = [tokenizer.encode(prompt, return_tensors="pt") for prompt in prompts]
# for i in range(len(prompt_ids)):
# prompt_ids[0][:, 0]

tl_logits = [model(prompt_ids).detach().cpu() for prompt_ids in tqdm(prompt_ids)]

# hf logits are really slow as it's on CPU. If you have a big/multi-GPU machine, run `hf_model = hf_model.to("cuda")` to speed this up
logits = [hf_model(prompt_ids).logits.detach().cpu() for prompt_ids in tqdm(prompt_ids)]

for i in range(len(prompts)): 
    assert torch.allclose(logits[i], tl_logits[i], atol=1e-3, rtol=1e-3)

# In[73]:


# print(str(hf_model.model)[:200])
# torch.testing.assert_close(
#     hf_model.model.embed_tokens.weight,
#     model.W_E.cpu(),
# )
# hf_model.model.layers[0].input_layernorm.variance_epsilon
# model.blocks[0]


# In[75]:

from contextlib import contextmanager

# Get ModuleList from arbitrary transformer model
# (Alternatively, we could pick the module list containing >50% of model params.)
def get_hookable_blocks(model):
    assert isinstance(model, LlamaForCausalLM)
    return [model.model.layers[i].self_attn for i in range(len(model.model.layers))] + [layer for layer in model.model.layers]

@contextmanager
def not_pre_hooks(hooks):
    try:
        print(str(hooks[0][0])[:50])
        handles = [mod.register_forward_hook(hook) for mod, hook in hooks]
        yield handles
    except Exception as e:
        print(e, "error")
    finally:
        for handle in handles:
            print("Removin")
            handle.remove()

@contextmanager
def residual_stream(model: LlamaForCausalLM, layers: Optional[List[int]] = None):
    "Context manager to track residual stream activations in the model."
    # Plausibly could be replaced by "output_hidden_states=True" in model call.

    stream = [None] * len(get_hookable_blocks(model))
    layers = layers or range(len(stream))

    def _make_pre_hook(i):
        def _hook(_, inputs):
            # concat along the sequence dimension
            stream[i] = inputs[0] if stream[i] is None else t.cat([stream[i], inputs[0]], dim=1)
        return _hook

    def _make_hook(i):
        def _hook(_, inputs, outputs):
            # concat along the sequence dimension
            stream[i] = outputs if stream[i] is None else t.cat([stream[i], outputs], dim=1)
        return _hook

    hooks = [(layer, _make_hook(i)) for i, layer in enumerate(get_hookable_blocks(model)) if i in layers]
    with not_pre_hooks(hooks):
        yield stream

# Sample text
text = "Hello, how are you?"

# Tokenize input and convert to tensor
input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Add batch dimension

# Get activations
with residual_stream(hf_model) as activations:
    outputs = hf_model(input_ids)

assert len(activations) != activations.count(None)

#%%

# Sample text
text = "Hello, how are you?"

# Tokenize input and convert to tensor
input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Add batch dimension

outputs = hf_model(input_ids)


# In[ ]:

_, cache = model.run_with_cache(
    input_ids.cuda(),
    names_filter = lambda name: name=="blocks.0.attn.hook_q", # name.endswith("rot_q"), #  or name.endswith("ln2.hook_normalized"),
    device="cpu",
)

#%%

tl_query_rot = einops.rearrange(
    cache["blocks.0.attn.hook_rot_q"],
    "batch seq head d_head -> batch head seq d_head",
)

#%%

tl_query_pre_rot = einops.rearrange(
    cache["blocks.0.attn.hook_q"],
    "batch seq head d_head -> batch head seq d_head",
)

#%%

fig = go.Figure()

# add lines for each layer
fig.add_trace(go.Scatter(
    x=np.arange(len(activations)//2),
    y=torch.tensor([activations[i][0].norm() for i in range(len(activations)//2)]) / torch.tensor([cache[transformer_lens.utils.get_act_name("attn_out", i)].norm() for i in range(len(activations)//2)]),
    name="Ratio of resid_pre norms between HF model and TL",
))

# fig.add_trace(go.Scatter(
#     x=np.arange(len(activations)),
#     y=torch.tensor([activations[i + len(activations)//2].norm() for i in range(len(activations)//2)]) / torch.tensor([(model.blocks[i].ln1.w * cache[transformer_lens.utils.get_act_name("normalized", i, 'ln1')]).norm() for i in range(len(activations)//2)]),
#     name="Ratio of post_RMS norms between HF model and TL",
# ))

# fig.update_layout(
#     title="Residual Stream Norms",
# )

# fig.add_trace(go.Scatter(
#     x=np.arange(len(activations)),
#     y=[cache[transformer_lens.utils.get_act_name("resid_pre", i)].norm() for i in range(len(activations))],
# ))


#%%

# LLAMA FUNCTIONS

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_embed_just_to_q(q, cos, sin, position_ids):

    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

# In[42]:

# TL FUNCTIONS

def apply_rotary(
    self,
    x: Float[torch.Tensor, "batch pos head_index d_head"],
    past_kv_pos_offset=0,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)
    x_pos = x.size(1)
    x_rot = x[..., : self.cfg.rotary_dim]
    x_pass = x[..., self.cfg.rotary_dim :]
    x_flip = self.rotate_every_two(x_rot)
    x_rotated = (
        x_rot
        * self.rotary_cos[past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :]
        + x_flip
        * self.rotary_sin[past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :]
    )
    return torch.cat([x_rotated, x_pass], dim=-1)

#%%

for i in range(len(prompts)): 
    # print(torch.allclose(logits[i], tl_logits[i], atol=1e-1, rtol=1e-1))
    print(logits[i].norm(), tl_logits[i].norm())

print(model(torch.arange(3, 8)[None].cuda()).norm(), hf_model(torch.arange(3, 8)[None]).logits.norm())

# with 1e-6 we get
# tensor(1531.4062) tensor(1509.5721)
# tensor(1727.2130) tensor(2080.0579)
# tensor(854.9116) tensor(964.2944)

# with 1e-5 we get: 
# tensor(1531.4062) tensor(1488.4860)
# tensor(1727.2130) tensor(2133.8892)
# tensor(854.9116) tensor(993.9772)

from transformer_lens.loading_from_pretrained import get_official_model_name
print(model.cfg.eps, get_official_model_name(model.cfg.model_name))

# model.tokenizer.pad_token_id, model.tokenizer.bos_token_id, model.tokenizer.eos_token_id = 0, 1, 2 # ??? Why are these so wrong?
# tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id = 0, 1, 2 # ??? Why are these so wrong?


# In[46]:


prompt_ids


# ## TransformerLens Demo

# ### Reading from hooks

# In[17]:


llama_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
llama_tokens = model.to_tokens(llama_text)
llama_logits, llama_cache = model.run_with_cache(llama_tokens, remove_batch_dim=True)

attention_pattern = llama_cache["pattern", 0, "attn"]
llama_str_tokens = model.to_str_tokens(llama_text)

print("Layer 0 Head Attention Patterns:")
cv.attention.attention_patterns(tokens=llama_str_tokens, attention=attention_pattern)


# ### Writing to hooks

# In[11]:


layer_to_ablate = 0
head_index_to_ablate = 31

# We define a head ablation hook
# The type annotations are NOT necessary, they're just a useful guide to the reader
# 
def head_ablation_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    print(f"Shape of the value tensor: {value.shape}")
    value[:, :, head_index_to_ablate, :] = 0.
    return value

original_loss = model(llama_tokens, return_type="loss")
ablated_loss = model.run_with_hooks(
    llama_tokens, 
    return_type="loss", 
    fwd_hooks=[(
        utils.get_act_name("v", layer_to_ablate), 
        head_ablation_hook
        )]
    )
print(f"Original Loss: {original_loss.item():.3f}")
print(f"Ablated Loss: {ablated_loss.item():.3f}")


# In[ ]:




