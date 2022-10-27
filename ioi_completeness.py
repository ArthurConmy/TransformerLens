#%%
import os
import torch

from ioi_circuit_extraction import ALEX_NAIVE

if os.environ["USER"] in ["exx", "arthur"]:  # so Arthur can safely use octobox
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
assert torch.cuda.device_count() == 1
import json
from statistics import mean
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
import warnings
import json
from numpy import sin, cos, pi
from time import ctime
from dataclasses import dataclass
from ioi_utils import logit_diff
from tqdm import tqdm
import pandas as pd
import torch
import torch as t
from easy_transformer.utils import (
    gelu_new,
    to_numpy,
    get_corner,
    print_gpu_mem,
)  # helper functions
from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.EasyTransformer import (
    EasyTransformer,
    TransformerBlock,
    MLP,
    Attention,
    LayerNormPre,
    PosEmbed,
    Unembed,
    Embed,
)
from easy_transformer.experiments import (
    ExperimentMetric,
    AblationConfig,
    EasyAblation,
    EasyPatching,
    PatchingConfig,
    get_act_hook,
)
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional, Iterable
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from sklearn.linear_model import LinearRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import spacy
import re
from einops import rearrange
import einops
from pprint import pprint
import gc
from datasets import load_dataset
import matplotlib.pyplot as plt


from ioi_dataset import (
    IOIDataset,
    NOUNS_DICT,
    NAMES,
    gen_flipped_prompts,
    gen_prompt_uniform,
    BABA_TEMPLATES,
    ABBA_TEMPLATES,
)
from ioi_utils import (
    basis_change,
    add_arrow,
    CLASS_COLORS,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
    plot_ellipse,
    probs,
)
from copy import deepcopy

plotly_colors = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
]

from functools import partial

#%% [markdown]
# # <h1><b>Setup</b></h1>
# Import model and dataset
#%% # plot writing in the IO - S direction
model_name = "gpt2"  # Here we used gpt-2 small ("gpt2")

print_gpu_mem("About to load model")
model = EasyTransformer.from_pretrained(
    model_name,
)  # use_attn_result adds a hook blocks.{lay}.attn.hook_result that is before adding the biais of the attention layer
model.set_use_attn_result(True)
device = "cuda"
if torch.cuda.is_available():
    model.to(device)
print_gpu_mem("Gpt2 loaded")
# %% [markdown]
# Each prompts is a dictionnary containing 'IO', 'S' and the "text", the sentence that will be given to the model.
# The prompt type can be "ABBA", "BABA" or "mixed" (half of the previous two) depending on the pattern you want to study
# %%
# IOI Dataset initialisation
N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
# abca_dataset = ioi_dataset.gen_flipped_prompts(("S2", "RAND"))

# %%
# webtext = load_dataset("stas/openwebtext-10k")
# owb_seqs = [
#     "".join(show_tokens(webtext["train"]["text"][i][:2000], model, return_list=True)[: ioi_dataset.max_len])
#     for i in range(ioi_dataset.N)
# ]


#%%
from ioi_circuit_extraction import (
    join_lists,
    CIRCUIT,
    ALEX_NAIVE,
    RELEVANT_TOKENS,
    get_extracted_idx,
    get_heads_circuit,
    do_circuit_extraction,
    list_diff,
    ALL_NODES,
)


def get_all_nodes(circuit):
    nodes = []
    for circuit_class in circuit:
        for head in circuit[circuit_class]:
            nodes.append((head, RELEVANT_TOKENS[head][0]))
    return nodes


## define useful function


def get_heads_from_nodes(nodes, ioi_dataset):
    heads_to_keep_tok = {}
    for h, t in nodes:
        if h not in heads_to_keep_tok:
            heads_to_keep_tok[h] = []
        if t not in heads_to_keep_tok[h]:
            heads_to_keep_tok[h].append(t)

    heads_to_keep = {}
    for h in heads_to_keep_tok:
        heads_to_keep[h] = get_extracted_idx(heads_to_keep_tok[h], ioi_dataset)

    return heads_to_keep


def circuit_from_nodes_logit_diff(model, ioi_dataset, nodes):
    """Take a list of nodes, return the logit diff of the circuit described by the nodes"""
    heads_to_keep = get_heads_from_nodes(nodes, ioi_dataset)
    # print(heads_to_keep)
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    return logit_diff(model, ioi_dataset, all=False)


def circuit_from_heads_logit_diff(
    model, ioi_dataset, mean_dataset, heads_to_rmv=None, heads_to_kp=None, all=False
):
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_kp,
        heads_to_remove=heads_to_rmv,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=mean_dataset,
    )
    return logit_diff(model, ioi_dataset, all=all)


def greed_search_max_broken(get_circuit_logit_diff):
    NODES_PER_STEP = 10
    NB_SETS = 5
    NB_ITER = 10
    current_nodes = ALL_NODES.copy()
    all_sets = []
    all_node_baseline = get_circuit_logit_diff(ALL_NODES)

    for step in range(NB_SETS):
        current_nodes = ALL_NODES.copy()
        nodes_removed = []
        baseline = all_node_baseline

        for iter in range(NB_ITER):
            to_test = random.sample(current_nodes, NODES_PER_STEP)

            results = []
            for node in to_test:  # check wich heads in to_test causes the biggest drop
                circuit_minus_node = current_nodes.copy()
                circuit_minus_node.remove(node)
                results.append(get_circuit_logit_diff(circuit_minus_node))

            diff_to_baseline = [(results[i] - baseline) for i in range(len(results))]
            best_node_idx = np.argmin(diff_to_baseline)

            best_node = to_test[best_node_idx]
            current_nodes.remove(best_node)  # we remove the best node from the circuit
            nodes_removed.append(best_node)

            if (
                iter > NB_ITER // 2 - 1
            ):  # we begin to save the sets after half of the iterations
                all_sets.append(
                    {
                        "circuit_nodes": current_nodes.copy(),
                        "removed_nodes": nodes_removed.copy(),
                    }
                )

            print(
                f"iter: {iter} - best node:{best_node} - drop:{min(diff_to_baseline)} - baseline:{baseline}"
            )
            print_gpu_mem(f"iter {iter}")
            baseline = results[best_node_idx]  # new baseline for the next iteration
    return all_sets


def test_minimality(model, ioi_dataset, v, J, absolute=True):
    """Compute |Metric( (C\J) U {v}) - Metric(C\J)| where J is a list of nodes, v is a node"""
    C_minus_J = list(set(ALL_NODES.copy()) - set(J.copy()))

    LD_C_m_J = circuit_from_nodes_logit_diff(
        model, ioi_dataset, C_minus_J
    )  # metric(C\J)
    C_minus_J_plus_v = set(C_minus_J.copy())
    C_minus_J_plus_v.add(v)
    C_minus_J_plus_v = list(C_minus_J_plus_v)

    LD_C_m_J_plus_v = circuit_from_nodes_logit_diff(
        model, ioi_dataset, C_minus_J_plus_v
    )  # metric( (C\J) U {v})
    if absolute:
        return np.abs(LD_C_m_J - LD_C_m_J_plus_v)
    else:
        return LD_C_m_J - LD_C_m_J_plus_v


def add_key_to_json_dict(fname, key, value):
    with open(fname, "r") as f:
        d = json.load(f)
    d[key] = value
    with open(fname, "w") as f:
        json.dump(d, f)


def greed_search_max_brok_cob_diff(
    get_cob_brok_from_nodes,
    init_set=[],
    NODES_PER_STEP=10,
    NB_SETS=5,
    NB_ITER=10,
    verbose=True,
    save_to_file=False,
):
    """Greedy search to find G that maximize the difference between broken and cobbled circuit |metric(C\G) - metric(M\G)| . Return a list of node sets."""
    all_sets = []

    neg_head_in_G = False
    if neg_head_in_G:
        init_set = list(set(init_set) + set([((10, 7), "end"), ((11, 10), "end")]))

    all_node_baseline = get_cob_brok_from_nodes(
        nodes=init_set
    )  # |metric(C) - metric(M)|

    C_minus_G_init = ALL_NODES.copy()
    for n in init_set:
        C_minus_G_init.remove(n)

    for step in range(NB_SETS):

        C_minus_G = C_minus_G_init.copy()
        G = init_set.copy()

        old_diff = all_node_baseline

        for iter in range(NB_ITER):

            to_test = random.sample(C_minus_G, min(NODES_PER_STEP, len(C_minus_G)))

            results = []
            for node in to_test:  # check wich heads in to_test causes the biggest drop
                G_plus_node = G.copy()
                G_plus_node.append(node)
                results.append(get_cob_brok_from_nodes(G_plus_node))

            best_node_idx = np.argmax(results)
            max_diff = results[best_node_idx]
            if max_diff > old_diff:
                best_node = to_test[best_node_idx]
                C_minus_G.remove(best_node)  # we remove the best node from the circuit
                G.append(best_node)
                old_diff = max_diff

                all_sets.append(
                    {"circuit_nodes": C_minus_G.copy(), "removed_nodes": G.copy()}
                )
                if verbose:
                    print(
                        f"iter: {iter} - best node:{best_node} - max brok cob diff:{max(results)} - baseline:{all_node_baseline}"
                    )
                    print_gpu_mem(f"iter {iter}")
        all_sets.append({"circuit_nodes": C_minus_G.copy(), "removed_nodes": G.copy()})
        if save_to_file:
            with open(
                f"jsons/greed_search_max_brok_cob_diff_{step}_{ctime()}.json", "w"
            ) as f:
                json.dump(all_sets, f)
    return all_sets


#%% [markdown]
# # <h1><b>Setup</b></h1>
# Import model and dataset
#%% # plot writing in the IO - S direction

if __name__ == "__main__":

    model_name = "gpt2"  # Here we used gpt-2 small ("gpt2")

    print_gpu_mem("About to load model")
    # model = EasyTransformer(
    #     model_name, use_attn_result=True
    # )  # use_attn_result adds a hook blocks.{lay}.attn.hook_result that is before adding the biais of the attention layer
    device = "cuda"
    if torch.cuda.is_available():
        model.to(device)
    print_gpu_mem("Gpt2 loaded")

    # IOI Dataset initialisation
    N = 150
    ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
    abca_dataset = ioi_dataset.gen_flipped_prompts(("S2", "RAND"))
    print("CIRCUIT STUDIED : ", CIRCUIT)

cde_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)

mean_dataset = cde_dataset

# %%
# webtext = load_dataset("stas/openwebtext-10k")
# owb_seqs = [
#     "".join(show_tokens(webtext["train"]["text"][i][:2000], model, return_list=True)[: ioi_dataset.max_len])
#     for i in range(ioi_dataset.N)
# ]

run_original = True

if __name__ != "__main__":
    run_original = False
#%% [markdown] Do some faithfulness
model.reset_hooks()
logit_diff_M = logit_diff(model, ioi_dataset)
print(f"logit_diff_M: {logit_diff_M}")

for circuit in [CIRCUIT.copy(), ALEX_NAIVE.copy()]:
    all_nodes = get_all_nodes(circuit)
    heads_to_keep = get_heads_circuit(ioi_dataset, excluded=[], circuit=circuit)
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=mean_dataset,
    )

    logit_diff_circuit = logit_diff(model, ioi_dataset)
    print(f"{logit_diff_circuit}")
# %% [markdown] select CIRCUIT or ALEX_NAIVE in otder to choose between the two circuits studied in the paper. Look at the `perf_by_sets.append` line to see how the results are saved
circuit = deepcopy(ALEX_NAIVE)
print("Working with", circuit)
cur_metric = logit_diff

run_original = True
print("Are we running the original experiment?", run_original)

if run_original:
    circuit_perf = []
    perf_by_sets = []
    for G in tqdm(list(circuit.keys()) + ["none"]):
        if G == "ablation":
            continue
        print_gpu_mem(G)
        # compute METRIC( C \ G )
        # excluded_classes = ["negative"]
        excluded_classes = []
        if G != "none":
            excluded_classes.append(G)
        heads_to_keep = get_heads_circuit(
            ioi_dataset, excluded=excluded_classes, circuit=circuit
        )  # TODO check the MLP stuff
        model.reset_hooks()
        model, _ = do_circuit_extraction(
            model=model,
            heads_to_keep=heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            mean_dataset=mean_dataset,
        )
        torch.cuda.empty_cache()
        cur_metric_broken_circuit, std_broken_circuit = cur_metric(
            model, ioi_dataset, std=True, all=True
        )
        torch.cuda.empty_cache()
        # metric(C\G)
        # adding back the whole model

        excl_class = list(circuit.keys())
        if G != "none":
            excl_class.remove(G)
        G_heads_to_remove = get_heads_circuit(
            ioi_dataset, excluded=excl_class, circuit=circuit
        )  # TODO check the MLP stuff
        torch.cuda.empty_cache()

        model.reset_hooks()
        model, _ = do_circuit_extraction(
            model=model,
            heads_to_remove=G_heads_to_remove,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            mean_dataset=mean_dataset,
        )

        torch.cuda.empty_cache()
        cur_metric_cobble, std_cobble_circuit = cur_metric(
            model, ioi_dataset, std=True, all=True
        )
        print(cur_metric_cobble.mean(), cur_metric_broken_circuit.mean())
        torch.cuda.empty_cache()

        # metric(M\G)
        on_diagonals = []
        off_diagonals = []
        for i in range(len(cur_metric_cobble)):
            circuit_perf.append(
                {
                    "removed_set_id": G,
                    "ldiff_broken": float(cur_metric_broken_circuit[i].cpu().numpy()),
                    "ldiff_cobble": float(cur_metric_cobble[i].cpu().numpy()),
                    "sentence": ioi_dataset.sentences[i],
                    "template": ioi_dataset.templates_by_prompt[i],
                }
            )

            x, y = basis_change(
                circuit_perf[-1]["ldiff_broken"],
                circuit_perf[-1]["ldiff_cobble"],
            )
            circuit_perf[-1]["on_diagonal"] = x
            circuit_perf[-1]["off_diagonal"] = y
            on_diagonals.append(x)
            off_diagonals.append(y)

        perf_by_sets.append(
            {
                "removed_group": G,
                "mean_cur_metric_broken": cur_metric_broken_circuit.mean(),
                "mean_cur_metric_cobble": cur_metric_cobble.mean(),
                "std_cur_metric_broken": cur_metric_broken_circuit.std(),
                "std_cur_metric_cobble": cur_metric_cobble.std(),
                "on_diagonal": np.mean(on_diagonals),
                "off_diagonal": np.mean(off_diagonals),
                "std_on_diagonal": np.std(on_diagonals),
                "std_off_diagonal": np.std(off_diagonals),
                "color": CLASS_COLORS[G],
                "symbol": "diamond-x",
            }
        )

        perf_by_sets[-1]["mean_abs_diff"] = abs(
            perf_by_sets[-1]["mean_cur_metric_broken"]
            - perf_by_sets[-1]["mean_cur_metric_cobble"]
        ).mean()

    df_circuit_perf = pd.DataFrame(circuit_perf)
    circuit_classes = sorted(perf_by_sets, key=lambda x: -x["mean_abs_diff"])
    df_perf_by_sets = pd.DataFrame(perf_by_sets)


with open(f"sets/perf_by_classes_{ctime()}.json", "w") as f:
    json.dump(circuit_perf, f)

#%% [markdown] UH SKIP THIS IF YA WANT TO HAVE GOOD PLOT? Load in a .csv file or .json file; this preprocesses things in the rough format of Alex's files, see the last "if" for what happens to the additions to perf_by_sets

if False:

    def get_df_from_csv(fname):
        df = pd.read_csv(fname)
        return df

    def get_list_of_dicts_from_df(df):
        return [dict(x) for x in df.to_dict("records")]

    def read_json_from_file(fname):
        with open(fname) as f:
            return json.load(f)

    perf_by_sets = []

    circuit_to_import = "natural"

    fnames = [
        # f"sets/greedy_circuit_perf_{circuit_to_import}_circuit_rd_Search.json",
        # f"sets/greedy_circuit_perf_{circuit_to_import}_circuit_max_brok_cob_diff.json",
        f"sets/perf_{circuit_to_import}_circuit_by_classes.json",
    ]

    sets_type = ["random_search", "greedy", "class"]
    for k, fname in enumerate(fnames):
        if fname[-4:] == ".csv":
            dat = get_list_of_dicts_from_df(get_df_from_csv(fname))
            avg_things = {"Empty set": {"mean_ldiff_broken": 0, "mean_ldiff_cobble": 0}}
            for i in range(1, 62):
                avg_things[f"Set {i}"] = deepcopy(avg_things["Empty set"])
            for x in dat:
                avg_things[x["removed_set_id"]]["mean_ldiff_broken"] += x[
                    "ldiff_broken"
                ]
                avg_things[x["removed_set_id"]]["mean_ldiff_cobble"] += x[
                    "ldiff_cobble"
                ]
            for x in avg_things.keys():
                avg_things[x]["mean_ldiff_broken"] /= 150
                avg_things[x]["mean_ldiff_cobble"] /= 150
            avg_things.pop("Empty set")

        elif fname[-5:] == ".json":
            dat = read_json_from_file(fname)
            avg_things = {}
            for x in dat:
                if not x["removed_set_id"] in avg_things:
                    avg_things[x["removed_set_id"]] = {
                        "mean_ldiff_broken": 0,
                        "mean_ldiff_cobble": 0,
                        "mean_dist": 0,
                    }

                avg_things[x["removed_set_id"]]["mean_ldiff_broken"] += x[
                    "ldiff_broken"
                ]
                avg_things[x["removed_set_id"]]["mean_ldiff_cobble"] += x[
                    "ldiff_cobble"
                ]
            for x in avg_things.keys():
                avg_things[x]["mean_ldiff_broken"] /= 150
                avg_things[x]["mean_ldiff_cobble"] /= 150
                avg_things[x]["mean_dist"] = np.abs(
                    avg_things[x]["mean_ldiff_broken"]
                    - avg_things[x]["mean_ldiff_cobble"]
                )
        else:
            raise ValueError("Unknown file type")

        all_dists = [avg_things[x]["mean_dist"] for x in avg_things.keys()]
        print(f"Max dist {circuit_to_import} - {sets_type[k]}: {max(all_dists)}")

        nb_set = 0
        for x, y in avg_things.items():
            new_y = deepcopy(y)
            new_y["removed_group"] = x
            new_y["symbol"] = "arrow-bar-left"
            if x == "Empty set":
                new_y["color"] = "red"
                new_y["name"] = "Empty set"
            else:
                if sets_type[k] == "random_search":
                    new_y["color"] = "green"
                    new_y["name"] = "Random set"
                elif sets_type[k] == "greedy":
                    new_y["color"] = "blue"
                    new_y["name"] = "Greedy search set"
                    new_y["symbol"] = "square"
                elif sets_type[k] == "class":
                    new_y["color"] = CLASS_COLORS[x]
                    new_y["name"] = x
                    new_y["symbol"] = "circle"
            new_y["mean_cur_metric_broken"] = new_y.pop("mean_ldiff_broken")
            new_y["mean_cur_metric_cobble"] = new_y.pop("mean_ldiff_cobble")

            if (
                (nb_set - 1 not in [0, 5, 3, 6, 23])
                and sets_type[k] == "greedy"
                and circuit_to_import == "natural"
            ):
                nb_set += 1
                continue

            if x == "Empty set":
                nb_set += 1
                continue
            perf_by_sets.append(new_y)
            nb_set += 1
#%% [markdown] make the figure

fig = go.Figure()

## add the grey region
# make the dotted line
minx = -2
maxx = 6
eps = 1.0
xs = np.linspace(minx - 1, maxx + 1, 100)
ys = xs

fig.add_trace(
    go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        name=f"x=y",
        line=dict(color="grey", width=2, dash="dash"),
    )
)

rd_set_added = False
for i, perf in enumerate(perf_by_sets):
    fig.add_trace(
        go.Scatter(
            x=[perf["mean_cur_metric_broken"]],
            y=[perf["mean_cur_metric_cobble"]],
            mode="markers",
            name=perf[
                "removed_group"  # change to "name" or something for the greedy sets
            ],
            marker=dict(symbol=perf["symbol"], size=10, color=perf["color"]),
            showlegend=(
                (" 1" in perf["removed_group"][-2:])
                or ("Set" not in perf["removed_group"])
            ),
        )
    )
    continue


# fig.update_layout(showlegend=False) #

fig.update_xaxes(title_text="F(C \ K)")
fig.update_yaxes(title_text="F(M \ K)")
fig.update_xaxes(showgrid=True, gridcolor="black", gridwidth=1)
fig.update_yaxes(showgrid=True, gridcolor="black", gridwidth=1)
fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")

# USE THESE LINES TO SCALE SVGS PROPERLY
fig.update_xaxes(range=[minx, maxx])
fig.update_yaxes(range=[minx, maxx])
fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")

fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)
import os

circuit_to_export = "natural"
fpath = f"circuit_completeness_{circuit_to_export}_CIRCUIT_at_{ctime()}.svg"
if os.path.exists(
    "/home/ubuntu/my_env/lib/python3.9/site-packages/easy_transformer/svgs"
):
    fpath = "svgs/" + fpath

fig.write_image(fpath)
fig.show()

#%% [markdown]
# We begin by gathering all the hooks we need
# NOTE THESE HOOOKS ARE TOTALLY IOI DATASET DEPENDENT
# AND CIRCUIT DEPENDENT

do_asserts = False

for doover in range(int(1e9)):
    for raw_circuit_idx, raw_circuit in enumerate([CIRCUIT, ALEX_NAIVE]):

        if doover == 0 and raw_circuit_idx == 0:
            print("Starting with the NAIVE!")
            continue

        circuit = deepcopy(raw_circuit)
        all_nodes = get_all_nodes(circuit)
        all_circuit_nodes = [head[0] for head in all_nodes]
        circuit_size = len(all_circuit_nodes)

        complement_hooks = (
            do_circuit_extraction(  # these are the default edit-all things
                model=model,
                heads_to_keep={},
                mlps_to_remove={},
                ioi_dataset=ioi_dataset,
                mean_dataset=mean_dataset,
                return_hooks=True,
                hooks_dict=True,
            )
        )

        assert len(complement_hooks) == 144

        heads_to_keep = get_heads_from_nodes(all_nodes, ioi_dataset)
        assert len(heads_to_keep) == circuit_size, (
            len(heads_to_keep),
            circuit_size,
        )

        circuit_hooks = do_circuit_extraction(
            model=model,
            heads_to_keep=heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            mean_dataset=mean_dataset,
            return_hooks=True,
            hooks_dict=True,
        )

        model_rem_hooks = do_circuit_extraction(
            model=model,
            heads_to_remove=heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            mean_dataset=mean_dataset,
            return_hooks=True,
            hooks_dict=True,
        )

        circuit_hooks_keys = list(circuit_hooks.keys())

        for layer, head_idx in circuit_hooks_keys:
            if (layer, head_idx) not in heads_to_keep.keys():
                circuit_hooks.pop((layer, head_idx))
        assert len(circuit_hooks) == circuit_size, (len(circuit_hooks), circuit_size)

        # [markdown] needed functions ...

        def cobble_eval(model, nodes):
            """Eval M\nodes"""
            model.reset_hooks()
            for head in nodes:
                model.add_hook(*model_rem_hooks[head])
            cur_logit_diff = logit_diff(model, ioi_dataset)
            model.reset_hooks()
            return cur_logit_diff

        def circuit_eval(model, nodes):
            """Eval C\nodes"""
            model.reset_hooks()
            for head in all_circuit_nodes:
                if head not in nodes:
                    model.add_hook(*circuit_hooks[head])
            for head in complement_hooks:
                if head not in all_circuit_nodes or head in nodes:
                    model.add_hook(*complement_hooks[head])
            cur_logit_diff = logit_diff(model, ioi_dataset)
            model.reset_hooks()
            return cur_logit_diff

        def difference_eval(model, nodes):
            """Eval completeness metric | F(C\nodes) - F(M\nodes) |"""
            c = circuit_eval(model, nodes)
            m = cobble_eval(model, nodes)
            return torch.abs(c - m)

        # actual experiments

        if do_asserts:
            c = circuit_eval(model, [])
            m = cobble_eval(model, [])
            print(f"{c}, {m} {torch.abs(c-m)}")

            for entry in perf_by_sets:  # check backwards compatibility
                circuit_class = entry["removed_group"]  # includes "none"
                assert circuit_class in list(circuit.keys()) + ["none"], circuit_class
                nodes = (
                    circuit[circuit_class] if circuit_class in circuit.keys() else []
                )

                c = circuit_eval(model, nodes)
                m = cobble_eval(model, nodes)

                assert torch.allclose(entry["mean_cur_metric_cobble"], m), (
                    entry["mean_cur_metric_cobble"],
                    m,
                    circuit_class,
                )
                assert torch.allclose(entry["mean_cur_metric_broken"], c), (
                    entry["mean_cur_metric_broken"],
                    c,
                    circuit_class,
                )

                print(f"{circuit_class} {c}, {m} {torch.abs(c-m)}")

        # [markdown] now do the greedy set experiments

        def add_key_to_json_dict(fname, key, value):
            """Thanks copilot"""
            with open(fname, "r") as f:
                d = json.load(f)
            d[key] = value
            with open(fname, "w") as f:
                json.dump(d, f)

        def new_greedy_search(
            no_runs,
            no_iters,
            no_samples,
            save_to_file=True,
            verbose=True,
        ):
            """
            Greedy search to find G that maximizes the difference between broken and cobbled circuit: |metric(C\G) - metric(M\G)|
            """
            all_sets = [{"circuit_nodes": [], "removed_nodes": []}]  # not mantained
            C_minus_G_init = deepcopy(all_nodes)
            C_minus_G_init = [head[0] for head in C_minus_G_init]

            c = circuit_eval(model, [])
            m = cobble_eval(model, [])
            baseline = torch.abs(c - m)

            metadata = {
                "no_runs": no_runs,
                "no_iters": no_iters,
                "no_samples": no_samples,
            }
            fname = (
                f"jsons/greedy_search_results_{raw_circuit_idx}_{doover}_{ctime()}.json"
            )
            print(fname)

            # write to JSON file
            if save_to_file:
                with open(
                    fname,
                    "w",
                ) as outfile:
                    json.dump(metadata, outfile)

            for run in tqdm(range(no_runs)):
                C_minus_G = deepcopy(C_minus_G_init)
                G = []
                old_diff = baseline.clone()

                for iter in range(no_iters):
                    print("iter", iter)
                    to_test = random.sample(C_minus_G, min(no_samples, len(C_minus_G)))
                    # sample without replacement

                    cevals = []
                    mevals = []

                    results = []
                    for (
                        node
                    ) in (
                        to_test
                    ):  # check which heads in to_test causes the biggest drop
                        G_plus_node = deepcopy(G) + [node]

                        cevals.append(circuit_eval(model, G_plus_node).item())
                        mevals.append(cobble_eval(model, G_plus_node).item())
                        results.append(abs(cevals[-1] - mevals[-1]))

                    best_node_idx = np.argmax(results)
                    max_diff = results[best_node_idx]
                    if max_diff > old_diff:
                        best_node = to_test[best_node_idx]
                        C_minus_G.remove(
                            best_node
                        )  # we remove the best node from the circuit
                        G.append(best_node)
                        old_diff = max_diff

                        all_sets.append(
                            {
                                "circuit_nodes": deepcopy(C_minus_G),
                                "removed_nodes": deepcopy(G),
                                "ceval": cevals[best_node_idx],
                                "meval": mevals[best_node_idx],
                            }
                        )
                        if verbose:
                            print(
                                f"iter: {iter} - best node:{best_node} - max brok cob diff:{max(results)} - baseline:{baseline}"
                            )
                            print_gpu_mem(f"iter {iter}")

                run_results = {
                    "result": old_diff,
                    "best set": all_sets[-1]["removed_nodes"],
                    "ceval": all_sets[-1]["ceval"],
                    "meval": all_sets[-1]["meval"],
                }

                if save_to_file:
                    add_key_to_json_dict(fname, f"run {run}", run_results)

        new_greedy_search(
            no_runs=10,
            no_iters=10,
            no_samples=10 if circuit_size == 26 else 5,
            save_to_file=True,
            verbose=True,
        )
#%% [markdown] do random search too

mode = "naive"
if mode == "naive":
    circuit = deepcopy(ALEX_NAIVE)
else:
    circuit = deepcopy(CIRCUIT)
all_nodes = get_all_nodes(circuit)

xs = []
ys = []

for _ in range(100):
    indicator = torch.randint(0, 2, (len(all_nodes),))
    nodes = [node[0] for node, ind in zip(all_nodes, indicator) if ind == 1]
    c = circuit_eval(model, nodes)
    m = cobble_eval(model, nodes)
    print(f"{c}, {m} {torch.abs(c-m)}")

    xs.append(c)
    ys.append(m)

torch.save(xs, f"pts/{mode}_random_xs.pt")
torch.save(ys, f"pts/{mode}_random_ys.pt")
# %% gready circuit breaking
def get_heads_from_nodes(nodes, ioi_dataset):
    heads_to_keep_tok = {}
    for h, t in nodes:
        if h not in heads_to_keep_tok:
            heads_to_keep_tok[h] = []
        if t not in heads_to_keep_tok[h]:
            heads_to_keep_tok[h].append(t)

    heads_to_keep = {}
    for h in heads_to_keep_tok:
        heads_to_keep[h] = get_extracted_idx(heads_to_keep_tok[h], ioi_dataset)

    return heads_to_keep


def circuit_from_nodes_logit_diff(model, ioi_dataset, nodes):
    """Take a list of nodes, return the logit diff of the circuit described by the nodes"""
    assert False  # I don't want to be redefining ALL_NODES
    heads_to_keep = get_heads_from_nodes(nodes, ioi_dataset)
    # print(heads_to_keep)
    model.reset_hooks()
    small_N = 40
    small_ioi_dataset = IOIDataset(
        prompt_type="mixed", N=small_N, tokenizer=model.tokenizer, nb_templates=2
    )
    small_cde_dataset = (
        small_ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
        .gen_flipped_prompts(("S", "RAND"))
        .gen_flipped_prompts(("S1", "RAND"), manual_word_idx=ioi_dataset.word_idx)
    )

    circuit_to_study = "natural_circuit"

    assert circuit_to_study in ["auto_search", "natural_circuit", "naive_circuit"]

    ALL_NODES_AUTO_SEARCH = [
        ((4, 0), "IO"),
        ((1, 5), "S+1"),
        ((6, 8), "S"),
        ((10, 6), "IO"),
        ((10, 10), "end"),
        ((8, 10), "end"),
        ((9, 2), "S+1"),
        ((5, 3), "and"),
        ((2, 10), "S2"),
        ((10, 4), "S2"),
        ((0, 9), "S"),
        ((7, 8), "S"),
        ((1, 8), "and"),
        ((2, 7), "S2"),
        ((1, 5), "end"),
        ((8, 7), "end"),
        ((7, 0), "S+1"),
    ]

    if circuit_to_study == "auto_search":
        ALL_NODES = ALL_NODES_AUTO_SEARCH.copy()
    elif circuit_to_study == "natural_circuit":
        ALL_NODES = []  # a node is a tuple (head, token)
        for h in RELEVANT_TOKENS:
            for tok in RELEVANT_TOKENS[h]:
                ALL_NODES.append((h, tok))
    elif circuit_to_study == "naive_circuit":
        CIRCUIT = {
            "name mover": [(9, 6), (9, 9), (10, 0)],
            "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
            "induction": [(5, 5), (5, 9)],
            "duplicate token": [(3, 0), (0, 10)],
            "previous token": [(2, 2), (4, 11)],
            "negative": [],
        }
        ALL_NODES = []
        RELEVANT_TOKENS = {}
        for head in (
            CIRCUIT["name mover"] + CIRCUIT["negative"] + CIRCUIT["s2 inhibition"]
        ):
            RELEVANT_TOKENS[head] = ["end"]

        for head in CIRCUIT["induction"]:
            RELEVANT_TOKENS[head] = ["S2"]

        for head in CIRCUIT["duplicate token"]:
            RELEVANT_TOKENS[head] = ["S2"]

        for head in CIRCUIT["previous token"]:
            RELEVANT_TOKENS[head] = ["S+1"]
        ALL_NODES = []  # a node is a tuple (head, token)
        for h in RELEVANT_TOKENS:
            for tok in RELEVANT_TOKENS[h]:
                ALL_NODES.append((h, tok))


def circuit_from_heads_logit_diff(
    model, ioi_dataset, mean_dataset, heads_to_rmv=None, heads_to_kp=None, all=False
):
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_kp,
        heads_to_remove=heads_to_rmv,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=mean_dataset,
    )
    return logit_diff(model, ioi_dataset, all=all)


# %% Run experiment


def logit_diff_from_nodes(model, ioi_dataset, mean_dataset, nodes):
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=get_heads_from_nodes(nodes, ioi_dataset),
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=mean_dataset,
    )
    ldiff_broken = logit_diff(model, ioi_dataset, all=False)  # Metric(C\nodes)
    return ldiff_broken


def compute_cobble_broken_diff(
    model,
    ioi_dataset,
    mean_dataset,
    nodes,
    return_both=False,
    all_node=None,  # TODO add this
):  # red teaming the circuit by trying
    """Compute |Metric(C\ nodes) - Metric(M\ nodes)|"""
    if all_node is None:
        nodes_to_keep = ALL_NODES.copy()
    else:
        nodes_to_keep = all_node.copy()

    for n in nodes:
        try:
            nodes_to_keep.remove(n)  # C\nodes
        except:
            print(n)
            raise ValueError("Node not in all nodes")
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=get_heads_from_nodes(nodes_to_keep, ioi_dataset),
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=mean_dataset,
    )
    ldiff_broken = logit_diff(model, ioi_dataset, all=False)  # Metric(M\nodes)

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_remove=get_heads_from_nodes(nodes, ioi_dataset),  # M\nodes
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=mean_dataset,
    )
    ldiff_cobble = logit_diff(model, ioi_dataset, all=False)  # Metric(C\nodes)

    print(f"ldiff_broken: {ldiff_broken}")
    print(f"ldiff_cobble: {ldiff_cobble}")

    if return_both:
        return ldiff_broken, ldiff_cobble
    else:
        return np.abs(ldiff_broken - ldiff_cobble)


#%% [markdown] hopefully ignoarable plottig proceessin

# style of file
"""
{"no_runs": 10, "no_iters": 10, "no_samples": 5, "run 0": {"result": 2.9001526832580566, "best set": [[0, 10], [2, 2], [8, 10], [4, 11]], "ceval": -1.0231037139892578, "meval": 1.8770489692687988}}
"""

assert os.getcwd().endswith("Easy-Transformer"), os.getcwd
fnames = os.listdir("jsons")
fnames = [fname for fname in fnames if "greedy_search_results" in fname]

xs = [[], []]
ys = [[], []]
names = []

for circuit_idx in range(0, 2):  # 0 is our circuit, 1 is naive
    for fname in fnames:
        with open(f"jsons/{fname}", "r") as f:
            data = json.load(f)
        for idx in range(100):
            key = f"run {idx}"
            if key in data:
                if (
                    f"results_{circuit_idx}" in fname and "ceval" in data[key]
                ):  # our circuit, not naive
                    xs[circuit_idx].append(data[key]["ceval"])
                    ys[circuit_idx].append(data[key]["meval"])
                    names.append(
                        str(data[key]["best set"]) + " " + str(data[key]["result"])
                    )

                else:
                    pass
#%%

mode = "complete"
# mode = "naive"

fig = go.Figure()
## add the grey region
# make the dotted line
minx = -2
maxx = 6
eps = 1.0
xs = np.linspace(minx - 1, maxx + 1, 100)
ys = xs

fig.add_trace(
    go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        name=f"x=y",
        line=dict(color="grey", width=2, dash="dash"),
    )
)

perf_by_sets = torch.load(
    f"pts/{mode}_perf_by_sets.pt"
)  # see the format of this file to overwrite plots

rd_set_added = False
for i, perf in enumerate(perf_by_sets):
    fig.add_trace(
        go.Scatter(
            x=[perf["mean_cur_metric_broken"]],
            y=[perf["mean_cur_metric_cobble"]],
            mode="markers",
            name=perf["removed_group"],
            marker=dict(symbol="circle", size=10, color=perf["color"]),
            showlegend=(
                (" 1" in perf["removed_group"][-2:])
                or ("Set" not in perf["removed_group"])
            ),
        )
    )
    continue

# add the greedy
greedy_xs = torch.load(f"pts/{mode}_xs.pt")[:30]
greedy_ys = torch.load(f"pts/{mode}_ys.pt")[:30]

fig.add_trace(
    go.Scatter(
        x=greedy_xs,
        y=greedy_ys,
        mode="markers",
        name="Greedy",
        marker=dict(symbol="square", size=6, color="blue"),
    )
)

# add the random
random_xs = torch.load(f"pts/{mode}_random_xs.pt")  # [:10]
random_ys = torch.load(f"pts/{mode}_random_ys.pt")  # [:10]

fig.add_trace(
    go.Scatter(
        x=random_xs,
        y=random_ys,
        mode="markers",
        name="Random",
        marker=dict(symbol="triangle-left", size=10, color="green"),
    )
)

# fig.update_layout(showlegend=False)
fig.update_xaxes(title_text="F(C \ K)")
fig.update_yaxes(title_text="F(M \ K)")
fig.update_xaxes(showgrid=True, gridcolor="black", gridwidth=1)
fig.update_yaxes(showgrid=True, gridcolor="black", gridwidth=1)
fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")

# USE THESE LINES TO SCALE SVGS PROPERLY
fig.update_xaxes(range=[-1, 6])
fig.update_yaxes(range=[-1, 6])
fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")

fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)

circuit_to_export = "natural"
fpath = f"circuit_completeness_{circuit_to_export}_CIRCUIT_at_{ctime()}.svg"
if os.path.exists(
    "/home/ubuntu/my_env/lib/python3.9/site-packages/easy_transformer/svgs"
):
    fpath = "svgs/" + fpath

fig.write_image(fpath)
fig.show()
