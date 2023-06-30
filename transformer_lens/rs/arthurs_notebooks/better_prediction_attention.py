# %% [markdown] [4]:

"""
Cribbed from key_and_query_projection.py
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project, get_effective_embedding_2
from transformer_lens.rs.arthurs_notebooks.arthur_utils import *
import argparse

#%%

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)
model.set_use_attn_result(True)

# %%

MAX_SEQ_LEN = 512 # half of 1024 as
BATCH_SIZE = 30
batched_tokens, targets = get_filtered_webtext(model, batch_size=BATCH_SIZE, seed=1729, device="cuda", max_seq_len=MAX_SEQ_LEN)
effective_embeddings = get_effective_embedding_2(model)

# %%

# Find the top 5% of things by importance
# Do this crap
# See change in loss

NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
# NEGATIVE_HEAD_IDX, NEGATIVE_LAYER_IDX = 9, 9
# for NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX in [(10, 0), (10, 7), (9, 9), (11, 10)] + list(itertools.product(range(11, -1, -1), range(12))):

END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
names_filter1 = (
    lambda name: name == END_STATE_HOOK
    or name==f"blocks.{NEGATIVE_LAYER_IDX}.hook_resid_pre"
    or name==f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_result"
    or name==f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_attn_scores"
)
model = model.to("cuda:1")
logits, cache = model.run_with_cache(
    batched_tokens.to("cuda:1"),
    names_filter=names_filter1,
    device="cpu",
)
model = model.to("cuda:0")
cache.to("cuda:0")
print("Done")
cpu_logits = logits.cpu()
del logits
gc.collect()
torch.cuda.empty_cache()

# %%

original_end_state = cache[get_act_name("resid_post", model.cfg.n_layers-1)]

batched_tokens_loss = get_loss_from_end_state(
    model=model,
    end_state=original_end_state,
    targets=targets,
)

#%%

head_output = cache[get_act_name("result", NEGATIVE_LAYER_IDX)][:, :, NEGATIVE_HEAD_IDX]
assert head_output.shape == (BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model)

#%%

if ipython is not None:
    unembed = einops.einsum(
        head_output, 
        model.W_U,
        "b s d_model, d_model d_vocab -> b s d_vocab",
    )

#%% 

if ipython is not None:
    the_topk = torch.topk(
        -unembed,
        k=10,
        dim=-1,
    ).indices


#%%

mean_head_output = einops.reduce(head_output, "b s d -> d", reduction="mean")

#%%

mean_ablated_end_states = cache[get_act_name("resid_post", model.cfg.n_layers-1)] - head_output + einops.repeat(mean_head_output, "d -> b s d", b=BATCH_SIZE, s=MAX_SEQ_LEN)
mean_ablated_loss = get_loss_from_end_state(
    model=model,
    end_state=mean_ablated_end_states,
    targets=targets,
)

# %%

max_importance_examples = sorted(
    [
        (
            batch_idx,
            seq_idx,
            (mean_ablated_loss-batched_tokens_loss)[batch_idx, seq_idx].item(),
        )
        for batch_idx, seq_idx in itertools.product(
            range(BATCH_SIZE), range(MAX_SEQ_LEN)
        )
    ],
    key=lambda x: x[2],
    reverse=True,
)

# %%

# Get the top 5% of things by importance

TOP5P_BATCH_SIZE = len(max_importance_examples) // 20
all_top_5_percent = max_importance_examples[:TOP5P_BATCH_SIZE]
top5p_batch_indices = [x[0] for x in all_top_5_percent]
top5p_seq_indices = [x[1] for x in all_top_5_percent]

#%%

top5p_tokens = batched_tokens[top5p_batch_indices]
top5p_targets = torch.LongTensor([targets[top5p_batch_idx, top5p_seq_idx] for top5p_batch_idx, top5p_seq_idx in zip(top5p_batch_indices, top5p_seq_indices)])

#%%

top5p_losses = batched_tokens_loss[top5p_batch_indices, top5p_seq_indices]

# %%

# Do the key-side thing where we project onto W_U

keyside_projections = t.zeros((BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model))
keyside_orthogonals = t.zeros((BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model))

for batch_idx, seq_idx in tqdm(list(itertools.product(range(BATCH_SIZE), range(MAX_SEQ_LEN)))):
    keyside_vector, keyside_orthogonal = project(
        cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][batch_idx, seq_idx],
        effective_embeddings["W_E (including MLPs)"][batched_tokens[batch_idx, seq_idx]],
    )
    keyside_projections[batch_idx, seq_idx] = keyside_vector
    keyside_orthogonals[batch_idx, seq_idx] = keyside_orthogonal

#%% 

queryside_vectors = t.ones((TOP5P_BATCH_SIZE, model.cfg.d_model)).cuda() * (-420)
queryside_components = [] # t.ones((TOP5P_BATCH_SIZE, MAX_SEQ_LEN+1)) * (-420)
NORMY=False
for batch_batch_idx, (top5p_batch_idx, top5p_seq_idx) in tqdm(list(enumerate(list(zip(top5p_batch_indices, top5p_seq_indices))))):
    t.cuda.empty_cache()

    my_direction_indices = list(set([batched_tokens[top5p_batch_idx, earlier_seq_idx].item() for earlier_seq_idx in range(top5p_seq_idx+1)]))
    my_directions_lookup = {}
    for idx in range(top5p_seq_idx+1):
        for dir_idx, tok in enumerate(my_direction_indices):
            if batched_tokens[top5p_batch_idx, idx].item() == tok:
                assert idx not in my_directions_lookup
                my_directions_lookup[idx] = dir_idx
    assert len(my_directions_lookup) == top5p_seq_idx+1
    my_directions = [model.W_U.T[my_direction_idx] for my_direction_idx in my_direction_indices]

    queryside_vector, queryside_orthogonal, queryside_component = project(
        cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][top5p_batch_idx, top5p_seq_idx],
        dir=my_directions,
        return_component=True,
    )
    queryside_vectors[batch_batch_idx] = queryside_vector
    assert len(queryside_component) == len(my_direction_indices) # number of distinct tokens

    if NORMY:
        queryside_norms = [model.W_U.T[batched_tokens[top5p_batch_idx, earlier_seq_idx]].norm(dim=0).item() for earlier_seq_idx in range(top5p_seq_idx+1)]
        queryside_norms = torch.tensor(queryside_norms)
        assert queryside_component.shape == queryside_norms.shape
        queryside_components.append(queryside_component * queryside_norms.cuda())
    else:
        queryside_components.append([queryside_component[my_directions_lookup[idx]] for idx in range(top5p_seq_idx+1)])

    # warnings.warn("Another lock on")
    # queryside_vectors[batch_batch_idx] = model.W_U.T[top5p_tokens[batch_idx, seq_idx]]

#%%

all_queryside_norms = []
for batch_batch_idx, (top5p_batch_idx, top5p_seq_idx) in tqdm(list(enumerate(list(zip(top5p_batch_indices, top5p_seq_indices))))):
    queryside_norms = [model.W_U.T[batched_tokens[top5p_batch_idx, earlier_seq_idx]].norm(dim=0).item() for earlier_seq_idx in range(top5p_seq_idx+1)]
    queryside_components[batch_batch_idx] = torch.tensor(queryside_components[batch_batch_idx]) * torch.tensor(queryside_norms)
    # queryside_norms = torch.tensor(queryside_norms)
    # assert queryside_component.shape == queryside_norms.shape
    # queryside_components.append(queryside_component * queryside_norms.cuda())

#%%

fig = go.Figure()
colors = px.colors.qualitative.Plotly 
colors = colors * 10

for i in range(10):
    fig.add_trace(
        go.Scatter(
            x = (cache[get_act_name("attn_scores", NEGATIVE_LAYER_IDX)][top5p_batch_indices[i], NEGATIVE_HEAD_IDX, top5p_seq_indices[i], :top5p_seq_indices[i]+1]).cpu(),
            y = torch.tensor(queryside_components[i]).cpu(),
            mode="markers",
            text=[(j, model.to_str_tokens(batched_tokens[top5p_batch_indices[i], max(0,j-1):j+2])) for j in range(top5p_seq_indices[i]+1)],
            marker=dict(
                color=colors[i],
            ),
        )
    )
fig.show()

# %%
