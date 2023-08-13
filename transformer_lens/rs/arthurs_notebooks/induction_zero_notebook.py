#%%

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.induction_zero_utils import *

# %%

torch.manual_seed(2794979691)
(
    model,
    validation_data,
    validation_patch_data, 
    kl_divergence,
    negative_log_probs,
) = get_all_induction_things(
    num_examples=50, 
    seq_len=300, 
    device="cuda:0", 
    data_seed=42, 
    return_one_element=True, 
)

#%%

model.reset_hooks()
cached_value = model.run_with_cache(
    validation_patch_data, 
    names_filter = lambda name: name==get_act_name("result", 1),
)[1][get_act_name("result", 1)]
cached_ln = model.run_with_cache(
    validation_data,
    names_filter = lambda name: name == "ln_final.hook_scale",
)[1]["ln_final.hook_scale"]

#%%

def setter_hook(z, hook, head_idx, new_val):
    assert len(z.shape)==4
    assert z.shape[2]==8 # n heads
    assert list(z.shape[:2]) + list(z.shape[3:]) == list(new_val.shape), (z.shape, new_val.shape)
    z[:, :, head_idx] = new_val
    return z

def freeze_setter_hook(z, hook, new_val):
    assert z.shape == new_val.shape
    z[:] = new_val
    return z

FREEZE_LN = True

keys = []
xvals = []
yvals = []

for freeze_ln in [True, False]:
    for head_indices in [[5], [6], [5, 6]]:
        for ablation_type in ["random", "zero"]:

            model.reset_hooks()

            for head_idx in head_indices:
                patching_tensor = cached_value[:, :, head_idx].clone()

                if ablation_type == "zero":
                    patching_tensor = torch.zeros_like(patching_tensor)
                
                model.add_hook(
                    get_act_name("result", 1),
                    partial(
                        setter_hook,
                        head_idx = head_idx,
                        new_val = patching_tensor,
                    ),
                )

            if freeze_ln:
                model.add_hook(
                    "ln_final.hook_scale",
                    partial(
                        freeze_setter_hook,
                        new_val = cached_ln,
                    ),
                )

            logits = model(validation_data)

            val = kl_divergence(
                logits,
            )

            if ablation_type == "zero":
                xvals.append(val.item())
                keys.append("Ablating Induction Head " + str(head_indices[0]) + (" freezing LN" if freeze_ln else " recomputing LN"))
            if ablation_type == "random":
                yvals.append(val.item())

            print(
                f"{head_indices=} {ablation_type=}: {val.item()}"
            )
# %%

fig = go.Figure()
fig.add_scatter(
    x=xvals,
    y=yvals,
    mode="markers+text",
    text=keys,
)
fig.add_trace( # y = x
    go.Scatter(
        x=[-1, 3],
        y=[-1, 3],
        mode="lines",
    )
)

fig.update_layout(
    xaxis_title="KL Divergence When Zero Ablating",
    yaxis_title="KL Divergence When Resample Ablating", 
)

# %%
