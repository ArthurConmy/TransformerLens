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
    num_examples=10, 
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

#%%

def setter_hook(z, hook, head_idx, new_val):
    assert len(z.shape)==4
    assert z.shape[2]==8 # n heads
    assert list(z.shape[:2]) + list(z.shape[3:]) == list(new_val.shape), (z.shape, new_val.shape)
    z[:, :, head_idx] = new_val
    return z

for head_indices in [[0, 1, 2, 3, 4, 6]]:
# [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6], [0, 1, 2, 3, 4, 5, 6]]:
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

        logits = model(validation_data)
        val = kl_divergence(
            logits,
        )
        print(
            f"{head_indices=} {ablation_type=}: {val.item()}"
        )
# %%
