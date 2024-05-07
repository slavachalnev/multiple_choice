# %%
from transformer_lens import HookedTransformer


# %%
model = HookedTransformer.from_pretrained('gemma-7b')

print(model)


# %%
