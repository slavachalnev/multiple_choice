# %%
import torch
import numpy as np
import random
from transformer_lens import HookedTransformer, ActivationCache
import transformer_lens.utils as utils
import plotly.express as px

torch.set_grad_enabled(False)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# %%
model = HookedTransformer.from_pretrained('gemma-7b', dtype=torch.float16)
print(model)

# %%
out = model.generate("Hi, my name is")
print(out)

# %%

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab


# %%
prelude = "A highly knowledgeable and intelligent AI answers multiple-choice questions about Biology. "
question = "To prevent desiccation and injury, the embryos of terrestrial vertebrates are encased within a fluid secreted by the:"
answers = """
(A) amnion
(B) chorion
(C) allantois
(D) yolk sac
"""
post_text = "Answer: ("

answer_token = "A"

text = prelude + question + answers + post_text
print(model.to_str_tokens(text))



# %%

utils.test_prompt(text, answer_token, model, prepend_space_to_answer=False)


# %%
# DLA

logits, cache = model.run_with_cache(text)
cache: ActivationCache



# %%

decomposed, labels = cache.decompose_resid(layer=-1, return_labels=True, pos_slice=-1)
stacked_mlps = decomposed[2::2, :, :]
labels = labels[2::2]

stacked = cache.stack_head_results(layer=-1, pos_slice=-1)


# %%
print(decomposed.shape)
print(stacked.shape)
print('n_heads', n_heads)

# %%

print(labels)
# %%

dla = cache.logit_attrs(torch.cat([stacked_mlps, stacked]), answer_token)

# %%

fig = px.line(dla.cpu())
fig.show()


# %%
