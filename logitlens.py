import pandas as pd
import numpy as np
import torch
from fancy_einsum import einsum
from torch.nn.functional import kl_div

import plotly.graph_objects as go
import plotly.express as px


def kl_div(p, q):
    return torch.sum(p * torch.log(p / q), dim=-1)

def to_array(x):
    return x.type(torch.float32).detach().cpu().numpy()

# Create the heatmap
def plot_logit_lens(model, prompt, what='probs', component='resid_post', tok_id=None, cutoff=0):

    device = model.W_U.device

    tokens = model.to_tokens(prompt)
    layers = len(model.blocks)
    
    # Extract resid_post from cache
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.to(device))
    activ = torch.cat([cache[f'blocks.{l}.hook_{component}'].to(device) for l in range(layers)], dim=0) # [layers pos d_model]

    # Compute logits and probabilities [check apply_ln_to_stack()]
    logits = einsum("... d_model, d_model d_vocab -> ... d_vocab", model.ln_final(activ.to(model.W_U.device)), model.W_U)
    proba = (logits + model.b_U).softmax(-1).detach().type(torch.float32).cpu() # [layers pos d_vocab]

    if tok_id is None:
        tok_id = proba[-1, -1, :].argmax()
        print(f"Next token: {model.to_string(tok_id)}", tok_id)

    if what == 'probs':
        mx = proba.max(-1)
        z = mx.values
        token_ids = mx.indices
        text = np.vectorize(lambda x: model.tokenizer.decode([x]))(token_ids)

    if what == 'ranks':
        ranks = proba - torch.gather(proba, -1, torch.cat([tokens.cpu(), tok_id[None, None]], -1).repeat([layers, 1])[:, 1:, None])
        text = (ranks >= 0).sum(-1).cpu()
        z = torch.log(text)

    if what == 'kl':
        z = to_array(kl_div(proba[-1, ...], proba))
        text = np.round(z, 2)

    if what == 'angle':
        last_resid_post = cache[f'blocks.{layers-1}.hook_resid_post'].to(device) # [1 pos d_model]
        angles = torch.diagonal(torch.matmul(activ, last_resid_post.mT), 0, 1, 2) # [layers pos]
        norms = torch.norm(activ, dim=-1) * torch.norm(last_resid_post, dim=-1) # [layers pos]
        z = to_array(torch.acos(angles / norms) / torch.pi * 180)
        text = np.round(z, 2)

    if what == 'perplexity':
        z = 2 ** (- proba * torch.log2(proba)).sum(-1)
        text = np.round(z, 2)

    fig = go.Figure(data=go.Heatmap(
        z=z[:, 1+cutoff:],
        text=text[:, 1+cutoff:], 
        texttemplate="%{text}",
        showscale=True
    ))

    # Update layout if needed
    fig.update_layout(
        title=f"LogitLens - {what} - {component}",
        xaxis_title="Tokens",
        yaxis_title="Layers",
    )

    fig.update_xaxes(
        tickvals=list(range(len(tokens[0][1+cutoff:]))),
        ticktext=model.to_str_tokens(prompt)[1+cutoff:]
    )

    return fig


def top_tokens_lens(model, prompt, component='resid_post', k=5, **kwargs):
    tokens = model.to_tokens(prompt)
    layers = len(model.blocks)
    # Extract resid_post from cache
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    idxs = logits[0,-1, :].argsort(-1)[-k:].cpu()
    
    activ = torch.cat([cache[f'blocks.{l}.hook_{component}'].to(device) for l in range(layers)], dim=0)
    logits = einsum("... d_model, d_model d_vocab -> ... d_vocab", model.ln_final(activ.to(model.W_U.device)), model.W_U)
    proba = (logits + model.b_U).softmax(-1).detach().type(torch.float32).cpu()
    proba = to_array(proba[:, -1, idxs])
    
    df = pd.DataFrame(proba, columns=model.to_str_tokens(idxs)).reset_index()
    df = pd.melt(df, id_vars=['index'], value_vars=df.columns, var_name='token', value_name='proba')
    fig = px.line(df, x='index', y='proba', color='token')
    
    for i in range(k):
        fig.add_trace(go.Scatter(x=np.arange(layers), y=proba[:, i], mode='markers', marker_color=fig.data[i].line.color, showlegend=False))

    fig.update_layout(**kwargs)
    
    return fig