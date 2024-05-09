import numpy as np
import torch
from transformer_lens import HookedTransformer, ActivationCache, utils
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
import einops
from jaxtyping import Float
from typing_extensions import Literal
from torchtyping import TensorType as TT
from functools import partial
from IPython.display import HTML, Markdown
import pysvelte
from plotly.subplots import make_subplots
import ipywidgets as widgets

import plotly.express as px

def patching_hook(
    resid_pre: Float[torch.Tensor, "batch pos d_component"],
    hook: HookPoint,
    position: int,
    corrupted_cache: ActivationCache
) -> Float[torch.Tensor, "batch pos d_component"]:

    corrupted_resid_pre = corrupted_cache[hook.name]
    resid_pre[:, position, :] = corrupted_resid_pre[:, position, :]
    return resid_pre


class Patching:
    def __init__(self, model, how, metric=None) -> None:
        
        assert how in ['ap', 'atp', 'atp*'], f'Unknown patching method {how}'
        
        self.model = model
        self.how = how
        self.a_corr = None
        self.x_corr = None
        self.a_clean = None
        self.x_clean = None
        self.component = None

        self.head_names = [
            f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
        ]
        self.head_names_signed = [f"{name}{sign}" for name in self.head_names for sign in ["+", "-"]]
        self.head_names_qkv = [
            f"{name}{act_name}" for name in self.head_names for act_name in ["Q", "K", "V"]
        ]

        if metric is None:
            metric = self.logits_to_logit_diff
        self.metric = metric

    def patching(self, x_clean, a_clean, x_corr=None, a_corr=None, corr='adv', component='resid_pre'):
        
        assert corr in ['adv', 'zero', 'rand'], f'Unknown corruption method {corr}'

        try:
            utils.get_act_name("resid_pre", 0)
        except KeyError: 
            raise ValueError(f'Unknown component {component}')

        if corr == 'adv':
            assert x_corr is not None and a_corr is not None, 'Corrupted data must be provided for adversarial corruption'
        elif x_corr is not None or a_corr is not None:
            print('Warning: corrupted data provided but corruption method is not adversarial, ignoring corrupted data')

        self.x_clean = x_clean
        self.a_clean = a_clean
        self.a_corr = a_corr
        self.x_corr = x_corr
        self.component = component

        # Tokenization and caching
        self.clean_tokens = self.model.to_tokens(self.x_clean)
        clean_logits, clean_cache = self.model.run_with_cache(self.clean_tokens)
        self.clean_cache = clean_cache
        self.clean_logit_diff = self.metric(clean_logits)
        print(f"Clean logit difference: {self.clean_logit_diff.item():.3f}")

        if self.x_corr is not None:
            self.corrupted_tokens = self.model.to_tokens(self.x_corr)
            corrupted_logits, corrupted_cache = self.model.run_with_cache(self.corrupted_tokens)
            self.corrupted_cache = corrupted_cache
            self.corrupted_logit_diff = self.metric(corrupted_logits)
            print(f"Corrupted logit difference: {self.corrupted_logit_diff.item():.3f}")

        # Patching
        print('Patching...')
        if self.how == 'ap':
            self.patch_ap()
        elif self.how == 'atp':
            self.patch_atp()
        elif self.how == 'atp*':
            self.patch_atp_star()

    def logits_to_logit_diff(self, logits):
        a_clean_id = self.model.to_single_token(self.a_clean)
        a_corr_id = self.model.to_single_token(self.a_corr)
        return logits[0, -1, a_clean_id] - logits[0, -1, a_corr_id]
    
    def get_cache_fwd_and_bwd(self, tokens):
        filter_not_qkv_input = lambda name: "_input" not in name
        self.model.reset_hooks()
        cache = {}

        def forward_cache_hook(act, hook):
            cache[hook.name] = act.detach()

        self.model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")

        grad_cache = {}

        def backward_cache_hook(act, hook):
            grad_cache[hook.name] = act.detach()

        self.model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

        value = self.metric(self.model(tokens))
        value.backward()
        self.model.reset_hooks()
        return (
            value.item(),
            ActivationCache(cache, self.model),
            ActivationCache(grad_cache, self.model),
        )
    
    def stack_head_vector_from_cache(
        self, cache, activation_name: Literal["q", "k", "v", "z"]
    ) -> TT["layer_and_head_index", "batch", "pos", "d_head"]:
        """Stacks the head vectors from the cache from a specific activation (key, query, value or mixed_value (z)) into a single tensor."""
        stacked_head_vectors = torch.stack(
            [cache[activation_name, l] for l in range(self.model.cfg.n_layers)], dim=0
        )
        stacked_head_vectors = einops.rearrange(
            stacked_head_vectors,
            "layer batch pos head_index d_head -> (layer head_index) batch pos d_head",
        )
        return stacked_head_vectors
    
    def stack_head_pattern_from_cache(
        self, cache,
    ) -> TT["layer_and_head_index", "batch", "dest_pos", "src_pos"]:
        """Stacks the head patterns from the cache into a single tensor."""
        stacked_head_pattern = torch.stack(
            [cache["pattern", l] for l in range(self.model.cfg.n_layers)], dim=0
        )
        stacked_head_pattern = einops.rearrange(
            stacked_head_pattern,
            "layer batch head_index dest_pos src_pos -> (layer head_index) batch dest_pos src_pos",
        )
        return stacked_head_pattern

    @torch.no_grad()
    def patch_ap(self):
        
        num_positions = len(self.clean_tokens[0])
        patching_result = torch.zeros((self.model.cfg.n_layers, num_positions), device=self.model.cfg.device)

        for layer in tqdm(range(self.model.cfg.n_layers)):
            for position in range(num_positions):
                temp_hook_fn = partial(patching_hook, position=position, corrupted_cache=self.corrupted_cache)
                
                patched_logits = self.model.run_with_hooks(self.clean_tokens, fwd_hooks=[
                    (utils.get_act_name(self.component, layer), temp_hook_fn)
                ])

                patched_logit_diff = self.metric(patched_logits).detach()
                patching_result[layer, position] = (patched_logit_diff - self.clean_logit_diff)/(self.corrupted_logit_diff - self.clean_logit_diff)

    
    def patch_atp(self):
        
        _, corrupted_cache = self.model.run_with_cache(self.corrupted_tokens)

        _, clean_cache, clean_grad_cache = self.get_cache_fwd_and_bwd(self.clean_tokens)

        if self.component in ["resid_pre", "resid_post"]:
            corrupted_act, labels = corrupted_cache.accumulated_resid(-1, incl_mid=True, return_labels=True)
            clean_act = clean_cache.accumulated_resid(-1, incl_mid=True, return_labels=False)
            clean_grad_act = clean_grad_cache.accumulated_resid(-1, incl_mid=True, return_labels=False)
        elif self.component in ["attn_all"]:
            corrupted_act = corrupted_cache.stack_head_results(-1)
            clean_act = clean_cache.stack_head_results(-1)
            clean_grad_act = clean_grad_cache.stack_head_results(-1)
        elif self.component in ["attn_q", "attn_k", "attn_v", "attn_z"]:
            activation_name = self.component[-1]
            corrupted_act = self.stack_head_vector_from_cache(corrupted_cache, activation_name)
            clean_act = self.stack_head_vector_from_cache(clean_cache, activation_name)
            clean_grad_act = self.stack_head_vector_from_cache(clean_grad_cache, activation_name)
        elif self.component in ["attn_pattern"]:
            corrupted_act = self.stack_head_pattern_from_cache(corrupted_cache)
            clean_act = self.stack_head_pattern_from_cache(clean_cache)
            clean_grad_act = self.stack_head_pattern_from_cache(clean_grad_cache)
        
        if self.component in ["resid_pre", "resid_post", "attn_all", "attn_q", "attn_k", "attn_v", "attn_z"]:
            self.patch = einops.reduce(
                - clean_grad_act * (corrupted_act - clean_act),
                "component batch pos d_model -> component pos",
                "sum",
            )
        elif self.component in ["attn_pattern"]:
            self.patch = einops.reduce(
                - clean_grad_act * (corrupted_act - clean_act),
                "component batch dest_pos src_pos -> component dest_pos src_pos",
                "sum",
            )
            self.patch = einops.rearrange(
                self.patch,
                "(layer head) dest src -> layer head dest src",
                layer=self.model.cfg.n_layers,
                head=self.model.cfg.n_heads,
            )
        if "sum" in self.component:
            self.patch = einops.reduce(
                self.patch,
                "(layer head) pos -> layer head",
                "sum",
                layer=self.model.cfg.n_layers,
                head=self.model.cfg.n_heads,
            )

    ############
    # Plotting #
    ############
    def plot_single_pattern(self, layer, tokens, **kwargs):
        fig = make_subplots(
            rows=self.model.cfg.n_heads // 4 + 1,
            cols=4,
            shared_yaxes=True
        )
        for i in range(self.model.cfg.n_heads):
            data = self.patch[layer, i].cpu().numpy()
            #data = np.fliplr(np.triu(np.fliplr(data), k=0)) - 1
            data[data == -1] = np.nan

            fig.add_trace(
                px.imshow(
                    data,
                    x=[f"{tok} ({j})" for j, tok in enumerate(self.model.to_str_tokens(tokens))],
                    y=list(reversed([f"{tok} ({j})" for j, tok in enumerate(self.model.to_str_tokens(tokens))])),
                    color_continuous_scale="RdBu",
                    zmin=0,
                    zmax=1,
                    title=f"Head {i}",
                    aspect="auto"
                ).data[0],
                row=i // 4 + 1,
                col=i % 4 + 1
            )
        
        fig.update_layout(**kwargs)
        fig.show()
    
    def plot_attention_attr(self, tokens, **kwargs):

        attention_attr_pos = self.patch.clamp(min=-1e-5)
        attention_attr_neg = -self.patch.clamp(max=1e-5)
        attention_attr_signed = torch.stack([attention_attr_pos, attention_attr_neg], dim=0)
        attention_attr_signed = einops.rearrange(
            attention_attr_signed,
            "sign layer head_index dest src -> (layer head_index sign) dest src",
        )
        attention_attr_signed = attention_attr_signed / attention_attr_signed.max()
        attention_attr_indices = (
            attention_attr_signed.max(-1).values.max(-1).values.argsort(descending=True)
        )

        attention_attr_signed = attention_attr_signed[attention_attr_indices, :, :] # [2*layer*head, pos, pos]
        head_labels = [self.head_names_signed[i.item()] for i in attention_attr_indices]
        head_labels_by_layer = [[] for l in range(self.model.cfg.n_layers)]
        k = 0
        for h in head_labels:
            l = int(h[1])
            head_labels_by_layer[l].append((h, k))
            k += 1

        # Create a dropdown widget for layer selection
        layer_selector = widgets.Dropdown(
            options=[(f'Layer {i}', i) for i in range(self.model.cfg.n_layers)],
            value=0,
            description='Layer:',
        )
        
        # Function to update the plot based on the selected layer
        def update_plot(layer):

            rows = len(head_labels_by_layer[layer])

            # Update the plot  
            fig = make_subplots(
                rows=rows // 4 + 1,
                cols=4,
                shared_yaxes=True,
                horizontal_spacing=0.01,
            )
            for i, (h, k) in enumerate(head_labels_by_layer[layer]):
                data = attention_attr_signed[k].cpu().numpy() + 1
                data = np.fliplr(np.triu(np.fliplr(data), k=0)) - 1

                # Replace the lower right triangle with np.nan
                data[data == -1] = np.nan
                fig.add_trace(
                    px.imshow(
                        data,
                        x=[f"{tok} ({j})" for j, tok in enumerate(self.model.to_str_tokens(tokens))],
                        y=list(reversed([f"{tok} ({j})" for j, tok in enumerate(self.model.to_str_tokens(tokens))])),
                        color_continuous_scale="RdBu",
                        zmin=-1,
                        zmax=1,
                        title=h,
                        aspect="auto"
                    ).data[0],
                    row=i // 4 + 1,
                    col=i % 4 + 1
                )
                fig.update_layout(**kwargs)
            fig.show()

        # Display the widget and attach the update function
        widgets.interact(update_plot, layer=layer_selector)

    def plot_patch(self, layer=None, what=None, **kwargs):
        if self.component in ["resid_pre", "resid_post"]:
            ys = []
            for i in range(self.model.cfg.n_layers):
                ys.append(f'RS-pre L{i}')
                ys.append(f'RS-mid L{i}')
            ys.append('RS-final')
        elif self.component in ["attn_all", "attn_q", "attn_k", "attn_v", "attn_z"]:
            ys = [f'Attn L{i} H{j}' for i in range(self.model.cfg.n_layers) for j in range(self.model.cfg.n_heads)]
        
        if "pattern" in self.component:
            if what == 'top':
                self.plot_attention_attr(self.model.to_tokens(self.x_clean), title="Patching results", **kwargs)
            else:
                assert layer is not None, 'Layer must be specified for pattern component'
                self.plot_single_pattern(layer, self.model.to_tokens(self.x_clean), title="Patching results", **kwargs)
                
        else:
            fig = px.imshow(
                self.patch.cpu().numpy(), 
                x=[f"{tok} ({i})" for i, tok in enumerate(self.model.to_str_tokens(self.x_clean))],
                y=ys,
                title=f"Patching results for {self.how} method",
                color_continuous_scale='RdBu', zmin=-1, zmax=1, aspect='auto'
            )
        
            fig.update_layout(**kwargs)
            fig.show()