# gemma3_controlled_model.py
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gemma3 import Model as Gemma3ShellModel, ModelArgs as Gemma3ShellConfig
from mlx_lm.models.gemma3_text import ModelArgs as Gemma3TextConfig
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.utils import get_model_path
from mlx_lm import load as load_mlx_model
from mlx.utils import tree_flatten, tree_unflatten
from typing import Optional, Dict, Tuple, Any
import logging

from gemma3_control_core import ControlledGemma3DecoderLayer 

logger = logging.getLogger(__name__)

class ControlledGemma3TextModel(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.args = config 
        self.model_type = config.model_type 
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            ControlledGemma3DecoderLayer(config, i) 
            for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None, 
        mask: Optional[mx.array] = None, # This is the outer mask argument
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        
        h *= mx.array(self.args.hidden_size**0.5, dtype=h.dtype)

        if cache is None:
            cache = self.make_cache()

        prepared_full_mask = None
        prepared_sliding_window_mask = None

        if mask is None and h.shape[1] > 1: 
            from mlx_lm.models.base import create_attention_mask 

            global_layer_idx_for_mask = self.args.sliding_window_pattern - 1
            if global_layer_idx_for_mask < 0: global_layer_idx_for_mask = 0 

            cache_for_full_mask_list = None
            if cache and len(cache) > global_layer_idx_for_mask and cache[global_layer_idx_for_mask] is not None:
                 cache_for_full_mask_list = [cache[global_layer_idx_for_mask]]
            
            prepared_full_mask = create_attention_mask(h, cache_for_full_mask_list)
            if isinstance(prepared_full_mask, str) and prepared_full_mask == "causal": 
                prepared_full_mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            if prepared_full_mask is not None:
                prepared_full_mask = prepared_full_mask.astype(h.dtype)

            cache_for_sliding_mask_list = [cache[0]] if cache and len(cache) > 0 and cache[0] is not None else None
            prepared_sliding_window_mask = create_attention_mask(h, cache_for_sliding_mask_list)
            if isinstance(prepared_sliding_window_mask, str) and prepared_sliding_window_mask == "causal":
                prepared_sliding_window_mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            if prepared_sliding_window_mask is not None:
                prepared_sliding_window_mask = prepared_sliding_window_mask.astype(h.dtype)
        
        for i, layer_instance in enumerate(self.layers):
            layer_cache_obj = cache[i] if cache and i < len(cache) else None
            
            current_layer_mask = mask 
            if current_layer_mask is None: 
                is_global_layer = (i % self.args.sliding_window_pattern == self.args.sliding_window_pattern - 1)
                if is_global_layer:
                    current_layer_mask = prepared_full_mask
                else:
                    current_layer_mask = prepared_sliding_window_mask
            
            h = layer_instance(h, mask=current_layer_mask, cache=layer_cache_obj)
        
        final_norm_output = self.norm(h)
        out_logits = self.lm_head(final_norm_output)
        return out_logits

    @property
    def layers_prop(self): 
        return self.layers

    def make_cache(self): 
        caches = []
        for i in range(self.args.num_hidden_layers):
            if (i % self.args.sliding_window_pattern == self.args.sliding_window_pattern - 1):
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window, keep=0))
        return caches

    def sanitize(self, weights: Dict[str, Any]):
        # Weights are from original_shell_model_instance.language_model.parameters()
        # These keys are like "model.embed_tokens.weight", "model.layers.0...", "lm_head.weight"
        # Our ControlledGemma3TextModel expects "embed_tokens.weight", "layers.0...", "lm_head.weight"
        sanitized_weights = {}
        for k, v in weights.items():
            if k.startswith("model."):
                sanitized_weights[k[len("model."):]] = v
            elif k == "lm_head.weight": # lm_head is already top-level in OriginalGemma3TextModel
                sanitized_weights[k] = v
            else:
                # This case should ideally not happen if weights come from OriginalGemma3TextModel
                logger.warning(f"Unexpected weight key during sanitize: {k}. Using as is.")
                sanitized_weights[k] = v
        
        # Tie lm_head to embed_tokens if lm_head is missing (common practice)
        if "lm_head.weight" not in sanitized_weights and "embed_tokens.weight" in sanitized_weights:
            logger.info("Sanitizing in ControlledGemma3TextModel: Tying lm_head.weight to embed_tokens.weight.")
            sanitized_weights["lm_head.weight"] = sanitized_weights["embed_tokens.weight"]
        return sanitized_weights

def load_controlled_gemma3_model(model_name_or_path: str, tokenizer_config: Optional[Dict] = {}, trust_remote_code: bool = True) -> Tuple[Any, Any]:
    model_path_resolved = get_model_path(model_name_or_path) 
    original_shell_model_instance, tokenizer = load_mlx_model(str(model_path_resolved), tokenizer_config=tokenizer_config)

    if not isinstance(original_shell_model_instance, Gemma3ShellModel):
        logger.error(f"Loaded model from {model_name_or_path} is not a Gemma3ShellModel. Type is {type(original_shell_model_instance)}")
        raise TypeError(f"Loaded model from {model_name_or_path} is not a Gemma3ShellModel. Type is {type(original_shell_model_instance)}")

    shell_config: Gemma3ShellConfig = original_shell_model_instance.args 
    text_model_config_dict = shell_config.text_config
    
    if "vocab_size" not in text_model_config_dict or text_model_config_dict["vocab_size"] != shell_config.vocab_size:
        text_model_config_dict["vocab_size"] = shell_config.vocab_size

    text_model_args = Gemma3TextConfig.from_dict(text_model_config_dict)
    controlled_text_model = ControlledGemma3TextModel(text_model_args)
    
    original_language_model = original_shell_model_instance.language_model 
    original_language_model_params_flat = dict(tree_flatten(original_language_model.parameters()))
    
    # Sanitize parameter keys before updating the controlled_text_model
    sanitized_params_flat = controlled_text_model.sanitize(original_language_model_params_flat)
    
    controlled_text_model.update(tree_unflatten(list(sanitized_params_flat.items())))
    mx.eval(controlled_text_model.parameters()) 

    original_shell_model_instance.language_model = controlled_text_model
    
    logger.info(f"Successfully loaded and wrapped '{model_name_or_path}'. Its language_model is now ControlledGemma3TextModel.")
    return original_shell_model_instance, tokenizer
