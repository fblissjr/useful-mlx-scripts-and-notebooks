# gemma3_control_utils.py
import mlx.core as mx
from typing import Optional, List, Any, Dict
import logging

from gemma3_controlled_model import ControlledGemma3TextModel 
from gemma3_control_core import ControlledGemma3DecoderLayer, ALL_CONTROL_POINTS 
from mlx_lm.models.gemma3 import Model as Gemma3ShellModel


logger = logging.getLogger(__name__)

def derive_control_vector(
    model_shell: Gemma3ShellModel, 
    tokenizer: Any,
    positive_prompts: List[str],
    negative_prompts: List[str],
    layer_idx: int,
    control_point: str,
    average_over_tokens: bool = True, 
) -> Optional[mx.array]:
    if not isinstance(model_shell.language_model, ControlledGemma3TextModel):
        logger.error("Model's language_model is not a ControlledGemma3TextModel instance during derivation.")
        raise TypeError("Model's language_model is not a ControlledGemma3TextModel instance.")
    
    controlled_text_model: ControlledGemma3TextModel = model_shell.language_model

    def get_activations_for_prompts(prompts: List[str]) -> List[mx.array]:
        all_activations = []
        if not (0 <= layer_idx < len(controlled_text_model.layers)): 
             logger.error(f"Invalid layer_idx {layer_idx} for ControlledGemma3TextModel during derivation.")
             raise ValueError(f"Invalid layer_idx {layer_idx} for ControlledGemma3TextModel")
        
        target_layer = controlled_text_model.layers[layer_idx]
        if not isinstance(target_layer, ControlledGemma3DecoderLayer):
            logger.error(f"Layer {layer_idx} in ControlledGemma3TextModel is not a ControlledGemma3DecoderLayer during derivation.")
            raise TypeError(f"Layer {layer_idx} in ControlledGemma3TextModel is not a ControlledGemma3DecoderLayer.")

        target_layer.start_capture([control_point])
        
        for prompt_text in prompts:
            input_ids = tokenizer.encode(prompt_text, return_tensors="np")
            input_ids_mx = mx.array(input_ids)
            
            if input_ids_mx.shape[0] == 0 or input_ids_mx.shape[1] == 0 : 
                logger.warning(f"Empty tokenization for prompt: '{prompt_text}'. Skipping.")
                continue

            _ = model_shell(input_ids_mx) 
            
            captured_data = target_layer.stop_capture()
            activation = captured_data.get(control_point)
            
            if activation is not None:
                if average_over_tokens:
                    if activation.ndim == 3 and activation.shape[0] == 1: 
                        activation = mx.mean(activation.squeeze(0), axis=0) 
                    elif activation.ndim == 2: 
                        activation = mx.mean(activation, axis=0)
                all_activations.append(activation)
            else:
                logger.warning(f"No activation captured for point '{control_point}' in layer {layer_idx} for prompt: '{prompt_text}'")
        
        return [act for act in all_activations if act is not None]

    logger.info(f"Deriving control vector for layer {layer_idx}, point '{control_point}'...")
    pos_activations = get_activations_for_prompts(positive_prompts)
    neg_activations = get_activations_for_prompts(negative_prompts)

    if not pos_activations or not neg_activations:
        logger.warning("Could not retrieve sufficient activations for deriving control vector.")
        return None

    try:
        mean_pos_act = mx.mean(mx.stack(pos_activations, axis=0), axis=0)
        mean_neg_act = mx.mean(mx.stack(neg_activations, axis=0), axis=0)
    except Exception as e:
        logger.error(f"Error stacking/averaging activations: {e}")
        logger.error(f"Positive activation shapes: {[a.shape for a in pos_activations]}")
        logger.error(f"Negative activation shapes: {[a.shape for a in neg_activations]}")
        return None

    control_vec = mean_pos_act - mean_neg_act
    logger.info(f"Derived control vector for layer {layer_idx}, point '{control_point}'. Shape: {control_vec.shape}, Norm: {mx.linalg.norm(control_vec):.4f}")
    return control_vec
