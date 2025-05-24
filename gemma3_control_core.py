# gemma3_control_core.py
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gemma3_text import ModelArgs as Gemma3TextConfig, TransformerBlock as OriginalGemma3DecoderLayer
from typing import Optional, Dict, Tuple, List, Any
import logging

logger = logging.getLogger(__name__)

# Define standardized names for control points within a decoder layer
CONTROL_POINT_PRE_ATTN_LN = "pre_attention_layernorm_input"
CONTROL_POINT_ATTN_OUTPUT = "attention_output" # After o_proj
CONTROL_POINT_POST_ATTN_RESIDUAL = "post_attention_residual" # Input to MLP LN
CONTROL_POINT_PRE_MLP_LN = "pre_mlp_layernorm_input"
CONTROL_POINT_MLP_OUTPUT = "mlp_output" # After down_proj
CONTROL_POINT_POST_MLP_RESIDUAL = "post_mlp_residual" # Final output of layer

ALL_CONTROL_POINTS = [
    CONTROL_POINT_PRE_ATTN_LN,
    CONTROL_POINT_ATTN_OUTPUT,
    CONTROL_POINT_POST_ATTN_RESIDUAL,
    CONTROL_POINT_PRE_MLP_LN,
    CONTROL_POINT_MLP_OUTPUT,
    CONTROL_POINT_POST_MLP_RESIDUAL,
]

def clip_residual(x: mx.array, y: mx.array) -> mx.array:
    """
    Performs a clipped residual addition, similar to the internal implementation
    in gemma3_text.py, which is crucial for float16 stability.
    """
    if x.dtype == mx.float16:
        bound = mx.finfo(mx.float16).max # Use finfo for robustness
        return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(mx.float16)
    return x + y

class ControlledGemma3DecoderLayer(OriginalGemma3DecoderLayer):
    """
    A Gemma3DecoderLayer (gemma3_text.TransformerBlock) modified to allow 
    application of control vectors at specified points in its forward pass.
    """
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self._control_vectors: Dict[str, List[Tuple[mx.array, float]]] = {cp: [] for cp in ALL_CONTROL_POINTS}
        self._active_control_points: Dict[str, bool] = {cp: False for cp in ALL_CONTROL_POINTS}

        self._capture_activations_map: Dict[str, Optional[mx.array]] = {cp: None for cp in ALL_CONTROL_POINTS}
        self._is_capturing: bool = False
        self._capture_targets: List[str] = []

    def add_control(self, control_point: str, vector: mx.array, strength: float):
        if control_point not in self._control_vectors:
            logger.error(f"Unknown control point: {control_point} in layer {self.layer_idx}. Available: {ALL_CONTROL_POINTS}")
            raise ValueError(f"Unknown control point: {control_point} in layer {self.layer_idx}. Available: {ALL_CONTROL_POINTS}")
        self._control_vectors[control_point].append((vector, strength))
        self._active_control_points[control_point] = True
        logger.info(f"Layer {self.layer_idx}: Added control to '{control_point}' with strength {strength:.2f}, num_controls_at_point: {len(self._control_vectors[control_point])}")

    def clear_controls(self, control_point: Optional[str] = None):
        if control_point:
            if control_point in self._control_vectors:
                self._control_vectors[control_point] = []
                self._active_control_points[control_point] = False
                logger.info(f"Layer {self.layer_idx}: Cleared controls for '{control_point}'")
            else:
                logger.error(f"Unknown control point: {control_point} in layer {self.layer_idx}")
                raise ValueError(f"Unknown control point: {control_point} in layer {self.layer_idx}")
        else:
            for cp in ALL_CONTROL_POINTS:
                self._control_vectors[cp] = []
                self._active_control_points[cp] = False
            logger.info(f"Layer {self.layer_idx}: Cleared all controls")

    def _apply_controls_at_point(self, x: mx.array, point_name: str) -> mx.array:
        if self._active_control_points.get(point_name, False):
            for vec, strength in self._control_vectors[point_name]:
                if x.ndim == 3 and vec.ndim == 1: 
                    x = x + (vec * strength).astype(x.dtype)
                elif x.ndim == vec.ndim: 
                     x = x + (vec * strength).astype(x.dtype)
                else:
                    logger.warning(f"Layer {self.layer_idx}, Point '{point_name}': Vector shape {vec.shape} not directly broadcastable to activation shape {x.shape}. Skipping this vector.")
        return x

    def _capture_if_needed(self, x: mx.array, point_name: str):
        if self._is_capturing and point_name in self._capture_targets:
            self._capture_activations_map[point_name] = x.copy() 

    def start_capture(self, target_points: List[str]):
        self._is_capturing = True
        self._capture_targets = target_points
        for cp in ALL_CONTROL_POINTS: 
            self._capture_activations_map[cp] = None
        logger.info(f"Layer {self.layer_idx}: Started capturing for {target_points}")

    def stop_capture(self) -> Dict[str, Optional[mx.array]]:
        self._is_capturing = False
        captured_data = {k: v for k, v in self._capture_activations_map.items() if v is not None}
        self._capture_targets = [] 
        logger.info(f"Layer {self.layer_idx}: Stopped capturing. Captured: {list(captured_data.keys())}")
        return captured_data

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None, 
    ) -> mx.array: 
        
        # 1. Input Layernorm for Attention
        x_ln1_input = x
        self._capture_if_needed(x_ln1_input, CONTROL_POINT_PRE_ATTN_LN)
        x_ln1_input_controlled = self._apply_controls_at_point(x_ln1_input, CONTROL_POINT_PRE_ATTN_LN)
        normed_x_for_attn = self.input_layernorm(x_ln1_input_controlled)

        # 2. Self-Attention
        attn_out_raw = self.self_attn(normed_x_for_attn, mask=mask, cache=cache)
        self._capture_if_needed(attn_out_raw, CONTROL_POINT_ATTN_OUTPUT)
        attn_out_controlled = self._apply_controls_at_point(attn_out_raw, CONTROL_POINT_ATTN_OUTPUT)
        
        # 3. First Residual Connection
        post_attn_ln_out = self.post_attention_layernorm(attn_out_controlled)
        h_after_attn = clip_residual(x, post_attn_ln_out) 
        
        self._capture_if_needed(h_after_attn, CONTROL_POINT_POST_ATTN_RESIDUAL)
        h_after_attn_controlled = self._apply_controls_at_point(h_after_attn, CONTROL_POINT_POST_ATTN_RESIDUAL)

        # 4. Pre-Feedforward Layernorm
        mlp_ln_input = h_after_attn_controlled
        self._capture_if_needed(mlp_ln_input, CONTROL_POINT_PRE_MLP_LN)
        mlp_ln_input_controlled = self._apply_controls_at_point(mlp_ln_input, CONTROL_POINT_PRE_MLP_LN)
        normed_h_for_mlp = self.pre_feedforward_layernorm(mlp_ln_input_controlled)

        # 5. MLP
        mlp_out_raw = self.mlp(normed_h_for_mlp)
        self._capture_if_needed(mlp_out_raw, CONTROL_POINT_MLP_OUTPUT)
        mlp_out_controlled = self._apply_controls_at_point(mlp_out_raw, CONTROL_POINT_MLP_OUTPUT)

        # 6. Second Residual Connection
        post_ffw_ln_out = self.post_feedforward_layernorm(mlp_out_controlled)
        final_out = clip_residual(h_after_attn_controlled, post_ffw_ln_out) 

        self._capture_if_needed(final_out, CONTROL_POINT_POST_MLP_RESIDUAL)
        final_out_controlled = self._apply_controls_at_point(final_out, CONTROL_POINT_POST_MLP_RESIDUAL)
        
        return final_out_controlled
