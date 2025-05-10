# credits: https://gist.github.com/senstella/77178bb5d6ec67bf8c54705a5f490bed
import torch 
from safetensors.torch import save_file

INPUT_NAME = "model_weights.ckpt"
OUTPUT_NAME = "model.safetensors"

state = torch.load(INPUT_NAME, map_location="cpu")
new_state = {}

for key, value in state.items():
    if key.startswith("preprocessor"): continue
    if 'num_batches_tracked' in key: continue
    if 'conv' in key or 'ctc_decoder' in key or key == "decoder.decoder_layers.0.weight":
        if len(value.shape) == 4:
            value = value.permute((0, 2, 3, 1))
        elif len(value.shape) == 3:
            value = value.permute((0, 2, 1))
    if 'weight_ih_l' in key:
        key = key.replace('weight_ih_l', '') + '.Wx'
    if 'weight_hh_l' in key:
        key = key.replace('weight_hh_l', '') + '.Wh'
    if 'bias_ih_l' in key or 'bias_hh_l' in key:
        key = key.replace('bias_ih_l', '').replace('bias_hh_l', '') + '.bias'
        new_state[key] = value if new_state.get(key) is None else value + new_state[key]
    else:
        new_state[key] = value
  
save_file(new_state, OUTPUT_NAME)
