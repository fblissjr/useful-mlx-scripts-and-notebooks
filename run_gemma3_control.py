# run_gemma3_controlled.py
import argparse
import logging
import mlx.core as mx
from mlx_lm import generate as generate_text_mlx
from mlx_lm.sample_utils import make_sampler
import json 
import os # For checking file paths

# Assuming gemma3_controlled_model.py and gemma3_control_core.py are in the same directory or PYTHONPATH
from gemma3_controlled_model import load_controlled_gemma3_model, ControlledGemma3TextModel
from gemma3_control_core import ControlledGemma3DecoderLayer, ALL_CONTROL_POINTS
from gemma3_control_utils import derive_control_vector

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

def apply_controls_for_experiment(model_internal, experiment_controls, tokenizer, model_shell, hidden_size, add_special_tokens_for_encode_func, process_derivation_prompts_func):
    """Applies control vectors defined in an experiment's 'controls' list."""
    active_controls_info = []
    for control_spec in experiment_controls:
        layer_idx = control_spec["layer_idx"]
        control_point = control_spec["control_point"]
        strength = control_spec["strength"]
        vector_source = control_spec["vector_source"]
        control_vector = None

        if not (0 <= layer_idx < len(model_internal.layers)):
            logger.error(f"Experiment '{experiment_controls.get('name', 'Unnamed')}': Invalid layer_idx {layer_idx}. Skipping this control.")
            continue
        if control_point not in ALL_CONTROL_POINTS:
            logger.error(f"Experiment '{experiment_controls.get('name', 'Unnamed')}': Invalid control_point '{control_point}'. Skipping this control.")
            continue

        if vector_source["type"] == "derive":
            logger.info(f"Deriving vector for experiment control: L{layer_idx}, P:{control_point}...")
            pos_prompts_raw = vector_source["positive_prompts_raw"]
            neg_prompts_raw = vector_source["negative_prompts_raw"]
            
            # Process prompts with chat template if needed
            pos_prompts_processed = process_derivation_prompts_func(pos_prompts_raw)
            neg_prompts_processed = process_derivation_prompts_func(neg_prompts_raw)

            control_vector = derive_control_vector(
                model_shell, # Pass the shell model instance
                tokenizer,
                pos_prompts_processed,
                neg_prompts_processed,
                layer_idx,
                control_point,
                average_over_tokens=vector_source.get("average_over_tokens", True)
            )
        elif vector_source["type"] == "load_from_file":
            file_path = vector_source["file_path"]
            if os.path.exists(file_path):
                try:
                    control_vector = mx.load(file_path) # Assuming .npy or .npz with a single array
                    if isinstance(control_vector, dict): # Handle npz
                        if len(control_vector.keys()) == 1:
                            control_vector = list(control_vector.values())[0]
                        else: # Try to find 'arr_0' or a common key
                            control_vector = control_vector.get('arr_0', control_vector.get('vector'))
                            if control_vector is None:
                                logger.error(f"Experiment '{experiment_controls.get('name', 'Unnamed')}': NPZ file {file_path} has multiple arrays and no 'arr_0' or 'vector' key. Skipping this control.")
                                continue
                    
                    # Basic shape check
                    if control_vector.shape != (hidden_size,):
                         logger.error(f"Experiment '{experiment_controls.get('name', 'Unnamed')}': Loaded vector from {file_path} has shape {control_vector.shape}, expected ({hidden_size},). Skipping.")
                         control_vector = None # Invalidate vector
                    else:
                        logger.info(f"Loaded control vector from {file_path} for L{layer_idx}, P:{control_point}.")
                except Exception as e:
                    logger.error(f"Experiment '{experiment_controls.get('name', 'Unnamed')}': Failed to load control vector from {file_path}: {e}. Skipping this control.")
                    control_vector = None
            else:
                logger.error(f"Experiment '{experiment_controls.get('name', 'Unnamed')}': Control vector file not found: {file_path}. Skipping this control.")
                control_vector = None
        elif vector_source["type"] == "random_positive":
            control_vector = mx.random.normal(shape=(hidden_size,)).astype(mx.float16) * 0.1
        elif vector_source["type"] == "random_negative":
            control_vector = mx.random.normal(shape=(hidden_size,)).astype(mx.float16) * -0.1
        else:
            logger.error(f"Experiment '{experiment_controls.get('name', 'Unnamed')}': Unknown vector_source type '{vector_source['type']}'. Skipping this control.")
            continue

        if control_vector is not None:
            model_internal.layers[layer_idx].add_control(control_point, control_vector.astype(mx.float16), strength)
            active_controls_info.append(f"L{layer_idx}|{control_point}|S{strength:.2f}")
    return active_controls_info

def clear_all_experiment_controls(model_internal):
    """Clears all controls from all layers."""
    for layer in model_internal.layers:
        if isinstance(layer, ControlledGemma3DecoderLayer):
            layer.clear_controls()
    logger.info("Cleared all experiment controls from all layers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemma 3 model with granular control vectors.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the MLX-converted Gemma 3 model directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of Japan?",
        help="The initial prompt for text generation (used if no experiment file or for baseline)."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate."
    )
    # Sampling parameters
    parser.add_argument(
        "--temp",
        type=float,
        default=0.1, # Changed default to be slightly less deterministic for better showcase
        help="Temperature for text generation."
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Sampling top-p. Default is 1.0 (no top-p filtering)."
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Sampling min-p. Default is 0.0 (no min-p filtering)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Sampling top-k. Default is 0 (no top-k filtering)."
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=0.0,
        help="Probability of XTC sampling to happen each next token. Default is 0.0 (no XTC)."
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.0, 
        help="Threshold the probs of each next token candidate to be sampled by XTC. Default is 0.0."
    )
    parser.add_argument(
        "--min-tokens-to-keep",
        type=int,
        default=1, 
        help="Minimum tokens to keep for min-p sampling. Default is 1."
    )

    # Other arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    # Arguments for chat template
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Use the tokenizer's chat template to format the prompt. If not set, the raw prompt is used."
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to use with the chat template (if --use-chat-template is active)."
    )
    parser.add_argument(
        "--chat-template-args", 
        type=json.loads,
        help="A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{\"enable_thinking\":false}'",
        default="{}"
    )
    # Arguments for single control vector application OR experiment file
    parser.add_argument(
        "--experiments-file",
        type=str,
        default=None,
        help="Path to a JSON file defining experiments (collections of control vectors and test prompts)."
    )
    parser.add_argument(
        "--control-layer-idx",
        type=int,
        default=None,
        help="Layer index to apply a single CLI-specified control vector (used if --experiments-file is not provided)."
    )
    parser.add_argument(
        "--control-point",
        type=str,
        default=None,
        choices=ALL_CONTROL_POINTS,
        help="Control point for single CLI-specified control vector."
    )
    parser.add_argument(
        "--control-strength",
        type=float,
        default=1.0,
        help="Strength for single CLI-specified control vector."
    )
    parser.add_argument(
        "--control-vector-type",
        type=str,
        default="random_positive",
        choices=["random_positive", "random_negative"],
        help="Type of dummy control vector for single CLI-specified control."
    )


    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())
    model_identifier = args.model_path

    try:
        logger.info(f"Loading model from: {model_identifier}...")
        gemma_shell_model_instance, tokenizer = load_controlled_gemma3_model(model_identifier) 
        
        if not isinstance(gemma_shell_model_instance.language_model, ControlledGemma3TextModel):
            logger.error("Model loading did not result in the expected controlled language_model structure.")
            raise TypeError("Model loading did not result in the expected controlled language_model structure.")
            
        controlled_text_model_internal: ControlledGemma3TextModel = gemma_shell_model_instance.language_model
        
        if not hasattr(controlled_text_model_internal, 'args') or \
           not hasattr(controlled_text_model_internal.args, 'hidden_size') or \
           not hasattr(controlled_text_model_internal.args, 'num_hidden_layers'):
            logger.error("Loaded controlled text model's .args object does not have expected attributes (hidden_size, num_hidden_layers).")
            raise ValueError("Loaded controlled text model's .args object does not have expected attributes (hidden_size, num_hidden_layers).")

        hidden_size = controlled_text_model_internal.args.hidden_size
        num_layers = controlled_text_model_internal.args.num_hidden_layers
        logger.info(f"Model loaded successfully. Text model hidden size: {hidden_size}, Num layers: {num_layers}")

    except Exception as e:
        logger.exception(f"Error loading model: {e}") 
        logger.error("Please ensure you have provided a valid path or Hugging Face ID for an MLX-converted Gemma 3 model.")
        exit(1)

    # Define helper for processing prompts with chat template (used by various parts)
    # This needs to be defined after tokenizer and args are available.
    def _process_prompt_with_template(raw_prompt_str):
        _add_special_tokens = True
        _processed_prompt_str = raw_prompt_str
        if args.use_chat_template:
            if tokenizer.chat_template is None:
                logger.warning("Tokenizer does not have a chat_template. Using Hugging Face default chat template.")
                tokenizer.chat_template = tokenizer.default_chat_template 
            
            _messages = []
            if args.system_prompt:
                _messages.append({"role": "system", "content": args.system_prompt})
            _messages.append({"role": "user", "content": raw_prompt_str}) # Assuming raw_prompt_str is the user content
            
            try:
                _processed_prompt_str = tokenizer.apply_chat_template(
                    _messages,
                    tokenize=False, 
                    add_generation_prompt=True, 
                    **args.chat_template_args
                )
                _add_special_tokens = False 
            except Exception as e:
                logger.error(f"Error applying chat template to prompt '{raw_prompt_str[:50]}...': {e}. Using raw prompt string.")
                _processed_prompt_str = raw_prompt_str
                _add_special_tokens = True
        return _processed_prompt_str, _add_special_tokens

    # Create sampler using all provided CLI arguments
    sampler = make_sampler(
        temp=args.temp,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
        xtc_probability=args.xtc_probability,
        xtc_threshold=args.xtc_threshold,
        min_tokens_to_keep=args.min_tokens_to_keep,
    )

    # --- Run Experiments from File OR Single CLI-defined Control ---
    if args.experiments_file:
        logger.info(f"Loading experiments from: {args.experiments_file}")
        try:
            with open(args.experiments_file, 'r') as f:
                experiments_config = json.load(f)
        except Exception as e:
            logger.exception(f"Failed to load or parse experiments file '{args.experiments_file}'. Error: {e}")
            exit(1)

        for exp_idx, experiment in enumerate(experiments_config.get("experiments", [])):
            exp_name = experiment.get("name", f"Unnamed Experiment {exp_idx + 1}")
            exp_desc = experiment.get("description", "No description.")
            logger.info(f"\n--- Running Experiment: {exp_name} ---")
            logger.info(f"Description: {exp_desc}")

            # Apply controls for this experiment
            active_controls_str = "None"
            if "controls" in experiment and experiment["controls"]:
                active_controls_info = apply_controls_for_experiment(
                    controlled_text_model_internal, 
                    experiment["controls"], 
                    tokenizer, 
                    gemma_shell_model_instance,
                    hidden_size,
                    lambda p: _process_prompt_with_template(p)[1], # Pass func to get add_special_tokens flag
                    lambda prompts_list: [_process_prompt_with_template(p)[0] for p in prompts_list] # Pass func to get processed strings
                )
                if active_controls_info:
                    active_controls_str = ", ".join(active_controls_info)
            
            logger.info(f"Active controls for '{exp_name}': {active_controls_str}")

            for test_prompt_raw in experiment.get("test_prompts", [args.prompt]): # Default to CLI prompt if none in exp
                test_prompt_str, add_spec_tokens = _process_prompt_with_template(test_prompt_raw)
                test_prompt_tokens = tokenizer.encode(test_prompt_str, add_special_tokens=add_spec_tokens)
                
                logger.info(f"Generating for prompt: '{test_prompt_raw}' (Formatted: '{test_prompt_str[:60]}...')")
                response = generate_text_mlx(
                    gemma_shell_model_instance, 
                    tokenizer, 
                    prompt=test_prompt_tokens, 
                    max_tokens=args.max_tokens, 
                    sampler=sampler
                )
                logger.info(f"Response for '{exp_name}' (Prompt: '{test_prompt_raw[:30]}...'): {response}")

            # Clear controls after this experiment
            clear_all_experiment_controls(controlled_text_model_internal)
            logger.info(f"--- Finished Experiment: {exp_name} ---")

    elif args.control_layer_idx is not None and args.control_point is not None:
        # Run single CLI-defined control experiment
        logger.info("\n--- Running Single CLI-Specified Control Experiment ---")
        if not (0 <= args.control_layer_idx < num_layers):
            logger.error(f"Invalid --control-layer-idx {args.control_layer_idx}. Must be between 0 and {num_layers - 1}.")
        else:
            logger.info(f"Applying CLI control: Layer {args.control_layer_idx}, Point '{args.control_point}', Strength {args.control_strength}, Type '{args.control_vector_type}'")
            
            vec_multiplier = 1.0 if args.control_vector_type == "random_positive" else -1.0
            cli_control_vector = mx.random.normal(shape=(hidden_size,)).astype(mx.float16) * 0.1 * vec_multiplier
            
            controlled_text_model_internal.layers[args.control_layer_idx].add_control(
                args.control_point, 
                cli_control_vector, 
                args.control_strength
            )
            
            main_prompt_str, main_add_spec_tokens = _process_prompt_with_template(args.prompt)
            main_prompt_tokens = tokenizer.encode(main_prompt_str, add_special_tokens=main_add_spec_tokens)

            logger.info(f"Generating with CLI-specified control: '{args.prompt}' (Formatted: '{main_prompt_str[:60]}...')")
            controlled_response_cli = generate_text_mlx(gemma_shell_model_instance, tokenizer, prompt=main_prompt_tokens, max_tokens=args.max_tokens, sampler=sampler)
            logger.info(f"CLI Controlled Response: {controlled_response_cli}")

            controlled_text_model_internal.layers[args.control_layer_idx].clear_controls(args.control_point)
            logger.info("Cleared CLI-specified control.")
    else:
        # Just run baseline if no experiments file and no single control CLI args
        logger.info("\n--- No experiments file or specific CLI control provided. Running baseline generation. ---")
    
    # Always run a final baseline generation if no experiment file was processed or after all experiments
    if not args.experiments_file :
        logger.info(f"Generating final baseline response (no active controls): '{args.prompt}'")
        main_prompt_str, main_add_spec_tokens = _process_prompt_with_template(args.prompt)
        main_prompt_tokens = tokenizer.encode(main_prompt_str, add_special_tokens=main_add_spec_tokens)
        baseline_response = generate_text_mlx(gemma_shell_model_instance, tokenizer, prompt=main_prompt_tokens, max_tokens=args.max_tokens, sampler=sampler)
        logger.info(f"Final Baseline Response: {baseline_response}")


    logger.info("\n--- Experimentation Finished ---")
