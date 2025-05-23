# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import logging
import platform
import time
import uuid
import warnings
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
from huggingface_hub import scan_cache_dir

from ._version import __version__
from .models.cache import make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache, KVCache, QuantizedKVCache
from .sample_utils import make_logits_processors, make_sampler
from .utils import load, stream_generate

def get_system_fingerprint():
    gpu_arch = mx.metal.device_info()["architecture"] if mx.metal.is_available() else ""
    return f"{__version__}-{mx.__version__}-{platform.platform()}-{gpu_arch}"

class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int

def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[List[int]],
    eos_token_id: Union[int, None],
) -> StopCondition:
    """
    Determines whether the token generation should stop based on predefined
    conditions.

    Args:
        tokens (List[int]): The current sequence of generated tokens.
        stop_id_sequences (List[List[[int]]): A list of integer lists, each
          representing a sequence of token IDs. If the end of the `tokens`
          list matches any of these sequences, the generation should stop.
        eos_token_id (Union[int, None]): The token ID that represents the
          end-of-sequence. If the last token in `tokens` matches this, the
          generation should stop.

    Returns:
        StopCondition: A named tuple indicating whether the stop condition has
          been met (`stop_met`) and how many tokens should be trimmed from the
          end if it has (`trim_length`).
    """
    if tokens and tokens[-1] == eos_token_id:
        return StopCondition(stop_met=True, trim_length=0)

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids) :] == stop_ids:
                return StopCondition(stop_met=True, trim_length=len(stop_ids))

    return StopCondition(stop_met=False, trim_length=0)

def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    Checks if a suffix of s1 has overlap with a prefix of s2

    Args:
        s1 (Sequence): The first sequence
        s2 (Sequence): The second sequence

    Returns:
        bool: If the two sequences have overlap
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))

@dataclass
class PromptCache:
    cache: List[Any] = field(default_factory=list)
    model_key: Tuple[str, Optional[str]] = ("", None)
    tokens: List[int] = field(default_factory=list)

# Key changes from original server.py
class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        self.cli_args = cli_args
        self.model_key = None
        self.model = None
        self.tokenizer = None

        # Preload the default model if provided
        if self.cli_args.model is not None:
            self.load("default_model")

    def load(self, model_path, adapter_path=None):
        if self.model_key == (model_path, adapter_path):
            return self.model, self.tokenizer

        # Reset state
        self.model = None
        self.tokenizer = None
        self.model_key = None

        # Simple tokenizer config
        tokenizer_config = {
            "trust_remote_code": True if self.cli_args.trust_remote_code else None,
            "local_files_only": True
        }

        try:
            if model_path == "default_model" and self.cli_args.model is not None:
                model, tokenizer = load(
                    self.cli_args.model,
                    adapter_path=adapter_path or self.cli_args.adapter_path,
                    tokenizer_config=tokenizer_config,
                )
            else:
                self._validate_model_path(model_path)
                model, tokenizer = load(
                    model_path, 
                    adapter_path=adapter_path, 
                    tokenizer_config=tokenizer_config
                )

            self.model_key = (model_path, adapter_path)
            self.model = model
            self.tokenizer = tokenizer
            return model, tokenizer

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

class APIHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        model_provider: ModelProvider,
        *args,
        prompt_cache: Optional[PromptCache] = None,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        """Initialize request specific metadata"""
        self.created = int(time.time())
        self.model_provider = model_provider
        self.prompt_cache = prompt_cache or PromptCache()
        self.system_fingerprint = system_fingerprint or get_system_fingerprint()
        super().__init__(*args, **kwargs)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _set_completion_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()

    def _set_stream_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()

    def do_OPTIONS(self):
        self._set_completion_headers(204)
        self.end_headers()

    def do_POST(self):
        """Handle POST request"""
        endpoints = {
            "/v1/completions": self.handle_text_completions,
            "/v1/chat/completions": self.handle_chat_completions,
            "/chat/completions": self.handle_chat_completions,
        }

        if self.path not in endpoints:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Parse request body
        content_length = int(self.headers["Content-Length"])
        raw_body = self.rfile.read(content_length)
        self.body = json.loads(raw_body.decode())
        
        # Extract request parameters from the body
        self.stream = self.body.get("stream", False)
        self.max_tokens = self.body.get("max_tokens", 512)
        self.temperature = self.body.get("temperature", 0.0)
        self.top_p = self.body.get("top_p", 0.95)
        self.repetition_penalty = self.body.get("repetition_penalty", 1.0)
        self.repetition_context_size = self.body.get("repetition_context_size", 20)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", -1)
        self.stream_options = self.body.get("stream_options", None)
        self.requested_model = self.body.get("model", "default_model")
        self.adapter = self.body.get("adapters", None)
        self.max_tokens = self.body.get("max_completion_tokens", None)
        if self.max_tokens is None:
            self.max_tokens = self.body.get("max_tokens", 512)
        self.validate_model_parameters()
        
        # Validate parameters
        self.validate_model_parameters()

        # Load model
        try:
            self.model, self.tokenizer = self.model_provider.load(
                self.body.get("model", "default_model"),
                self.body.get("adapters", None)
            )
        except Exception as e:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(str(e).encode())
            return

        # Get stop sequences
        stop_words = self.body.get("stop", [])
        if isinstance(stop_words, str):
            stop_words = [stop_words]
        stop_id_sequences = [
            self.tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # Set headers based on stream mode
        if self.stream:
            self._set_stream_headers(200)
        else:
            self._set_completion_headers(200)

        # Process request
        prompt = endpoints[self.path]()
        self.handle_completion(prompt, stop_id_sequences)
        
    def validate_model_parameters(self):
        """
        Validate the model parameters passed in the request for the correct types and values.
        """
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")

        if not isinstance(self.max_tokens, int) or self.max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")

        if not isinstance(self.temperature, (float, int)) or self.temperature < 0:
            raise ValueError("temperature must be a non-negative float")

        if not isinstance(self.top_p, (float, int)) or self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be a float between 0 and 1")

        if (
            not isinstance(self.repetition_penalty, (float, int))
            or self.repetition_penalty < 0
        ):
            raise ValueError("repetition_penalty must be a non-negative float")

        if self.logprobs != -1 and not (0 < self.logprobs <= 10):
            raise ValueError(
                f"logprobs must be between 1 and 10 but got {self.logprobs:,}"
            )

        if (
            not isinstance(self.repetition_context_size, int)
            or self.repetition_context_size < 0
        ):
            raise ValueError("repetition_context_size must be a non-negative integer")

        if self.logit_bias is not None:
            if not isinstance(self.logit_bias, dict):
                raise ValueError("logit_bias must be a dict of int to float")

            try:
                self.logit_bias = {int(k): v for k, v in self.logit_bias.items()}
            except ValueError:
                raise ValueError("logit_bias must be a dict of int to float")

        if not isinstance(self.requested_model, str):
            raise ValueError("model must be a string")
        if self.adapter is not None and not isinstance(self.adapter, str):
            raise ValueError("adapter must be a string")

    def generate_response(
        self,
        text: str,
        finish_reason: Union[Literal["length", "stop"], None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None,
        top_tokens: Optional[List[Dict[int, float]]] = None,
        tokens: Optional[List[int]] = None,
    ) -> dict:
        """
        Generate a single response packet based on response type (stream or
        not), completion type and parameters.

        Args:
            text (str): Text generated by model
            finish_reason (Union[Literal["length", "stop"], None]): The reason the
              response is being sent: "length", "stop" or `None`.
            prompt_token_count (Optional[int]): The number of tokens in the prompt,
              used to populate the "usage" field (not used when stream).
            completion_token_count (Optional[int]): The number of tokens in the
              response, used to populate the "usage" field (not used when stream).
            token_logprobs (Optional[List[float]]): The log probabilities per token,
              in token order.
            top_tokens (Optional[List[Dict[int, float]]]): List of dictionaries mapping
              tokens to logprobs for the top N tokens at each token position.
            tokens (Optional[List[int]]): List of tokens to return with logprobs structure

        Returns:
            dict: A dictionary containing the response, in the same format as
              OpenAI's API.
        """
        token_logprobs = token_logprobs if token_logprobs else []
        top_logprobs = top_tokens if top_tokens else []

        # Static response
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [
                {
                    "index": 0,
                    "logprobs": {
                        "token_logprobs": token_logprobs,
                        "top_logprobs": top_logprobs,
                        "tokens": tokens,
                    },
                    "finish_reason": finish_reason,
                }
            ],
        }

        if not self.stream:
            if not (
                isinstance(prompt_token_count, int)
                and isinstance(completion_token_count, int)
            ):
                raise ValueError(
                    "Response type is complete, but token counts not provided"
                )

            response["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }

        choice = response["choices"][0]

        # Add dynamic response
        if self.object_type.startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            choice[key_name] = {"role": "assistant", "content": text}
        elif self.object_type == "text_completion":
            choice.update(text=text)
        else:
            ValueError(f"Unsupported response type: {self.object_type}")

        return response

    def handle_chat_completions(self) -> List[int]:
        """Handle chat completion request"""
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"

        if self.model_provider.cli_args.raw_text:
            # Raw text mode - use content directly
            prompt = body["messages"][-1]["content"]
            if isinstance(prompt, list):
                prompt = self.tokenizer.decode(prompt)
            return self.tokenizer.encode(prompt)
        
        # Standard chat template handling
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                body["messages"],
                add_generation_prompt=True,
            )
        else:
            return self.tokenizer.encode(body["messages"][-1]["content"])

    def handle_text_completions(self) -> List[int]:
        """Handle text completion request"""
        self.request_id = f"cmpl-{uuid.uuid4()}"
        self.object_type = "text_completion"
        assert "prompt" in self.body, "Request did not contain a prompt"
        return self.tokenizer.encode(self.body["prompt"])

    def validate_model_parameters(self):
        """Validate request parameters"""
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")

        if not isinstance(self.max_tokens, int) or self.max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")

        if not isinstance(self.temperature, (float, int)) or self.temperature < 0:
            raise ValueError("temperature must be a non-negative float")

        if not isinstance(self.top_p, (float, int)) or self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be a float between 0 and 1")

        if not isinstance(self.repetition_penalty, (float, int)) or self.repetition_penalty < 0:
            raise ValueError("repetition_penalty must be a non-negative float")

        if self.logprobs != -1 and not (0 < self.logprobs <= 10):
            raise ValueError(f"logprobs must be between 1 and 10 but got {self.logprobs}")

        if not isinstance(self.repetition_context_size, int) or self.repetition_context_size < 0:
            raise ValueError("repetition_context_size must be a non-negative integer")

        if self.logit_bias is not None:
            if not isinstance(self.logit_bias, dict):
                raise ValueError("logit_bias must be a dict of int to float")
            try:
                self.logit_bias = {int(k): v for k, v in self.logit_bias.items()}
            except ValueError:
                raise ValueError("logit_bias must be a dict of int to float")
            
    def get_prompt_cache(self, prompt):
        cache_len = len(self.prompt_cache.tokens)
        if (
            self.prompt_cache.model_key != self.model_provider.model_key
            or cache_len >= len(prompt)
            or self.prompt_cache.tokens != prompt[:cache_len]
        ):
            self.prompt_cache.model_key = self.model_provider.model_key
            self.prompt_cache.cache = make_prompt_cache(self.model_provider.model)  # <-- Here's make_prompt_cache
        else:
            prompt = prompt[cache_len:]
        self.prompt_cache.tokens.extend(prompt)
        return prompt

    def handle_completion(
        self,
        prompt: List[int],
        stop_id_sequences: List[List[int]],
    ):
        """
        Generate a response to a prompt and send it to the client in a single batch.
        """
        tokens = []
        finish_reason = "length" 
        stop_sequence_suffix = None
        
        if self.stream:
            self.end_headers()
            logging.debug("Starting stream:")
        else:
            logging.debug("Starting completion:")
            
        token_logprobs = []
        top_tokens = []

        # Create base cache and convert to quantized if needed
        cache = make_prompt_cache(
            self.model_provider.model,
            max_kv_size=self.model_provider.cli_args.max_kv_size
        )
        
        # Convert to quantized cache if bits specified 
        if hasattr(self.model_provider.cli_args, 'kv_bits') and self.model_provider.cli_args.kv_bits:
            for i in range(len(cache)):
                if isinstance(cache[i], KVCache):
                    cache[i] = cache[i].to_quantized(
                        group_size=self.model_provider.cli_args.kv_group_size,
                        bits=self.model_provider.cli_args.kv_bits
                    )

        text = ""
        tic = time.perf_counter()
        
        # Get prompt cache
        prompt = self.get_prompt_cache(prompt)
        
        # Setup sampling and processing
        sampler = make_sampler(self.temperature, top_p=self.top_p)
        logits_processors = make_logits_processors(
            self.logit_bias,
            self.repetition_penalty,
            self.repetition_context_size
        )

        # Generation loop
        for gen_response in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=cache,
        ):
            segment = gen_response.text
            text += segment
            logging.debug(text)
            token = gen_response.token
            logprobs = gen_response.logprobs
            tokens.append(token)

            # Handle logprobs if requested
            if self.logprobs > 0:
                sorted_indices = mx.argpartition(-logprobs, kth=self.logprobs - 1)
                top_indices = sorted_indices[: self.logprobs]
                top_logprobs = logprobs[top_indices]
                top_token_info = zip(top_indices.tolist(), top_logprobs.tolist())
                top_tokens.append(tuple(top_token_info))

            token_logprobs.append(logprobs[token].item())

            # Check stop conditions
            stop_condition = stopping_criteria(
                tokens, stop_id_sequences, self.tokenizer.eos_token_id
            )
            if stop_condition.stop_met:
                finish_reason = "stop"
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(
                        tokens[-stop_condition.trim_length :]
                    )
                    text = text[: -len(stop_sequence_suffix)]
                break

            # Handle streaming output
            if self.stream:
                if any(
                    sequence_overlap(tokens, sequence)
                    for sequence in stop_id_sequences
                ):
                    continue
                elif segment:
                    response = self.generate_response(segment, None)
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                    self.wfile.flush()

        # Update cache and send final response
        self.prompt_cache.tokens.extend(tokens)

        logging.debug(f"Prompt: {gen_response.prompt_tps:.3f} tokens-per-sec")
        logging.debug(f"Generation: {gen_response.generation_tps:.3f} tokens-per-sec")
        logging.debug(f"Peak memory: {gen_response.peak_memory:.3f} GB")

        if self.stream:
            # Send final streaming responses
            response = self.generate_response(segment, finish_reason)
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()
            
            if self.stream_options is not None and self.stream_options["include_usage"]:
                response = self.completion_usage_response(len(prompt), len(tokens))
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.flush()
                
            self.wfile.write("data: [DONE]\n\n".encode())
            self.wfile.flush()
        else:
            # Send complete response
            response = self.generate_response(
                text,
                finish_reason,
                len(prompt),
                len(tokens),
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
            )
            response_json = json.dumps(response).encode()
            
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json)
            self.wfile.flush()

    def completion_usage_response(
        self,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
    ):
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": "chat.completion",
            "model": self.requested_model,
            "created": self.created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        }
        return response

    def handle_models_request(self):
        """
        Handle a GET request for the /v1/models endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        # Scan the cache directory for downloaded mlx models
        hf_cache_info = scan_cache_dir()
        downloaded_models = [
            repo for repo in hf_cache_info.repos if "mlx" in repo.repo_id
        ]

        # Create a list of available models
        models = [
            {
                "id": repo.repo_id,
                "object": "model",
                "created": self.created,
            }
            for repo in downloaded_models
        ]

        response = {"object": "list", "data": models}

        response_json = json.dumps(response).encode()
        self.wfile.write(response_json)
        self.wfile.flush()

def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=HTTPServer,
    handler_class=APIHandler,
):
    server_address = (host, port)
    prompt_cache = PromptCache()
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(
            model_provider,
            prompt_cache=prompt_cache,
            system_fingerprint=get_system_fingerprint(),
            *args,
            **kwargs,
        ),
    )
    warnings.warn(
        "mlx_lm.server is not recommended for production as "
        "it only implements basic security checks."
    )
    logging.info(f"Starting httpd at {host} on port {port}...")
    httpd.serve_forever()

def main():
    parser = argparse.ArgumentParser(description="MLX Server with KV Cache Optimization")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--cache-limit-gb",
        type=int,
        default=None,
        help="Set the MLX cache limit in GB",
        required=False,
    )
    # Add KV cache optimization arguments
    parser.add_argument(
        "--kv-bits",
        type=int, 
        default=None,
        help="Number of bits for KV cache quantization (e.g. 8 for 8-bit quantization)",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization",
    )
    parser.add_argument(
        "--kv-quantize-after",
        type=int,
        default=5000,
        help="Start KV cache quantization after this many tokens",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None, 
        help="Maximum size of the KV cache before rotation",
    )
    parser.add_argument(
        "--raw-text",
        action="store_true",
        help="Disable chat templating for raw text generation",
    )
    args = parser.parse_args()

    if args.cache_limit_gb is not None:
        logging.debug(f"Setting cache limit to {args.cache_limit_gb} GB")
        mx.metal.set_cache_limit(args.cache_limit_gb * 1024 * 1024 * 1024)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    run(args.host, args.port, ModelProvider(args))

if __name__ == "__main__":
    main()