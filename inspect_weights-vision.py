# gemma3_wrapper/inspect_model.py (Corrected Version)

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from huggingface_hub import hf_hub_download
import safetensors.torch
import os

# Use rich for pretty printing
console = Console()

def inspect_model(model_identifier: str):
    """
    Downloads (if necessary) and inspects a model repository.
    Handles both local file paths and Hugging Face Hub IDs.

    Args:
        model_identifier (str): A Hugging Face repo ID or a local path to a model directory.
    """
    console.print(Panel(f"Inspecting Model: [bold cyan]{model_identifier}[/bold cyan]", title="Model Inspector", border_style="green"))

    model_path = Path(model_identifier).expanduser()
    is_local = model_path.is_dir()

    try:
        if is_local:
            console.print(f"[green]Detected local model directory at: {model_path}[/green]")
            config_path = model_path / "config.json"
            processor_config_path = model_path / "processor_config.json"
            index_path = model_path / "model.safetensors.index.json"
        else:
            console.print(f"[green]Detected Hugging Face Hub ID: {model_identifier}[/green]")
            # Download the main configuration files first
            config_path = hf_hub_download(repo_id=model_identifier, filename="config.json")
            processor_config_path = hf_hub_download(repo_id=model_identifier, filename="processor_config.json", repo_type="model", fatal_ok=True)
            index_path = hf_hub_download(repo_id=model_identifier, filename="model.safetensors.index.json", fatal_ok=True)


        # --- Load and Print Configurations ---
        if not Path(config_path).exists():
             console.print(f"[bold red]Error: config.json not found at {config_path}.[/bold red]")
             return

        with open(config_path, 'r') as f:
            config = json.load(f)
        console.print(Panel(json.dumps(config, indent=2), title="[bold]config.json[/bold]", border_style="blue"))

        if processor_config_path and Path(processor_config_path).exists():
            with open(processor_config_path, 'r') as f:
                processor_config = json.load(f)
            console.print(Panel(json.dumps(processor_config, indent=2), title="[bold]processor_config.json[/bold]", border_style="yellow"))
        else:
            console.print("[yellow]No processor_config.json found.[/yellow]")

        # --- Load and Inspect Weight Keys ---
        weights_path = None
        weight_file_name = None

        if is_local:
            if Path(index_path).exists():
                 with open(index_path, 'r') as f:
                    index_data = json.load(f)
                 weight_file_name = next(iter(index_data["weight_map"].values()))
                 weights_path = model_path / weight_file_name
            else:
                 weight_files = list(model_path.glob("*.safetensors"))
                 if weight_files:
                     weights_path = weight_files[0]
                     weight_file_name = weights_path.name
        else: # Remote repo
             if index_path:
                 with open(index_path, 'r') as f:
                    index_data = json.load(f)
                 weight_file_name = next(iter(index_data["weight_map"].values()))
                 weights_path = hf_hub_download(repo_id=model_identifier, filename=weight_file_name)
             else:
                # This part is slow as it downloads everything to find a file
                from huggingface_hub import list_files_info
                repo_files = list_files_info(repo_id=model_identifier)
                safetensor_files = [f.rfilename for f in repo_files if f.rfilename.endswith(".safetensors")]
                if safetensor_files:
                    weight_file_name = safetensor_files[0]
                    weights_path = hf_hub_download(repo_id=model_identifier, filename=weight_file_name)

        if not weights_path or not Path(weights_path).exists():
            console.print("[bold red]Error: Could not find any .safetensors weight files.[/bold red]")
            return

        # Load the tensor keys without loading the actual data into memory
        with open(weights_path, 'rb') as f:
            header_size_bytes = f.read(8)
            header_size = int.from_bytes(header_size_bytes, 'little')
            header_data = f.read(header_size)
            header = json.loads(header_data)

        tensor_keys = list(header.keys())

        # Organize keys into a tree structure for better readability
        tree = Tree("Model Weights Structure", guide_style="bold bright_blue")
        paths = {}
        for key in sorted(tensor_keys):
            parts = key.split('.')
            current_level = paths
            for part in parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

        def add_to_tree(tree_node, path_dict):
            for key, value in sorted(path_dict.items()):
                if not value: # Leaf node
                    tree_node.add(f"[green]:file_folder: {key}[/green]")
                else:
                    child_node = tree_node.add(f"[bold blue]:open_file_folder: {key}[/bold blue]")
                    add_to_tree(child_node, value)

        add_to_tree(tree, paths)
        console.print(Panel(tree, title=f"[bold]Weight Keys from [yellow]{weight_file_name}[/yellow][/bold]", border_style="magenta"))

    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}[/bold red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Hugging Face model repositories or local directories.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face repo ID or a local path to a model directory."
    )
    args = parser.parse_args()
    inspect_model(args.model)
