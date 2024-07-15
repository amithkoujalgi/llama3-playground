#!/usr/bin/env python
import argparse
import os
import subprocess
import sys

from llama3_playground.core.utils import ModelManager


def main():
    parser = argparse.ArgumentParser(description="CLI for playground")

    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '-l',
        '--list-models',
        action='store_true',
        dest='list_models_option',
        help='List models'
    )
    group.add_argument(
        '-t',
        '--train',
        action='store_true',
        dest='train_model_option',
        help='Train the base model with dataset and create a trained model (with LoRA adapters and full/merged model)'
    )
    group.add_argument(
        '-i',
        '--infer',
        action='store_true',
        dest='infer_model_option',
        help='Infer from the model'
    )

    # Add subparsers for the inference options
    subparsers = parser.add_subparsers(dest='infer_command', required=False)

    # Subparser for inference with a specific model
    infer_model_parser = subparsers.add_parser('model', help='Infer using a specific model')
    infer_model_parser.add_argument(
        '-m', '--model-name',
        type=str,
        required=True,
        help='Name of the model to infer from'
    )
    args: argparse.Namespace = parser.parse_args()

    # then check which option was selected
    if args.list_models_option:
        list_models()
    elif args.train_model_option:
        train_model()


def train_model():
    import llama3_playground
    module_path = llama3_playground.__file__.replace('__init__.py', '')
    module_path = os.path.join(module_path, 'core', 'train.py')
    train_cmd = [sys.executable, module_path]
    subprocess.Popen(train_cmd).communicate()


def list_models():
    trained_models = ModelManager.list_trained_models(include_lora_adapters=True)
    if len(trained_models) == 0:
        print('No trained models found')
    else:
        print(f"Listing {len(trained_models)} trained models:")
        latest = ModelManager.get_latest_model(lora_adapters_only=False)

        for model in trained_models:
            if latest in model:
                print(f'- {model} (latest)')
            else:
                print(f'- {model}')


if __name__ == "__main__":
    main()
