"""CLI tool for post-training quantization of Hugging Face models using ONNX Runtime.

All code comments must be in English.
This script automates the process of exporting a PyTorch model to ONNX
and applying 8-bit dynamic quantization to optimize inference speed and size.
"""

import argparse
import os
import sys
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig


def main():
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description="Quantize Hugging Face models to 8-bit ONNX format for optimized CPU inference.")

    # Required positional arguments
    parser.add_argument("model_path", type=str, help="Path to the local PyTorch model directory (e.g., ./best_model)")
    parser.add_argument("output_path", type=str, help="Destination directory to save the final quantized model")

    # Optional arguments for infrastructure matching
    parser.add_argument("--arch", type=str, choices=["arm64", "avx2", "avx512", "avx512_vnni"], default="arm64", help="CPU architecture target optimization. Default is arm64 (suitable for Mac/ARM servers).")
    parser.add_argument("--temp_dir", type=str, default="./tmp_onnx", help="Temporary directory used during raw ONNX export. Default is ./tmp_onnx")

    args = parser.parse_args()

    # Validate source path existence
    if not os.path.exists(args.model_path):
        print(f"Error: Source model path '{args.model_path}' does not exist.")
        sys.exit(1)

    print(f"[*] Starting quantization pipeline for: {args.model_path}")
    print(f"[*] Target CPU architecture profile: {args.arch}")

    try:
        # Step 1: Load tokenizer and export base model to intermediate ONNX format
        print("\n[1/3] Exporting original PyTorch model to raw ONNX...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = ORTModelForSequenceClassification.from_pretrained(args.model_path, export=True)

        # Save intermediate files
        model.save_pretrained(args.temp_dir)
        tokenizer.save_pretrained(args.temp_dir)
        print(f" -> Raw ONNX successfully saved to temporary directory: {args.temp_dir}")

        # Step 2: Configure quantization strategy based on user architecture input
        print("\n[2/3] Preparing quantization configuration matrix...")
        quantizer = ORTQuantizer.from_pretrained(args.temp_dir)

        # Match config profile dynamically
        if args.arch == "arm64":
            qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
        elif args.arch == "avx2":
            qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        elif args.arch == "avx512":
            qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=False)
        elif args.arch == "avx512_vnni":
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        else:
            qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

        # Step 3: Run INT8 dynamic quantization execution block
        print("\n[3/3] Executing INT8 dynamic quantization process...")
        quantizer.quantize(
            save_dir=args.output_path,
            quantization_config=qconfig,
        )

        # Explicitly save the tokenizer to the final production folder
        tokenizer.save_pretrained(args.output_path)
        print(f"\n[+] Success! Quantized model and tokenizer saved at: {args.output_path}")

    except Exception as e:
        print(f"\n[-] Critical pipeline failure occurred: {str(e)}")
        sys.exit(1)

    finally:
        # Optional housecleaning: automated cleanup of the intermediate staging area
        if os.path.exists(args.temp_dir):
            import shutil
            shutil.rmtree(args.temp_dir)
            print("[*] Staging area cleaned up successfully.")


if __name__ == "__main__":
    main()