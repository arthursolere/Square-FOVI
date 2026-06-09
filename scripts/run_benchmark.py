# Copyright (c) 2026 Arthur Solère. All rights reserved.
# This file is part of the Square-FOVI repository, released under the MIT License.

#!/usr/bin/env python3
import argparse
import json
import torch
import pprint
from pathlib import Path

# Adjust these imports to match your project's module structure
from fovi import get_trainer_from_base_fn, get_model_from_base_fn
from fovi.utils.benchmark import Profiler, measure_setup_cost
from fovi.utils.flops import FlopWrapper

class BenchmarkWrapper(torch.nn.Module):
    """A clean wrapper to inject kwargs and freeze gradients for pure model benchmarking."""
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.kwargs = kwargs
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, inputs):
        return self.model(*inputs, **self.kwargs) if isinstance(inputs, tuple) else self.model(inputs, **self.kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Profiler for FOVI Compute Benchmarking")
    
    # Model & Data args
    parser.add_argument("--base_fn", type=str, default=None, help="Path to the model checkpoint/base function")
    parser.add_argument("--hf_model", type=str, default=None, help="Directly load a HuggingFace model")
    parser.add_argument("--use_trainer", action="store_true", help="Initialize full Trainer (loads datasets, higher CPU memory)")
    parser.add_argument("--n_fixations", type=int, nargs='+', default=None, help="List of fixations to evaluate (e.g., 1 3 5)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run benchmark on (e.g., 'cuda:0')")
    
    # Random Input Override
    parser.add_argument("--random_input_shape", type=int, nargs='+', default=None, 
                        help="Bypass dataloader and use a random tensor (e.g., 128 3 224 224)")

    # Benchmark args
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations for latency")
    parser.add_argument("--throughput_seconds", type=float, default=5.0, help="Duration for throughput test")
    parser.add_argument("--use_autocast", action="store_true", help="Enable AMP (overrides/combines with config)")
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="Dtype for autocast (e.g., 'bfloat16')")
    parser.add_argument("--sub_module_paths", type=str, nargs='*', default=[], 
                        help="Space-separated list of module paths to track (e.g., network.backbone.embeddings)")
    
    # Output args
    parser.add_argument("--output_json", type=str, default=None, help="Path to save results")
    parser.add_argument("--export_trace", type=str, default=None, help="Path to export Chrome trace (optional)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.base_fn is None and args.hf_model is None:
        raise ValueError("You must provide either --base_fn or --hf_model.")

    # Parse GPU index for NVML energy polling
    gpu_index = int(args.device.split(":")[1]) if ":" in args.device else 0

    all_results = {}
    
    # ==========================================
    # BRANCH 1: RAW HUGGINGFACE BASELINE
    # ==========================================
    if args.hf_model:
        print(f"=== Starting Benchmark Pipeline for HuggingFace Model: {args.hf_model} ===")
        from transformers import AutoModel
        
        if args.random_input_shape is None:
            raise ValueError("You MUST provide --random_input_shape when using --hf_model.")
            
        print("\n--- Measuring Setup & Precomputation Cost ---")
        def load_hf_model():
            model = AutoModel.from_pretrained(args.hf_model)
            return model.to(args.device) 
            
        model = measure_setup_cost(load_hf_model)
        wrapper = BenchmarkWrapper(model)
        wrapper.eval()
        
        inputs = torch.randn(*args.random_input_shape, device=args.device)
        if args.use_autocast:
            amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        else:
            amp_dtype = None
        
        print(f"\n--- Running End-to-End Profiler (Autocast: {args.use_autocast}) ---")
        profiler = Profiler(model=wrapper, device=args.device)
        results = profiler.run_benchmark(
            inputs=inputs, warmup=args.warmup, iters=args.iters, throughput_seconds=args.throughput_seconds,
            export_trace_path=args.export_trace, use_autocast=args.use_autocast, amp_dtype=amp_dtype,
            gpu_index=gpu_index, sub_module_paths=args.sub_module_paths
        )
        
        h, w = args.random_input_shape[2], args.random_input_shape[3]
        patches = (h // 16) * (w // 16)
        results.update({
            "hf_model": args.hf_model,
            "input_source": "random_tensor",
            "patches_processed": patches,
            "random_input_shape": args.random_input_shape
        })
        all_results["baseline"] = results

    # ==========================================
    # BRANCH 2: FOVI MODEL (WITH OR WITHOUT TRAINER)
    # ==========================================
    else:
        print(f"=== Starting Benchmark Pipeline for FOVI Model: {args.base_fn} ===")
        print(f"--- Mode: {'Trainer (Full Pipeline)' if args.use_trainer else 'Pure Model (Compute Only)'} ---")
        
        # --- Loading Logic ---
        print("\n--- Measuring Setup & Precomputation Cost ---")
        if args.use_trainer:
            def load_trainer(): return get_trainer_from_base_fn(args.base_fn, quiet=True, load=True)
            core_obj = measure_setup_cost(load_trainer)
            cfg = getattr(core_obj, 'cfg', None)
        else:
            if args.random_input_shape is None:
                raise ValueError("You MUST provide --random_input_shape when NOT using --use_trainer.")
            def load_model(): return get_model_from_base_fn(args.base_fn, quiet=True, load=True, device=args.device)
            core_obj = measure_setup_cost(load_model)
            cfg = getattr(core_obj, 'cfg', None)

        # --- AMP & Fixations Extraction ---
        cfg_use_amp = getattr(cfg.training, 'use_amp', False) if cfg and hasattr(cfg, 'training') else False
        cfg_amp_dtype_str = getattr(cfg.training, 'amp_dtype', "float16") if cfg and hasattr(cfg, 'training') else "float16"
        
        use_autocast = args.use_autocast or cfg_use_amp
        amp_dtype = torch.bfloat16 if cfg_amp_dtype_str == "bfloat16" else (torch.float32 if cfg_amp_dtype_str == "float32" else torch.float16)
        if amp_dtype == torch.float32: use_autocast = False
            
        print(f"\n--- AMP Settings: use_autocast={use_autocast}, dtype={amp_dtype} ---")
        
        if args.n_fixations is not None:
            use_n_fixations_list = args.n_fixations
        elif cfg and hasattr(cfg, 'saccades') and hasattr(cfg.saccades, 'n_fixations_val'):
            use_n_fixations_list = cfg.saccades.n_fixations_val
        else:
            use_n_fixations_list = [None]
                
        # --- Inputs Generation ---
        if args.random_input_shape is not None:
            print(f"--- Bypassing DataLoader: Using Random Tensor of shape {args.random_input_shape} ---")
            inputs = torch.randn(*args.random_input_shape, device=args.device)
        else:
            print("--- Fetching representative inputs from val_loader ---")
            temp_wrapper = FlopWrapper(core_obj)
            inputs = temp_wrapper.get_inputs(core_obj.val_loader)
        
        # --- Benchmark Loop ---
        for n_fix in use_n_fixations_list:
            run_key = f"n_fixations_{n_fix}" if n_fix is not None else "baseline"
            print(f"\n=== Benchmarking {run_key} ===")
            kwargs = {'n_fixations': n_fix} if n_fix is not None else {}
            
            # Use appropriate wrapper
            if args.use_trainer:
                wrapper = FlopWrapper(core_obj, **kwargs)
            else:
                wrapper = BenchmarkWrapper(core_obj, **kwargs)
            wrapper.eval()
            
            profiler = Profiler(model=wrapper, device=args.device)
            results = profiler.run_benchmark(
                inputs=inputs, warmup=args.warmup, iters=args.iters, throughput_seconds=args.throughput_seconds,
                export_trace_path=args.export_trace, use_autocast=use_autocast, amp_dtype=amp_dtype,
                gpu_index=gpu_index, sub_module_paths=args.sub_module_paths
            )
            
            # Extract dynamically from FoviNet or Trainer
            if hasattr(core_obj, 'num_coords'):
                patches = core_obj.num_coords
            elif hasattr(core_obj, 'model') and hasattr(core_obj.model, 'num_coords'):
                patches = core_obj.model.num_coords
            else:
                patches = "Unknown"

            results.update({
                "base_fn": args.base_fn,
                "n_fixations": n_fix,
                "device": args.device, 
                "use_autocast": use_autocast,
                "amp_dtype": str(amp_dtype),
                "input_source": "random_tensor" if args.random_input_shape else "val_loader",
                "patches_processed": patches
            })
            if args.random_input_shape: results["random_input_shape"] = args.random_input_shape
                
            all_results[run_key] = results
    
    # ==========================================
    # SAVE & EXPORT
    # ==========================================
    print("\n=== Final Benchmark Report ===")
    pprint.pprint(all_results, sort_dicts=False)
    
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nResults successfully saved to {out_path}")

if __name__ == "__main__":
    main()