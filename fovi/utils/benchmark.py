# Copyright (c) 2026 Arthur Solère. All rights reserved.
# This file is part of the Square-FOVI repository, released under the MIT License.

import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
from statistics import mean, median
import gc
import time
import psutil
import os
import threading
from .flops import make_flop_counter
from . import add_to_all

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

__all__ = []

@add_to_all(__all__)
def measure_setup_cost(setup_fn, *args, **kwargs):
    """
    Measures CPU and GPU memory and time taken to initialize the model/dataset.
    Answers the reviewer's 'precomputation and where it lives' question.
    """
    process = psutil.Process(os.getpid())
    gc.collect()
    torch.cuda.empty_cache()
    
    cpu_mem_before = process.memory_info().rss / (1024**2)
    gpu_mem_before = torch.cuda.memory_allocated() / (1024**2)
    
    t0 = time.perf_counter()
    result = setup_fn(*args, **kwargs)
    t1 = time.perf_counter()
    
    gc.collect()
    cpu_mem_after = process.memory_info().rss / (1024**2)
    gpu_mem_after = torch.cuda.memory_allocated() / (1024**2)
    
    print(f"[Setup] Time: {t1 - t0:.2f}s | "
          f"CPU RAM Delta: {cpu_mem_after - cpu_mem_before:.2f} MB | "
          f"GPU RAM Delta: {gpu_mem_after - gpu_mem_before:.2f} MB")
    return result

@add_to_all(__all__)
class Profiler:
    """
    An end-to-end benchmarking suite for exposing architectural bottlenecks.
    Tracks wall-clock latency, sub-module execution time, compute MACs/FLOPs, 
    and true memory hoarding using real DataLoader inputs.
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.sub_module_events = {}
        self.hooks = []
        self.process = psutil.Process(os.getpid())

    def _get_module_by_path(self, path):
        mod = self.model
        for p in path.split('.'):
            mod = getattr(mod, p)
        return mod

    def _register_submodule_hooks(self, module_paths):
        self.sub_module_events = {
            path: {'starts': [], 'ends': []} for path in module_paths
        }
        
        for path in module_paths:
            mod = self._get_module_by_path(path)
            
            def pre_hook(module, input, p=path):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.sub_module_events[p]['starts'].append(event)
                
            def post_hook(module, input, output, p=path):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.sub_module_events[p]['ends'].append(event)

            self.hooks.append(mod.register_forward_pre_hook(pre_hook))
            self.hooks.append(mod.register_forward_hook(post_hook))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _measure_precomputation_memory(self):
        """Captures the static memory footprint before any forward passes."""
        gc.collect()
        torch.cuda.empty_cache()
        pre_gpu_MB = torch.cuda.memory_allocated(self.device) / (1024**2)
        pre_cpu_MB = self.process.memory_info().rss / (1024**2)
        return pre_gpu_MB, pre_cpu_MB

    def _warmup(self, inputs, amp_ctx, warmup_iters):
        """Warms up the CUDA cache and cuDNN auto-tuner."""
        with amp_ctx:
            for _ in range(warmup_iters):
                _ = self.model(*inputs) if isinstance(inputs, tuple) else self.model(inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def _phase_1a_latency(self, inputs, amp_ctx, iters):
        """Pure performance run: No overhead, no NVML polling."""
        e2e_starts, e2e_ends = [], []
        
        with amp_ctx:
            for _ in range(iters):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                _ = self.model(*inputs) if isinstance(inputs, tuple) else self.model(inputs)
                end_event.record()
                
                e2e_starts.append(start_event)
                e2e_ends.append(end_event)

        torch.cuda.synchronize()
        
        e2e_times = [s.elapsed_time(e) for s, e in zip(e2e_starts, e2e_ends)]
        mean_ms = mean(e2e_times)
        
        return {
            "e2e_mean_ms": mean_ms,
            "e2e_median_ms": median(e2e_times),
            "e2e_p99_ms": np.percentile(e2e_times, 99),
        }

    def _phase_1b_energy(self, inputs, amp_ctx, duration_seconds, batch_size, gpu_index):
        """Time-bounded run to solely average power via NVML using a background thread."""
        if not PYNVML_AVAILABLE:
            return {"energy_joules_per_img": "N/A (pynvml not installed)"}

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        
        power_readings_mw = []
        stop_event = threading.Event()
        
        def poll_power():
            while not stop_event.is_set():
                try:
                    power_readings_mw.append(pynvml.nvmlDeviceGetPowerUsage(handle))
                except pynvml.NVMLError:
                    pass
                time.sleep(0.05)  # Poll every 50ms

        monitor_thread = threading.Thread(target=poll_power)
        
        throughput_iters = 0
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        
        monitor_thread.start()
        with amp_ctx:
            while time.perf_counter() - t_start < duration_seconds:
                _ = self.model(*inputs) if isinstance(inputs, tuple) else self.model(inputs)
                throughput_iters += 1
                    
        torch.cuda.synchronize()
        actual_duration = time.perf_counter() - t_start
        
        stop_event.set()
        monitor_thread.join()
        pynvml.nvmlShutdown()

        if not power_readings_mw:
            return {"energy_joules_per_img": "N/A (duration too short)"}
        
        if throughput_iters == 0:
            return {"energy_joules_per_img": "N/A (0 iterations completed)"}

        avg_power_w = mean(power_readings_mw) / 1000.0
        energy_j_per_img = (avg_power_w * actual_duration) / (batch_size * throughput_iters)

        return {"energy_joules_per_img": energy_j_per_img}
    
    def _phase_1c_throughput(self, inputs, amp_ctx, iters, batch_size):
        """
        Pure throughput run: A tight loop with zero per-iteration tracking overhead.
        Provides the most accurate measurement of maximum images/sec.
        """
        torch.cuda.synchronize()
        t_start_wall = time.perf_counter()
        
        with amp_ctx:
            for _ in range(iters):
                _ = self.model(*inputs) if isinstance(inputs, tuple) else self.model(inputs)
                
        torch.cuda.synchronize()
        total_time_s = time.perf_counter() - t_start_wall
        
        pure_throughput_ips = (batch_size * iters) / total_time_s
        
        return {
            "throughput_images_per_sec": pure_throughput_ips,
            "throughput_total_time_s": total_time_s
        }

    def _phase_2_profiling(self, inputs, amp_ctx, sub_module_paths, export_trace_path):
        """Short run with hooks and tracing enabled to map overheads."""
        sub_module_stats = {}
        if not export_trace_path and not sub_module_paths:
            return {"sub_modules": {}}

        self._register_submodule_hooks(sub_module_paths)
        
        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=False, with_stack=False,
        ) if export_trace_path else nullcontext()

        with profiler_ctx as prof, amp_ctx:
            with record_function(f"E2E_Inference"):
                _ = self.model(*inputs) if isinstance(inputs, tuple) else self.model(inputs)
        
        torch.cuda.synchronize()
        
        if export_trace_path:
            prof.export_chrome_trace(export_trace_path)

        for path in sub_module_paths:
            times = [s.elapsed_time(e) for s, e in zip(self.sub_module_events[path]['starts'], self.sub_module_events[path]['ends'])]
            mean_time = mean(times)
            sub_module_stats[path] = {
                "mean_ms_total": mean_time,
            }
            
        self._remove_hooks()
        return {"sub_modules": sub_module_stats}

    def _phase_3_compute_flops(self, inputs):
        """Computes FLOPs (MACs) via fvcore following standard ML conventions."""
        fvcore_inputs = inputs if isinstance(inputs, tuple) else (inputs,)
        flops_counter = make_flop_counter(self.model, fvcore_inputs)
        
        gflops = flops_counter.total() / 1e9
        
        return {
            "GFLOPS_total": gflops,
        }

    @torch.inference_mode()
    def run_benchmark(
        self, 
        inputs,
        warmup=20, 
        iters=100, 
        throughput_seconds=5.0,
        sub_module_paths=None,
        export_trace_path=None,
        use_autocast=True,
        amp_dtype=torch.float16,
        gpu_index=0
    ):
        """Executes the full benchmarking pipeline."""
        sub_module_paths = sub_module_paths or []
                
        # Format inputs for device and extract batch size dynamically
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
            batch_size = inputs.shape[0]
        elif isinstance(inputs, (list, tuple)):
            inputs = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in inputs)
            batch_size = inputs[0].shape[0]
        else:
            raise ValueError("Inputs must be a torch.Tensor or a tuple/list of tensors.")
        
        # Helper to slice inputs to batch size 1 for true latency measurement
        def _slice_to_bs1(x):
            if isinstance(x, torch.Tensor):
                return x[0:1] # Preserve the batch dimension
            elif isinstance(x, (list, tuple)):
                return type(x)(_slice_to_bs1(t) if isinstance(t, torch.Tensor) else t for t in x)
            return x

        inputs_bs1 = _slice_to_bs1(inputs)

        amp_dtype = amp_dtype if use_autocast else None
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext()

        torch.backends.cudnn.benchmark = True
        self.model.eval()

        # Step 1: Precomputation Memory Track
        pre_gpu_MB, pre_cpu_MB = self._measure_precomputation_memory()

        # Step 2: Warmup
        self._warmup(inputs, amp_ctx, warmup)

        # Step 3: Phase 1A - Pure Latency
        latency_metrics = self._phase_1a_latency(inputs_bs1, amp_ctx, iters)

        peak_gpu_MB = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        activation_memory_MB = peak_gpu_MB - pre_gpu_MB

        # Step 4: Phase 1B - Energy Only (NVML polling isolated here)
        energy_metrics = self._phase_1b_energy(
            inputs, amp_ctx, throughput_seconds, batch_size, gpu_index
        )

        # Step 5: Phase 1C - Pure Throughput
        throughput_metrics = self._phase_1c_throughput(
            inputs, amp_ctx, iters, batch_size
        )

        # Step 6: Phase 2 - Sub-module Overheads & Tracing
        trace_metrics = self._phase_2_profiling(
            inputs, amp_ctx, sub_module_paths, export_trace_path
        )

        # Step 7: Phase 3 - Compute FLOPs
        compute_metrics = self._phase_3_compute_flops(inputs_bs1)

        # Merge and return all metrics
        final_report = {
            "precomputed_static_GPU_MB": pre_gpu_MB,
            "precomputed_static_CPU_MB": pre_cpu_MB,
            "peak_activation_GPU_MB": activation_memory_MB
        }
        final_report.update(latency_metrics)
        final_report.update(throughput_metrics)
        final_report.update(energy_metrics)
        final_report.update(trace_metrics)
        final_report.update(compute_metrics)

        return final_report