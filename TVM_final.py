import numpy as np
import tvm
from tvm import te, topi
import psutil
import time

# Parameters
N, CI, H, W = 1, 64, 56, 56   # batch, in_channels, height, width
CO, KH, KW = 128, 3, 3        # out_channels, kernel height/width
stride, padding = 1, 1

# Placeholders
data = te.placeholder((N, CI, H, W), name="data")
kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

# ---- Winograd Convolution ----
winograd_out = topi.nn.conv2d_winograd_nchw(
    data, kernel, stride, padding, dilation=1, out_dtype="float32"
)
s_winograd = te.create_schedule(winograd_out.op)
func_winograd = tvm.build(s_winograd, [data, kernel, winograd_out], target="llvm")

# ---- Standard Convolution ----
standard_out = topi.nn.conv2d_nchw(
    data, kernel, stride, padding, dilation=1, out_dtype="float32"
)
s_standard = te.create_schedule(standard_out.op)
func_standard = tvm.build(s_standard, [data, kernel, standard_out], target="llvm")

# ---- Test data ----
dev = tvm.cpu()
a_np = np.random.uniform(-1, 1, (N, CI, H, W)).astype("float32")
w_np = np.random.uniform(-1, 1, (CO, CI, KH, KW)).astype("float32")

a_tvm = tvm.nd.array(a_np, device=dev)
w_tvm = tvm.nd.array(w_np, device=dev)
b_winograd = tvm.nd.empty((N, CO, H, W), device=dev)
b_standard = tvm.nd.empty((N, CO, H, W), device=dev)

# ---- Warmup ----
func_winograd(a_tvm, w_tvm, b_winograd)
func_standard(a_tvm, w_tvm, b_standard)

# ---- Power estimation ----
def get_cpu_power_estimate():
    """Crude estimate of CPU power based on utilization"""
    idle_power = 5.0   # W
    max_power = 45.0   # W (replace with your CPU’s TDP if known)
    usage = psutil.cpu_percent(interval=0.1) / 100.0
    return idle_power + (max_power - idle_power) * usage

# ---- Runtime measurement ----
def benchmark(func, a, w, b, name):
    # power before
    power_before = get_cpu_power_estimate()

    # measure with tvm.time_evaluator
    time_f = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
    prof_res = time_f(a, w, b).results

    # power after
    power_after = get_cpu_power_estimate()
    avg_power = (power_before + power_after) / 2.0

    # latency / throughput
    mean_ms = np.mean(prof_res) * 1e3
    std_ms = np.std(prof_res) * 1e3
    num_ops = N * CO * H * W * CI * KH * KW
    gops = (2 * num_ops) / (mean_ms * 1e-3) / 1e9

    print(f"✅ {name} Conv2D executed")
    print("Output shape:", b.shape)
    print(f"Mean runtime: {mean_ms:.3f} ms (std {std_ms:.3f} ms)")
    print(f"Throughput: {gops:.2f} GOPS")
    print(f"Estimated Avg Power: {avg_power:.2f} W")
    print("-" * 60)

# Run benchmarks
benchmark(func_winograd, a_tvm, w_tvm, b_winograd, "Winograd")
benchmark(func_standard, a_tvm, w_tvm, b_standard, "Standard")
