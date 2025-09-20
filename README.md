Winograd Convolution Performance Analysis
This repository contains implementations and performance analysis of Winograd convolution algorithms on CPU and GPU platforms, as described in the seminar paper "Exploring CPU and GPU Implementations of Winograd Convolution Algorithms for EdgeAI Inference".

📄 Project Overview
The project investigates the performance of Winograd convolution compared to standard direct convolution methods, focusing on:

Latency (execution time)

Throughput (GFLOPs)

Power consumption (Watts)

Experiments were conducted on both CPU (using TVM) and GPU (using cuDNN) platforms using the CIFAR-100 dataset.

📁 Repository Structure
text
├── 4dtensor_to_winograd.cu      # GPU implementation using Winograd convolution (cuDNN)
├── direct_convolution_GPU.cu    # GPU implementation using direct convolution (cuDNN)
├── TVM_final.py                 # CPU implementation using TVM (Winograd + Standard)
├── Winograd convolution report.pdf     # Seminar paper with detailed analysis
└── README.md                    # This file
🛠️ Requirements
GPU Implementation (CUDA/cuDNN)
NVIDIA GPU with CUDA support

CUDA Toolkit

cuDNN library

NVML for power monitoring

CPU Implementation (TVM)
Python 3.x

TVM with LLVM backend

NumPy

psutil

🚀 How to Run
GPU Programs (Winograd vs Direct Convolution)
Compile and run the CUDA programs with:

bash
# Compile
nvcc -o winograd_conv 4dtensor_to_winograd.cu -lcudnn -lnvml
nvcc -o direct_conv direct_convolution_GPU.cu -lcudnn -lnvml

# Run with default batch size (64)
./winograd_conv
./direct_conv

# Run with custom batch size
./winograd_conv --batch 128
./direct_conv --batch 128
CPU Program (TVM Implementation)
Run the Python script:

bash
python TVM_final.py
📊 Key Findings
From the seminar paper and implementations:

Winograd convolution reduces arithmetic operations by approximately 2.25-3× compared to direct convolution for 3×3 kernels

GPU implementations generally show higher throughput but also higher power consumption

CPU implementations using TVM demonstrate the flexibility of Winograd optimization on general-purpose hardware

The trade-off between performance and power consumption is crucial for EdgeAI applications

📈 Performance Metrics
The implementations measure and report:

Latency: Execution time per convolution operation (ms)

Throughput: Computational throughput (GFLOPs/GOPS)

Power consumption: Average power draw during execution (Watts)

