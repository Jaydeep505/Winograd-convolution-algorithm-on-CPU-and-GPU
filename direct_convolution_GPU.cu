#include <iostream> // for printing to console
#include <fstream> // for reading files
#include <vector> // for dynamic arrays (std::vector)
#include <chrono> // for timing
#include <cstdint> // for fixed size types like uint8_t
#include <cuda_runtime.h> // CUDA Runtime API
#include <cudnn.h> // NVIDIA cuDNN library (deep learning)
#include <nvml.h>  // for power monitoring

// Macros for error checking 
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDNN(call) { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_NVML(call) { \
    nvmlReturn_t status = call; \
    if (status != NVML_SUCCESS) { \
        std::cerr << "NVML Error: " << nvmlErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

const int IMG_SIZE = 1024;   // One channnel
const int IMAGE_BYTES = 3074; // 3 channels 
const int NUM_IMAGES = 50000;
const int in_channels = 3;   // RGB images, 3 channels
const int out_channels = 64; // Number of output feature maps after convolution
const int input_h = 32;
const int input_w = 32;
const int kernel_h = 3;
const int kernel_w = 3;
const int pad_h = 1;
const int pad_w = 1;
const int stride_h = 1;
const int stride_w = 1;

struct CifarImage {
    uint8_t coarse_label;
    uint8_t fine_label;
    std::vector<float> data; // 3*1024 normalized floats
};

CifarImage parse_image(const std::vector<uint8_t>& raw) {// Reads the raw binary data of one image 
    CifarImage img;
    img.coarse_label = raw[0]; // Coarse label
    img.fine_label = raw[1];   // Fine label
    img.data.resize(3 * IMG_SIZE);
    for (int i = 0; i < IMG_SIZE; ++i)
        img.data[i] = raw[2 + i] / 255.0f;
    for (int i = 0; i < IMG_SIZE; ++i)
        img.data[IMG_SIZE + i] = raw[2 + IMG_SIZE + i] / 255.0f;
    for (int i = 0; i < IMG_SIZE; ++i)
        img.data[2 * IMG_SIZE + i] = raw[2 + 2 * IMG_SIZE + i] / 255.0f;
    return img;
}

std::vector<CifarImage> load_cifar100_bin(const std::string& filepath, int max_images = NUM_IMAGES) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filepath << std::endl;
        exit(1);
    }
    std::vector<CifarImage> dataset;
    std::vector<uint8_t> buffer(IMAGE_BYTES);
    for (int i = 0; i < max_images; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), IMAGE_BYTES);
        if (file.gcount() < IMAGE_BYTES) break;
        dataset.push_back(parse_image(buffer));
    }
    return dataset;
}

int main(int argc, char* argv[]) { // this lets you parse arguments from terminal
    int batch_size = 64; // Default value

    if(argc >= 3 && std::string(argv[1]) == "--batch"){
        batch_size = std::stoi(argv[2]);
        std::cout << "Using batch size: " << batch_size << std::endl;
    } else{
        std::cout << "Using default batch size: 64\n";
    }

    CHECK_NVML(nvmlInit());  // Initialize NVIDIA Management Library
    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &device)); // Get handle for GPU 0

    CHECK_CUDA(cudaSetDevice(0)); // Set CUDA device 0 for processing

    auto images = load_cifar100_bin("../cifar-100-binary/train.bin", batch_size);
    if (images.size() < batch_size) {
        std::cerr << "No enough images loaded." << std::endl;
        return -1;
    }

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn)); // Create cuDNN context

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, in_channels, input_h, input_w));

    cudnnFilterDescriptor_t filterDesc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_channels, in_channels, kernel_h, kernel_w));

    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_h, stride_w,
        1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    int out_n, out_c, out_h, out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &out_n, &out_c, &out_h, &out_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

    size_t input_size = batch_size * in_channels * input_h * input_w * sizeof(float);
    size_t filter_size = out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
    size_t output_size = out_n * out_c * out_h * out_w * sizeof(float);

    float* d_input;
    CHECK_CUDA(cudaMalloc(&d_input, input_size));

    std::vector<float> input_tensor(batch_size * 3 * 32 * 32);

    // Prepare Input Tensor in NCHW Format ready for cuDNN
    for (int n = 0; n < batch_size; ++n){
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 32; ++h) {
                for (int w = 0; w < 32; ++w) {
                    int idx = n * (3* 32 * 32) + c * 32 * 32 + h * 32 + w;
                    int src_idx = c * 32 * 32 + h * 32 +w;
                    input_tensor[idx] = images[n].data[src_idx];
                }
            }
        }
    }
    CHECK_CUDA(cudaMemcpy(d_input, input_tensor.data(), input_size, cudaMemcpyHostToDevice));

    float* d_filter;
    CHECK_CUDA(cudaMalloc(&d_filter, filter_size));
    std::vector<float> h_filter(filter_size / sizeof(float), 0.1f);
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter.data(), filter_size, cudaMemcpyHostToDevice));

    float* d_output;
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspace_size));
    void* d_workspace = nullptr;
    if (workspace_size > 0) CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));

    // Warm-up convolution
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_filter, convDesc, algo, d_workspace, workspace_size, &beta, outputDesc, d_output));
    CHECK_CUDA(cudaDeviceSynchronize());

    unsigned int power_mW;
    CHECK_NVML(nvmlDeviceGetPowerUsage(device, &power_mW));
    float power_before = power_mW / 1000.0f;
std::cout << "convolution is running " << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_filter, convDesc, algo, d_workspace, workspace_size, &beta, outputDesc, d_output));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    CHECK_NVML(nvmlDeviceGetPowerUsage(device, &power_mW));
    float power_after = power_mW / 1000.0f;
    float avg_power = (power_before + power_after) / 2.0f;

    float latency_ms = std::chrono::duration<double>(end - start).count() * 1000 / 100.0f;
    float GFLOPs = 2.0f * batch_size * out_channels * input_h * input_w * in_channels * kernel_h * kernel_w / 1e9;
    float throughput = GFLOPs / (latency_ms / 1000.0f);

    std::cout << "Latency: " << latency_ms << " ms, Throughput: " << throughput << " GFLOPs, Avg. Power: " << avg_power << " W" << std::endl;

    if (d_workspace) cudaFree(d_workspace);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);
    CHECK_NVML(nvmlShutdown());

    return 0;
}
