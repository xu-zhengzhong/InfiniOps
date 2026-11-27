#ifndef INFINI_OPS_NVIDIA_SPARSE_CUH_
#define INFINI_OPS_NVIDIA_SPARSE_CUH_

#include <cuda_runtime.h>

#include <algorithm>

#include "common/sparse_buffers.h"

namespace infini::ops::native::nvidia::sparse {

constexpr int kThreads = 256;

inline int Blocks(std::size_t n) {
  return static_cast<int>(
      std::min<std::size_t>((n + kThreads - 1) / kThreads, 65535));
}

__device__ inline std::size_t Thread() {
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ inline std::size_t Step() {
  return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

struct CudaFailure {
  cudaError_t error;
};

inline void Check(cudaError_t error) {
  if (error != cudaSuccess) throw CudaFailure{error};
}

inline ValidationResult Failure(cudaError_t error) {
  if (error == cudaErrorMemoryAllocation)
    return {ValidationCode::kAllocationFailure};
  if (error == cudaErrorStreamCaptureImplicit ||
      error == cudaErrorStreamCaptureUnsupported)
    return {ValidationCode::kUnsupportedCapability};
  return {ValidationCode::kExecutionFailure};
}

// One diagnostic per invocation. The winning thread writes position and
// component. The host observes all fields only after stream completion.
struct Diagnostic {
  int code;
  ValidationComponent component;
  std::size_t position;
};

ValidationResult Validate(const SparseTensor& input, cudaStream_t stream,
                          Diagnostic* scratch);

class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    Check(cudaGetDevice(&previous_));
    changed_ = previous_ != device;
    if (changed_) Check(cudaSetDevice(device));
  }
  ~DeviceScope() {
    if (changed_) (void)cudaSetDevice(previous_);
  }

 private:
  int previous_{};
  bool changed_{};
};

template <typename Work>
ValidationResult WithInput(void* raw_stream, const SparseTensor& input,
                           Diagnostic* scratch, Work work) {
  try {
    DeviceScope guard(input.device().index());
    auto stream = static_cast<cudaStream_t>(raw_stream);
    cudaStreamCaptureStatus capture;
    Check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      return {ValidationCode::kUnsupportedCapability};
    try {
      auto status = Validate(input, stream, scratch);
      if (status) status = work(stream);
      Check(cudaStreamSynchronize(stream));
      return status;
    } catch (...) {
      // Drain earlier successful launches before releasing borrowed buffers.
      (void)cudaStreamSynchronize(stream);
      throw;
    }
  } catch (CudaFailure failure) {
    return Failure(failure.error);
  }
}

template <typename T>
const T* Read(const Tensor& tensor) {
  return static_cast<const T*>(tensor.data());
}

template <typename T>
T* Write(Tensor tensor) {
  return static_cast<T*>(tensor.data());
}

template <typename F>
void VisitTypes(const SparseTensor& input, F f) {
  if (input.index_dtype() == DataType::kInt32) {
    if (input.dtype() == DataType::kFloat32)
      f(std::int32_t{}, float{});
    else
      f(std::int32_t{}, double{});
  } else {
    if (input.dtype() == DataType::kFloat32)
      f(std::int64_t{}, float{});
    else
      f(std::int64_t{}, double{});
  }
}

inline void Copy(const Tensor& src, const Tensor& dst, cudaStream_t stream) {
  const auto bytes = src.numel() * src.element_size();
  if (bytes)
    Check(cudaMemcpyAsync(Write<void>(dst), src.data(), bytes,
                          cudaMemcpyDeviceToDevice, stream));
}

}  // namespace infini::ops::native::nvidia::sparse

#endif
