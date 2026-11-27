#include <cuda_runtime_api.h>
#include <infini/ops.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace infini::ops;
using Code = ValidationCode;

#define CHECK(...)                                           \
  do {                                                       \
    if (!(__VA_ARGS__)) {                                    \
      std::cerr << __LINE__ << ": " << #__VA_ARGS__ << '\n'; \
      return 1;                                              \
    }                                                        \
  } while (false)

int main(int argc, char**) {
  const Device gpu{Device::Type::kNvidia};
  Config config;
  config.set_implementation_index(0);
  Handle handle;
  SparseOutputPlan plan;
  std::optional<SparseTensor> input;
  const auto f32 = DataType::kFloat32, i64 = DataType::kInt64;

  // These cases finish during metadata/selection checks, before any device
  // access.
  const auto n = std::size_t{1} + std::numeric_limits<int>::max();
  std::int64_t coord{};
  float value{};
  CHECK(SparseTensor::CreateCoo({1, 1}, {&coord, {2, n}, i64, gpu},
                                {&value, {n}, f32, gpu}, input));
  CHECK(Coalesce::Prepare(handle, config, *input, {}, plan).code ==
        Code::kUnsupportedCapability);
  const auto huge =
      std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t);
  CHECK(SparseTensor::CreateCoo({huge, 0}, {nullptr, {2, 0}, i64, gpu},
                                {nullptr, {0}, f32, gpu}, input));
  CHECK(ToSparseCsr::Prepare(handle, config, *input, {}, plan).code ==
        Code::kIntegerOverflow);
  CHECK(SparseTensor::CreateCoo({1, 1}, {&coord, {2, 1}, i64, gpu},
                                {&value, {1}, f32, gpu}, input));
  const Tensor alias{&value, {1, 1}, f32, gpu};
  CHECK(ToDense::CheckedCall(handle, config, *input, alias).code ==
        Code::kAliasing);
  Config missing;
  missing.set_implementation_index(8);
  CHECK(ToDense::CheckedCall(handle, missing, *input, alias).code ==
        Code::kUnsupportedImplementation);
  if (argc == 1) return 0;

  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
        cudaSuccess);
  handle.set_stream(stream);
  // Coordinate and value owners outlive the synchronous checked calls.
  std::int64_t* indices;
  float *values, *out;
  CHECK(cudaMalloc(&indices, 4 * sizeof(*indices)) == cudaSuccess);
  CHECK(cudaMalloc(&values, 2 * sizeof(*values)) == cudaSuccess);
  CHECK(cudaMalloc(&out, 2 * sizeof(*out)) == cudaSuccess);
  std::array<std::int64_t, 4> coords{0, 0, 1, 1};
  std::array<float, 2> entries{4, -1};
  CHECK(cudaMemcpyAsync(indices, coords.data(), sizeof(coords),
                        cudaMemcpyHostToDevice, stream) == cudaSuccess);
  CHECK(cudaMemcpyAsync(values, entries.data(), sizeof(entries),
                        cudaMemcpyHostToDevice, stream) == cudaSuccess);
  CHECK(SparseTensor::CreateCoo({1, 2}, {indices, {2, 2}, i64, gpu},
                                {values, {2}, f32, gpu}, input));
  const Tensor dense{out, {1, 2}, f32, gpu};
  CHECK(ToDense::CheckedCall(handle, config, *input, dense));
  CHECK(cudaStreamQuery(stream) == cudaSuccess);
  std::array<float, 2> result;
  CHECK(cudaMemcpy(result.data(), out, sizeof(result),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(result[0] == 0 && result[1] == 3);
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) ==
        cudaSuccess);
  CHECK(ToDense::CheckedCall(handle, config, *input, dense).code ==
        Code::kUnsupportedCapability);
  cudaGraph_t graph;
  CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  CHECK(cudaGraphDestroy(graph) == cudaSuccess);
  CHECK(cudaFree(out) == cudaSuccess);
  CHECK(cudaFree(values) == cudaSuccess);
  CHECK(cudaFree(indices) == cudaSuccess);
  CHECK(cudaStreamDestroy(stream) == cudaSuccess);
  return 0;
}
