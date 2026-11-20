#include <infini/ops.h>

int main() {
  using namespace infini::ops;
  const Device cpu{Device::Type::kCpu};
  const Tensor indices{nullptr, {2, 0}, DataType::kInt64, cpu};
  const Tensor values{nullptr, {0}, DataType::kFloat32, cpu};
  std::optional<SparseTensor> sparse;
  if (!SparseTensor::CreateCoo({2, 3}, indices, values, sparse)) return 1;

  const Handle handle;
  const Config config;
  if (!SparseProbe::CanMake(config, *sparse, 2)) return 2;
  if (!SparseProbe::Make(*sparse, 2)) return 3;
  if (!SparseProbe::CheckedCall(handle, config, *sparse, 2).ok()) return 4;
  if (SparseProbe::CheckedCall(handle, config, *sparse, 0.).code !=
      ValidationCode::kInvalidMetadata)
    return 5;

  std::vector<std::optional<SparseTensor>> masks{std::nullopt, sparse};
  if (!SparseProbe::CanMake(config, values, masks)) return 6;
  if (!SparseProbe::Make(values, masks)) return 7;
  if (!SparseProbe::CheckedCall(handle, config, values, masks).ok()) return 8;
  masks.clear();
  if (SparseProbe::CheckedCall(handle, config, values, masks).code !=
      ValidationCode::kInvalidStructure)
    return 9;

  std::optional<std::vector<SparseTensor>> list;
  if (!SparseProbe::Make(config, list, values)) return 10;
  if (SparseProbe::CheckedCall(handle, config, list, values).code !=
      ValidationCode::kInvalidMetadata)
    return 11;
  list.emplace();
  if (SparseProbe::CheckedCall(handle, config, list, values).code !=
      ValidationCode::kInvalidStructure)
    return 12;
  list->push_back(*sparse);
  if (!SparseProbe::CheckedCall(handle, config, list, values).ok()) return 13;

  SparseProbe::Call(handle, config, values, 1.);
  return 0;
}
