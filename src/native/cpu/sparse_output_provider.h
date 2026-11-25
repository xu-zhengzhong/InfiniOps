#ifndef INFINI_OPS_NATIVE_CPU_SPARSE_OUTPUT_PROVIDER_H_
#define INFINI_OPS_NATIVE_CPU_SPARSE_OUTPUT_PROVIDER_H_

#include "sparse_output.h"

namespace infini::ops::native::cpu {

struct SparseOutputProvider {
  static ValidationResult Prepare(const SparseTensor&, TensorLayout,
                                  SparseOutputPlan*);
};

}  // namespace infini::ops::native::cpu

#endif  // INFINI_OPS_NATIVE_CPU_SPARSE_OUTPUT_PROVIDER_H_
