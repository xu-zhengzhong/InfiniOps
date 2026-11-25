#ifndef INFINI_OPS_BASE_COALESCE_H_
#define INFINI_OPS_BASE_COALESCE_H_

#include "operator.h"
#include "sparse_output.h"

namespace infini::ops {

class Coalesce : public Operator<Coalesce> {
 public:
  static ValidationResult Prepare(const Handle& handle, const Config& config,
                                  const SparseTensor& input,
                                  const SparsePrepareOptions& options,
                                  SparseOutputPlan& plan);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_COALESCE_H_
