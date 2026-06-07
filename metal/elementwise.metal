// Elementwise kernels — one thread per output element.
//
// Generic shape: tensors are flattened into a 1D length `n`. Higher-rank
// shape and strides are the dispatcher's problem (we assume row-major
// contiguous storage at this stage; Transpose/Broadcast materialize new
// buffers).

#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add_bf16(
    device const bfloat *a   [[buffer(0)]],
    device const bfloat *b   [[buffer(1)]],
    device bfloat       *out [[buffer(2)]],
    constant uint       &n   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] + b[gid];
}
