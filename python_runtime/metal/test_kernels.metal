// Test kernels used by the buffer-bridge smoke path. Kept minimal —
// just enough math to confirm a kernel actually wrote to the buffer
// the host can see via mx.array.

#include <metal_stdlib>
using namespace metal;

// `out[i] = a[i] * scale` over `n` fp32 elements. The simplest kernel
// that exercises read + write of host-shared memory.
kernel void scale_f32(
    device const float* a    [[buffer(0)]],
    device float*       out  [[buffer(1)]],
    constant float&     scale[[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] * scale;
}

// `out[i] = a[i] + b[i]`. Three separate buffers — exercises the
// multi-buffer dispatch path through pyo3.
kernel void add_f32(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] + b[gid];
}
