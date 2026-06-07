#![allow(dead_code)]

//! MSL string-building helper.
//!
//! Kernel templates accumulate code into a `MslWriter`, which manages
//! indentation and gives us a single chokepoint if we ever want to
//! lint or pretty-print the generated source.

use std::fmt::Write;

/// Append-only buffer with indentation tracking.
pub struct MslWriter {
    buf: String,
    indent: usize,
}

impl Default for MslWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl MslWriter {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            indent: 0,
        }
    }

    pub fn push_indent(&mut self) {
        self.indent += 1;
    }

    pub fn pop_indent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    /// Write a line with current indentation.
    pub fn line(&mut self, s: impl AsRef<str>) {
        for _ in 0..self.indent {
            self.buf.push_str("    ");
        }
        self.buf.push_str(s.as_ref());
        self.buf.push('\n');
    }

    /// Write a blank line (no indentation).
    pub fn blank(&mut self) {
        self.buf.push('\n');
    }

    /// `writeln!`-style with current indentation.
    pub fn writeln(&mut self, args: std::fmt::Arguments<'_>) {
        for _ in 0..self.indent {
            self.buf.push_str("    ");
        }
        let _ = self.buf.write_fmt(args);
        self.buf.push('\n');
    }

    pub fn finish(self) -> String {
        self.buf
    }
}

/// Common preamble — Metal stdlib include and namespace pull.
pub fn prelude(w: &mut MslWriter) {
    w.line("#include <metal_stdlib>");
    w.line("#include <metal_atomic>");
    w.line("#include <metal_simdgroup_matrix>");
    w.line("#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>) && __has_include(\"mlx/backend/metal/kernels/steel/gemm/gemm.h\") && __has_include(\"mlx/backend/metal/kernels/steel/gemm/nax.h\") && __has_include(\"mlx/backend/metal/kernels/quantized_nax.h\")");
    w.line("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>");
    w.line("#include \"mlx/backend/metal/kernels/utils.h\"");
    w.line("#include \"mlx/backend/metal/kernels/steel/utils.h\"");
    w.line("#include \"mlx/backend/metal/kernels/steel/gemm/gemm.h\"");
    w.line("#include \"mlx/backend/metal/kernels/steel/gemm/nax.h\"");
    w.line("#include \"mlx/backend/metal/kernels/quantized_nax.h\"");
    w.line("#define FUSEKIT_HAVE_MLX_NAX 1");
    if std::env::var("FUSEKIT_QMV_M_QDOT").as_deref() == Ok("1") {
        w.line("#define FUSEKIT_QMV_M_QDOT 1");
    } else {
        w.line("#define FUSEKIT_QMV_M_QDOT 0");
    }
    let qmv_fast_qdot_all = std::env::var("FUSEKIT_QMV_FAST_QDOT").ok();
    let qmv_fast_qdot_bit = |bits: u32, default_on: bool| -> bool {
        let key = format!("FUSEKIT_QMV_FAST_QDOT_Q{bits}");
        if let Ok(v) = std::env::var(key) {
            return v.as_str() != "0";
        }
        if let Some(v) = &qmv_fast_qdot_all {
            return v.as_str() != "0";
        }
        default_on
    };
    for (bits, default_on) in [(4u32, false), (5u32, true), (6u32, true), (8u32, true)] {
        let enabled = if qmv_fast_qdot_bit(bits, default_on) {
            1
        } else {
            0
        };
        w.line(format!("#define FUSEKIT_QMV_FAST_QDOT_Q{bits} {enabled}"));
    }
    let qmv_fast_qparam_broadcast = std::env::var("FUSEKIT_QMV_FAST_QPARAM_BROADCAST")
        .ok()
        .map(|s| s.as_str() != "0")
        .unwrap_or(false);
    w.line(format!(
        "#define FUSEKIT_QMV_FAST_QPARAM_BROADCAST {}",
        if qmv_fast_qparam_broadcast { 1 } else { 0 }
    ));
    mlx_nax_helpers(w);
    w.line("#else");
    w.line("#define FUSEKIT_HAVE_MLX_NAX 0");
    w.line("#endif");
    w.line("using namespace metal;");
    // mlx's CustomKernel sources reference C++-style type names
    // (`bfloat16_t`, `float16_t`) that Metal-SL doesn't ship by
    // default. Alias them to the Metal-native scalar types so the
    // verbatim source compiles inside our metallib.
    w.line("typedef bfloat bfloat16_t;");
    w.line("typedef half   float16_t;");
    w.blank();
}

fn mlx_nax_helpers(w: &mut MslWriter) {
    w.line(
        r#"
template <typename T, typename U, int scale_step_per_thread>
METAL_FUNC void fusekit_qparam_pair_load(
    const device T* scales,
    const device T* biases,
    uint simd_lid,
    thread U* scale,
    thread U* bias) {
  if constexpr (scale_step_per_thread > 1) {
    if constexpr (FUSEKIT_QMV_FAST_QPARAM_BROADCAST) {
      uint src_lane =
          (simd_lid / uint(scale_step_per_thread)) * uint(scale_step_per_thread);
      U scale_src = (simd_lid == src_lane) ? U(scales[0]) : U(0);
      U bias_src = (simd_lid == src_lane) ? U(biases[0]) : U(0);
      *scale = simd_broadcast(scale_src, src_lane);
      *bias = simd_broadcast(bias_src, src_lane);
      return;
    }
  }
  *scale = U(scales[0]);
  *bias = U(biases[0]);
}

	template <
	    typename T,
	    int group_size,
	    int bits,
	    int num_simdgroups,
	    int results_per_simdgroup>
	METAL_FUNC void fusekit_qmv_fast_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
	  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;

      U s;
      U b;
      fusekit_qparam_pair_load<T, U, scale_step_per_thread>(
          sl, bl, simd_lid, &s, &b);
      if (bits == 4 && !FUSEKIT_QMV_FAST_QDOT_Q4) {
        const device uint16_t* ws16 = (const device uint16_t*)wl;
        U accum = 0;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          uint16_t wi = ws16[i];
          accum +=
              (x_thread[4 * i] * ((wi      ) & 0x000f) +
               x_thread[4 * i + 1] * ((wi      ) & 0x00f0) +
               x_thread[4 * i + 2] * ((wi      ) & 0x0f00) +
               x_thread[4 * i + 3] * ((wi      ) & 0xf000));
        }
        result[row] += s * accum + sum * b;
      } else if (bits == 5 && !FUSEKIT_QMV_FAST_QDOT_Q5) {
        U accum = 0;
        for (int i = 0; i < (values_per_thread / 8); i++) {
          const device uint8_t* wb = wl + 5 * i;
          const thread U* xt = x_thread + 8 * i;
          accum += (wb[0] & 0x1f) * xt[0];
          accum += (wb[0] & 0xe0) * xt[1];
          accum += (wb[1] & 0x03) * (xt[1] * 256.0f);
          accum += (wb[1] & 0x7c) * xt[2];
          accum += (wb[1] & 0x80) * xt[3];
          accum += (wb[2] & 0x0f) * (xt[3] * 256.0f);
          accum += (wb[2] & 0xf0) * xt[4];
          accum += (wb[3] & 0x01) * (xt[4] * 256.0f);
          accum += (wb[3] & 0x3e) * xt[5];
          accum += (wb[3] & 0xc0) * xt[6];
          accum += (wb[4] & 0x07) * (xt[6] * 256.0f);
          accum += (wb[4] & 0xf8) * xt[7];
        }
        result[row] += s * accum + sum * b;
      } else if (bits == 6 && !FUSEKIT_QMV_FAST_QDOT_Q6) {
        U accum = 0;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const device uint8_t* wb = wl + 3 * i;
          const thread U* xt = x_thread + 4 * i;
          accum += (wb[0] & 0x3f) * xt[0];
          accum += (wb[0] & 0xc0) * xt[1];
          accum += (wb[1] & 0x0f) * (xt[1] * 256.0f);
          accum += (wb[1] & 0xf0) * xt[2];
          accum += (wb[2] & 0x03) * (xt[2] * 256.0f);
          accum += (wb[2] & 0xfc) * xt[3];
        }
        result[row] += s * accum + sum * b;
      } else if (bits == 8 && !FUSEKIT_QMV_FAST_QDOT_Q8) {
        U accum = 0;
        for (int i = 0; i < values_per_thread; i++) {
          uint8_t wi = wl[i];
          accum += x_thread[i] * wi;
        }
        result[row] += s * accum + sum * b;
      } else {
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
	  }
	}

	template <
	    typename T,
	    int group_size,
	    int bits,
	    int num_simdgroups>
	METAL_FUNC void fusekit_qmv_fast2_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
	  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  U result0 = 0;
  U result1 = 0;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * 2) + simd_gid * 2;

  const device uint8_t* ws0 =
      ws + out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  const device uint8_t* ws1 = ws0 + in_vec_size_w;
  const device T* scales0 = scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* scales1 = scales0 + in_vec_size_g;
  const device T* biases0 = biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* biases1 = biases0 + in_vec_size_g;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    U s0;
    U b0;
    U s1;
    U b1;
    fusekit_qparam_pair_load<T, U, scale_step_per_thread>(
        scales0, biases0, simd_lid, &s0, &b0);
    fusekit_qparam_pair_load<T, U, scale_step_per_thread>(
        scales1, biases1, simd_lid, &s1, &b1);
    result0 += qdot<U, values_per_thread, bits>(ws0, x_thread, s0, b0, sum);
    result1 += qdot<U, values_per_thread, bits>(ws1, x_thread, s1, b1, sum);

    ws0 += block_size * bytes_per_pack / pack_factor;
    ws1 += block_size * bytes_per_pack / pack_factor;
    scales0 += block_size / group_size;
    scales1 += block_size / group_size;
    biases0 += block_size / group_size;
    biases1 += block_size / group_size;
    x += block_size;
  }

  result0 = simd_sum(result0);
  result1 = simd_sum(result1);
  if (simd_lid == 0) {
    y[0] = static_cast<T>(result0);
    y[1] = static_cast<T>(result1);
  }
	}

	template <
	    typename T,
	    int group_size,
	    int bits,
	    int num_simdgroups,
	    int results_per_simdgroup>
	METAL_FUNC void fusekit_qmv_fanout2_exact_same_tile_impl(
    const device uint32_t* w0,
    const device T* scales0,
    const device T* biases0,
    const device uint32_t* w1,
    const device T* scales1,
    const device T* biases1,
    const device T* x,
    device T* y0,
    device T* y1,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws0 = (const device uint8_t*)w0;
  const device uint8_t* ws1 = (const device uint8_t*)w1;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result0[results_per_simdgroup] = {0};
  thread U result1[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws0 +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    ws1 +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales0 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    scales1 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases0 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases1 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y0 += tid.x * out_vec_size + out_row;
    y1 += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl0 = (const device uint8_t*)(ws0 + row * in_vec_size_w);
        auto wl1 = (const device uint8_t*)(ws1 + row * in_vec_size_w);
        const device T* sl0 = scales0 + row * in_vec_size_g;
        const device T* sl1 = scales1 + row * in_vec_size_g;
        const device T* bl0 = biases0 + row * in_vec_size_g;
        const device T* bl1 = biases1 + row * in_vec_size_g;

        U s0 = U(sl0[0]);
        U s1 = U(sl1[0]);
        U b0 = U(bl0[0]);
        U b1 = U(bl1[0]);
        result0[row] +=
            qdot<U, values_per_thread, bits>(wl0, x_thread, s0, b0, sum);
        result1[row] +=
            qdot<U, values_per_thread, bits>(wl1, x_thread, s1, b1, sum);
      }

      ws0 += block_size * bytes_per_pack / pack_factor;
      ws1 += block_size * bytes_per_pack / pack_factor;
      scales0 += block_size / group_size;
      scales1 += block_size / group_size;
      biases0 += block_size / group_size;
      biases1 += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl0 = (const device uint8_t*)(ws0 + row * in_vec_size_w);
        auto wl1 = (const device uint8_t*)(ws1 + row * in_vec_size_w);
        const device T* sl0 = scales0 + row * in_vec_size_g;
        const device T* sl1 = scales1 + row * in_vec_size_g;
        const device T* bl0 = biases0 + row * in_vec_size_g;
        const device T* bl1 = biases1 + row * in_vec_size_g;

        U s0 = U(sl0[0]);
        U s1 = U(sl1[0]);
        U b0 = U(bl0[0]);
        U b1 = U(bl1[0]);
        result0[row] += qdot_safe<U, values_per_thread, bits>(
            wl0, x_thread, s0, b0, sum, remaining);
        result1[row] += qdot_safe<U, values_per_thread, bits>(
            wl1, x_thread, s1, b1, sum, remaining);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result0[row] = simd_sum(result0[row]);
      result1[row] = simd_sum(result1[row]);
      if (simd_lid == 0) {
        y0[row] = static_cast<T>(result0[row]);
        y1[row] = static_cast<T>(result1[row]);
      }
    }
  }

  else {
    ws0 += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    ws1 += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales0 += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    scales1 += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases0 += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases1 += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y0 += tid.x * out_vec_size + used_out_row;
    y1 += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl0 = (const device uint8_t*)(ws0 + row * in_vec_size_w);
        auto wl1 = (const device uint8_t*)(ws1 + row * in_vec_size_w);
        const device T* sl0 = scales0 + row * in_vec_size_g;
        const device T* sl1 = scales1 + row * in_vec_size_g;
        const device T* bl0 = biases0 + row * in_vec_size_g;
        const device T* bl1 = biases1 + row * in_vec_size_g;

        U s0 = U(sl0[0]);
        U s1 = U(sl1[0]);
        U b0 = U(bl0[0]);
        U b1 = U(bl1[0]);
        result0[row] +=
            qdot<U, values_per_thread, bits>(wl0, x_thread, s0, b0, sum);
        result1[row] +=
            qdot<U, values_per_thread, bits>(wl1, x_thread, s1, b1, sum);
      }

      ws0 += block_size * bytes_per_pack / pack_factor;
      ws1 += block_size * bytes_per_pack / pack_factor;
      scales0 += block_size / group_size;
      scales1 += block_size / group_size;
      biases0 += block_size / group_size;
      biases1 += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl0 = (const device uint8_t*)(ws0 + row * in_vec_size_w);
        auto wl1 = (const device uint8_t*)(ws1 + row * in_vec_size_w);
        const device T* sl0 = scales0 + row * in_vec_size_g;
        const device T* sl1 = scales1 + row * in_vec_size_g;
        const device T* bl0 = biases0 + row * in_vec_size_g;
        const device T* bl1 = biases1 + row * in_vec_size_g;

        U s0 = U(sl0[0]);
        U s1 = U(sl1[0]);
        U b0 = U(bl0[0]);
        U b1 = U(bl1[0]);
        result0[row] += qdot_safe<U, values_per_thread, bits>(
            wl0, x_thread, s0, b0, sum, remaining);
        result1[row] += qdot_safe<U, values_per_thread, bits>(
            wl1, x_thread, s1, b1, sum, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] = simd_sum(result0[row]);
      result1[row] = simd_sum(result1[row]);
      if (simd_lid == 0) {
        y0[row] = static_cast<T>(result0[row]);
        y1[row] = static_cast<T>(result1[row]);
      }
    }
  }
	}

	template <
	    typename T,
	    int group_size,
	    int bits,
	    int num_simdgroups,
	    int results_per_simdgroup>
	METAL_FUNC void fusekit_qmv_fanout2_same_tile_impl(
    const device uint32_t* w0,
    const device T* scales0,
    const device T* biases0,
    const device uint32_t* w1,
    const device T* scales1,
    const device T* biases1,
    const device T* x,
    device T* y0,
    device T* y1,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
	  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws0 = (const device uint8_t*)w0;
  const device uint8_t* ws1 = (const device uint8_t*)w1;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result0[results_per_simdgroup] = {0};
  thread U result1[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws0 += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  ws1 += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales0 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  scales1 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases0 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases1 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y0 += tid.x * out_vec_size + out_row;
  y1 += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl0 = (const device uint8_t*)(ws0 + row * in_vec_size_w);
      auto wl1 = (const device uint8_t*)(ws1 + row * in_vec_size_w);
      const device T* sl0 = scales0 + row * in_vec_size_g;
      const device T* sl1 = scales1 + row * in_vec_size_g;
      const device T* bl0 = biases0 + row * in_vec_size_g;
      const device T* bl1 = biases1 + row * in_vec_size_g;

      U s0;
      U b0;
      U s1;
      U b1;
      fusekit_qparam_pair_load<T, U, scale_step_per_thread>(
          sl0, bl0, simd_lid, &s0, &b0);
      fusekit_qparam_pair_load<T, U, scale_step_per_thread>(
          sl1, bl1, simd_lid, &s1, &b1);
      if (bits == 4 && !FUSEKIT_QMV_FAST_QDOT_Q4) {
        const device uint16_t* ws160 = (const device uint16_t*)wl0;
        const device uint16_t* ws161 = (const device uint16_t*)wl1;
        U accum0 = 0;
        U accum1 = 0;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          uint16_t w0i = ws160[i];
          uint16_t w1i = ws161[i];
          accum0 +=
              (x_thread[4 * i] * ((w0i      ) & 0x000f) +
               x_thread[4 * i + 1] * ((w0i      ) & 0x00f0) +
               x_thread[4 * i + 2] * ((w0i      ) & 0x0f00) +
               x_thread[4 * i + 3] * ((w0i      ) & 0xf000));
          accum1 +=
              (x_thread[4 * i] * ((w1i      ) & 0x000f) +
               x_thread[4 * i + 1] * ((w1i      ) & 0x00f0) +
               x_thread[4 * i + 2] * ((w1i      ) & 0x0f00) +
               x_thread[4 * i + 3] * ((w1i      ) & 0xf000));
        }
        result0[row] += s0 * accum0 + sum * b0;
        result1[row] += s1 * accum1 + sum * b1;
      } else if (bits == 5 && !FUSEKIT_QMV_FAST_QDOT_Q5) {
        U accum0 = 0;
        U accum1 = 0;
        for (int i = 0; i < (values_per_thread / 8); i++) {
          const device uint8_t* wb0 = wl0 + 5 * i;
          const device uint8_t* wb1 = wl1 + 5 * i;
          const thread U* xt = x_thread + 8 * i;
          accum0 += (wb0[0] & 0x1f) * xt[0];
          accum0 += (wb0[0] & 0xe0) * xt[1];
          accum0 += (wb0[1] & 0x03) * (xt[1] * 256.0f);
          accum0 += (wb0[1] & 0x7c) * xt[2];
          accum0 += (wb0[1] & 0x80) * xt[3];
          accum0 += (wb0[2] & 0x0f) * (xt[3] * 256.0f);
          accum0 += (wb0[2] & 0xf0) * xt[4];
          accum0 += (wb0[3] & 0x01) * (xt[4] * 256.0f);
          accum0 += (wb0[3] & 0x3e) * xt[5];
          accum0 += (wb0[3] & 0xc0) * xt[6];
          accum0 += (wb0[4] & 0x07) * (xt[6] * 256.0f);
          accum0 += (wb0[4] & 0xf8) * xt[7];
          accum1 += (wb1[0] & 0x1f) * xt[0];
          accum1 += (wb1[0] & 0xe0) * xt[1];
          accum1 += (wb1[1] & 0x03) * (xt[1] * 256.0f);
          accum1 += (wb1[1] & 0x7c) * xt[2];
          accum1 += (wb1[1] & 0x80) * xt[3];
          accum1 += (wb1[2] & 0x0f) * (xt[3] * 256.0f);
          accum1 += (wb1[2] & 0xf0) * xt[4];
          accum1 += (wb1[3] & 0x01) * (xt[4] * 256.0f);
          accum1 += (wb1[3] & 0x3e) * xt[5];
          accum1 += (wb1[3] & 0xc0) * xt[6];
          accum1 += (wb1[4] & 0x07) * (xt[6] * 256.0f);
          accum1 += (wb1[4] & 0xf8) * xt[7];
        }
        result0[row] += s0 * accum0 + sum * b0;
        result1[row] += s1 * accum1 + sum * b1;
      } else if (bits == 6 && !FUSEKIT_QMV_FAST_QDOT_Q6) {
        U accum0 = 0;
        U accum1 = 0;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const device uint8_t* wb0 = wl0 + 3 * i;
          const device uint8_t* wb1 = wl1 + 3 * i;
          const thread U* xt = x_thread + 4 * i;
          accum0 += (wb0[0] & 0x3f) * xt[0];
          accum0 += (wb0[0] & 0xc0) * xt[1];
          accum0 += (wb0[1] & 0x0f) * (xt[1] * 256.0f);
          accum0 += (wb0[1] & 0xf0) * xt[2];
          accum0 += (wb0[2] & 0x03) * (xt[2] * 256.0f);
          accum0 += (wb0[2] & 0xfc) * xt[3];
          accum1 += (wb1[0] & 0x3f) * xt[0];
          accum1 += (wb1[0] & 0xc0) * xt[1];
          accum1 += (wb1[1] & 0x0f) * (xt[1] * 256.0f);
          accum1 += (wb1[1] & 0xf0) * xt[2];
          accum1 += (wb1[2] & 0x03) * (xt[2] * 256.0f);
          accum1 += (wb1[2] & 0xfc) * xt[3];
        }
        result0[row] += s0 * accum0 + sum * b0;
        result1[row] += s1 * accum1 + sum * b1;
      } else if (bits == 8 && !FUSEKIT_QMV_FAST_QDOT_Q8) {
        U accum0 = 0;
        U accum1 = 0;
        for (int i = 0; i < values_per_thread; i++) {
          uint8_t w0i = wl0[i];
          uint8_t w1i = wl1[i];
          accum0 += x_thread[i] * w0i;
          accum1 += x_thread[i] * w1i;
        }
        result0[row] += s0 * accum0 + sum * b0;
        result1[row] += s1 * accum1 + sum * b1;
      } else {
        result0[row] += qdot<U, values_per_thread, bits>(wl0, x_thread, s0, b0, sum);
        result1[row] += qdot<U, values_per_thread, bits>(wl1, x_thread, s1, b1, sum);
      }
    }

    ws0 += block_size * bytes_per_pack / pack_factor;
    ws1 += block_size * bytes_per_pack / pack_factor;
    scales0 += block_size / group_size;
    scales1 += block_size / group_size;
    biases0 += block_size / group_size;
    biases1 += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result0[row] = simd_sum(result0[row]);
    result1[row] = simd_sum(result1[row]);
    if (simd_lid == 0) {
      y0[row] = static_cast<T>(result0[row]);
      y1[row] = static_cast<T>(result1[row]);
    }
	  }
		}

		template <
		    typename T,
		    int group_size,
		    int bits,
		    int num_simdgroups,
		    int results_per_simdgroup,
		    int token_rows>
		METAL_FUNC void fusekit_qmv_swiglu_pair_m_impl(
	    const device uint32_t* w_gate,
	    const device T* scales_gate,
	    const device T* biases_gate,
	    const device uint32_t* w_up,
	    const device T* scales_up,
	    const device T* biases_up,
	    const device T* x,
	    device T* y,
	    const int in_vec_size,
	    const int out_vec_size,
	    const int active_rows,
	    uint3 tid,
	    uint simd_gid,
	    uint simd_lid) {
	  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
	  constexpr int pack_factor = get_pack_factor<bits, 32>();
	  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
	  constexpr int values_per_thread = pack_factor * packs_per_thread;
	  constexpr int block_size = values_per_thread * SIMD_SIZE;
	  constexpr int scale_step_per_thread = group_size / values_per_thread;

	  const int row_base = tid.x * token_rows;
	  if (row_base >= active_rows) {
	    return;
	  }

	  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
	  const int in_vec_size_g = in_vec_size / group_size;
	  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
	      simd_gid * results_per_simdgroup;

	  const device uint8_t* wg_base = (const device uint8_t*)w_gate +
	      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
	  const device uint8_t* wu_base = (const device uint8_t*)w_up +
	      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
	  const device T* sg_base =
	      scales_gate + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
	  const device T* su_base =
	      scales_up + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
	  const device T* bg_base =
	      biases_gate + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
	  const device T* bu_base =
	      biases_up + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

	  typedef float U;

	  thread U x_thread[token_rows][values_per_thread];
	  thread U x_sum[token_rows];
	  thread U result_gate[token_rows][results_per_simdgroup];
	  thread U result_up[token_rows][results_per_simdgroup];
	  for (int r = 0; r < token_rows; r++) {
	    x_sum[r] = 0;
	    for (int c = 0; c < results_per_simdgroup; c++) {
	      result_gate[r][c] = 0;
	      result_up[r][c] = 0;
	    }
	  }

	  for (int k = 0; k < in_vec_size; k += block_size) {
	    for (int r = 0; r < token_rows; r++) {
	      if (row_base + r < active_rows) {
	        const device T* xr =
	            x + (row_base + r) * in_vec_size + k + simd_lid * values_per_thread;
	        x_sum[r] = load_vector<T, U, values_per_thread, bits>(xr, x_thread[r]);
	      } else {
	        x_sum[r] = 0;
	        for (int i = 0; i < values_per_thread; i++) {
	          x_thread[r][i] = 0;
	        }
	      }
	    }

	    const device uint8_t* wg_k =
	        wg_base + k * bytes_per_pack / pack_factor;
	    const device uint8_t* wu_k =
	        wu_base + k * bytes_per_pack / pack_factor;
	    const device T* sg_k = sg_base + k / group_size;
	    const device T* su_k = su_base + k / group_size;
	    const device T* bg_k = bg_base + k / group_size;
	    const device T* bu_k = bu_base + k / group_size;

	    for (int col = 0; col < results_per_simdgroup; col++) {
	      const device uint8_t* wgl = wg_k + col * in_vec_size_w;
	      const device uint8_t* wul = wu_k + col * in_vec_size_w;
	      const device T* sgl = sg_k + col * in_vec_size_g;
	      const device T* sul = su_k + col * in_vec_size_g;
	      const device T* bgl = bg_k + col * in_vec_size_g;
	      const device T* bul = bu_k + col * in_vec_size_g;
	      U sg = U(sgl[0]);
	      U su = U(sul[0]);
	      U bg = U(bgl[0]);
	      U bu = U(bul[0]);

		      if (bits == 4 && !FUSEKIT_QMV_M_QDOT) {
	        const device uint16_t* wg16 = (const device uint16_t*)wgl;
	        const device uint16_t* wu16 = (const device uint16_t*)wul;
	        thread U accum_gate[token_rows];
	        thread U accum_up[token_rows];
	        for (int r = 0; r < token_rows; r++) {
	          accum_gate[r] = 0;
	          accum_up[r] = 0;
	        }
	        for (int i = 0; i < (values_per_thread / 4); i++) {
	          uint16_t wig = wg16[i];
	          uint16_t wiu = wu16[i];
	          for (int r = 0; r < token_rows; r++) {
	            accum_gate[r] +=
	                (x_thread[r][4 * i] * ((wig      ) & 0x000f) +
	                 x_thread[r][4 * i + 1] * ((wig      ) & 0x00f0) +
	                 x_thread[r][4 * i + 2] * ((wig      ) & 0x0f00) +
	                 x_thread[r][4 * i + 3] * ((wig      ) & 0xf000));
	            accum_up[r] +=
	                (x_thread[r][4 * i] * ((wiu      ) & 0x000f) +
	                 x_thread[r][4 * i + 1] * ((wiu      ) & 0x00f0) +
	                 x_thread[r][4 * i + 2] * ((wiu      ) & 0x0f00) +
	                 x_thread[r][4 * i + 3] * ((wiu      ) & 0xf000));
	          }
	        }
	        for (int r = 0; r < token_rows; r++) {
	          result_gate[r][col] += sg * accum_gate[r] + x_sum[r] * bg;
	          result_up[r][col] += su * accum_up[r] + x_sum[r] * bu;
	        }
				      } else if (bits == 8 && !FUSEKIT_QMV_M_QDOT) {
	        thread U accum_gate[token_rows];
	        thread U accum_up[token_rows];
	        for (int r = 0; r < token_rows; r++) {
	          accum_gate[r] = 0;
	          accum_up[r] = 0;
	        }
	        for (int i = 0; i < values_per_thread; i++) {
	          uint8_t wig = wgl[i];
	          uint8_t wiu = wul[i];
	          for (int r = 0; r < token_rows; r++) {
	            accum_gate[r] += x_thread[r][i] * wig;
	            accum_up[r] += x_thread[r][i] * wiu;
	          }
	        }
	        for (int r = 0; r < token_rows; r++) {
	          result_gate[r][col] += sg * accum_gate[r] + x_sum[r] * bg;
	          result_up[r][col] += su * accum_up[r] + x_sum[r] * bu;
	        }
	      } else {
	        for (int r = 0; r < token_rows; r++) {
	          result_gate[r][col] +=
	              qdot<U, values_per_thread, bits>(wgl, x_thread[r], sg, bg, x_sum[r]);
	          result_up[r][col] +=
	              qdot<U, values_per_thread, bits>(wul, x_thread[r], su, bu, x_sum[r]);
	        }
	      }
	    }
	  }

	  for (int r = 0; r < token_rows; r++) {
	    if (row_base + r < active_rows) {
	      for (int col = 0; col < results_per_simdgroup; col++) {
	        U gate = simd_sum(result_gate[r][col]);
	        U up = simd_sum(result_up[r][col]);
	        if (simd_lid == 0) {
	          U g = U(static_cast<T>(gate));
	          U u = U(static_cast<T>(up));
	          U sig = 1.0f / (1.0f + exp(-g));
	          y[(row_base + r) * out_vec_size + out_row + col] =
	              static_cast<T>((sig * g) * u);
	        }
	      }
	    }
	  }
	}

	template <
	    typename T,
	    int group_size,
	    int bits,
	    int num_simdgroups,
	    int results_per_simdgroup,
	    int token_rows>
	METAL_FUNC void fusekit_qmv_fast_m_impl(
	    const device uint32_t* w,
	    const device T* scales,
	    const device T* biases,
	    const device T* x,
	    device T* y,
	    const int in_vec_size,
	    const int out_vec_size,
	    const int active_rows,
	    uint3 tid,
	    uint simd_gid,
	    uint simd_lid) {
	  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
	  constexpr int pack_factor = get_pack_factor<bits, 32>();
	  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
	  constexpr int values_per_thread = pack_factor * packs_per_thread;
	  constexpr int block_size = values_per_thread * SIMD_SIZE;
	  constexpr int scale_step_per_thread = group_size / values_per_thread;

	  const int row_base = tid.x * token_rows;
	  if (row_base >= active_rows) {
	    return;
	  }

	  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
	  const int in_vec_size_g = in_vec_size / group_size;
	  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
	      simd_gid * results_per_simdgroup;

	  const device uint8_t* ws_base = (const device uint8_t*)w +
	      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
	  const device T* scales_base =
	      scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
	  const device T* biases_base =
	      biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

	  typedef float U;

	  thread U x_thread[token_rows][values_per_thread];
	  thread U x_sum[token_rows];
	  thread U result[token_rows][results_per_simdgroup];
	  for (int r = 0; r < token_rows; r++) {
	    x_sum[r] = 0;
	    for (int c = 0; c < results_per_simdgroup; c++) {
	      result[r][c] = 0;
	    }
	  }

	  for (int k = 0; k < in_vec_size; k += block_size) {
	    for (int r = 0; r < token_rows; r++) {
	      if (row_base + r < active_rows) {
	        const device T* xr =
	            x + (row_base + r) * in_vec_size + k + simd_lid * values_per_thread;
	        x_sum[r] = load_vector<T, U, values_per_thread, bits>(xr, x_thread[r]);
	      } else {
	        x_sum[r] = 0;
	        for (int i = 0; i < values_per_thread; i++) {
	          x_thread[r][i] = 0;
	        }
	      }
	    }

	    const device uint8_t* ws_k =
	        ws_base + k * bytes_per_pack / pack_factor;
	    const device T* scales_k = scales_base + k / group_size;
	    const device T* biases_k = biases_base + k / group_size;

	    for (int col = 0; col < results_per_simdgroup; col++) {
	      const device uint8_t* wl = ws_k + col * in_vec_size_w;
	      const device T* sl = scales_k + col * in_vec_size_g;
	      const device T* bl = biases_k + col * in_vec_size_g;
	      U s;
	      U b;
	      fusekit_qparam_pair_load<T, U, scale_step_per_thread>(
	          sl, bl, simd_lid, &s, &b);

		      if (bits == 4 && !FUSEKIT_QMV_M_QDOT) {
	        const device uint16_t* ws16 = (const device uint16_t*)wl;
	        thread U accum[token_rows];
	        for (int r = 0; r < token_rows; r++) {
	          accum[r] = 0;
	        }
	        for (int i = 0; i < (values_per_thread / 4); i++) {
	          uint16_t wi = ws16[i];
	          for (int r = 0; r < token_rows; r++) {
	            accum[r] +=
	                (x_thread[r][4 * i] * ((wi      ) & 0x000f) +
	                 x_thread[r][4 * i + 1] * ((wi      ) & 0x00f0) +
	                 x_thread[r][4 * i + 2] * ((wi      ) & 0x0f00) +
	                 x_thread[r][4 * i + 3] * ((wi      ) & 0xf000));
	          }
	        }
	        for (int r = 0; r < token_rows; r++) {
	          result[r][col] += s * accum[r] + x_sum[r] * b;
	        }
      } else if (bits == 5 && !FUSEKIT_QMV_M_QDOT) {
        thread U accum[token_rows];
        for (int r = 0; r < token_rows; r++) {
          accum[r] = 0;
        }
        for (int i = 0; i < (values_per_thread / 8); i++) {
          const device uint8_t* wb = wl + 5 * i;
          for (int r = 0; r < token_rows; r++) {
            const thread U* xt = x_thread[r] + 8 * i;
            accum[r] += (wb[0] & 0x1f) * xt[0];
            accum[r] += (wb[0] & 0xe0) * xt[1];
            accum[r] += (wb[1] & 0x03) * (xt[1] * 256.0f);
            accum[r] += (wb[1] & 0x7c) * xt[2];
            accum[r] += (wb[1] & 0x80) * xt[3];
            accum[r] += (wb[2] & 0x0f) * (xt[3] * 256.0f);
            accum[r] += (wb[2] & 0xf0) * xt[4];
            accum[r] += (wb[3] & 0x01) * (xt[4] * 256.0f);
            accum[r] += (wb[3] & 0x3e) * xt[5];
            accum[r] += (wb[3] & 0xc0) * xt[6];
            accum[r] += (wb[4] & 0x07) * (xt[6] * 256.0f);
            accum[r] += (wb[4] & 0xf8) * xt[7];
          }
        }
        for (int r = 0; r < token_rows; r++) {
          result[r][col] += s * accum[r] + x_sum[r] * b;
        }
      } else if (bits == 6 && !FUSEKIT_QMV_M_QDOT) {
        thread U accum[token_rows];
        for (int r = 0; r < token_rows; r++) {
          accum[r] = 0;
        }
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const device uint8_t* wb = wl + 3 * i;
          for (int r = 0; r < token_rows; r++) {
            const thread U* xt = x_thread[r] + 4 * i;
            accum[r] += (wb[0] & 0x3f) * xt[0];
            accum[r] += (wb[0] & 0xc0) * xt[1];
            accum[r] += (wb[1] & 0x0f) * (xt[1] * 256.0f);
            accum[r] += (wb[1] & 0xf0) * xt[2];
            accum[r] += (wb[2] & 0x03) * (xt[2] * 256.0f);
            accum[r] += (wb[2] & 0xfc) * xt[3];
          }
        }
        for (int r = 0; r < token_rows; r++) {
          result[r][col] += s * accum[r] + x_sum[r] * b;
        }
			      } else if (bits == 8 && !FUSEKIT_QMV_M_QDOT) {
		        thread U accum[token_rows];
	        for (int r = 0; r < token_rows; r++) {
	          accum[r] = 0;
	        }
	        for (int i = 0; i < values_per_thread; i++) {
	          uint8_t wi = wl[i];
	          for (int r = 0; r < token_rows; r++) {
	            accum[r] += x_thread[r][i] * wi;
	          }
	        }
	        for (int r = 0; r < token_rows; r++) {
	          result[r][col] += s * accum[r] + x_sum[r] * b;
	        }
	      } else {
	        for (int r = 0; r < token_rows; r++) {
	          result[r][col] +=
	              qdot<U, values_per_thread, bits>(wl, x_thread[r], s, b, x_sum[r]);
	        }
	      }
	    }
	  }

	  for (int r = 0; r < token_rows; r++) {
	    if (row_base + r < active_rows) {
	      for (int col = 0; col < results_per_simdgroup; col++) {
	        result[r][col] = simd_sum(result[r][col]);
	        if (simd_lid == 0) {
	          y[(row_base + r) * out_vec_size + out_row + col] =
	              static_cast<T>(result[r][col]);
	        }
	      }
	    }
	  }
	}

	template <
	    typename T,
	    int group_size,
	    int bits,
	    int num_simdgroups,
	    int results_per_simdgroup,
	    int token_rows>
	METAL_FUNC void fusekit_qmv_fanout2_same_tile_m_impl(
	    const device uint32_t* w0,
	    const device T* scales0,
	    const device T* biases0,
	    const device uint32_t* w1,
	    const device T* scales1,
	    const device T* biases1,
	    const device T* x,
	    device T* y0,
	    device T* y1,
	    const int in_vec_size,
	    const int out_vec_size,
	    const int active_rows,
	    uint3 tid,
	    uint simd_gid,
	    uint simd_lid) {
	  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
	  constexpr int pack_factor = get_pack_factor<bits, 32>();
	  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
	  constexpr int values_per_thread = pack_factor * packs_per_thread;
	  constexpr int block_size = values_per_thread * SIMD_SIZE;
	  constexpr int scale_step_per_thread = group_size / values_per_thread;

	  const int row_base = tid.x * token_rows;
	  if (row_base >= active_rows) {
	    return;
	  }

	  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
	  const int in_vec_size_g = in_vec_size / group_size;
	  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
	      simd_gid * results_per_simdgroup;

	  const device uint8_t* ws0_base = (const device uint8_t*)w0 +
	      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
	  const device uint8_t* ws1_base = (const device uint8_t*)w1 +
	      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
	  const device T* scales0_base =
	      scales0 + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
	  const device T* scales1_base =
	      scales1 + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
	  const device T* biases0_base =
	      biases0 + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
	  const device T* biases1_base =
	      biases1 + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

	  typedef float U;

	  thread U x_thread[token_rows][values_per_thread];
	  thread U x_sum[token_rows];
	  thread U result0[token_rows][results_per_simdgroup];
	  thread U result1[token_rows][results_per_simdgroup];
	  for (int r = 0; r < token_rows; r++) {
	    x_sum[r] = 0;
	    for (int c = 0; c < results_per_simdgroup; c++) {
	      result0[r][c] = 0;
	      result1[r][c] = 0;
	    }
	  }

	  for (int k = 0; k < in_vec_size; k += block_size) {
	    for (int r = 0; r < token_rows; r++) {
	      if (row_base + r < active_rows) {
	        const device T* xr =
	            x + (row_base + r) * in_vec_size + k + simd_lid * values_per_thread;
	        x_sum[r] = load_vector<T, U, values_per_thread, bits>(xr, x_thread[r]);
	      } else {
	        x_sum[r] = 0;
	        for (int i = 0; i < values_per_thread; i++) {
	          x_thread[r][i] = 0;
	        }
	      }
	    }

	    const device uint8_t* ws0_k =
	        ws0_base + k * bytes_per_pack / pack_factor;
	    const device uint8_t* ws1_k =
	        ws1_base + k * bytes_per_pack / pack_factor;
	    const device T* scales0_k = scales0_base + k / group_size;
	    const device T* scales1_k = scales1_base + k / group_size;
	    const device T* biases0_k = biases0_base + k / group_size;
	    const device T* biases1_k = biases1_base + k / group_size;

	    for (int col = 0; col < results_per_simdgroup; col++) {
	      const device uint8_t* wl0 = ws0_k + col * in_vec_size_w;
	      const device uint8_t* wl1 = ws1_k + col * in_vec_size_w;
	      const device T* sl0 = scales0_k + col * in_vec_size_g;
	      const device T* sl1 = scales1_k + col * in_vec_size_g;
	      const device T* bl0 = biases0_k + col * in_vec_size_g;
	      const device T* bl1 = biases1_k + col * in_vec_size_g;
	      U s0;
	      U b0;
	      U s1;
	      U b1;
	      fusekit_qparam_pair_load<T, U, scale_step_per_thread>(
	          sl0, bl0, simd_lid, &s0, &b0);
	      fusekit_qparam_pair_load<T, U, scale_step_per_thread>(
	          sl1, bl1, simd_lid, &s1, &b1);

		      if (bits == 4 && !FUSEKIT_QMV_M_QDOT) {
	        const device uint16_t* ws016 = (const device uint16_t*)wl0;
	        const device uint16_t* ws116 = (const device uint16_t*)wl1;
	        thread U accum0[token_rows];
	        thread U accum1[token_rows];
	        for (int r = 0; r < token_rows; r++) {
	          accum0[r] = 0;
	          accum1[r] = 0;
	        }
	        for (int i = 0; i < (values_per_thread / 4); i++) {
	          uint16_t wi0 = ws016[i];
	          uint16_t wi1 = ws116[i];
	          for (int r = 0; r < token_rows; r++) {
	            accum0[r] +=
	                (x_thread[r][4 * i] * ((wi0      ) & 0x000f) +
	                 x_thread[r][4 * i + 1] * ((wi0      ) & 0x00f0) +
	                 x_thread[r][4 * i + 2] * ((wi0      ) & 0x0f00) +
	                 x_thread[r][4 * i + 3] * ((wi0      ) & 0xf000));
	            accum1[r] +=
	                (x_thread[r][4 * i] * ((wi1      ) & 0x000f) +
	                 x_thread[r][4 * i + 1] * ((wi1      ) & 0x00f0) +
	                 x_thread[r][4 * i + 2] * ((wi1      ) & 0x0f00) +
	                 x_thread[r][4 * i + 3] * ((wi1      ) & 0xf000));
	          }
	        }
	        for (int r = 0; r < token_rows; r++) {
	          result0[r][col] += s0 * accum0[r] + x_sum[r] * b0;
	          result1[r][col] += s1 * accum1[r] + x_sum[r] * b1;
	        }
			      } else if (bits == 8 && !FUSEKIT_QMV_M_QDOT) {
		        thread U accum0[token_rows];
	        thread U accum1[token_rows];
	        for (int r = 0; r < token_rows; r++) {
	          accum0[r] = 0;
	          accum1[r] = 0;
	        }
	        for (int i = 0; i < values_per_thread; i++) {
	          uint8_t wi0 = wl0[i];
	          uint8_t wi1 = wl1[i];
	          for (int r = 0; r < token_rows; r++) {
	            accum0[r] += x_thread[r][i] * wi0;
	            accum1[r] += x_thread[r][i] * wi1;
	          }
	        }
	        for (int r = 0; r < token_rows; r++) {
	          result0[r][col] += s0 * accum0[r] + x_sum[r] * b0;
	          result1[r][col] += s1 * accum1[r] + x_sum[r] * b1;
	        }
	      } else {
	        for (int r = 0; r < token_rows; r++) {
	          result0[r][col] +=
	              qdot<U, values_per_thread, bits>(wl0, x_thread[r], s0, b0, x_sum[r]);
	          result1[r][col] +=
	              qdot<U, values_per_thread, bits>(wl1, x_thread[r], s1, b1, x_sum[r]);
	        }
	      }
	    }
	  }

	  for (int r = 0; r < token_rows; r++) {
	    if (row_base + r < active_rows) {
	      for (int col = 0; col < results_per_simdgroup; col++) {
	        result0[r][col] = simd_sum(result0[r][col]);
	        result1[r][col] = simd_sum(result1[r][col]);
	        if (simd_lid == 0) {
	          y0[(row_base + r) * out_vec_size + out_row + col] =
	              static_cast<T>(result0[r][col]);
	          y1[(row_base + r) * out_vec_size + out_row + col] =
	              static_cast<T>(result1[r][col]);
	        }
	      }
	    }
	  }
	}

	template <
	    typename T,
	    int group_size,
	    int bits,
	    int num_simdgroups,
	    int results_per_simdgroup,
	    int token_rows>
	METAL_FUNC void fusekit_qmv_fast_m_argmax_scan_impl(
	    const device uint32_t* w,
	    const device T* scales,
	    const device T* biases,
	    const device T* x,
	    device uint* tg_argmax,
	    const int in_vec_size,
	    const int out_vec_size,
	    const int active_rows,
	    const int n_out_groups,
	    threadgroup float* max_buf,
	    threadgroup uint* idx_buf,
	    uint3 tid,
	    uint simd_gid,
	    uint simd_lid) {
	  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
	  constexpr int pack_factor = get_pack_factor<bits, 32>();
	  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
	  constexpr int values_per_thread = pack_factor * packs_per_thread;
	  constexpr int block_size = values_per_thread * SIMD_SIZE;
	  constexpr int scale_step_per_thread = group_size / values_per_thread;

	  const int row_base = tid.x * token_rows;
	  if (row_base >= active_rows) {
	    return;
	  }

	  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
	  const int in_vec_size_g = in_vec_size / group_size;
	  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
	      simd_gid * results_per_simdgroup;

	  const device uint8_t* ws_base = (const device uint8_t*)w +
	      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
	  const device T* scales_base =
	      scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
	  const device T* biases_base =
	      biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

	  typedef float U;

	  thread U x_thread[token_rows][values_per_thread];
	  thread U x_sum[token_rows];
	  thread U result[token_rows][results_per_simdgroup];
	  for (int r = 0; r < token_rows; r++) {
	    x_sum[r] = 0;
	    for (int c = 0; c < results_per_simdgroup; c++) {
	      result[r][c] = 0;
	    }
	  }

	  for (int k = 0; k < in_vec_size; k += block_size) {
	    for (int r = 0; r < token_rows; r++) {
	      if (row_base + r < active_rows) {
	        const device T* xr =
	            x + (row_base + r) * in_vec_size + k + simd_lid * values_per_thread;
	        x_sum[r] = load_vector<T, U, values_per_thread, bits>(xr, x_thread[r]);
	      } else {
	        x_sum[r] = 0;
	        for (int i = 0; i < values_per_thread; i++) {
	          x_thread[r][i] = 0;
	        }
	      }
	    }

	    const device uint8_t* ws_k =
	        ws_base + k * bytes_per_pack / pack_factor;
	    const device T* scales_k = scales_base + k / group_size;
	    const device T* biases_k = biases_base + k / group_size;

	    for (int col = 0; col < results_per_simdgroup; col++) {
	      const device uint8_t* wl = ws_k + col * in_vec_size_w;
	      const device T* sl = scales_k + col * in_vec_size_g;
	      const device T* bl = biases_k + col * in_vec_size_g;
	      U s;
	      U b;
	      fusekit_qparam_pair_load<T, U, scale_step_per_thread>(
	          sl, bl, simd_lid, &s, &b);

	      if (bits == 4) {
	        const device uint16_t* ws16 = (const device uint16_t*)wl;
	        thread U accum[token_rows];
	        for (int r = 0; r < token_rows; r++) {
	          accum[r] = 0;
	        }
	        for (int i = 0; i < (values_per_thread / 4); i++) {
	          uint16_t wi = ws16[i];
	          for (int r = 0; r < token_rows; r++) {
	            accum[r] +=
	                (x_thread[r][4 * i] * ((wi      ) & 0x000f) +
	                 x_thread[r][4 * i + 1] * ((wi      ) & 0x00f0) +
	                 x_thread[r][4 * i + 2] * ((wi      ) & 0x0f00) +
	                 x_thread[r][4 * i + 3] * ((wi      ) & 0xf000));
	          }
	        }
	        for (int r = 0; r < token_rows; r++) {
	          result[r][col] += s * accum[r] + x_sum[r] * b;
	        }
		      } else if (bits == 8) {
	        thread U accum[token_rows];
	        for (int r = 0; r < token_rows; r++) {
	          accum[r] = 0;
	        }
	        for (int i = 0; i < values_per_thread; i++) {
	          uint8_t wi = wl[i];
	          for (int r = 0; r < token_rows; r++) {
	            accum[r] += x_thread[r][i] * wi;
	          }
	        }
	        for (int r = 0; r < token_rows; r++) {
	          result[r][col] += s * accum[r] + x_sum[r] * b;
	        }
	      } else {
	        for (int r = 0; r < token_rows; r++) {
	          result[r][col] +=
	              qdot<U, values_per_thread, bits>(wl, x_thread[r], s, b, x_sum[r]);
	        }
	      }
	    }
	  }

	  for (int r = 0; r < token_rows && r < 4; r++) {
	    float local_max = -INFINITY;
	    uint local_idx = (out_row < out_vec_size) ? uint(out_row) : 0xFFFFFFFFu;
	    if (row_base + r < active_rows) {
	      for (int col = 0; col < results_per_simdgroup; col++) {
	        int out_idx = out_row + col;
	        if (out_idx >= out_vec_size) {
	          continue;
	        }
	        result[r][col] = simd_sum(result[r][col]);
	        if (simd_lid == 0) {
	          float v = float(static_cast<T>(result[r][col]));
	          if (v > local_max || (v == local_max && uint(out_idx) < local_idx)) {
	            local_max = v;
	            local_idx = uint(out_idx);
	          }
	        }
	      }
	    }
	    if (simd_lid == 0) {
	      max_buf[r * 32 + simd_gid] = local_max;
	      idx_buf[r * 32 + simd_gid] = local_idx;
	    }
	  }
	  threadgroup_barrier(mem_flags::mem_threadgroup);

	  if (simd_gid == 0) {
	    for (int r = 0; r < token_rows && r < 4; r++) {
	      if (row_base + r >= active_rows) {
	        continue;
	      }
	      float v = (simd_lid < num_simdgroups) ? max_buf[r * 32 + simd_lid] : -INFINITY;
	      uint i = (simd_lid < num_simdgroups) ? idx_buf[r * 32 + simd_lid] : 0xFFFFFFFFu;
	      float gm = simd_max(v);
	      uint mi = (v == gm && i != 0xFFFFFFFFu) ? i : 0xFFFFFFFFu;
	      uint gi = simd_min(mi);
	      if (simd_lid == 0) {
	        uint vbits = as_type<uint>(gm);
	        uint key = (vbits & 0x80000000u) ? (~vbits) : (vbits ^ 0x80000000u);
	        uint row = uint(row_base + r);
	        uint out_base = (row * uint(n_out_groups) + tid.y) * 2u;
	        tg_argmax[out_base + 0u] = key;
	        tg_argmax[out_base + 1u] = gi;
	      }
	    }
	  }
	}

	template <int n_out_groups, int active_rows>
	METAL_FUNC void fusekit_qmv_argmax_finalize_impl(
	    const device uint* tg_argmax,
	    device uint* next_tok,
	    threadgroup uint* key_buf,
	    threadgroup uint* idx_buf,
	    uint tid,
	    uint tg_size,
	    uint row) {
	  if (row >= uint(active_rows)) {
	    return;
	  }
	  uint row_base = row * uint(n_out_groups) * 2u;

	  uint local_key = 0u;
	  uint local_idx = 0xFFFFFFFFu;
	  for (uint i = tid; i < uint(n_out_groups); i += tg_size) {
	    uint k = tg_argmax[row_base + i * 2u + 0u];
	    uint v = tg_argmax[row_base + i * 2u + 1u];
	    if (k > local_key) {
	      local_key = k;
	      local_idx = v;
	    }
	  }

	  uint simd_lane = tid % 32u;
	  uint simd_id = tid / 32u;
	  uint sgs_per_tg = tg_size / 32u;
	  uint sg_max = simd_max(local_key);
	  uint sg_arg = (local_key == sg_max && local_idx != 0xFFFFFFFFu) ? local_idx : 0xFFFFFFFFu;
	  uint sg_idx = simd_min(sg_arg);
	  if (simd_lane == 0u) {
	    key_buf[simd_id] = sg_max;
	    idx_buf[simd_id] = sg_idx;
	  }
	  threadgroup_barrier(mem_flags::mem_threadgroup);
	  if (simd_id == 0u) {
	    uint v = (simd_lane < sgs_per_tg) ? key_buf[simd_lane] : 0u;
	    uint i = (simd_lane < sgs_per_tg) ? idx_buf[simd_lane] : 0xFFFFFFFFu;
	    uint gm = simd_max(v);
	    uint mi = (v == gm && i != 0xFFFFFFFFu) ? i : 0xFFFFFFFFu;
	    uint gi = simd_min(mi);
	    if (simd_lane == 0u) {
	      next_tok[row] = gi;
	    }
	  }
	}
	
	template <
	    typename T,
	    int group_size,
	    int bits,
	    int num_simdgroups,
	    int results_per_simdgroup>
	METAL_FUNC void fusekit_qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = U(sl[0]);
        U b = U(bl[0]);
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = U(sl[0]);
        U b = U(bl[0]);
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }

  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = U(sl[0]);
        U b = U(bl[0]);
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = U(sl[0]);
        U b = U(bl[0]);
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void fusekit_qmm_t_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    const int K_eff,
    uint3 tid,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <
    typename T,
    int group_size,
    int bits,
    int num_simdgroups,
    int results_per_simdgroup,
    int no_prod_round>
METAL_FUNC void fusekit_qmv_fast_residual_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device T* residual,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  residual += tid.x * out_vec_size + out_row;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;

      U s;
      U b;
      fusekit_qparam_pair_load<T, U, scale_step_per_thread>(
          sl, bl, simd_lid, &s, &b);
      if (bits == 4 && !FUSEKIT_QMV_FAST_QDOT_Q4) {
        const device uint16_t* ws16 = (const device uint16_t*)wl;
        U accum = 0;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          uint16_t wi = ws16[i];
          accum +=
              (x_thread[4 * i] * ((wi      ) & 0x000f) +
               x_thread[4 * i + 1] * ((wi      ) & 0x00f0) +
               x_thread[4 * i + 2] * ((wi      ) & 0x0f00) +
               x_thread[4 * i + 3] * ((wi      ) & 0xf000));
        }
        result[row] += s * accum + sum * b;
      } else if (bits == 5 && !FUSEKIT_QMV_FAST_QDOT_Q5) {
        U accum = 0;
        for (int i = 0; i < (values_per_thread / 8); i++) {
          const device uint8_t* wb = wl + 5 * i;
          const thread U* xt = x_thread + 8 * i;
          accum += (wb[0] & 0x1f) * xt[0];
          accum += (wb[0] & 0xe0) * xt[1];
          accum += (wb[1] & 0x03) * (xt[1] * 256.0f);
          accum += (wb[1] & 0x7c) * xt[2];
          accum += (wb[1] & 0x80) * xt[3];
          accum += (wb[2] & 0x0f) * (xt[3] * 256.0f);
          accum += (wb[2] & 0xf0) * xt[4];
          accum += (wb[3] & 0x01) * (xt[4] * 256.0f);
          accum += (wb[3] & 0x3e) * xt[5];
          accum += (wb[3] & 0xc0) * xt[6];
          accum += (wb[4] & 0x07) * (xt[6] * 256.0f);
          accum += (wb[4] & 0xf8) * xt[7];
        }
        result[row] += s * accum + sum * b;
      } else if (bits == 6 && !FUSEKIT_QMV_FAST_QDOT_Q6) {
        U accum = 0;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const device uint8_t* wb = wl + 3 * i;
          const thread U* xt = x_thread + 4 * i;
          accum += (wb[0] & 0x3f) * xt[0];
          accum += (wb[0] & 0xc0) * xt[1];
          accum += (wb[1] & 0x0f) * (xt[1] * 256.0f);
          accum += (wb[1] & 0xf0) * xt[2];
          accum += (wb[2] & 0x03) * (xt[2] * 256.0f);
          accum += (wb[2] & 0xfc) * xt[3];
        }
        result[row] += s * accum + sum * b;
      } else if (bits == 8 && !FUSEKIT_QMV_FAST_QDOT_Q8) {
        U accum = 0;
        for (int i = 0; i < values_per_thread; i++) {
          uint8_t wi = wl[i];
          accum += x_thread[i] * wi;
        }
        result[row] += s * accum + sum * b;
      } else {
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      if (no_prod_round != 0) {
        y[row] = static_cast<T>(
            result[row] + static_cast<U>(residual[row]));
      } else {
        T prod = static_cast<T>(result[row]);
        y[row] = static_cast<T>(
            static_cast<U>(prod) + static_cast<U>(residual[row]));
      }
    }
  }
}

template <
    typename T,
    int group_size,
	    int bits,
	    int num_simdgroups,
	    int results_per_simdgroup,
	    int token_rows,
	    int no_prod_round>
METAL_FUNC void fusekit_qmv_fast_residual_m_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device T* residual,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    const int active_rows,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int row_base = tid.x * token_rows;
  if (row_base >= active_rows) {
    return;
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const device uint8_t* ws_base = (const device uint8_t*)w +
      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  const device T* scales_base =
      scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* biases_base =
      biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  typedef float U;

  thread U x_thread[token_rows][values_per_thread];
  thread U x_sum[token_rows];
  thread U result[token_rows][results_per_simdgroup];
  for (int r = 0; r < token_rows; r++) {
    x_sum[r] = 0;
    for (int c = 0; c < results_per_simdgroup; c++) {
      result[r][c] = 0;
    }
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    for (int r = 0; r < token_rows; r++) {
      if (row_base + r < active_rows) {
        const device T* xr =
            x + (row_base + r) * in_vec_size + k + simd_lid * values_per_thread;
        x_sum[r] = load_vector<T, U, values_per_thread, bits>(xr, x_thread[r]);
      } else {
        x_sum[r] = 0;
        for (int i = 0; i < values_per_thread; i++) {
          x_thread[r][i] = 0;
        }
      }
    }

    const device uint8_t* ws_k =
        ws_base + k * bytes_per_pack / pack_factor;
    const device T* scales_k = scales_base + k / group_size;
    const device T* biases_k = biases_base + k / group_size;

    for (int col = 0; col < results_per_simdgroup; col++) {
      const device uint8_t* wl = ws_k + col * in_vec_size_w;
      const device T* sl = scales_k + col * in_vec_size_g;
      const device T* bl = biases_k + col * in_vec_size_g;
      U s;
      U b;
      fusekit_qparam_pair_load<T, U, scale_step_per_thread>(
          sl, bl, simd_lid, &s, &b);

      if (bits == 4) {
        const device uint16_t* ws16 = (const device uint16_t*)wl;
        thread U accum[token_rows];
        for (int r = 0; r < token_rows; r++) {
          accum[r] = 0;
        }
        for (int i = 0; i < (values_per_thread / 4); i++) {
          uint16_t wi = ws16[i];
          for (int r = 0; r < token_rows; r++) {
            accum[r] +=
                (x_thread[r][4 * i] * ((wi      ) & 0x000f) +
                 x_thread[r][4 * i + 1] * ((wi      ) & 0x00f0) +
                 x_thread[r][4 * i + 2] * ((wi      ) & 0x0f00) +
                 x_thread[r][4 * i + 3] * ((wi      ) & 0xf000));
          }
        }
        for (int r = 0; r < token_rows; r++) {
          result[r][col] += s * accum[r] + x_sum[r] * b;
        }
      } else if (bits == 5) {
        thread U accum[token_rows];
        for (int r = 0; r < token_rows; r++) {
          accum[r] = 0;
        }
        for (int i = 0; i < (values_per_thread / 8); i++) {
          const device uint8_t* wb = wl + 5 * i;
          for (int r = 0; r < token_rows; r++) {
            const thread U* xt = x_thread[r] + 8 * i;
            accum[r] += (wb[0] & 0x1f) * xt[0];
            accum[r] += (wb[0] & 0xe0) * xt[1];
            accum[r] += (wb[1] & 0x03) * (xt[1] * 256.0f);
            accum[r] += (wb[1] & 0x7c) * xt[2];
            accum[r] += (wb[1] & 0x80) * xt[3];
            accum[r] += (wb[2] & 0x0f) * (xt[3] * 256.0f);
            accum[r] += (wb[2] & 0xf0) * xt[4];
            accum[r] += (wb[3] & 0x01) * (xt[4] * 256.0f);
            accum[r] += (wb[3] & 0x3e) * xt[5];
            accum[r] += (wb[3] & 0xc0) * xt[6];
            accum[r] += (wb[4] & 0x07) * (xt[6] * 256.0f);
            accum[r] += (wb[4] & 0xf8) * xt[7];
          }
        }
        for (int r = 0; r < token_rows; r++) {
          result[r][col] += s * accum[r] + x_sum[r] * b;
        }
      } else if (bits == 6) {
        thread U accum[token_rows];
        for (int r = 0; r < token_rows; r++) {
          accum[r] = 0;
        }
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const device uint8_t* wb = wl + 3 * i;
          for (int r = 0; r < token_rows; r++) {
            const thread U* xt = x_thread[r] + 4 * i;
            accum[r] += (wb[0] & 0x3f) * xt[0];
            accum[r] += (wb[0] & 0xc0) * xt[1];
            accum[r] += (wb[1] & 0x0f) * (xt[1] * 256.0f);
            accum[r] += (wb[1] & 0xf0) * xt[2];
            accum[r] += (wb[2] & 0x03) * (xt[2] * 256.0f);
            accum[r] += (wb[2] & 0xfc) * xt[3];
          }
        }
        for (int r = 0; r < token_rows; r++) {
          result[r][col] += s * accum[r] + x_sum[r] * b;
        }
      } else if (bits == 8) {
        thread U accum[token_rows];
        for (int r = 0; r < token_rows; r++) {
          accum[r] = 0;
        }
        for (int i = 0; i < values_per_thread; i++) {
          uint8_t wi = wl[i];
          for (int r = 0; r < token_rows; r++) {
            accum[r] += x_thread[r][i] * wi;
          }
        }
        for (int r = 0; r < token_rows; r++) {
          result[r][col] += s * accum[r] + x_sum[r] * b;
        }
      } else {
        for (int r = 0; r < token_rows; r++) {
          result[r][col] +=
              qdot<U, values_per_thread, bits>(wl, x_thread[r], s, b, x_sum[r]);
        }
      }
    }
  }

  for (int r = 0; r < token_rows; r++) {
    if (row_base + r < active_rows) {
      for (int col = 0; col < results_per_simdgroup; col++) {
        int out_idx = out_row + col;
        if (out_idx >= out_vec_size) {
          continue;
        }
	        result[r][col] = simd_sum(result[r][col]);
	        if (simd_lid == 0) {
	          int row_off = (row_base + r) * out_vec_size + out_idx;
	          if (no_prod_round != 0) {
	            y[row_off] = static_cast<T>(
	                result[r][col] + static_cast<U>(residual[row_off]));
	          } else {
	            T prod = static_cast<T>(result[r][col]);
	            y[row_off] = static_cast<T>(
	                static_cast<U>(prod) + static_cast<U>(residual[row_off]));
	          }
	        }
	      }
	    }
  }
}

template <int values_per_thread, int bits>
inline float fusekit_load_vector_tg_float(
    const threadgroup float* x,
    thread float* x_thread) {
  float sum = 0.0f;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, int group_size, int bits>
METAL_FUNC float fusekit_qmv_fast_tg_partial_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const threadgroup float* x,
    const int in_vec_size,
    const int row,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;

  ws += row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

  thread float x_thread[values_per_thread];
  float result = 0.0f;

  for (int k = 0; k < in_vec_size; k += block_size) {
    float sum = fusekit_load_vector_tg_float<values_per_thread, bits>(x, x_thread);
    float s = float(scales[0]);
    float b = float(biases[0]);
    result += qdot<float, values_per_thread, bits>(ws, x_thread, s, b, sum);

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  return result;
}

template <typename T, int group_size, int bits>
METAL_FUNC void fusekit_qmv_fast_tg2_partial_impl(
    const device uint32_t* w0,
    const device T* scales0,
    const device T* biases0,
    const device uint32_t* w1,
    const device T* scales1,
    const device T* biases1,
    const threadgroup float* x,
    const int in_vec_size,
    const int row,
    uint simd_lid,
    thread float* out0,
    thread float* out1) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const device uint8_t* ws0 = (const device uint8_t*)w0;
  const device uint8_t* ws1 = (const device uint8_t*)w1;

  ws0 += row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  ws1 += row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales0 += row * in_vec_size_g + simd_lid / scale_step_per_thread;
  scales1 += row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases0 += row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases1 += row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

  thread float x_thread[values_per_thread];
  float result0 = 0.0f;
  float result1 = 0.0f;

  for (int k = 0; k < in_vec_size; k += block_size) {
    float sum = fusekit_load_vector_tg_float<values_per_thread, bits>(x, x_thread);
    float s0;
    float b0;
    float s1;
    float b1;
    fusekit_qparam_pair_load<T, float, scale_step_per_thread>(
        scales0, biases0, simd_lid, &s0, &b0);
    fusekit_qparam_pair_load<T, float, scale_step_per_thread>(
        scales1, biases1, simd_lid, &s1, &b1);
    result0 += qdot<float, values_per_thread, bits>(ws0, x_thread, s0, b0, sum);
    result1 += qdot<float, values_per_thread, bits>(ws1, x_thread, s1, b1, sum);

    ws0 += block_size * bytes_per_pack / pack_factor;
    ws1 += block_size * bytes_per_pack / pack_factor;
    scales0 += block_size / group_size;
    scales1 += block_size / group_size;
    biases0 += block_size / group_size;
    biases1 += block_size / group_size;
    x += block_size;
  }

  *out0 = result0;
  *out1 = result1;
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void fusekit_qmm_t_nax_tgp_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    uint3 tid,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm = min(SM, short(M - (y_row + tm)));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : min(SN, short(N - (y_col + tn)));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(x + kk1, K);
          } else {
            Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
          }

          Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }

        x += BK;
        loader_w.next();
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}
"#,
    );
}
