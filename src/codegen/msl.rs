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
    let qmv_m_qdot_all = std::env::var("FUSEKIT_QMV_M_QDOT").ok();
    let qmv_m_qdot_bit = |bits: u32, default_on: bool| -> bool {
        let key = format!("FUSEKIT_QMV_M_QDOT_Q{bits}");
        if let Ok(v) = std::env::var(key) {
            return v.as_str() != "0";
        }
        if let Some(v) = &qmv_m_qdot_all {
            return v.as_str() != "0";
        }
        default_on
    };
    w.line("#define FUSEKIT_QMV_M_QDOT 0");
    // M-row qMV defaults are selected from quant metadata in kernels::,
    // keeping shape admission and emitted helper macros in sync.
    for bits in [4u32, 5u32, 6u32, 8u32] {
        let default_on = crate::codegen::kernels::qmv_m_qdot_default(bits);
        let enabled = if qmv_m_qdot_bit(bits, default_on) {
            1
        } else {
            0
        };
        w.line(format!("#define FUSEKIT_QMV_M_QDOT_Q{bits} {enabled}"));
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
    for bits in [4u32, 5u32, 6u32, 8u32] {
        let default_on = crate::codegen::kernels::qmv_fast_qdot_default(bits);
        let enabled = if qmv_fast_qdot_bit(bits, default_on) {
            1
        } else {
            0
        };
        w.line(format!("#define FUSEKIT_QMV_FAST_QDOT_Q{bits} {enabled}"));
    }
    for bits in [4u32, 8u32] {
        let key = format!("FUSEKIT_QMV_FAST_PACKS_Q{bits}");
        let default_packs =
            crate::codegen::kernels::qmv_fast_packs_per_thread_default(bits).unwrap_or(2);
        let packs = std::env::var(key)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| matches!(n, 1 | 2 | 4))
            .unwrap_or(default_packs);
        w.line(format!("#define FUSEKIT_QMV_FAST_PACKS_Q{bits} {packs}"));
    }
    for bits in [4u32, 8u32] {
        let key = format!("FUSEKIT_QMV_FAST_TG2_PACKS_Q{bits}");
        let default_packs =
            crate::codegen::kernels::qmv_fast_tg2_packs_per_thread_default(bits).unwrap_or(2);
        let packs = std::env::var(key)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| matches!(n, 1 | 2 | 4))
            .unwrap_or(default_packs);
        w.line(format!(
            "#define FUSEKIT_QMV_FAST_TG2_PACKS_Q{bits} {packs}"
        ));
    }
    let residual_q4_qdot = std::env::var("FUSEKIT_QMV_FAST_RESIDUAL_QDOT_Q4")
        .ok()
        .map(|s| s.as_str() != "0")
        .unwrap_or(crate::codegen::kernels::qmv_fast_residual_qdot_default(4));
    w.line(format!(
        "#define FUSEKIT_QMV_FAST_RESIDUAL_QDOT_Q4 {}",
        if residual_q4_qdot { 1 } else { 0 }
    ));
    let residual_q8_qdot = std::env::var("FUSEKIT_QMV_FAST_RESIDUAL_QDOT_Q8")
        .ok()
        .map(|s| s.as_str() != "0")
        .unwrap_or(crate::codegen::kernels::qmv_fast_residual_qdot_default(8));
    w.line(format!(
        "#define FUSEKIT_QMV_FAST_RESIDUAL_QDOT_Q8 {}",
        if residual_q8_qdot { 1 } else { 0 }
    ));
    let qmv_fast_qparam_broadcast = std::env::var("FUSEKIT_QMV_FAST_QPARAM_BROADCAST")
        .ok()
        .map(|s| s.as_str() != "0")
        .unwrap_or(false);
    w.line(format!(
        "#define FUSEKIT_QMV_FAST_QPARAM_BROADCAST {}",
        if qmv_fast_qparam_broadcast { 1 } else { 0 }
    ));
    let qparam_broadcast_bit = |bits: u32, default_on: bool| -> bool {
        let key = format!("FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q{bits}");
        std::env::var(key)
            .ok()
            .map(|s| s.as_str() != "0")
            .unwrap_or(default_on)
    };
    for bits in [4u32, 5u32, 6u32, 8u32] {
        let default_on = crate::codegen::kernels::qmv_fast_qparam_broadcast_default(bits);
        let enabled = if qparam_broadcast_bit(bits, default_on) {
            1
        } else {
            0
        };
        w.line(format!(
            "#define FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q{bits} {enabled}"
        ));
    }
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
template <typename T, typename U, int scale_step_per_thread, int bits>
METAL_FUNC void fusekit_qparam_pair_load(
    const device T* scales,
    const device T* biases,
    uint simd_lid,
    thread U* scale,
    thread U* bias) {
  if constexpr (scale_step_per_thread > 1) {
    if constexpr (
        FUSEKIT_QMV_FAST_QPARAM_BROADCAST ||
        (bits == 4 && FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q4) ||
        (bits == 5 && FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q5) ||
        (bits == 6 && FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q6) ||
        (bits == 8 && FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q8)) {
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

	template <typename U, int values_per_thread>
	METAL_FUNC U fusekit_q8_dot_byte_accum(
	    const device uint8_t* ws,
	    thread U* x_thread) {
	  U accum = U(0);
	  for (int i = 0; i < values_per_thread; ++i) {
	    accum += x_thread[i] * U(ws[i]);
	  }
	  return accum;
	}

	template <typename U, int values_per_thread>
	METAL_FUNC U fusekit_q8_dot_u32_load(
	    const device uint8_t* ws,
	    thread U* x_thread) {
	  U accum = U(0);
	  const device uint32_t* ws32 = (const device uint32_t*)ws;
	  constexpr int words = values_per_thread / 4;
	  for (int i = 0; i < words; ++i) {
	    uint w = ws32[i];
	    int base = 4 * i;
	    // Keep MLX qdot's exact accumulation order while amortizing the
	    // weight load over four q8 values. Grouping the four products into
	    // one expression is faster-looking but changes fp32 roundoff enough
	    // to flip near-tie greedy tokens in q8 models.
	    accum += x_thread[base + 0] * U(w & 0x000000ffu);
	    accum += x_thread[base + 1] * U((w >> 8) & 0x000000ffu);
	    accum += x_thread[base + 2] * U((w >> 16) & 0x000000ffu);
	    accum += x_thread[base + 3] * U((w >> 24) & 0x000000ffu);
	  }
	  for (int i = words * 4; i < values_per_thread; ++i) {
	    accum += x_thread[i] * U(ws[i]);
	  }
	  return accum;
	}

	template <typename U, int values_per_thread>
	METAL_FUNC U fusekit_q4_dot_u32_unshifted(
	    const device uint8_t* ws,
	    thread U* x_thread) {
	  U accum = U(0);
	  const device uint32_t* ws32 = (const device uint32_t*)ws;
	  for (int i = 0; i < (values_per_thread / 8); ++i) {
	    uint wi = ws32[i];
	    uint wlo = wi & 0x0000ffffu;
	    uint whi = wi >> 16u;
	    accum +=
	        (x_thread[8 * i + 0] * U(wlo & 0x000fu) +
	         x_thread[8 * i + 1] * U(wlo & 0x00f0u) +
	         x_thread[8 * i + 2] * U(wlo & 0x0f00u) +
	         x_thread[8 * i + 3] * U(wlo & 0xf000u) +
	         x_thread[8 * i + 4] * U(whi & 0x000fu) +
	         x_thread[8 * i + 5] * U(whi & 0x00f0u) +
	         x_thread[8 * i + 6] * U(whi & 0x0f00u) +
	         x_thread[8 * i + 7] * U(whi & 0xf000u));
	  }
	  return accum;
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
  constexpr int packs_per_thread =
      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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
	      fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
	          sl, bl, simd_lid, &s, &b);
	      if (bits == 4 && !FUSEKIT_QMV_FAST_QDOT_Q4) {
	        const device uint32_t* ws32 = (const device uint32_t*)wl;
	        U accum = 0;
	        for (int i = 0; i < (values_per_thread / 8); i++) {
	          uint wi = ws32[i];
	          uint wlo = wi & 0x0000ffffu;
	          uint whi = wi >> 16u;
	          accum +=
	              (x_thread[8 * i] * (wlo & 0x000fu) +
	               x_thread[8 * i + 1] * (wlo & 0x00f0u) +
	               x_thread[8 * i + 2] * (wlo & 0x0f00u) +
	               x_thread[8 * i + 3] * (wlo & 0xf000u) +
	               x_thread[8 * i + 4] * (whi & 0x000fu) +
	               x_thread[8 * i + 5] * (whi & 0x00f0u) +
	               x_thread[8 * i + 6] * (whi & 0x0f00u) +
	               x_thread[8 * i + 7] * (whi & 0xf000u));
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
        U accum = fusekit_q8_dot_u32_load<U, values_per_thread>(wl, x_thread);
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
  constexpr int packs_per_thread =
      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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
    fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
        scales0, biases0, simd_lid, &s0, &b0);
    fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
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
        if constexpr (bits == 4) {
          result0[row] += s0 * fusekit_q4_dot_u32_unshifted<U, values_per_thread>(wl0, x_thread) + sum * b0;
          result1[row] += s1 * fusekit_q4_dot_u32_unshifted<U, values_per_thread>(wl1, x_thread) + sum * b1;
        } else if constexpr (bits == 8) {
          result0[row] += s0 * fusekit_q8_dot_u32_load<U, values_per_thread>(wl0, x_thread) + sum * b0;
          result1[row] += s1 * fusekit_q8_dot_u32_load<U, values_per_thread>(wl1, x_thread) + sum * b1;
        } else {
          result0[row] +=
              qdot<U, values_per_thread, bits>(wl0, x_thread, s0, b0, sum);
          result1[row] +=
              qdot<U, values_per_thread, bits>(wl1, x_thread, s1, b1, sum);
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
        if constexpr (bits == 4) {
          result0[row] += s0 * fusekit_q4_dot_u32_unshifted<U, values_per_thread>(wl0, x_thread) + sum * b0;
          result1[row] += s1 * fusekit_q4_dot_u32_unshifted<U, values_per_thread>(wl1, x_thread) + sum * b1;
        } else if constexpr (bits == 8) {
          result0[row] += s0 * fusekit_q8_dot_u32_load<U, values_per_thread>(wl0, x_thread) + sum * b0;
          result1[row] += s1 * fusekit_q8_dot_u32_load<U, values_per_thread>(wl1, x_thread) + sum * b1;
        } else {
          result0[row] +=
              qdot<U, values_per_thread, bits>(wl0, x_thread, s0, b0, sum);
          result1[row] +=
              qdot<U, values_per_thread, bits>(wl1, x_thread, s1, b1, sum);
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
  constexpr int packs_per_thread =
      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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
      fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
          sl0, bl0, simd_lid, &s0, &b0);
	      fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
	          sl1, bl1, simd_lid, &s1, &b1);
	      if (bits == 4 && !FUSEKIT_QMV_FAST_QDOT_Q4) {
	        const device uint32_t* ws320 = (const device uint32_t*)wl0;
	        const device uint32_t* ws321 = (const device uint32_t*)wl1;
	        U accum0 = 0;
	        U accum1 = 0;
	        for (int i = 0; i < (values_per_thread / 8); i++) {
	          uint w0i = ws320[i];
	          uint w1i = ws321[i];
	          uint w0lo = w0i & 0x0000ffffu;
	          uint w0hi = w0i >> 16u;
	          uint w1lo = w1i & 0x0000ffffu;
	          uint w1hi = w1i >> 16u;
	          accum0 +=
	              (x_thread[8 * i] * (w0lo & 0x000fu) +
	               x_thread[8 * i + 1] * (w0lo & 0x00f0u) +
	               x_thread[8 * i + 2] * (w0lo & 0x0f00u) +
	               x_thread[8 * i + 3] * (w0lo & 0xf000u) +
	               x_thread[8 * i + 4] * (w0hi & 0x000fu) +
	               x_thread[8 * i + 5] * (w0hi & 0x00f0u) +
	               x_thread[8 * i + 6] * (w0hi & 0x0f00u) +
	               x_thread[8 * i + 7] * (w0hi & 0xf000u));
	          accum1 +=
	              (x_thread[8 * i] * (w1lo & 0x000fu) +
	               x_thread[8 * i + 1] * (w1lo & 0x00f0u) +
	               x_thread[8 * i + 2] * (w1lo & 0x0f00u) +
	               x_thread[8 * i + 3] * (w1lo & 0xf000u) +
	               x_thread[8 * i + 4] * (w1hi & 0x000fu) +
	               x_thread[8 * i + 5] * (w1hi & 0x00f0u) +
	               x_thread[8 * i + 6] * (w1hi & 0x0f00u) +
	               x_thread[8 * i + 7] * (w1hi & 0xf000u));
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
	        U accum0 = fusekit_q8_dot_u32_load<U, values_per_thread>(wl0, x_thread);
	        U accum1 = fusekit_q8_dot_u32_load<U, values_per_thread>(wl1, x_thread);
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
	  constexpr int packs_per_thread =
	      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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
	      U sg;
	      U su;
	      U bg;
	      U bu;
	      if constexpr (bits == 8) {
	        fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
	            sgl, bgl, simd_lid, &sg, &bg);
	        fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
	            sul, bul, simd_lid, &su, &bu);
	      } else {
	        sg = U(sgl[0]);
	        su = U(sul[0]);
	        bg = U(bgl[0]);
	        bu = U(bul[0]);
	      }

		      if (bits == 4 && !FUSEKIT_QMV_M_QDOT_Q4) {
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
				      } else if (bits == 8 && !FUSEKIT_QMV_M_QDOT_Q8) {
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
	  constexpr int packs_per_thread =
	      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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
	      fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
	          sl, bl, simd_lid, &s, &b);

		      if (bits == 4 && !FUSEKIT_QMV_M_QDOT_Q4) {
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
      } else if (bits == 5 && !FUSEKIT_QMV_M_QDOT_Q5) {
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
      } else if (bits == 6 && !FUSEKIT_QMV_M_QDOT_Q6) {
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
				      } else if (bits == 8 && !FUSEKIT_QMV_M_QDOT_Q8) {
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
	  constexpr int packs_per_thread =
	      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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
	      fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
	          sl0, bl0, simd_lid, &s0, &b0);
	      fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
	          sl1, bl1, simd_lid, &s1, &b1);

		      if (bits == 4 && !FUSEKIT_QMV_M_QDOT_Q4) {
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
				      } else if (bits == 8 && !FUSEKIT_QMV_M_QDOT_Q8) {
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
	  constexpr int packs_per_thread =
	      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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
	      fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
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
	    if (k > local_key || (k == local_key && v < local_idx)) {
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

struct FusekitSdpaMaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct FusekitSdpaSumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct FusekitSdpaMulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct FusekitSdpaExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

template <
    typename T,
    int OUT_SEQ,
    int N_HEADS,
    int N_KV_HEADS,
    int HEAD_DIM,
    int HEADS_PER_KV,
    int BQ,
    int BK,
    int WM,
    int WN,
    typename AccumType = float>
METAL_FUNC void fusekit_sdpa_prefill_nax_impl(
    const device T* Q,
    const device T* K,
    const device T* V,
    const constant uint& kv_len,
    const constant uint& capacity,
    device T* O,
    uint grid_gid,
    uint simd_gid,
    uint simd_lid,
    float scale) {
  (void)simd_lid;
  constexpr short kU = 16;
  constexpr int kNWarps = WM * WN;
  static_assert(BQ >= (kNWarps * kU), "BQ must cover all simdgroups");
  static_assert(BQ % (kNWarps * kU) == 0, "BQ must tile simdgroups");
  static_assert(BK % kU == 0, "BK must be divisible by MMA fragment width");
  static_assert(HEAD_DIM % kU == 0, "HEAD_DIM must be divisible by MMA fragment width");
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "FuseKit block SDPA supports d64/d128");

  constexpr int TQ = BQ / (kNWarps * kU);
  constexpr int TK = BK / kU;
  constexpr int TD = HEAD_DIM / kU;
  static_assert(TQ == 1, "FuseKit block SDPA expects one Q fragment per simdgroup");

  uint q_block = grid_gid / uint(N_HEADS);
  uint head_id = grid_gid - q_block * uint(N_HEADS);
  uint kv_head = head_id / uint(HEADS_PER_KV);
  uint q_block_start = q_block * uint(BQ);
  if (q_block_start >= uint(OUT_SEQ) || head_id >= uint(N_HEADS) || kv_head >= uint(N_KV_HEADS)) {
    return;
  }

  const device T* Qh = Q + (q_block_start * uint(N_HEADS) + head_id) * uint(HEAD_DIM);
  const device T* Kh = K + kv_head * capacity * uint(HEAD_DIM);
  const device T* Vh = V + kv_head * capacity * uint(HEAD_DIM);
  device T* Oh = O + (head_id * uint(OUT_SEQ) + q_block_start) * uint(HEAD_DIM);

  using otile_t = NAXTile<AccumType, TQ, TD>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * short(simd_gid);
  Qh += uint(tm) * uint(N_HEADS * HEAD_DIM);
  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  constexpr short kRowsPT = otile_t::kRowsPerThread;
  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    int row = int(q_block_start) + int(tm) + int(i) * int(otile_t::kFragRowsJump) + int(sm);
    max_score[i] = (row < OUT_SEQ && (uint(kv_len) + uint(row)) > 0u) ? -INFINITY : 0.0f;
  }

  uint block_last_row = min(q_block_start + uint(BQ) - 1u, uint(OUT_SEQ - 1));
  uint kv_len_block = min(uint(kv_len) + block_last_row, capacity);
  int kb_lim = (int(kv_len_block) + BK - 1) / BK;

  for (int kb = 0; kb < kb_lim; ++kb) {
    const bool is_last_k = ((kb + 1) * BK > int(kv_len_block));
    const short lim_rows_k = short(int(kv_len_block) - kb * BK);

    using stile_t = NAXTile<AccumType, TQ, TK>;
    stile_t Stile;
    Stile.clear();

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;

          const int Q_load_off = iq * kU * int(N_HEADS * HEAD_DIM) + id * kU;
          const int K_load_off = kb * BK * int(HEAD_DIM) + ik * kU * int(HEAD_DIM) + id * kU;
          const short q_rows_left = short(int(OUT_SEQ) - int(q_block_start) - int(tm) - iq * kU);

          if (q_rows_left < kU) {
            Qtile.load_rows(Qh + Q_load_off, int(N_HEADS * HEAD_DIM), q_rows_left);
          } else {
            Qtile.load(Qh + Q_load_off, int(N_HEADS * HEAD_DIM));
          }

          if (is_last_k) {
            Ktile.load_rows(Kh + K_load_off, int(HEAD_DIM), lim_rows_k - ik * kU);
          } else {
            Ktile.load(Kh + K_load_off, int(HEAD_DIM));
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    const float scale2 = scale * 1.44269504089f;
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }

    if (is_last_k) {
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < lim_rows_k) ? fg[loc] : -INFINITY;
            }
          }
        }
      }
    }

    const int base_row = int(q_block_start) + int(tm);
    const int base_col = kb * BK;
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(iq, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto r = base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
            const auto c = base_col + ik * kU + jj + sn;
            const auto loc = ii * stile_t::kFragThrCols + jj;
            const int visible = int(kv_len) + max(r, 0);
            fg[loc] = (r >= OUT_SEQ || c >= visible) ? -INFINITY : fg[loc];
          }
        }
      }
    }

    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<FusekitSdpaMaxOp>(new_max);
    Stile.template row_bin_op<FusekitSdpaExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<FusekitSdpaSumOp>(sum_score);
    Otile.template row_bin_op<FusekitSdpaMulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id += 2) {
        if constexpr (HEAD_DIM == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;
          const int V_load_off = kb * BK * int(HEAD_DIM) + ik * kU * int(HEAD_DIM) + id * kU;
          if (is_last_k) {
            Vtile.load_rows(Vh + V_load_off, int(HEAD_DIM), lim_rows_k - ik * kU);
          } else {
            Vtile.load(Vh + V_load_off, int(HEAD_DIM));
          }
          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_none);
  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = (sum_score[i] > 0.0f) ? (1.0f / sum_score[i]) : 0.0f;
  }
  Otile.template row_bin_op<FusekitSdpaMulOp>(rcp);

  Oh += uint(tm) * uint(HEAD_DIM);
  const short lim_rows_q = short(int(OUT_SEQ) - int(q_block_start) - int(tm));
  if (lim_rows_q <= 0) {
    return;
  }
  if (lim_rows_q < kU) {
    Otile.store_rows(Oh, int(HEAD_DIM), lim_rows_q);
  } else {
    Otile.store(Oh, int(HEAD_DIM));
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

template <typename T, typename AccT>
struct FusekitSwigluEpilogue {
  METAL_FUNC T apply(AccT gate_acc, T up_val) const {
    float g = float(T(gate_acc));
    float sig = float(T(1.0f / (1.0f + exp(-g))));
    float gated = float(T(sig * g));
    return T(gated * float(up_val));
  }
};

template <typename T, typename GateMMA, typename UpMMA>
METAL_FUNC void fusekit_store_swiglu_pair_tile(
    thread const GateMMA& gate,
    thread const UpMMA& up,
    device T* y,
    const int ldd,
    short2 dst_tile_dims) {
  y += gate.sm * ldd + gate.sn;
  dst_tile_dims -= short2(gate.sn, gate.sm);
  if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
    return;
  }

  constexpr short kelems = decltype(gate.Ctile)::kElemsPerFrag;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < GateMMA::TM; i++) {
    if (i * GateMMA::TM_stride >= dst_tile_dims.y) {
      continue;
    }
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < GateMMA::TN; j++) {
      thread const auto& gate_frag = gate.Ctile.frag_at(i, j);
      thread const auto& up_frag = up.Ctile.frag_at(i, j);
      const int out_offset = (i * GateMMA::TM_stride) * ldd + (j * GateMMA::TN_stride);
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < kelems; k++) {
        if ((j * GateMMA::TN_stride + k) < dst_tile_dims.x) {
          float g = float(T(gate_frag[k]));
          float u = float(T(up_frag[k]));
          float sig = float(T(1.0f / (1.0f + exp(-g))));
          float gated = float(T(sig * g));
          y[out_offset + k] = T(gated * u);
        }
      }
    }
  }
}

template <
    typename T,
    int group_size,
    int bits,
    bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void fusekit_qmm_t_swiglu_pair_impl(
    const device uint32_t* gate_w,
    const device T* gate_scales,
    const device T* gate_biases,
    const device uint32_t* up_w,
    const device T* up_scales,
    const device T* up_biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* GateWs,
    threadgroup T* UpWs,
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

  const device T* x_tile = x + y_row * static_cast<int64_t>(K);
  device T* y_tile = y + y_row * static_cast<int64_t>(N) + y_col;
  auto gate_wl = (const device uint8_t*)gate_w + y_col * K_w;
  auto up_wl = (const device uint8_t*)up_w + y_col * K_w;
  const device T* gate_s_tile = gate_scales + y_col * K_g;
  const device T* gate_b_tile = gate_biases + y_col * K_g;
  const device T* up_s_tile = up_scales + y_col * K_g;
  const device T* up_b_tile = up_biases + y_col * K_g;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);

  mma_t mma_gate(simd_gid, simd_lid);
  mma_t mma_up(simd_gid, simd_lid);
  loader_x_t loader_x(x_tile, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_gate(gate_wl, gate_s_tile, gate_b_tile, K, GateWs, simd_gid, simd_lid);
  loader_w_t loader_up(up_wl, up_s_tile, up_b_tile, K, UpWs, simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_gate.load_safe(short2(BK, num_outs));
        loader_up.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_gate.mma(Xs, GateWs);
        mma_up.mma(Xs, UpWs);
        loader_x.next();
        loader_gate.next();
        loader_up.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_gate.load_unsafe();
        loader_up.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_gate.mma(Xs, GateWs);
        mma_up.mma(Xs, UpWs);
        loader_x.next();
        loader_gate.next();
        loader_up.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_gate.load_safe(short2(BK, num_outs));
        loader_up.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_gate.mma(Xs, GateWs);
        mma_up.mma(Xs, UpWs);
        loader_x.next();
        loader_gate.next();
        loader_up.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_gate.load_unsafe();
        loader_up.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_gate.mma(Xs, GateWs);
        mma_up.mma(Xs, UpWs);
        loader_x.next();
        loader_gate.next();
        loader_up.next();
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  fusekit_store_swiglu_pair_tile<T, mma_t, mma_t>(
      mma_gate, mma_up, y_tile, N, short2(num_outs, num_els));
}

template <int values_per_thread, int bits>
inline float fusekit_load_vector_tg_float(
    const threadgroup float* x,
    thread float* x_thread);

template <int values_per_thread, int bits, typename X>
inline float fusekit_load_vector_tg_float(
    const threadgroup X* x,
    thread float* x_thread);

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
  constexpr int packs_per_thread =
      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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
      fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
          sl, bl, simd_lid, &s, &b);
      if (bits == 4 && !FUSEKIT_QMV_FAST_RESIDUAL_QDOT_Q4) {
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
      } else if (bits == 8 && !FUSEKIT_QMV_FAST_RESIDUAL_QDOT_Q8) {
        U accum = fusekit_q8_dot_u32_load<U, values_per_thread>(wl, x_thread);
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
    int no_prod_round>
METAL_FUNC void fusekit_qmv_fast_residual_tgx_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device T* residual,
    device T* y,
    threadgroup float* x_smem,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tg_pos,
    uint thread_lid,
    uint tg_size,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread =
      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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
  const int out_row = tg_pos.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* x_row = x + tg_pos.x * in_vec_size;
  residual += tg_pos.x * out_vec_size + out_row;
  y += tg_pos.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    for (int idx = int(thread_lid); idx < block_size; idx += int(tg_size)) {
      x_smem[idx] = float(x_row[k + idx]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const threadgroup float* x_block = x_smem + simd_lid * values_per_thread;
    U sum = fusekit_load_vector_tg_float<values_per_thread, bits, float>(x_block, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;

      U s;
      U b;
      fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
          sl, bl, simd_lid, &s, &b);
      if (bits == 4 && !FUSEKIT_QMV_FAST_RESIDUAL_QDOT_Q4) {
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
      } else if (bits == 8 && !FUSEKIT_QMV_FAST_RESIDUAL_QDOT_Q8) {
        U accum = fusekit_q8_dot_u32_load<U, values_per_thread>(wl, x_thread);
        result[row] += s * accum + sum * b;
      } else {
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
  constexpr int packs_per_thread =
      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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
      fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
          sl, bl, simd_lid, &s, &b);

      if (bits == 4) {
        const device uint32_t* ws32 = (const device uint32_t*)wl;
        thread U accum[token_rows];
        for (int r = 0; r < token_rows; r++) {
          accum[r] = 0;
        }
        for (int i = 0; i < (values_per_thread / 8); i++) {
          uint wi = ws32[i];
          uint wlo = wi & 0x0000ffffu;
          uint whi = wi >> 16u;
          for (int r = 0; r < token_rows; r++) {
            accum[r] +=
                (x_thread[r][8 * i] * (wlo & 0x000fu) +
                 x_thread[r][8 * i + 1] * (wlo & 0x00f0u) +
                 x_thread[r][8 * i + 2] * (wlo & 0x0f00u) +
                 x_thread[r][8 * i + 3] * (wlo & 0xf000u) +
                 x_thread[r][8 * i + 4] * (whi & 0x000fu) +
                 x_thread[r][8 * i + 5] * (whi & 0x00f0u) +
                 x_thread[r][8 * i + 6] * (whi & 0x0f00u) +
                 x_thread[r][8 * i + 7] * (whi & 0xf000u));
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

template <typename T, int values_per_thread, int bits>
inline float fusekit_load_vector_tail_device(
    const device T* x,
    int lane_base,
    int in_vec_size,
    thread float* x_thread) {
  float sum = 0.0f;
  if constexpr (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      float x0 = (lane_base + i     < in_vec_size) ? float(x[i    ]) : 0.0f;
      float x1 = (lane_base + i + 1 < in_vec_size) ? float(x[i + 1]) : 0.0f;
      float x2 = (lane_base + i + 2 < in_vec_size) ? float(x[i + 2]) : 0.0f;
      float x3 = (lane_base + i + 3 < in_vec_size) ? float(x[i + 3]) : 0.0f;
      sum += x0 + x1 + x2 + x3;
      x_thread[i] = x0;
      x_thread[i + 1] = x1 / 16.0f;
      x_thread[i + 2] = x2 / 256.0f;
      x_thread[i + 3] = x3 / 4096.0f;
    }
  } else if constexpr (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      float xi = (lane_base + i < in_vec_size) ? float(x[i]) : 0.0f;
      sum += xi;
      x_thread[i] = xi;
    }
  }
  return sum;
}

template <
    typename T,
    int group_size,
    int bits,
    int num_simdgroups,
    int results_per_simdgroup,
    int token_rows,
    int no_prod_round>
METAL_FUNC void fusekit_qmv_fast_residual_m_tail_impl(
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
  constexpr int packs_per_thread =
      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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

  const int full_vec_size = (in_vec_size / block_size) * block_size;

  for (int k = 0; k < full_vec_size; k += block_size) {
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
      fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
          sl, bl, simd_lid, &s, &b);

      if constexpr (bits == 4) {
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
      } else if constexpr (bits == 8) {
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
      }
    }
  }

  if (full_vec_size < in_vec_size) {
    const int k = full_vec_size;
    const int lane_base = k + int(simd_lid) * values_per_thread;
    const bool lane_has_values = lane_base < in_vec_size;
    for (int r = 0; r < token_rows; r++) {
      if (row_base + r < active_rows) {
        const device T* xr =
            x + (row_base + r) * in_vec_size + lane_base;
        x_sum[r] = fusekit_load_vector_tail_device<T, values_per_thread, bits>(
            xr, lane_base, in_vec_size, x_thread[r]);
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
      U s = 0;
      U b = 0;
      if (lane_has_values) {
        fusekit_qparam_pair_load<T, U, scale_step_per_thread, bits>(
            sl, bl, simd_lid, &s, &b);
      }

      if constexpr (bits == 4) {
        const device uint16_t* ws16 = (const device uint16_t*)wl;
        thread U accum[token_rows];
        for (int r = 0; r < token_rows; r++) {
          accum[r] = 0;
        }
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const int elem_base = lane_base + 4 * i;
          uint16_t wi = (elem_base < in_vec_size) ? ws16[i] : 0;
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
      } else if constexpr (bits == 8) {
        thread U accum[token_rows];
        for (int r = 0; r < token_rows; r++) {
          accum[r] = 0;
        }
        for (int i = 0; i < values_per_thread; i++) {
          const int elem_idx = lane_base + i;
          uint8_t wi = (elem_idx < in_vec_size) ? wl[i] : 0;
          for (int r = 0; r < token_rows; r++) {
            accum[r] += x_thread[r][i] * wi;
          }
        }
        for (int r = 0; r < token_rows; r++) {
          result[r][col] += s * accum[r] + x_sum[r] * b;
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

template <int values_per_thread, int bits, typename X>
inline float fusekit_load_vector_tg_float(
    const threadgroup X* x,
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
      float x0 = float(x[i]);
      float x1 = float(x[i + 1]);
      float x2 = float(x[i + 2]);
      float x3 = float(x[i + 3]);
      sum += x0 + x1 + x2 + x3;
      x_thread[i] = x0;
      x_thread[i + 1] = x1 / 16.0f;
      x_thread[i + 2] = x2 / 256.0f;
      x_thread[i + 3] = x3 / 4096.0f;
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
      float xi = float(x[i]);
      sum += xi;
      x_thread[i] = xi;
    }
  }

  return sum;
}

template <typename T, int group_size, int bits, typename X>
METAL_FUNC float fusekit_qmv_fast_tg_partial_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const threadgroup X* x,
    const int in_vec_size,
    const int row,
    uint simd_lid) {
  constexpr int packs_per_thread =
      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_PACKS_Q8 : 2));
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
    float sum = fusekit_load_vector_tg_float<values_per_thread, bits, X>(x, x_thread);
    float s;
    float b;
    fusekit_qparam_pair_load<T, float, scale_step_per_thread, bits>(
        scales, biases, simd_lid, &s, &b);
    if (bits == 4 && !FUSEKIT_QMV_FAST_QDOT_Q4) {
      const device uint32_t* ws32 = (const device uint32_t*)ws;
      float accum = 0.0f;
      for (int i = 0; i < (values_per_thread / 8); i++) {
        uint wi = ws32[i];
        uint wlo = wi & 0x0000ffffu;
        uint whi = wi >> 16u;
        float x0 = x_thread[8 * i];
        float x1 = x_thread[8 * i + 1];
        float x2 = x_thread[8 * i + 2];
        float x3 = x_thread[8 * i + 3];
        float x4 = x_thread[8 * i + 4];
        float x5 = x_thread[8 * i + 5];
        float x6 = x_thread[8 * i + 6];
        float x7 = x_thread[8 * i + 7];
        accum +=
            (x0 * (wlo & 0x000fu) +
             x1 * (wlo & 0x00f0u) +
             x2 * (wlo & 0x0f00u) +
             x3 * (wlo & 0xf000u) +
             x4 * (whi & 0x000fu) +
             x5 * (whi & 0x00f0u) +
             x6 * (whi & 0x0f00u) +
             x7 * (whi & 0xf000u));
      }
      result += s * accum + sum * b;
    } else if (bits == 8 && !FUSEKIT_QMV_FAST_RESIDUAL_QDOT_Q8) {
      float accum = 0.0f;
      for (int i = 0; i < values_per_thread; i++) {
        accum += x_thread[i] * ws[i];
      }
      result += s * accum + sum * b;
    } else {
      result += qdot<float, values_per_thread, bits>(ws, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  return result;
}

template <typename T, int group_size, int bits, typename X>
METAL_FUNC void fusekit_qmv_fast_tg2_partial_impl(
    const device uint32_t* w0,
    const device T* scales0,
    const device T* biases0,
    const device uint32_t* w1,
    const device T* scales1,
    const device T* biases1,
    const threadgroup X* x,
    const int in_vec_size,
    const int row,
    uint simd_lid,
    thread float* out0,
    thread float* out1) {
  constexpr int packs_per_thread =
      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_TG2_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_TG2_PACKS_Q8 : 2));
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

  float result0 = 0.0f;
	  float result1 = 0.0f;

	  for (int k = 0; k < in_vec_size; k += block_size) {
    if constexpr (bits == 4 && !FUSEKIT_QMV_FAST_QDOT_Q4) {
      float sum = 0.0f;
      const device uint32_t* ws0_32 = (const device uint32_t*)ws0;
      const device uint32_t* ws1_32 = (const device uint32_t*)ws1;
      float accum0 = 0.0f;
      float accum1 = 0.0f;
      for (int i = 0; i < (values_per_thread / 8); i++) {
        uint w0i = ws0_32[i];
        uint w1i = ws1_32[i];
        uint w0lo = w0i & 0x0000ffffu;
        uint w0hi = w0i >> 16u;
        uint w1lo = w1i & 0x0000ffffu;
        uint w1hi = w1i >> 16u;
        float x0r = float(x[8 * i]);
        float x1r = float(x[8 * i + 1]);
        float x2r = float(x[8 * i + 2]);
        float x3r = float(x[8 * i + 3]);
        float x4r = float(x[8 * i + 4]);
        float x5r = float(x[8 * i + 5]);
        float x6r = float(x[8 * i + 6]);
        float x7r = float(x[8 * i + 7]);
        sum += x0r + x1r + x2r + x3r + x4r + x5r + x6r + x7r;
        float x0 = x0r;
        float x1 = x1r / 16.0f;
        float x2 = x2r / 256.0f;
        float x3 = x3r / 4096.0f;
        float x4 = x4r;
        float x5 = x5r / 16.0f;
        float x6 = x6r / 256.0f;
        float x7 = x7r / 4096.0f;
        accum0 +=
            (x0 * (w0lo & 0x000fu) +
             x1 * (w0lo & 0x00f0u) +
             x2 * (w0lo & 0x0f00u) +
             x3 * (w0lo & 0xf000u) +
             x4 * (w0hi & 0x000fu) +
             x5 * (w0hi & 0x00f0u) +
             x6 * (w0hi & 0x0f00u) +
             x7 * (w0hi & 0xf000u));
        accum1 +=
            (x0 * (w1lo & 0x000fu) +
             x1 * (w1lo & 0x00f0u) +
             x2 * (w1lo & 0x0f00u) +
             x3 * (w1lo & 0xf000u) +
             x4 * (w1hi & 0x000fu) +
             x5 * (w1hi & 0x00f0u) +
             x6 * (w1hi & 0x0f00u) +
             x7 * (w1hi & 0xf000u));
      }
      float s0;
      float b0;
      float s1;
      float b1;
      fusekit_qparam_pair_load<T, float, scale_step_per_thread, bits>(
          scales0, biases0, simd_lid, &s0, &b0);
      fusekit_qparam_pair_load<T, float, scale_step_per_thread, bits>(
          scales1, biases1, simd_lid, &s1, &b1);
      result0 += s0 * accum0 + sum * b0;
      result1 += s1 * accum1 + sum * b1;
	    } else if (bits == 8 && !FUSEKIT_QMV_FAST_QDOT_Q8) {
      thread float x_thread[values_per_thread];
	    float sum = fusekit_load_vector_tg_float<values_per_thread, bits, X>(x, x_thread);
	    float s0;
	    float b0;
      float s1;
      float b1;
      s0 = float(scales0[0]);
      b0 = float(biases0[0]);
      s1 = float(scales1[0]);
      b1 = float(biases1[0]);
	      float accum0 = fusekit_q8_dot_u32_load<float, values_per_thread>(ws0, x_thread);
	      float accum1 = fusekit_q8_dot_u32_load<float, values_per_thread>(ws1, x_thread);
	      result0 += s0 * accum0 + sum * b0;
	      result1 += s1 * accum1 + sum * b1;
	    } else {
      thread float x_thread[values_per_thread];
	    float sum = fusekit_load_vector_tg_float<values_per_thread, bits, X>(x, x_thread);
	    float s0;
	    float b0;
      float s1;
      float b1;
      if constexpr (bits == 8) {
        float accum0 = fusekit_q8_dot_byte_accum<float, values_per_thread>(ws0, x_thread);
        float accum1 = fusekit_q8_dot_byte_accum<float, values_per_thread>(ws1, x_thread);
        // The two-output SwiGLU helper already carries two accumulator streams.
        // On M5 q8 decode the extra simd_broadcast live values slow this helper
        // down, while fanout/residual qMV still benefit from qparam broadcast.
        // Keep this policy scoped to the pair helper instead of changing the
        // global q8 qMV default.
        s0 = float(scales0[0]);
        b0 = float(biases0[0]);
        s1 = float(scales1[0]);
        b1 = float(biases1[0]);
        result0 += s0 * accum0 + sum * b0;
        result1 += s1 * accum1 + sum * b1;
      } else {
        fusekit_qparam_pair_load<T, float, scale_step_per_thread, bits>(
            scales0, biases0, simd_lid, &s0, &b0);
        fusekit_qparam_pair_load<T, float, scale_step_per_thread, bits>(
            scales1, biases1, simd_lid, &s1, &b1);
        result0 += qdot<float, values_per_thread, bits>(ws0, x_thread, s0, b0, sum);
        result1 += qdot<float, values_per_thread, bits>(ws1, x_thread, s1, b1, sum);
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

	  *out0 = result0;
	  *out1 = result1;
	}

template <int values_per_thread, int bits, typename X>
inline float fusekit_load_vector_tg_float_tail(
    const threadgroup X* x,
    int lane_base,
    int in_vec_size,
    thread float* x_thread) {
  float sum = 0.0f;
  if constexpr (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      float x0 = (lane_base + i     < in_vec_size) ? float(x[i    ]) : 0.0f;
      float x1 = (lane_base + i + 1 < in_vec_size) ? float(x[i + 1]) : 0.0f;
      float x2 = (lane_base + i + 2 < in_vec_size) ? float(x[i + 2]) : 0.0f;
      float x3 = (lane_base + i + 3 < in_vec_size) ? float(x[i + 3]) : 0.0f;
      sum += x0 + x1 + x2 + x3;
      x_thread[i] = x0;
      x_thread[i + 1] = x1 / 16.0f;
      x_thread[i + 2] = x2 / 256.0f;
      x_thread[i + 3] = x3 / 4096.0f;
    }
  } else if constexpr (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      float xi = (lane_base + i < in_vec_size) ? float(x[i]) : 0.0f;
      sum += xi;
      x_thread[i] = xi;
    }
  }
  return sum;
}

template <typename T, int group_size, int bits, typename X>
METAL_FUNC void fusekit_qmv_fast_tg2_partial_tail_impl(
    const device uint32_t* w0,
    const device T* scales0,
    const device T* biases0,
    const device uint32_t* w1,
    const device T* scales1,
    const device T* biases1,
    const threadgroup X* x,
    const int in_vec_size,
    const int row,
    uint simd_lid,
    thread float* out0,
    thread float* out1) {
  constexpr int packs_per_thread =
      bits == 2 ? 1 : (bits == 4 ? FUSEKIT_QMV_FAST_TG2_PACKS_Q4 : (bits == 8 ? FUSEKIT_QMV_FAST_TG2_PACKS_Q8 : 2));
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
    const int lane_base = k + int(simd_lid) * values_per_thread;
    const bool lane_has_values = lane_base < in_vec_size;
    float sum = fusekit_load_vector_tg_float_tail<values_per_thread, bits, X>(
        x, lane_base, in_vec_size, x_thread);
    float s0 = 0.0f;
    float b0 = 0.0f;
    float s1 = 0.0f;
    float b1 = 0.0f;
    if (lane_has_values) {
      fusekit_qparam_pair_load<T, float, scale_step_per_thread, bits>(
          scales0, biases0, simd_lid, &s0, &b0);
      fusekit_qparam_pair_load<T, float, scale_step_per_thread, bits>(
          scales1, biases1, simd_lid, &s1, &b1);
    }

    if constexpr (bits == 4) {
      const device uint16_t* ws016 = (const device uint16_t*)ws0;
      const device uint16_t* ws116 = (const device uint16_t*)ws1;
      float accum0 = 0.0f;
      float accum1 = 0.0f;
      for (int i = 0; i < (values_per_thread / 4); i++) {
        const int elem_base = lane_base + 4 * i;
        uint16_t wi0 = (elem_base < in_vec_size) ? ws016[i] : 0;
        uint16_t wi1 = (elem_base < in_vec_size) ? ws116[i] : 0;
        accum0 +=
            (x_thread[4 * i] * ((wi0      ) & 0x000f) +
             x_thread[4 * i + 1] * ((wi0      ) & 0x00f0) +
             x_thread[4 * i + 2] * ((wi0      ) & 0x0f00) +
             x_thread[4 * i + 3] * ((wi0      ) & 0xf000));
        accum1 +=
            (x_thread[4 * i] * ((wi1      ) & 0x000f) +
             x_thread[4 * i + 1] * ((wi1      ) & 0x00f0) +
             x_thread[4 * i + 2] * ((wi1      ) & 0x0f00) +
             x_thread[4 * i + 3] * ((wi1      ) & 0xf000));
      }
      result0 += s0 * accum0 + sum * b0;
      result1 += s1 * accum1 + sum * b1;
    } else if constexpr (bits == 8) {
      float accum0 = 0.0f;
      float accum1 = 0.0f;
      for (int i = 0; i < values_per_thread; i++) {
        const int elem_idx = lane_base + i;
        uint8_t wi0 = (elem_idx < in_vec_size) ? ws0[i] : 0;
        uint8_t wi1 = (elem_idx < in_vec_size) ? ws1[i] : 0;
        accum0 += x_thread[i] * wi0;
        accum1 += x_thread[i] * wi1;
      }
      result0 += s0 * accum0 + sum * b0;
      result1 += s1 * accum1 + sum * b1;
    }

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

template <typename T, typename Tile, bool residual_vector>
METAL_FUNC void fusekit_store_add_residual_nax_tile(
    thread const Tile& tile,
    const device T* residual,
    device T* y,
    const int ldd,
    short2 dst_tile_dims) {
  using Frag = typename Tile::NAXFrag_t;
  constexpr short tile_rows = Tile::kTileRows;
  constexpr short tile_cols = Tile::kTileCols;
  constexpr short elems_per_frag = Tile::kElemsPerFrag;

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < tile_rows; ++i) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < tile_cols; ++j) {
      thread const auto& frag = tile.frag_at(i, j);
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < elems_per_frag; ++k) {
        short2 coord = Frag::get_coord(k);
        short row = i * Frag::kFragRows + coord.y;
        short col = j * Frag::kFragCols + coord.x;
        if (row < dst_tile_dims.y && col < dst_tile_dims.x) {
          const int idx = row * ldd + col;
          // Match qMM followed by the generic Add kernel: qMM first
          // rounds to model dtype, then Add performs float accumulation.
          float q = float(T(frag[k]));
          float r;
          if constexpr (residual_vector) {
            r = float(residual[col]);
          } else {
            r = float(residual[idx]);
          }
          y[idx] = T(q + r);
        }
      }
    }
  }
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
METAL_FUNC void fusekit_qmm_t_nax_residual_tgp_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device T* residual,
    device T* y,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    const bool residual_vector,
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
  const short tgp_bn = aligned_N ? BN : min(BN, int(N - y_col));
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
      if (residual_vector) {
        fusekit_store_add_residual_nax_tile<T, decltype(Dtile), true>(
          Dtile,
          residual + y_col + tn,
          y + tm * N + tn,
          N,
          short2(sgp_sn, sgp_sm));
      } else {
        fusekit_store_add_residual_nax_tile<T, decltype(Dtile), false>(
          Dtile,
          residual + (y_row + tm) * static_cast<int64_t>(N) + y_col + tn,
          y + tm * N + tn,
          N,
          short2(sgp_sn, sgp_sm));
      }
    });
  });
}

template <typename T, typename Tile>
METAL_FUNC void fusekit_store_swiglu_nax_pair_tile(
    thread const Tile& gate,
    thread const Tile& up,
    device T* y,
    const int ldd,
    short2 dst_tile_dims) {
  using Frag = typename Tile::NAXFrag_t;
  constexpr short tile_rows = Tile::kTileRows;
  constexpr short tile_cols = Tile::kTileCols;
  constexpr short elems_per_frag = Tile::kElemsPerFrag;

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < tile_rows; ++i) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < tile_cols; ++j) {
      thread const auto& gate_frag = gate.frag_at(i, j);
      thread const auto& up_frag = up.frag_at(i, j);
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < elems_per_frag; ++k) {
        short2 coord = Frag::get_coord(k);
        short row = i * Frag::kFragRows + coord.y;
        short col = j * Frag::kFragCols + coord.x;
        if (row < dst_tile_dims.y && col < dst_tile_dims.x) {
          float g = float(T(gate_frag[k]));
          float u = float(T(up_frag[k]));
          float sig = float(T(1.0f / (1.0f + exp(-g))));
          float gated = float(T(sig * g));
          y[row * ldd + col] = T(gated * u);
        }
      }
    }
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 32,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void fusekit_qmm_t_swiglu_pair_nax_impl(
    const device uint32_t* gate_w,
    const device T* gate_scales,
    const device T* gate_biases,
    const device uint32_t* up_w,
    const device T* up_scales,
    const device T* up_biases,
    const device T* x,
    device T* y,
    threadgroup T* GateWs,
    threadgroup T* UpWs,
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

  const device uint8_t* gate_wl = (const device uint8_t*)gate_w + y_col * K_w;
  const device uint8_t* up_wl = (const device uint8_t*)up_w + y_col * K_w;
  gate_scales += y_col * K_g;
  gate_biases += y_col * K_g;
  up_scales += y_col * K_g;
  up_biases += y_col * K_g;
  x += y_row * static_cast<int64_t>(K);
  y += y_row * static_cast<int64_t>(N) + y_col;

  loader_w_t loader_gate(gate_wl, gate_scales, gate_biases, K, GateWs, simd_gid, simd_lid);
  loader_w_t loader_up(up_wl, up_scales, up_biases, K, UpWs, simd_gid, simd_lid);

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
  const short tgp_bn = aligned_N ? BN : min(BN, int(N - y_col));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;
  NAXTile<AccumType, TM, TN> GateD;
  NAXTile<AccumType, TM, TN> UpD;
  GateD.clear();
  UpD.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_gate.load_unsafe();
          loader_up.load_unsafe();
        } else {
          loader_gate.load_safe(short2(BK, tgp_bn));
          loader_up.load_safe(short2(BK, tgp_bn));
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

          Btile.template load<T, BK_padded, 1>(GateWs + tn * BK_padded + kk1);
          tile_matmad_nax(
              GateD,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          Btile.template load<T, BK_padded, 1>(UpWs + tn * BK_padded + kk1);
          tile_matmad_nax(
              UpD,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }

        x += BK;
        loader_gate.next();
        loader_up.next();
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      fusekit_store_swiglu_nax_pair_tile<T>(
          GateD, UpD, y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
    });
  });
}
"#,
    );
}
