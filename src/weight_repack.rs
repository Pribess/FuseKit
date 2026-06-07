//! Tile-major weight repack utility.
//!
//! Reshapes a row-major weight tensor `(out_dim, in_dim)` into
//! tile-major `(out_dim/T_O, in_dim/T_I, T_O, T_I)` layout so that
//! decode GEMV (32 lanes per SG × R rows per SG × stride-32 K access)
//! sees cache-line-aligned contiguous tiles instead of `R` separate
//! row streams. Each tile = T_O × T_I scalars.
//!
//! For Apple M5 with `gemv_rows_per_simdgroup() = 4` and simdgroup
//! width 32, the natural tile is (T_O=4, T_I=32) — 128 scalars per
//! tile. For half/bf16 that's 256 bytes = 4 cache lines, prefetchable
//! as a single stream.
//!
//! Currently this module operates on in-memory byte buffers; disk-IO
//! (read model.safetensors → write tile_major.safetensors) is the
//! Phase 2 integration. dispatch.py-side zerocopy mmap reassignment of
//! model.layer[i].weight is Phase 3.
//!
//! Repack invariant (host-side check):
//!   tile_major[(o/T_O, i/T_I, o%T_O, i%T_I)] == row_major[o, i]

use anyhow::{bail, Result};

/// Bytes per scalar for a given dtype name.
fn bytes_per_scalar(dtype: &str) -> Result<usize> {
    Ok(match dtype {
        "f16" | "float16" | "half" | "bf16" | "bfloat16" => 2,
        "f32" | "float32" => 4,
        "u32" | "uint32" => 4,
        "u16" | "uint16" => 2,
        "u8" | "uint8" => 1,
        _ => bail!("weight_repack: unsupported dtype `{}`", dtype),
    })
}

/// Repack a 2-D row-major weight buffer into 4-D tile-major.
/// Returns the repacked byte buffer plus the new shape
/// `(out_dim/T_O, in_dim/T_I, T_O, T_I)`. Caller checks divisibility.
pub fn tile_pack_2d(
    src: &[u8],
    out_dim: usize,
    in_dim: usize,
    t_out: usize,
    t_in: usize,
    dtype: &str,
) -> Result<(Vec<u8>, [usize; 4])> {
    let bytes = bytes_per_scalar(dtype)?;
    if out_dim % t_out != 0 {
        bail!("tile_pack_2d: out_dim {out_dim} not divisible by T_O {t_out}");
    }
    if in_dim % t_in != 0 {
        bail!("tile_pack_2d: in_dim {in_dim} not divisible by T_I {t_in}");
    }
    if src.len() != out_dim * in_dim * bytes {
        bail!(
            "tile_pack_2d: src len {} != {out_dim}×{in_dim}×{bytes}",
            src.len()
        );
    }
    let mut dst = vec![0u8; src.len()];
    let n_tiles_o = out_dim / t_out;
    let n_tiles_i = in_dim / t_in;
    for ot in 0..n_tiles_o {
        for it in 0..n_tiles_i {
            for io in 0..t_out {
                let o = ot * t_out + io;
                let src_row_byte_off = (o * in_dim + it * t_in) * bytes;
                let dst_tile_byte_off = (((ot * n_tiles_i + it) * t_out + io) * t_in) * bytes;
                let row_byte_len = t_in * bytes;
                dst[dst_tile_byte_off..dst_tile_byte_off + row_byte_len]
                    .copy_from_slice(&src[src_row_byte_off..src_row_byte_off + row_byte_len]);
            }
        }
    }
    Ok((dst, [n_tiles_o, n_tiles_i, t_out, t_in]))
}

/// Inverse of `tile_pack_2d`. Useful for correctness verification
/// (round-trip should be identity).
pub fn tile_unpack_2d(
    src: &[u8],
    out_dim: usize,
    in_dim: usize,
    t_out: usize,
    t_in: usize,
    dtype: &str,
) -> Result<Vec<u8>> {
    let bytes = bytes_per_scalar(dtype)?;
    let mut dst = vec![0u8; src.len()];
    let n_tiles_o = out_dim / t_out;
    let n_tiles_i = in_dim / t_in;
    for ot in 0..n_tiles_o {
        for it in 0..n_tiles_i {
            for io in 0..t_out {
                let o = ot * t_out + io;
                let dst_row_byte_off = (o * in_dim + it * t_in) * bytes;
                let src_tile_byte_off = (((ot * n_tiles_i + it) * t_out + io) * t_in) * bytes;
                let row_byte_len = t_in * bytes;
                dst[dst_row_byte_off..dst_row_byte_off + row_byte_len]
                    .copy_from_slice(&src[src_tile_byte_off..src_tile_byte_off + row_byte_len]);
            }
        }
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_pack_roundtrip_f16() {
        // Synthetic 8×32 weight with byte values = (o * 32 + i) % 256.
        let out_dim = 8usize;
        let in_dim = 32usize;
        let bytes_per = 2usize;
        let mut src = vec![0u8; out_dim * in_dim * bytes_per];
        for o in 0..out_dim {
            for i in 0..in_dim {
                let v = ((o * in_dim + i) % 256) as u8;
                let off = (o * in_dim + i) * bytes_per;
                src[off] = v;
                src[off + 1] = (v ^ 0x55) as u8;
            }
        }
        let (packed, packed_shape) = tile_pack_2d(&src, out_dim, in_dim, 4, 32, "f16").unwrap();
        assert_eq!(packed_shape, [2, 1, 4, 32]);
        assert_eq!(packed.len(), src.len());
        let unpacked = tile_unpack_2d(&packed, out_dim, in_dim, 4, 32, "f16").unwrap();
        assert_eq!(unpacked, src);
    }

    #[test]
    fn tile_pack_invariant_at_a_point() {
        // For an 8×32 buffer with T_O=4, T_I=32:
        //   - tile (ot=0, it=0) covers o=0..4, i=0..32
        //   - tile (ot=1, it=0) covers o=4..8, i=0..32
        // Verify src[o=5, i=7] lands at tile (ot=1, it=0, io=1, ii=7).
        let out_dim = 8usize;
        let in_dim = 32usize;
        let mut src = vec![0u8; out_dim * in_dim * 2];
        let o = 5;
        let i = 7;
        let src_off = (o * in_dim + i) * 2;
        src[src_off] = 0xAB;
        src[src_off + 1] = 0xCD;
        let (packed, _) = tile_pack_2d(&src, out_dim, in_dim, 4, 32, "f16").unwrap();
        // tile (ot=1, it=0, io=1, ii=7): flat index = (1*1*4 + 1)*32 + 7 = 5*32 + 7 = 167
        let expected_off = (1 * 1 * 4 + 1) * 32 * 2 + 7 * 2;
        assert_eq!(packed[expected_off], 0xAB);
        assert_eq!(packed[expected_off + 1], 0xCD);
    }
}
