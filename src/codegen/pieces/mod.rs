//! Composable kernel pieces — fusekit's shallow-IR for transformer
//! decode kernels.
//!
//! A piece is a self-contained MSL emit fragment that declares:
//!   - the SMEM regions it allocates (`smem_regions`),
//!   - the buffer arguments it needs (`buffer_slots`),
//!   - the SMEM regions it reads from / writes to other pieces
//!     (`smem_reads` / `smem_writes`),
//!   - the MSL body to emit inside the kernel scope (`emit_body`).
//!
//! [`compose`] stitches a linear sequence of pieces into one kernel:
//! it merges buffer slot lists, dedupes SMEM regions, inserts
//! threadgroup barriers when a piece reads SMEM another piece wrote,
//! and emits the wrapping `kernel void fn_name(...)`.
//!
//! The piece interface is the smallest viable IR for our codegen —
//! piece bodies stay hand-written MSL fragments, so we keep all the
//! tuning of the monolithic emitters while allowing new fusions to
//! be expressed as different piece sequences instead of brand-new
//! emitters.

use std::collections::HashSet;

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::ModelMeta;

pub mod global_barrier;
pub mod linear_residual;
pub mod multi_gemv;
pub mod pointwise_chain;
pub mod residual_add;
pub mod rmsnorm_reduce;
pub mod rope_kv_cache_store;
pub mod rope_q;
pub mod sdpa_decode;
pub mod swiglu_activation;

/// A threadgroup-shared memory region a piece allocates.
#[derive(Clone, Debug)]
pub struct SmemRegion {
    /// Variable name as it appears in MSL (e.g. `"norm_y"`).
    pub name: &'static str,
    /// MSL element type (e.g. `"float"`, `"bfloat"`).
    pub elem_ty: String,
    /// Element count (literal, may be a constexpr-resolved C-style expr).
    pub count: String,
}

/// Buffer-binding access pattern used to pick the right MSL prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferAccess {
    Read,
    Write,
}

/// A device-memory buffer slot the composed kernel needs at dispatch.
#[derive(Clone, Debug)]
pub struct BufferSlot {
    /// Variable name in MSL (e.g. `"x"`).
    pub var_name: String,
    /// MSL element type, including pointee qualifier if needed.
    pub elem_ty: String,
    pub access: BufferAccess,
    /// Stable role identifier — composer dedupes slots that share a
    /// role across pieces (e.g. two pieces both consume the same
    /// activation buffer).
    pub role: String,
}

/// Per-kernel emit context shared across all pieces.
pub struct EmitCtx<'a> {
    pub meta: &'a ModelMeta,
    /// Activation MSL type (`"bfloat"` / `"half"` / `"float"`).
    pub act_ty: &'static str,
}

/// A composable kernel fragment. Implementors describe their I/O
/// surface declaratively and emit the MSL body for their stage.
pub trait KernelPiece {
    /// Stable identifier for diagnostics.
    fn name(&self) -> &str;

    /// SMEM regions this piece allocates. Composer dedupes by name.
    fn smem_regions(&self, _ctx: &EmitCtx) -> Vec<SmemRegion> {
        Vec::new()
    }

    /// Buffer slots in the order this piece declares them. Composer
    /// dedupes by `role` so multiple pieces sharing an input slot
    /// only emit one argument.
    fn buffer_slots(&self, ctx: &EmitCtx) -> Vec<BufferSlot>;

    /// SMEM region names this piece reads from prior pieces.
    fn smem_reads(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// SMEM region names this piece writes to (consumed by later pieces).
    fn smem_writes(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// Emit the piece body inside the kernel scope. The composer has
    /// already emitted the kernel signature, SMEM declarations, and
    /// the standard `tid` / `tg_size` / `gid` / `simd_lane` / `simd_id`
    /// / `sgs_per_tg` locals before this is called.
    fn emit_body(&self, w: &mut MslWriter, ctx: &EmitCtx);
}

/// Compose a list of pieces into one kernel function `fn_name`.
///
/// Order matters: pieces are emitted top-to-bottom, with a
/// `threadgroup_barrier` inserted before any piece whose
/// `smem_reads()` overlap with `smem_writes()` of an earlier piece.
pub fn compose(fn_name: &str, pieces: &[Box<dyn KernelPiece>], ctx: &EmitCtx, w: &mut MslWriter) {
    // 1. Dedupe SMEM regions by name; later pieces' declarations are
    //    ignored when a region with the same name already exists.
    //    Different `count`s for the same name are caller error — we
    //    pick the first and let the MSL compiler complain.
    let mut smem: Vec<SmemRegion> = Vec::new();
    let mut seen_smem: HashSet<&'static str> = HashSet::new();
    for p in pieces {
        for r in p.smem_regions(ctx) {
            if seen_smem.insert(r.name) {
                smem.push(r);
            }
        }
    }

    // 2. Buffer slots — deduped + role-unified. Multiple pieces
    //    declaring the same `role` collapse into one buffer slot
    //    with access-flag unification: if any piece writes to the
    //    role, the slot is emitted without the `device const`
    //    prefix so the kernel can read+write. Pieces sharing a role
    //    MUST agree on `var_name` (piece-design contract — they're
    //    referring to the same logical buffer); we keep the first
    //    piece's name and assert no conflicting name later.
    //
    //    Distinct roles that happen to collide on `var_name` are a
    //    piece-design bug (Metal rejects redefinition); the composer
    //    renames colliding later entries with a numeric suffix and
    //    emits a warning so the piece can be fixed at source.
    let mut buffers: Vec<BufferSlot> = Vec::new();
    let mut role_to_idx: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut seen_var: HashSet<String> = HashSet::new();
    for p in pieces {
        for mut b in p.buffer_slots(ctx) {
            // Role merge path: already-seen role unifies access.
            if let Some(&idx) = role_to_idx.get(&b.role) {
                let existing = &mut buffers[idx];
                // Promote to Write if any piece writes.
                if matches!(b.access, BufferAccess::Write) {
                    existing.access = BufferAccess::Write;
                }
                // Var-name contract: pieces sharing a role must agree.
                if existing.var_name != b.var_name {
                    eprintln!(
                        "[fusekit] piece composer: role `{}` declared by \
                         multiple pieces with different var_names (`{}` vs \
                         `{}`) — keeping first, fix the piece",
                        b.role, existing.var_name, b.var_name
                    );
                }
                continue;
            }
            // New role — make sure the var_name doesn't collide with
            // another role's var_name (real piece-design bug).
            if !seen_var.insert(b.var_name.clone()) {
                let original = b.var_name.clone();
                let mut counter = 1u32;
                loop {
                    let candidate = format!("{original}_{counter}");
                    if seen_var.insert(candidate.clone()) {
                        eprintln!(
                            "[fusekit] piece composer: var_name `{}` collision \
                             across roles (latest role `{}`) — renaming to `{}`",
                            original, b.role, candidate
                        );
                        b.var_name = candidate;
                        break;
                    }
                    counter += 1;
                }
            }
            role_to_idx.insert(b.role.clone(), buffers.len());
            buffers.push(b);
        }
    }

    // 3. Kernel signature.
    w.line(format!("// Composed kernel: {} pieces", pieces.len()));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    for (i, b) in buffers.iter().enumerate() {
        let prefix = match b.access {
            BufferAccess::Read => "device const",
            BufferAccess::Write => "device",
        };
        w.line(format!(
            "{prefix} {ty} *{name} [[buffer({i})]],",
            ty = b.elem_ty,
            name = b.var_name,
        ));
    }
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]],");
    // `n_tgs` exposes the grid's TG count so pieces in a multi-TG
    // composition (e.g. LayerBlock with grid=N_HEADS) can stride
    // simdgroup work across all TGs. Single-TG composed kernels see
    // n_tgs=1 and any stride loop terminates after one iteration.
    w.line("uint n_tgs   [[threadgroups_per_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    // 4. Standard locals every piece can rely on.
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id   = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("uint sgs_total = n_tgs * sgs_per_tg;");
    w.blank();

    // 5. SMEM declarations.
    for r in &smem {
        w.line(format!(
            "threadgroup {ty} {name}[{count}];",
            ty = r.elem_ty,
            name = r.name,
            count = r.count,
        ));
    }
    if !smem.is_empty() {
        w.blank();
    }

    // 6. Emit pieces with auto-inserted barriers.
    //    A barrier fires before a piece if it touches (read OR write)
    //    a SMEM region a prior piece wrote to. Includes the
    //    write-after-write case — a piece that re-initialises a
    //    region (e.g. RmsNormReduce called twice in a layer for the
    //    pre-attn and post-attn norms) needs to wait for the prior
    //    consumer's reads to finish so it doesn't clobber values
    //    other threads in the TG are still reading.
    let mut written: HashSet<&'static str> = HashSet::new();
    for p in pieces {
        let reads_dirty = p.smem_reads().iter().any(|r| written.contains(r));
        let writes_dirty = p.smem_writes().iter().any(|r| written.contains(r));
        if reads_dirty || writes_dirty {
            w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        }
        w.line(format!("// piece: {}", p.name()));
        p.emit_body(w, ctx);
        for r in p.smem_writes() {
            written.insert(r);
        }
        w.blank();
    }

    w.pop_indent();
    w.line("}");
    w.blank();
}
