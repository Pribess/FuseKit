# fusekit — architecture

fusekit is a Rust compiler that turns mlx-lm models into Apple-Silicon megakernels. This document walks the full pipeline, names the moving parts, and links to the source.

It supersedes the older `fusion-algorithms.md` (which only covered the Stage 2 pattern-matching algorithm in isolation).

---

## Pipeline at a glance

```
            ┌───────────────────────────────────────────────────────────┐
            │  Stage 1 — Capture (in-process PyO3 + mlx-lm)             │
patch time  │    src/loader.rs        mx.export_function callback       │
            │                          → RawGraph (record stream)       │
            │                          → Graph IR (tensors + ops)       │
            ├───────────────────────────────────────────────────────────┤
            │  Stage 2 — Pattern fusion (hand-tuned matchers)           │
            │    src/fusion/patterns/*.rs                               │
            │                          → FusionGraph                    │
            │                          (FusionKind::RmsNormQkv, etc.)   │
            │                          + FusionKind::Single(op_name)    │
            │                              for unmatched ops             │
            ├───────────────────────────────────────────────────────────┤
            │  Stage 3 — General op-semantic fusion (NEW)               │
            │    src/fusion/category.rs   op kind table + chain rules   │
            │    src/fusion/general.rs    producer→consumer merge       │
            │                          → FusionKind::FusedChain         │
            │                              (Vec<ChainStep>)             │
            ├───────────────────────────────────────────────────────────┤
            │  Stage 4 — Codegen + Metal compile                        │
            │    src/codegen/kernels/*.rs   per-FusionKind MSL emitters │
            │    src/codegen/persistent.rs  task-graph dispatcher       │
            │                          → kernel.metal                   │
            │                          → kernel.metallib (xcrun metal)  │
            │                          → task_graph.json                │
            └───────────────────────────────────────────────────────────┘
            ┌───────────────────────────────────────────────────────────┐
runtime     │  Runtime — fusekit_runtime (Python ext + Rust dispatcher) │
(decode     │    python_runtime/        pyo3 cdylib                      │
 hot path)  │    runtime/fusekit_runtime/ Python hook into mlx_lm       │
            │       attach()  → install ICB on the loaded model         │
            │       per-token: submit_icb_async + wait_pending_icb      │
            └───────────────────────────────────────────────────────────┘
```

Patch time runs once per model: `fusekit patch <model>` produces `<model>/fusekit/kernel.metallib` plus `task_graph.json` plus the runtime payload. The runtime hook then attaches at every `mlx_lm.load(...)` and intercepts decode calls to run the megakernel.

---

## Stage 1 — Capture (`src/loader.rs`)

**Input:** model directory (e.g. `~/.omlx/models/mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit/`).

**Output:** typed `Graph` IR — tensors, ops, typed attrs.

### How

fusekit embeds the oMLX-bundled cpython-3.11 via pyo3 (`abi3-py311`, no `extension-module` feature). Build-time:
- `.cargo/config.toml` sets `PYO3_PYTHON` so `pyo3-build-config` finds the right interpreter.
- `build.rs` copies `libpython3.11.dylib` into `OUT_DIR` and rewrites its `install_name` from the bundled `/install/...` placeholder to `@rpath/libpython3.11.dylib`; adds `-rpath` pointing at the bundle.

Runtime:
- `PYTHONHOME` is set to the bundle root before `pyo3::prepare_freethreaded_python()`.
- `mlx_lm.load(model_dir)` returns the model; we call `mx.export_function(callback, model, dummy)` with a Rust closure as the callback.
- The closure receives a `PyDict` per primitive — name, inputs, outputs, **typed `arguments` list**. CustomKernel's `arguments` carries the full `state()` tuple (kernel name, Metal source, grid, threadgroup, …).
- We translate each callback dict into a `RawRecord` and lower the full stream into the `Graph` IR.

There is no JSON spill, no subprocess spawn. The earlier hand-written `scripts/dump_graph.py` driver and `cxx/mlx_bridge.{h,cc}` are gone (`commit: drop dead cxx mlx-bridge`).

### Key data shapes

- `loader::RawRecord`: `kind`, `inputs`, `outputs`, `name`, `arguments: Vec<serde_json::Value>`.
- `graph::Op`: `name: String`, `inputs: Vec<TensorId>`, `outputs: Vec<TensorId>`, `attrs: Attrs`.
- `graph::Attrs`: typed enum for ops that need typed parameters (RmsNorm, Softmax, RoPE, Quantized, **CustomKernel**, …). Unknown ops carry `Attrs::Unknown` and stay name-string only.

### CustomKernel handling

mlx's `mx.fast.metal_kernel(source=...)` produces a `CustomKernel` primitive. Its `state()` exposes name + Metal source + grid + threadgroup + shared-memory + ensure-row-contiguous. The pyo3 callback receives this as a 13-element list in `arguments`; `parse_attrs` packs it into `Attrs::CustomKernel { name, source, grid, threadgroup, shared_memory, ensure_row_contiguous }`. Stage 4 drops the source verbatim into our metallib alongside fusekit-emitted kernels — so an mlx-lm-defined kernel like Qwen3.6's `gated_delta_step` becomes "just another function" in our megakernel.

---

## Stage 2 — Pattern fusion (`src/fusion/patterns/`)

**Input:** `Graph`.
**Output:** `FusionGraph` whose fusions carry typed `FusionKind` (e.g. `RmsNormQkv`) and `FusionAttrs` (the typed parameters codegen needs).

### Pattern matchers

Hand-tuned recognisers for transformer-shaped sub-DAGs:

| Pattern | What it matches | Why hand-tuned |
|---|---|---|
| `RmsNormProj` | RMSNorm → 1 projection | Norm cached in SMEM, fed to projection |
| `RmsNormQkv` | RMSNorm → Q + K + V | Norm in SMEM **shared by 3 matmuls** |
| `RmsNormGateUp` / `…Swiglu` / `…SwigluDown` | MLP block variants | SMEM-resident silu, down_proj epilogue |
| `LinearResidual` | Matmul / QuantizedMatmul → Add (residual) | Matmul accumulator + residual in register, single DRAM write |
| `SdpaDecode` | Q·Kᵀ → scale → softmax → ·V | Decode-tuned scratch layout |
| `FinalRmsNormLmHead` | final RMSNorm → lm_head | Saves a hidden-sized DRAM round-trip |
| `RopeQDecode` / `RopeKvCacheStoreDecode` / `EmbedLookupDecode` / `ArgmaxLogitsDecode` | utility decode kernels | Synthesised at fusion-build time; not matched from trace |

Each pattern has a priority. The build loop walks ops in topological order, tries patterns by priority, and absorbs matched ops into one named fusion. Unmatched ops become `FusionKind::Single(op_name)` so every op in the graph is in exactly one fusion (the **compiler invariant**).

### Quantization-aware

All weight-matmul patterns accept both `Matmul` and `QuantizedMatmul` and capture the int4 `scales` / `biases` as additional boundaries. The QKV matcher also folds the post-QM bias-add into the fusion (mlx splits `quantized_matmul + bias` into two ops; we recombine them).

`fusion::ir::QuantInfo { bits, group_size }` is attached to each quantization-bearing `FusionAttrs::*` variant. Codegen branches on `Option<QuantInfo>` to pick the int4 dequant inner loop vs the bf16/fp16 path.

---

## Stage 3 — General op-semantic fusion (`src/fusion/general.rs`)

Runs **after** Stage 2. Walks the FusionGraph and merges adjacent `FusionKind::Single` (and `FusionKind::FusedChain`) fusions into bigger chains, using only op-semantic categories, single-consumer reachability, and shape compatibility.

Gated by `FUSEKIT_GENERAL_FUSION=1` env var while the default-on policy is being validated.

### Op categories (`src/fusion/category.rs`)

```rust
enum OpCategory {
    Pointwise,    // Add, Multiply, Sigmoid, Exp, Negative, LogAddExp,
                  // AsType, Select, GreaterEqual
    View,         // Reshape, Transpose, Broadcast, Flatten, Unflatten,
                  // Squeeze, ExpandDims, AsContiguous
    Reduction,    // Softmax, RMSNorm, LayerNorm
    Heavy,        // Matmul, AddMM, QuantizedMatmul, Convolution,
                  // RoPE, ScaledDotProductAttention
    Indexing,
    Materialized, // Gather, Slice, Concatenate, Split
    Generator,    // Arange, Full
    CustomKernel, // mx.fast.metal_kernel — opaque to general fusion
    Unknown,
}
```

`can_chain_into(self, consumer)` answers "can a producer of this category feed a consumer of that category in one kernel?". Today's rule set:

| Producer ↓ \ Consumer → | Pointwise | View | other |
|---|---|---|---|
| **Pointwise**         | ✅ | ✅ | depends on caller checks |
| **View** (absorbable) | ✅ | ✅ | — |
| **Reduction**         | ✅ (epilogue) | — | — |
| **Heavy**             | ✅ (epilogue) | — | — |
| **Generator**         | ✅ | — | — |
| **CustomKernel**      | ❌ (barrier, both sides) | ❌ | ❌ |

`is_chain_absorbable_view(op_name)` further restricts View to *element-contiguous* members (Reshape / Flatten / Unflatten / Squeeze / ExpandDims / AsContiguous). Broadcast and Transpose change element layout and would need stride math — still chain barriers for now.

### IR: `FusionKind::FusedChain` + `ChainStep`

A merged chain lives in one fusion whose `FusionAttrs::FusedChain { steps }` is an ordered list. Each `ChainStep` carries:

```rust
struct ChainStep {
    op_name: String,
    op_attrs: graph::Attrs,
    inputs: Vec<ChainOperand>,   // one per op operand
    output_shape: Vec<i32>,
}
enum ChainOperand {
    External { buffer_index: u32 },  // read kernel buffer N
    Register { step_index: u32 },    // read prior step's register
}
```

`buffer_index` indexes into the fusion's external `inputs` list (i.e. the kernel's bound buffers); `step_index` indexes prior steps in the same chain whose output lives in a thread-local register. The chain's last step writes to `out0`.

### Merge algorithm

```
loop:
    rebuild (tensor → producer_fusion, tensor → consumer_fusions)
            from the current FusionGraph
    for each fusion C in FusionGraph:
        if !is_chainable_consumer(C):       continue
        for each external input i of C:
            P = producer_of[i.source]
            if P is None or P == C:         continue
            if consumers_of[i.source] != [C]: continue
            if !is_chainable_producer(P):   continue
            if shapes_compatible(P.out, C.in[i]):
                merge P into C as FusedChain, remove P
                restart the loop
break when no more pairs merge
```

`is_chainable_consumer(C)`: every step's op must be Pointwise or absorbable-View.
`is_chainable_producer(P)`: same plus a Heavy step (Matmul or QuantizedMatmul) is allowed only at chain step 0, gated by `heavy_chain_shape_ok` per op.

The merge function (`merge_into_chain`) handles all four (Single | FusedChain) × (Single | FusedChain) cases via `extract_steps_and_inputs`, which synthesises a 1-step chain for a Single fusion so the rest of the merge sees one uniform shape.

The fixed-point loop keeps merging until nothing can grow. CustomKernel and any op outside the allowed categories acts as a hard barrier; merges stop there.

### Current scope

- ✅ Single↔Single pointwise pairs
- ✅ FusedChain↔Single & Single↔FusedChain (N-step chains)
- ✅ Heavy first-step (Matmul, QuantizedMatmul) → Pointwise tail
- ✅ Batched Matmul (any `a_batch == b_batch`) → Pointwise tail
- ✅ Reduction first-step (RMSNorm) → Pointwise tail. Body uses one
  threadgroup per row with simdgroup_sum reduction; tail external
  operands are shape-classified (`same-as-output`, `hidden-broadcast`,
  `scalar`) and indexed accordingly.
- ✅ View absorption (element-contiguous: Reshape, Flatten, Unflatten,
  Squeeze, ExpandDims, AsContiguous)
- ⏳ Softmax first-step
- ⏳ Broadcast / Transpose absorption (stride math)
- ⏳ Multi-output fan-out

### Empirical numbers

| Model | FusionGraph fusions (Stage 2 only) | After Stage 3 | Chains | Notable absorptions |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B-4bit | 878 | 878 | 0 | Stage 2 pattern matchers already absorb everything reachable |
| Qwen3.6-27B-4bit | 2469 | **1861** | **384** | 144 standalone QM → 0 (all chain heads), 16 batched Matmul (SDPA Q@Kᵀ) absorbed, 48 RMSNorms absorbed into Reduction-head chains, 320+ view+pointwise chains around the SSM body |

---

## Stage 4 — Codegen (`src/codegen/`)

**Input:** `FusionGraph` + `TaskGraph` + `ModelMeta`.
**Output:** `kernel.metallib` + `task_graph.json`.

### Per-FusionKind emitters

`codegen/kernels/mod.rs::emit_for(fusion, …)` dispatches on `FusionKind`. Each pattern fusion has its own MSL emitter (`linear_residual.rs`, `rmsnorm_qkv.rs`, `rmsnorm_gateup_swiglu.rs`, `sdpa_decode.rs`, …). Each emitter knows its hand-tuned thread shape, SMEM layout, vector loads, and quantization branches.

### Universal Single emitter (`codegen/kernels/single.rs`)

The 1:1 fallback for any `FusionKind::Single(op_name)` that escapes Stage 2 and didn't get absorbed by Stage 3. Dispatches by op name:
- **Pointwise** (Add, Mul, Sigmoid, Exp, …): one thread per element, expression-templated.
- **View** (Reshape, Flatten, Squeeze, …): identity passthrough (element-contiguous).
- **View** (Broadcast, Transpose): explicit stride math.
- **Heavy** (Matmul, Softmax, RMSNorm, Arange, Full, Gather, Concatenate): per-op specialised emit.
- **CustomKernel**: source verbatim. `function_name()` returns mlx's mangled name so the codegen dedup loop collapses N callsites with the same source into one emitted function.

The compiler invariant: every fusion emits a metallib function, never silently no-ops. Unknown ops still emit a stub kernel that's flagged for runtime refusal — fails loud rather than producing silent garbage.

### Chain emitter (`codegen/kernels/fused_chain.rs`)

Emits one MSL kernel function per `FusedChain`. Two body shapes by first step's category:

- **Pointwise-first chain**: flat-N grid, one thread per output element. Each step's value lives in a `float rN` register; only the last step writes `out0[i] = T(...)`.

- **Heavy-first chain** (Matmul or QuantizedMatmul at step 0): **simdgroup-cooperative GEMV** — 32 threads share one output element, K-axis reduction is striped across `simd_lane` and collapsed by `simd_sum`. After the reduction, lanes > 0 early-return and lane 0 runs the pointwise tail then writes. This matches what `LinearResidual` does by hand. The general path proves the same shape arises automatically *and* the same per-row work-sharing for the heavy reduction — closing the simdgroup-cooperation gap was the last piece keeping general chains slower than the hand-tuned pattern.

- **Reduction-first chain** (RMSNorm at step 0): one threadgroup per row. Threads iterate the hidden axis; `simd_sum` reduces `sum_sq` and a 2-stage cross-simd reduction lands the row-wide `rms_scale` in `reduce_buf[0]`. The same threads then loop the hidden axis again and apply the chain's pointwise tail per element. Tail external operands are shape-classified (same-shape → `[row * HIDDEN + i]`, hidden-broadcast → `[i]`, scalar → `[0]`) by `operand_expr_reduction`.

QM head uses the int4 affine dequant inner loop from `kernels::emit_int4_gemv_inner` (the helper already used by `linear_residual.rs::emit`), so dequant correctness is shared with the pattern emit.

Buffer signatures are typed from `ModelMeta::activation_dtype` (bf16 vs half) — quantized weights (`uint`) and quant scales/biases (`half`) are typed independently to match mlx's affine storage.

### Activation dtype awareness

`ModelMeta::activation_dtype` is auto-detected from `config.json`:
- `quantization.*` present → fp16 (mlx-lm dispatches `quantized_matmul` in fp16 regardless of `torch_dtype`).
- otherwise → `torch_dtype` (`bf16` / `fp16` / `fp32`).
- fallback bf16.

`codegen::kernels::act_ty(meta)` returns `"bfloat"` / `"half"` / `"float"`. Every emitter uses this for any tensor that flows between fused kernels (activations, norm weights, biases, KV cache, output buffers). Mismatch here is silent garbage (bf16 and fp16 share size but not exponent layout), so it's the one place we can't afford to hardcode.

### Metal compile + bfloat16_t alias

`codegen::msl::prelude` writes the `#include`s plus `typedef bfloat bfloat16_t;` / `typedef half float16_t;` aliases. mlx's CustomKernel sources reference the C++-style type names that Metal-SL doesn't ship by default; without the aliases the verbatim source fails to compile.

`xcrun metal -c -O3 -ffast-math` → `xcrun metallib` builds `kernel.metallib`.

---

## Runtime (`runtime/fusekit_runtime/` and `python_runtime/`)

This is the **currently incomplete piece** in the megakernel automation story.

### How runtime works today

`fusekit patch` extracts a Python package (`fusekit_runtime`) into `<model>/fusekit/runtime/` and installs a global hook (`~/.local/lib/python3.11/site-packages/fusekit.pth`) that monkey-patches `mlx_lm.load`. After load, fusekit's `attach()`:

1. Loads `kernel.metallib` + `task_graph.json` into a Rust dispatcher (`python_runtime/` pyo3 cdylib).
2. (Optionally) installs `fast_generate_step` to skip mlx-lm's per-token Python overhead.
3. **For Qwen2.5-style models only**, the demo `scripts/replace_layer_full.py` script hand-patches each layer's forward to dispatch fusekit kernels in an ICB.

### What's hand-written today vs what should be automatic

The hand-written part in `scripts/replace_layer_full.py` knows about Qwen2.5's layer structure (which attribute is `attn.q_proj`, which scratch buffer is `q_buf`, which uniform is `offset`, …). This is the only thing that ties fusekit to a particular model family.

To support arbitrary models — Qwen3.6 included — Stage 4 should emit, alongside `task_graph.json`, enough metadata for the runtime to build the ICB **without** any hand-written Python:

- per-task: kernel name, grid, threadgroup, list of buffer bindings
- per buffer binding: where the tensor lives at runtime (`model.layers[3].self_attn.q_proj.scales`, or a scratch buffer of given shape/dtype, or a uniform written each token)
- task dependency edges

The runtime hook then just walks task_graph.json, resolves tensors from the loaded model, allocates scratch and uniforms, builds the ICB once, and dispatches per token.

This is the "D" step in the project's near-term roadmap. None of Stage 1-3 needs to change to make D work — the IR already has enough information, it just doesn't get serialised into a runtime-consumable form yet.

---

## Where the layers sit, file map

```
src/
  loader.rs                    Stage 1: pyo3 capture + graph lowering
  graph.rs                     Graph IR (typed Attrs)

  fusion/
    ir.rs                      FusionKind, FusionAttrs, ChainStep, ModelMeta
    patterns/                  Stage 2: hand-tuned matchers
      rmsnorm_qkv.rs
      rmsnorm_gateup_swiglu.rs
      linear_residual.rs
      …
    safety.rs                  SMEM budget check per FusionKind
    build.rs                   Stage 2 driver
    category.rs                NEW — OpCategory + can_chain_into
    general.rs                 NEW — Stage 3 merge pass

  task/                        Stage 2.5: fusion → tasks (partitioning)
    build.rs

  codegen/
    msl.rs                     MslWriter + prelude (bfloat16_t alias)
    persistent.rs              megakernel dispatcher emit
    metal_compile.rs           xcrun metal + metallib
    kernels/
      mod.rs                   emit_for dispatcher + act_ty + int4 GEMV helper
      rmsnorm_qkv.rs           Stage 2 pattern emitters
      …
      single.rs                NEW — universal 1:1 op emitter (Pointwise, View,
                               Heavy, CustomKernel)
      fused_chain.rs           NEW — Stage 3 chain emitter (Pointwise-first,
                               Heavy-first)

python_runtime/              pyo3 cdylib — Rust ICB dispatcher (loaded by Python)
runtime/                     extracted into <model>/fusekit/runtime/ at patch time

scripts/replace_layer_full.py  hand-written demo dispatcher (Qwen2.5 only) —
                               will be removed once D lands.
```

---

## Roadmap

Near-term (in order of "what makes Qwen3.6 get real speedup"):

1. **D — Auto-install** (1–2 weeks)
   Stage 4 emits task_graph.json with full binding metadata. Runtime hook becomes model-agnostic. Without this, Stage 3 fusions exist in the metallib but never get dispatched on novel models. Removes the hand-written `scripts/replace_layer_full.py`.

2. **C — Reduction tail** (3–5 hours)
   Allow Softmax / RMSNorm as Heavy-equivalent chain head. SMEM scratch + simdgroup_sum already used by Stage 2 emitters; lift those into the chain emitter.

3. **B — Batched Matmul** (3–4 hours, plus C for the full SDPA win)
   Relax `matmul_chain_shape_ok` and emit `(batch, M, N)` grid. With C, the general path discovers SDPA's `Q@Kᵀ → Multiply → Softmax → @V` chain and SdpaDecode becomes deprecable.

4. **E (2 — Broadcast/Transpose absorption)** (2–3 hours)
   Stride math inside chains; extends absorption past the easy element-contiguous views.

Pattern matcher deprecation (when general can produce equal-or-better fusions automatically): the remaining gaps are

- **automatic SMEM caching** for fan-out (one producer feeding several consumers — what `RmsNormQkv` does with its norm output across Q/K/V),
- **vector loads** (`bfloat4` / `half4` × unroll based on shape divisibility),
- **cost model** for size-sensitive trade-offs (SMEM budget, register pressure, dispatch count vs cache miss).

Simdgroup-cooperative emit and parallel-axis selection are now in place for the Heavy/Reduction chain heads; closing fan-out + vector loads is what gets the chain emit on par with the hand-tuned `RmsNormQkv` / `RmsNormGateUpSwiglu` patterns. At that point `LinearResidual`, then `SdpaDecode`, then the rmsnorm-fused-MLP patterns become removable.
