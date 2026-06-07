# Fusion algorithms — reference

> **Status:** This file documents the Stage 2 pattern-matching algorithm in
> isolation, written before the Stage 3 general op-semantic fusion pass
> landed. For the full pipeline (Stage 1 capture → Stage 2 patterns →
> Stage 3 general fusion → Stage 4 codegen → runtime), see
> [`architecture.md`](architecture.md). The algorithm-family overview
> below remains useful background for the design choices behind that
> pipeline.

Four families of fusion algorithms, what they actually do, and where each
fits. Read this before considering "should we switch to algorithm X?" —
the answer is almost always "not yet."

The practical conclusion is at the bottom; the rest is context.

---

## 1. Pattern matching (declarative)

**How it works.** Hand-define a set of fusion templates (e.g. `rmsnorm_matmul`,
`sdpa`, `swiglu_mlp`). Walk the IR; at each op, try each template by
priority; if one matches, mark the matched ops as a fusion. Unmatched ops
become single-op fusions.

```
for op in topo_order(graph):
    if op already in a fusion: continue
    for pattern in patterns_by_priority:
        if pattern.try_match(op):
            mark matched ops as one fusion
            break
    else:
        single-op fusion(op)
```

**Information used:** op name, attrs, single-consumer (use-def) chains,
sometimes boundary shapes.

**Strengths**
- Transformers are highly structured — five or six patterns cover ~all
  of the meaningful fusions.
- Each pattern is a function you can read. Failure is debuggable.
- Maps 1:1 onto known hand-tuned kernels (FlashAttention, fused MLP, ...).

**Weaknesses**
- Novel architectures need new patterns. Slight variations
  (`RMSNorm → Bias → Matmul` vs `RMSNorm → Matmul`) bypass the matcher.
- Doesn't discover unfamiliar good fusions.
- Pattern overlap requires conflict resolution (longest-match, priority).

**Used by.** XLA HLO (fusion rules), TensorRT (hundreds of layer fusion
rules), MPK (humans write patterns directly as layer methods),
TorchInductor (patterns + bottom-up).

---

## 2. Bottom-up greedy

**How it works.** No prior pattern knowledge. Walk ops in topological
order; for each op, attempt to extend the current fusion or start a new
one based on local feasibility.

```
current = None
for op in topo_order(graph):
    if current is None:
        current = {op}
    elif can_extend(current, op):
        current.add(op)
    else:
        emit(current); current = {op}
emit(current)

fn can_extend(fusion, op):
    if fusion.outputs[last] has more than one consumer: return false
    if op uses tensor outside fusion: return false   # would need reload
    if op.threadgroup_shape != fusion.threadgroup_shape: return false
    if estimate_sram(fusion + op) > BUDGET: return false
    return true
```

**Information used:** use-def chains, SRAM budget, threadgroup shape
compatibility, access patterns (row vs column-major).

**Strengths**
- Catches elementwise chains (`add → multiply → sigmoid → multiply`)
  without anyone writing patterns.
- Generalizes to unseen graphs.
- Strong baseline when combined with patterns.

**Weaknesses**
- Greedy = locally optimal. `[A,B]` then `[C,D,E]` might lose to
  `[A,B,C]` + `[D,E]`.
- Can't see large patterns as one unit. Hits a reduction (softmax) and
  stops; never builds an SDPA-sized fusion from scratch.
- Threadgroup-shape constraints are awkward to encode.

**Used by.** TVM LoopFusion pass, XLA elementwise fusion, TorchInductor
(complements patterns), Halide (manual but similar spirit).

---

## 3. Dynamic programming / graph partitioning

**How it works.** Treat fusion as an optimization problem: partition the
graph into fusions that minimize total cost. For chains, simple DP. For
DAGs, heuristic / ILP / branch-and-bound.

```
# Linear chain version
dp[0] = 0
for i in 1..=n:
    dp[i] = ∞
    for j in 1..=i:
        if can_fuse(ops[j-1..i]) and fits_in_sram(ops[j-1..i]):
            c = dp[j-1] + cost_of_fusion(ops[j-1..i])
            if c < dp[i]: dp[i] = c; choice[i] = j-1
# trace choice[] back to recover fusions
```

The interesting (and hard) part is `cost_of_fusion`:

```
fn cost_of_fusion(ops):
    bytes_global = sum(intermediates that must materialize)
    bytes_sram   = sum(intermediates that stay on-chip)
    flops        = sum(op flops)

    time_mem   = bytes_global / BANDWIDTH
    time_compute = flops / PEAK_FLOPS / occupancy_estimate(ops)
    time_overhead = KERNEL_LAUNCH_OVERHEAD

    return max(time_mem, time_compute) + time_overhead
```

**Information used:** all static info, plus a hardware-specific cost
model (bandwidth, peak FLOPs, occupancy estimates).

**Strengths**
- Optimal under the cost model.
- Adapts to new hardware via cost model retuning.
- Explores the search space humans wouldn't enumerate.

**Weaknesses**
- Cost models are very hard to make accurate (±25% is good).
- An optimal-under-bad-model decision is just a wrong decision.
- DAG partitioning is approximately NP-hard; approximations needed for
  1500-op graphs.
- Debugging means reading the cost model — opaque.

**Used by.** Welder (MSR), Rammer, parts of XLA, AKG (Huawei).
Production rare — most teams conclude patterns + heuristics are enough.

---

## 4. E-graph rewriting (equality saturation)

**How it works.** Maintain an e-graph containing all programs equivalent
to the input. Apply rewrite rules to grow the e-graph until saturation
or time limit; then extract the lowest-cost program.

```
egraph = EGraph(input_program)
rewrites = [
    "norm(x, w) ; matmul(_, m) => norm_matmul(x, w, m)",
    "sigmoid(a) ; multiply(_, b) => silu_mul(a, b)",
    "matmul(reshape(x), w) => matmul(x, reshape(w))",  # associativity
    ...
]
while not saturated and time < budget:
    for r in rewrites:
        find matching e-classes; add r.rhs into them
        deduplicate
best = egraph.extract(cost_fn)   # NP-hard; ILP or heuristic
```

The selling point: e-graphs hold multiple equivalent representations
simultaneously, so a choice that looks bad locally doesn't preclude
finding a better global rewrite later.

**Information used:** rewrite rules (provably-equivalent transforms),
cost function.

**Strengths**
- Avoids irreversible greedy commits.
- Every rewrite is provably equivalent → strong correctness story.
- Can rewrite *and then* fuse (e.g., reshape commute + fusion).

**Weaknesses**
- E-graph size blows up super-linearly. 1500-op graphs would OOM
  realistically.
- Extraction is NP-hard.
- Almost no production deployments — mostly academic.
- Overkill for highly-structured workloads like transformers.

**Used by.** Tensat (research paper on TF graphs), `egg` crate (Rust
research tool), small DSL compilers. No major ML compiler uses it.

---

## Summary

| Algorithm | Inputs | Strengths | Weaknesses | Transformer fit |
|---|---|---|---|---|
| Pattern matching | op kind + use-def | Debuggable, maps to hand kernels | Needs per-architecture patterns | ⭐⭐⭐⭐⭐ |
| Bottom-up greedy | + static estimates | Pattern-free, finds chains | Local optima, misses large patterns | ⭐⭐⭐ (complement) |
| DP / partitioning | + cost model | Theoretically optimal | Cost model fragile, slow, opaque | ⭐⭐ |
| E-graph | + rewrite rules | Avoids irreversible choices | Memory/time blow up, no production | ⭐ |

---

## What fusekit will actually do

**Phase 1.** Pattern matching only. Define 4–5 transformer patterns
(`rmsnorm_matmul`, `qkv_rope`, `sdpa`, `out_proj_residual`, `swiglu_mlp`).
Validate with logit comparison against MLX. Anything unmatched gets a
single-op fusion. This mirrors MPK's task-definition style exactly,
just with the matching automated.

**Phase 2.** Add bottom-up greedy as a complement: after pattern matching,
greedy-fuse unmatched elementwise chains and small reductions. Adds value
without touching the pattern path.

**Phase 3 (maybe).** Introduce a simple cost model — initially just SRAM
bytes + estimated bandwidth — to make a few specific decisions (e.g.,
whether to split a large fusion across partition axes). Don't replace the
patterns; use cost only at the points where heuristics need a tiebreaker.

**Never (until proven needed).** Full DP/ILP partitioning. E-graph
rewriting. Both are research-grade for this domain.

## Lock-in implications

The pipeline stages (`Graph → FusionGraph → TaskGraph → MSL → metallib`)
are agnostic to which algorithm produces the `FusionGraph`. The fusion
builder is a single function for now:

```rust
pub fn build_fusions(graph: &Graph) -> Result<(FusionGraph, DecisionLog)>;
```

When Phase 2 introduces a second algorithm and they need to coexist (e.g.
patterns first, greedy second), extract a trait. Don't pre-design the
trait now — there's nothing to abstract over yet.
