# FuseKit Project Overview

## 1. FuseKit이 하는 일

FuseKit은 `mlx-lm` / `oMLX` 모델의 decode 경로를 Apple Silicon용 Metal 커널로 다시 컴파일하는 프로젝트다.

핵심 목표는 다음과 같다.

- MLX가 실행하는 모델 forward graph를 캡처한다.
- graph를 FuseKit IR로 낮춘다.
- decode에 중요한 연산들을 fusion한다.
- Metal 커널과 task graph를 생성한다.
- oMLX 런타임에 패치해서 실제 생성 경로에서 FuseKit 커널을 탄다.

한 줄로 요약하면:

```text
mlx-lm/oMLX 모델 graph -> FuseKit IR -> fused Metal kernels -> oMLX runtime patch
```

---

## 2. 전체 구조

```text
model directory
  |
  |  Stage 1: capture
  v
RawGraph
  |
  |  lower
  v
Graph IR
  |
  |  Stage 2: pattern fusion
  v
FusionGraph
  |
  |  Stage 3: general semantic fusion + IR passes
  v
Optimized FusionGraph
  |
  |  Stage 4: codegen
  v
kernel.metal + kernel.metallib + task_graph.json
  |
  |  runtime attach
  v
oMLX generation path
```

Patch time과 runtime이 분리되어 있다.

- Patch time: graph capture, fusion, Metal compile.
- Runtime: 이미 컴파일된 `kernel.metallib`와 `task_graph.json`을 읽고 decode step마다 dispatch한다.

---

## 3. 왜 IR이 필요한가

MLX trace를 그대로 커널로 바꾸면 구조 정보가 부족하다.

FuseKit IR은 다음 정보를 명시적으로 보존한다.

- tensor shape / dtype
- op name과 typed attrs
- tensor producer-consumer 관계
- model parameter source path
- cache role
- layer index
- quantization metadata
- MTP / cache / stateful op semantics

즉 IR은 단순 op list가 아니라, 최적화와 runtime binding이 같은 의미를 공유하게 만드는 계약이다.

```text
MLX primitive stream
  -> typed tensor graph
  -> semantic fusion graph
  -> dispatchable task graph
```

---

## 4. Stage 1: Graph Capture

FuseKit은 Python subprocess 없이 Rust 안에서 oMLX bundled CPython을 embed한다.

Capture 흐름:

```text
Rust + PyO3
  -> import mlx_lm
  -> load model
  -> mx.export_function(callback, model, dummy_input)
  -> callback records MLX primitives
```

캡처 결과는 `RawGraph`다.

이후 `RawGraph`를 `Graph IR`로 lower한다.

Graph IR은 다음을 가진다.

- `Tensor`: shape, dtype, optional producer
- `Op`: op name, inputs, outputs, attrs
- `Attrs`: RoPE, RMSNorm, Softmax, QuantizedMatmul, CustomKernel 등 typed attributes

---

## 5. Stage 2: FusionGraph

`Graph IR`의 개별 op들을 더 큰 의미 단위로 묶는다.

FuseKit에서 fusion 하나는 Metal kernel 하나의 후보가 된다.

```text
Graph IR ops
  Add, RMSNorm, QuantizedMatmul, RoPE, SDPA, ...

FusionGraph
  FusionKind::RmsNormQkv
  FusionKind::SdpaDecode
  FusionKind::LinearResidual
  FusionKind::FusedChain
  FusionKind::QmvFanout
  FusionKind::LayerBlock
```

Fusion은 다음을 가진다.

- `kind`: 어떤 fused kernel인지
- `inputs`: fusion 바깥에서 들어오는 tensor boundary
- `outputs`: fusion 밖으로 나가는 tensor boundary
- `attrs`: codegen에 필요한 구조화된 parameter
- `layer_hint`: 어떤 layer에 속하는지

---

## 6. FusionKind의 의미

대표적인 `FusionKind`:

| FusionKind | 의미 | 최적화 포인트 |
|---|---|---|
| `RmsNormQkv` | RMSNorm 후 Q/K/V projection | norm 결과를 한 번만 계산하고 3개 projection이 공유 |
| `SdpaDecode` | decode attention | QK, softmax, V path를 decode shape에 맞춤 |
| `LinearResidual` | projection + residual add | accumulator와 residual을 register에서 합침 |
| `RmsNormGateUpSwiglu` | MLP gate/up + SwiGLU | gate/up intermediate DRAM write 제거 |
| `FinalRmsNormLmHead` | final norm + lm head | hidden round-trip 제거 |
| `ArraysCacheShiftAppendDecode` | recurrent state cache update | GDN/SSM cache semantics를 runtime에 노출 |
| `QmvFanout` | 같은 input을 읽는 여러 quantized projection | activation 재로드와 dispatch 수 감소 |
| `FusedChain` | 일반 op-semantic chain | pattern 밖의 pointwise/view/heavy tail fusion |
| `LayerBlock` | layer 전체 megakernel 후보 | layer 내부 여러 round-trip 제거 |

---

## 7. Pattern Fusion과 General Fusion

FuseKit은 두 종류의 fusion을 같이 쓴다.

### Pattern fusion

Transformer에서 자주 나오는 구조를 명확히 인식한다.

예:

```text
RMSNorm -> Q/K/V projections
Matmul -> Add residual
RMSNorm -> gate/up -> SwiGLU
Q/K/V -> RoPE -> KV cache store -> SDPA
```

장점:

- hand-tuned kernel로 바로 연결된다.
- correctness와 디버깅이 쉽다.
- transformer 공통 경로에 강하다.

### General semantic fusion

모델 이름이나 특정 layer pattern이 아니라 op category와 graph 구조로 fusion한다.

예:

```text
Heavy op -> pointwise tail
Reduction -> pointwise tail
Pointwise -> pointwise
View -> pointwise
QuantizedMatmul siblings -> QmvFanout
```

사용하는 기준:

- producer-consumer 관계
- single-consumer 여부
- shape compatibility
- register budget
- shared memory budget
- spill / occupancy risk
- cache/state barrier 여부

즉 "특정 모델용 패턴"이 아니라, IR의 의미와 budget으로 fusion 여부를 결정한다.

---

## 8. TaskGraph와 Runtime Contract

Codegen 결과는 Metal source만이 아니다.

FuseKit은 `task_graph.json`을 같이 만든다.

`task_graph.json`은 runtime이 커널을 실행하기 위한 계약이다.

담는 정보:

- kernel function name
- grid / threadgroup shape
- input/output buffer binding
- model parameter path
- scratch buffer id
- layer index
- cache binding role
- scalar uniforms
- decode pipeline order
- active M value

Runtime은 이 파일만 보고 live oMLX model에서 buffer를 찾아 Metal dispatch를 구성한다.

```text
task_graph.json
  -> runtime resolves model.layers.N.xxx.weight
  -> binds KV cache / ArraysCache / scratch / uniforms
  -> builds Metal ICB
  -> executes per decode step
```

---

## 9. Runtime Patch 구조

FuseKit runtime은 모델 디렉터리에 설치된다.

```text
model/
  fusekit/
    kernel.metallib
    kernel.metal
    task_graph.json
    runtime/fusekit_runtime/
```

oMLX가 모델을 로드하면 FuseKit hook이 붙는다.

```text
mlx_lm.load / oMLX load
  -> FusekitWrapper(original_model)
  -> decode call intercept
  -> dispatcher.try_run(...)
  -> Metal ICB execution
```

지원하지 못하는 shape/cache 상태에서는 원래 model call로 돌아갈 수 있다.

하지만 목표는 decode hot path에서 가능한 많은 op를 FuseKit 커널로 처리하는 것이다.

---

## 10. Decode Hot Path 최적화

Decode는 token 하나씩 반복되므로 launch overhead, cache traffic, host sync가 중요하다.

FuseKit 최적화 축:

- per-token Python/host dispatch 감소
- kernel launch 수 감소
- intermediate tensor DRAM round-trip 감소
- activation 재로드 감소
- quantized matmul small-M 최적화
- KV cache / ArraysCache update를 semantic primitive로 처리
- argmax 결과를 GPU-side token buffer로 연결
- Metal ICB로 command encoding overhead 감소

Decode step의 목표 형태:

```text
token id
  -> embedding
  -> layer 0 fused pipeline
  -> layer 1 fused pipeline
  -> ...
  -> final norm + lm head
  -> argmax
  -> next token id
```

---

## 11. Quantized Matmul 최적화

MLX quantized weight layout:

```text
packed weight: uint32, shape = [out, in / 8]
scales: fp16, shape = [out, in / group_size]
biases: fp16, shape = [out, in / group_size]
```

FuseKit은 decode/MTP에서 작은 M에 특화된 qGEMV/qGEMM 경로를 만든다.

주요 아이디어:

- one row or small M rows 기준으로 tile 선택
- simdgroup 단위 K reduction
- int4/int8 dequant를 matmul inner loop 안에서 수행
- 같은 input activation을 읽는 여러 projection을 fanout으로 합침
- gate/up projection은 가능하면 SwiGLU까지 붙여 intermediate write 제거

예:

```text
norm(x)
  -> q_proj
  -> k_proj
  -> v_proj

becomes

QmvFanout(norm(x), [q_proj, k_proj, v_proj])
```

---

## 12. Cache/State 최적화

Decode는 KV cache와 recurrent state cache를 계속 갱신한다.

FuseKit은 cache를 단순 tensor가 아니라 semantic state로 본다.

대표 cache:

- KV cache: attention key/value append
- Quantized KV cache: packed/scales/biases 형태
- ArraysCache: Qwen3Next/GatedDeltaNet recurrent state
- MTP cache: draft head용 별도 cache

IR/runtime에서 cache role을 명시하면 다음이 가능하다.

- 현재 token K/V store를 커널 안에서 수행
- cache offset uniform을 decode step에 맞춰 갱신
- ArraysCache shift+append를 fused primitive로 처리
- MTP reject 시 rollback 가능한 state snapshot 제공
- speculative token과 visible cache length를 분리

---

## 13. MTP 최적화 방향

MTP는 target model이 다음 token을 검증하면서 draft token을 미리 사용하는 speculative decoding이다.

기본 cycle:

```text
state.next_main + state.draft
  -> 2-token verify backbone
  -> if draft accepted:
       emit draft + bonus
     else:
       emit verified token only
  -> MTP head proposes next draft
```

FuseKit에서 MTP를 최적화하려면 다음이 필요하다.

- M=1 single decode kernel
- M=2 or M=N verify kernel
- small-M qGEMV/qGEMM
- MTP head native path
- cache rollback semantics
- acceptance rate 보존
- Graph/ICB variant cache

중요한 점:

속도만 빨라져도 acceptance path가 raw와 달라지면 end-to-end 생성이 달라진다. 따라서 MTP 최적화는 kernel time과 함께 accept/reject sequence correctness를 같이 봐야 한다.

---

## 14. Generality 원칙

FuseKit의 최적화는 모델 이름에 직접 의존하지 않는 방향을 목표로 한다.

일반화 기준:

- op semantic
- tensor shape
- dtype
- quantization metadata
- cache role
- layer role
- active M
- register/shared-memory budget
- dispatch/ICB cost

예:

```text
QmvFanout
  model name: 사용하지 않음
  criterion: same input activation + compatible quant layout + budget fit

ArraysCacheShiftAppendDecode
  model name: 사용하지 않음
  criterion: recurrent cache slot + state length + append width

FusedChain
  model name: 사용하지 않음
  criterion: producer-consumer chain + op category + shape compatibility
```

---

## 15. 핵심 최적화 목록

현재 FuseKit이 노리는 최적화는 크게 다섯 축이다.

| 축 | 설명 | 효과 |
|---|---|---|
| Kernel fusion | 여러 op를 하나의 Metal function으로 합침 | launch 감소, DRAM intermediate 감소 |
| Quantized small-M kernels | decode/MTP의 M=1, M=2, M=N에 맞춘 qGEMV/qGEMM | quantized model decode throughput 개선 |
| Fanout fusion | 같은 activation을 읽는 여러 projection을 한 dispatch로 처리 | activation reload 감소 |
| Cache semantic primitive | KV/ArraysCache update를 runtime state로 직접 binding | cache round-trip/rollback 비용 감소 |
| ICB/runtime dispatch | per-token host encoding과 Python overhead 축소 | wall-clock decode latency 감소 |

---

## 16. 발표용 한 문장 요약

FuseKit은 MLX/oMLX 모델의 decode graph를 캡처해 semantic IR로 낮추고, pattern fusion과 budget 기반 general fusion을 통해 Apple Silicon용 Metal kernel set으로 재컴파일한 뒤, oMLX runtime에 패치해서 per-token generation hot path를 더 적은 dispatch와 더 적은 memory round-trip으로 실행하는 compiler/runtime 프로젝트다.

