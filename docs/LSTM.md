<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  File: docs/LSTM.md
  Package: github.com/lvlath/go/lstm
  Purpose: Executable inference guide, PyTorch IFGO weight ABI, numeric policy,
           streaming equivalence, error precedence, and deployment boundaries.
  Contract status: implemented single-layer unidirectional float64 inference;
                   source code and lstm/doc.go remain authoritative.
  Scope: explicitly imported finite weights, batch-first tensors, masked inference,
         detached state transfer, synthetic numerical oracles, and operational use.
  License: AGPL-3.0-only. See ../LICENSE.
-->

# LSTM - deterministic, single-layer recurrent inference

`github.com/lvlath/go/lstm` evaluates a **previously trained** LSTM layer using
finite `float64` weights and the PyTorch **input / forget / cell candidate /
output** (`i, f, g, o`, or **IFGO**) gate layout. It computes hidden features
for complete, variable-length, and streaming sequences. Its task is **forward
inference**: it does not train the weights, choose a forecast target, calibrate a
classifier, normalize raw physical units, or decide when an operator should act.

Use it when a trained, externally validated model needs the same frozen
parameters applied reproducibly to new time-ordered observations. An LSTM is
*not* automatically a better choice than ARIMA, HMM, ACF, PCA, or a simple
linear baseline; selecting and validating a learned model is an application
responsibility.

## Contents

1. [Purpose, scope, and comparison](#1-purpose-scope-and-comparison)
2. [Public API and result shapes](#2-public-api-and-result-shapes)
3. [Weight import and PyTorch IFGO ABI](#3-weight-import-and-pytorch-ifgo-abi)
4. [Recurrence and numeric safeguards](#4-recurrence-and-numeric-safeguards)
5. [Batch-first tensor and masking law](#5-batch-first-tensor-and-masking-law)
6. [Streaming chunks and exact replay](#6-streaming-chunks-and-exact-replay)
7. [Canonical ModelID and ownership](#7-canonical-modelid-and-ownership)
8. [Error precedence and cancellation](#8-error-precedence-and-cancellation)
9. [Executable Go scenario: fleet telemetry](#9-executable-go-scenario-fleet-telemetry)
10. [Independent oracles and PyTorch parity](#10-independent-oracles-and-pytorch-parity)
11. [Complexity and performance](#11-complexity-and-performance)
12. [Common production pitfalls](#12-common-production-pitfalls)
13. [Verification and related documentation](#13-verification-and-related-documentation)

---

## 1. Purpose, scope, and comparison

| Need                                                       | Correct boundary                                                | What is *not* provided                                               |
|:-----------------------------------------------------------|:----------------------------------------------------------------|:---------------------------------------------------------------------|
| Apply a frozen learned temporal representation             | `lstm.Import` → `Model.Infer`                                   | No weight fitting, gradient, optimizer, or neural output head        |
| Compare unequal session durations in one rectangular batch | `Model.InferSequence` with `lengths`                            | No automatic padding inference from zeros                            |
| Continue an online session without recomputing the prefix  | `Model.InferFinal` → `Model.InferChunk`                         | No mutable shared stream object or implicit session cursor           |
| Derive timestamp-aligned causal features                   | `series` for validity/windows; application constructs `Tensor3` | No timestamp processing inside `lstm`                                |
| Analyze probabilistic discrete regimes                     | `hmm` with causal filtering / retrospective smoothing           | No probabilistic regime probabilities from raw LSTM states           |
| Forecast an interpretable fixed-order conditional mean     | `arima`                                                         | No ARIMA coefficients or intervals from LSTM hidden states           |
| Forecast asymmetric conditional variance                   | `gjrgarch`                                                      | No variance, tail risk, or VaR just because hidden activations exist |
| Compress without learning a recurrent model                | `paa` / `pca`                                                   | No interchangeability of PCA coordinates and LSTM states             |

Supported: **one layer, one forward direction, no projection, `float64`,
explicitly imported weights**. Unsupported: bidirectional/multilayer LSTM,
dropout, training, autodiff, GPUs, dynamic quantization, pickle or arbitrary
model-file deserialization. The package does not read Python files; a caller
must transform a trusted export into numeric Go data before `Import`.

## 2. Public API and result shapes

The complete public inference surface is:

```go
func Import(weights lstm.Weights) (*lstm.Model, error)
func (m *lstm.Model) ID() string
func (m *lstm.Model) Infer(ctx context.Context, input lstm.Tensor3, lengths []int, initial *lstm.State, options lstm.Options) (*lstm.Result, error)
func (m *lstm.Model) InferDefault(ctx context.Context, input lstm.Tensor3) (*lstm.Result, error)
func (m *lstm.Model) InferFinal(ctx context.Context, input lstm.Tensor3, lengths []int, initial *lstm.State) (*lstm.Result, error)
func (m *lstm.Model) InferSequence(ctx context.Context, input lstm.Tensor3, lengths []int, initial *lstm.State) (*lstm.Result, error)
func (m *lstm.Model) InferChunk(ctx context.Context, chunk lstm.Tensor3, lengths []int, previous *lstm.Result, options lstm.Options) (*lstm.Result, error)
func (r *lstm.Result) IsNil() bool
```

The qualified declarations above are an API reference, **not** a compilable Go
source file. The executable program in [section 9](#9-executable-go-scenario-fleet-telemetry)
uses real package imports and checks errors. All convenience methods delegate
to the same canonical `Infer` kernel; no wrapper selects a different recurrence.

| Call                                                         | Active timesteps           | Initial state    | `Sequence`                        | `Final`                        |
|:-------------------------------------------------------------|:---------------------------|:-----------------|:----------------------------------|:-------------------------------|
| `Infer(ctx,x,lengths,initial, Options{ReturnSequence:true})` | Explicit local mask        | Explicit or zero | `[B,T,H]` with exact-zero padding | Always `[B,H]` hidden and cell |
| `InferDefault(ctx,x)`                                        | All `T`                    | Zero             | Zero-value `Tensor3`, no buffer   | Always                         |
| `InferFinal(ctx,x,lengths,initial)`                          | Explicit local mask        | Explicit or zero | Zero-value `Tensor3`, no buffer   | Always                         |
| `InferSequence(ctx,x,lengths,initial)`                       | Explicit local mask        | Explicit or zero | `[B,T,H]`                         | Always                         |
| `InferChunk(ctx,chunk,lengths,previous,opts)`                | Mask relative to **chunk** | `previous.Final` | Controlled by `opts`              | Always                         |

A nil `*Model` receiver's `ID()` returns the empty string; its `Infer`
returns `ErrNilInput` after context validation. Do not construct a usable
model with `&lstm.Model{}`; `Import` is the only supported builder.

`InferChunk` requires an existing **non-nil** `previous` result; it is not an
alias for a first chunk with zero state. Start with `InferFinal` or
`InferSequence`. `InferDefault` is a convenience for complete unmasked input;
it does not automatically infer valid lengths from sensor values.

The public carriers are:

```go
// Batch-first row-major flat tensor: (b*Time+t)*Features+feature.
type Tensor3 struct {
    Data     []float64
    Batch    int
    Time     int
    Features int
}

type Weights struct {
    Input     matrix.Matrix // [4H,D]
    Recurrent matrix.Matrix // [4H,H]
    Bias      []float64     // [4H], already combined
    Hidden    int           // H >= 1
}

type State struct {
    Hidden  matrix.Matrix // [B,H]
    Cell    matrix.Matrix // [B,H]
    ModelID string
}

type Options struct { ReturnSequence bool }
type Result struct { Sequence Tensor3; Final State }
```

These are structural excerpts; import the types rather than redeclaring them.
`Result.IsNil()` returns `true` **only** for a nil `*Result` receiver, including
a typed nil behind `core.Nilable`. An allocated but zero-valued `Result` is
not a valid inference simply because `IsNil()==false`; always check `error`.

## 3. Weight import and PyTorch IFGO ABI

Let `D` be the feature width and `H` the hidden/cell width. `Import` requires:

| Input               | Shape            | Physical layout                                |
|:--------------------|:-----------------|:-----------------------------------------------|
| `Weights.Input`     | `[4H,D]`         | Row-major input-to-gate coefficients           |
| `Weights.Recurrent` | `[4H,H]`         | Row-major previous-hidden-to-gate coefficients |
| `Weights.Bias`      | `[4H]`           | Combined gate biases, not two separate arrays  |
| `Weights.Hidden`    | integer `H >= 1` | Shared gate-block and recurrent state width    |

Gate rows are **contiguous H-row blocks**, never interleaved:

```text
             Input[4H,D]          Recurrent[4H,H]       Bias[4H]
           +---------------+     +---------------+     +--------+
  [0,H)    | i : sigmoid   |     | i : sigmoid   |     |   i    |
  [H,2H)   | f : sigmoid   |     | f : sigmoid   |     |   f    |
  [2H,3H)  | g : tanh      |     | g : tanh      |     |   g    |
  [3H,4H)  | o : sigmoid   |     | o : sigmoid   |     |   o    |
           +---------------+     +---------------+     +--------+
```

For a supported PyTorch `torch.nn.LSTM` (`num_layers=1`,
`bidirectional=False`, `proj_size=0`, `dtype=torch.float64`, no projection),
the *external trusted exporter* maps:

```text
Weights.Input     <- weight_ih_l0
Weights.Recurrent <- weight_hh_l0
Weights.Bias[k]   <- bias_ih_l0[k] + bias_hh_l0[k]
Weights.Hidden    <- hidden_size
```

That bias addition is performed **before** `Import`, with the selected
`float64` rounding convention and verified provenance. `lstm` has no
`ImportPyTorch`, auto-converter, serializer, or training-checkpoint reader;
unsupported architectures must be rejected by the producer, not silently
flattened into one forward layer. For an export that omits biases, an adapter
must explicitly construct the correct zero combined-bias vector: a nil Bias
is invalid.

See the [official PyTorch LSTM parameter documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
for the authoritative gate order and separate input/recurrent biases. Other
framework gate layouts must be deliberately transposed/reordered *outside*
this package and checked against independent outputs.

## 4. Recurrence and numeric safeguards

For each valid sample `(b,t)`, with preceding state `(h_prev,c_prev)`:

$$
\begin{aligned}
a &= W_{ih}x_{b,t}+W_{hh}h_{\mathrm{prev}}+b, \\
i &= \sigma(a_{0:H}),\quad f=\sigma(a_{H:2H}), \\
g &= \tanh(a_{2H:3H}),\quad o=\sigma(a_{3H:4H}),\\
c_{b,t} &= f\odot c_{\mathrm{prev}}+i\odot g,\\
h_{b,t} &= o\odot\tanh(c_{b,t}).
\end{aligned}
$$

![IFGO forward recurrence](https://latex.codecogs.com/svg.image?a%3DW_%7Bih%7Dx_t%2BW_%7Bhh%7Dh_%7Bt-1%7D%2Bb%2C%5Cquad%20c_t%3Df_t%5Codot%20c_%7Bt-1%7D%2Bi_t%5Codot%20g_t%2C%5Cquad%20h_t%3Do_t%5Codot%5Ctanh%28c_t%29)

The affine reduction reads the stored bias first, then ascending input
columns, then ascending recurrent columns. All `4H` preactivations are computed
from the **same old hidden state**, before the hidden cells are overwritten.
The outer traversal is batch → valid timestep → gate row → coefficient column.

The logistic gate uses two exponent-safe branches:

$$
\sigma(x)=\begin{cases}
\dfrac{1}{1+\exp(-x)}, & x\ge0,\\[4pt]
\dfrac{\exp(x)}{1+\exp(x)}, & x<0.
\end{cases}
$$

### Canonical batch and mask pseudocode

```text
validate context, dimensions, exact Data length, lengths, initial state
validate every source number, INCLUDING masked input cells
copy initial Hidden and Cell into private [B,H] working buffers
allocate optional zero-filled Sequence[B,T,H]

for b in 0..B-1:
    active = T if lengths is nil else lengths[b]
    for t in 0..active-1:
        x = input[b,t,:]
        for r in 0..4*H-1:
            a[r] = checked_affine(Input[r,:], x, Recurrent[r,:], old_hidden, Bias[r])
        for j in 0..H-1:
            i, f, g, o = sigmoid(a[j]), sigmoid(a[H+j]), tanh(a[2H+j]), sigmoid(a[3H+j])
            next_cell[j] = checked_cell(f*old_cell[j] + i*g)
            next_hidden[j] = o*tanh(next_cell[j])
        write next Hidden/Cell; optionally write Sequence[b,t,:]
    // Neither recurrent state nor Sequence[b,t>=active,:] is modified.

check cancellation; publish independently owned Final Hidden/Cell and Sequence
// Every error/cancellation above discards the per-call workspace and returns nil Result.
```

The pseudocode summarizes the implemented canonical kernel: *all* gate
preactivations use the unchanged preceding hidden state, and masked suffixes
are skipped rather than advanced with synthetic zero-valued inputs.

### Why saturation must follow affine overflow checks

A finite preactivation such as `x=1000` may legitimately round `sigmoid(x)`
to exactly `1` in binary64. An **overflowed** affine expression is different:
`Inf` is not a permitted preactivation, even if passing it to `sigmoid`
would also produce `1`. Treating the two as interchangeable hides invalid
parameter/data combinations.

The implementation uses a compensated `float64` sum in the common path. If
a product, intermediate sum, correction, or near-limit result threatens exact
classification, the same dot expression is recomputed using `math/big.Rat`
over the **original finite binary64 operands**. Its exact value is compared
against the largest finite binary64 magnitude *before* final rounding. This
also rejects cases such as an exact `MaxFloat64 + 1` that could otherwise
round back to a finite `MaxFloat64`. A similar exact boundary check protects
`f*c_prev+i*g` when the cell approaches the representable limit.

Two distinct failures must remain separate:

- `ErrNonFinite`: caller supplied NaN, positive infinity, or negative infinity
  in weights, state, or **any** input cell (including padding).
- `ErrOverflow`: supplied operands were finite, but an affine or recurrent
  output was mathematically outside the finite `float64` domain.

The use of compensated arithmetic and an exact exceptional fallback is **not**
a universal correctly rounded `float64` reference for every conditioned
nonexceptional dot product. It does not guarantee bitwise parity across
Go architectures, alternative CPU math libraries, or PyTorch CPU backends.

## 5. Batch-first tensor and masking law

`Tensor3.Data` represents `[B,T,D]` in row-major order:

$$ \operatorname{offset}(b,t,d)=(bT+t)D+d. $$

```text
Batch=2, Time=3, Features=2

Data = [x000 x001 | x010 x011 | x020 x021 | x100 x101 | x110 x111 | x120 x121]
         batch 0 / time 0,1,2                 batch 1 / time 0,1,2

lengths = [2,1]
          valid valid PAD                    valid PAD PAD
```

The geometry contract is `B>=1`, `D>=1`, `T>=0`, and
`len(Data)==B*T*D` after checked integer products. The feature dimension
must equal the model's imported `D`. `lengths=nil` means every row contains
exactly `T` valid steps. A nonnil mask must have `len(lengths)==B` and
`0<=lengths[b]<=T` for each batch row.

At every `t>=lengths[b]`, the kernel skips all gates, leaves **both** hidden
and cell states unchanged, and leaves the optional `Sequence.Data` cell at
exact positive zero. It does not repeat the last hidden activation. All
input elements, including masked elements, must nevertheless be finite.

| Case                    | State effect                                       | `Sequence` with `ReturnSequence=true`                |
|:------------------------|:---------------------------------------------------|:-----------------------------------------------------|
| `lengths=nil`           | Advance all `T` steps per batch row                | All steps contain computed hidden activations        |
| `lengths[b]=k`, `0<k<T` | Advance first `k`; keep final state during padding | Valid prefix followed by exactly zero hidden vectors |
| `lengths[b]=0`          | Preserve supplied initial state (or zeros)         | All `T` steps are exactly zero                       |
| `T=0`                   | No gate calls; copy valid `initial` or zeros       | Shape `[B,0,H]`, data length zero                    |
| `ReturnSequence=false`  | Same recurrence and same Final                     | **Entire** `Sequence` is the zero `Tensor3` value    |

With `T=0`, absent `Data` (`nil`) is permitted because the required element
count is zero. For positive required counts, absent `Data` is `ErrNilInput`;
a nonnil, incorrectly sized slice is `ErrShape`. Validation does not infer
session length from zero-filled samples. All buffers published after success
are detached from the input and from previously published results.

## 6. Streaming chunks and exact replay

`State{Hidden,Cell,ModelID}` is the only continuation record the numerical
kernel needs. It is an immutable-by-caller **snapshot during each call**,
not a mutable streaming object. A chunk begins with the `Final` from the
previous completed inference.

```text
full input  [0 ............................................... T)
             |----------- chunk 1 ---------|---- chunk 2 ------|
                            |                         |
                       InferFinal                InferChunk
                            |                         |
                         Final_1 ---------------> initial_2
                                                      |
                                                   Final_T
```

Let the global valid length of batch row `b` be `L_b`, a chunk start at
global time `s`, and its width be `C`. The local mask is:

$$
\ell_{b,\mathrm{chunk}}=\min\bigl(C,\max(0,L_b-s)\bigr).
$$

![Chunk-local valid length](https://latex.codecogs.com/svg.image?%5Cell_%7Bb%2C%5Cmathrm%7Bchunk%7D%7D%3D%5Cmin%28C%2C%5Cmax%280%2CL_b-s%29%29)

**Never reuse global `lengths` unchanged** for each chunk. A previous chunk
must consume the valid prefix first; masked steps cannot be used as fake
observations. A new independently started session requires a deliberate
zero-state `Infer` call, not passing a foreign checkpoint.

A formal recurrence argument is simple. For a valid observation `x_t`, let
`F_x(h,c)` be the deterministic IFGO state transition. The monolithic run is:

$$ S_T=F_{x_{T-1}}\circ\cdots\circ F_{x_0}(S_0). $$

Grouping that composition at the chunk boundary cannot change the evaluation
of any transition when the Go kernel, initial values, chronological input
order, local masks, and storage backend are the same. No extra rounding or
state reconstruction is performed by transferring the existing binary64
`Final`; `Final.Hidden` and `Final.Cell` are both required. Therefore the
implemented full-vs-chunk test checks `math.Float64bits` for exact equality
of both final states and concatenated valid outputs. This is an **intra-Go
kernel replay guarantee**, not a claim of identical bits across different
PyTorch CPU implementations.

For a normal stream:

```go
// After the initial completed chunk:
next, err := model.InferChunk(ctx, nextChunk, nextLocalLengths, first, lstm.Options{})
if err != nil {
    return err
}
// The next valid chunk can continue using `next`.
_ = next
```

The excerpt assumes `model`, `ctx`, `nextChunk`, `nextLocalLengths`, and
`first` are already initialized inside a function returning `error`.
`InferChunk` does not implement source-offset verification, external state
authentication, timestamp sequencing, or checkpoint persistence. Those are
application boundaries.

## 7. Canonical ModelID and ownership

`Import` copies every weight and bias into its own private row-major arrays.
It canonicalizes all stored negative zeros to positive zero. `Model.ID()`
returns a lowercase 64-character hexadecimal SHA-256 of the exact byte stream:

```text
ASCII("lstm/ifgo-f64/v1")
|| LE64(D)
|| LE64(H)
|| IEEE754LE(Input [4H,D] row-major)
|| IEEE754LE(Recurrent [4H,H] row-major)
|| IEEE754LE(Bias [4H] row-major)
```

The byte stream does not include a file name, URL, source dataset, timestamp,
PyTorch release, or runtime address. Changing the values of the source
matrices or `Bias` **after Import returns** cannot change either the model or
its ID. A mismatch between `initial.ModelID` and `Model.ID()` returns
`ErrStateVersion` before the initial matrices are examined. The same model
ID can be reused across independently initialized sessions; it is not a
session ID, signature, authorization token, or checkpoint integrity proof.

| Object                                              | Caller ownership after success                            | Concurrency rule                                        |
|:----------------------------------------------------|:----------------------------------------------------------|:--------------------------------------------------------|
| Supplied `Weights`, `Tensor3.Data`, `initial.State` | Caller retains originals; import/infer do not mutate them | Do not mutate concurrently while a call reads them      |
| `*Model`                                            | Owns private copied weights, no exported mutable view     | Multiple concurrent `Infer` calls are supported         |
| `Result.Sequence.Data`                              | Detached, caller-owned (when enabled)                     | Do not modify while another goroutine reads that result |
| `Result.Final.Hidden` / `.Cell`                     | Fresh independent `[B,H]` matrices                        | Treat as read-only during the next `InferChunk`         |

`matrix.Matrix` may be implemented by `*matrix.Dense` or a compatible custom
backend. The public interface does **not** expose the Dense backing slice.
Import deliberately uses checked `Matrix.At` reads once per element and
removes interface dispatch from recurrent loops by owning a private flat
`float64` copy. No reflection or unsafe memory alias is needed. This avoids
an ABI that would depend on mutable `Dense` internals. Custom `Matrix`
implementations must honor the interface's no-panic `At`/shape behavior;
if a custom Matrix can be stored as a typed-nil interface, it should
implement the `IsNil() bool` convention. Reflection-free input checking
cannot identify every arbitrary typed-nil pointer that omits `IsNil`.

## 8. Error precedence and cancellation

Errors are sentinel-classified using `errors.Is`, not message fragments.
The operation publishes **no partial** `Result` or state when an error
occurs; the caller's original input remains unchanged.

Canonical validation precedence for `Infer` is:

```text
nil context / pre-canceled context
    -> nil Model
    -> tensor B/T/D dimensions and D compatibility
    -> checked Data length and nil/non-nil storage
    -> local lengths mask shape and values
    -> initial.ModelID, initial matrix presence/shapes, state numeric values
    -> full input-data finiteness, including padding
    -> optional sequence allocation, recurrent compute, final publication
```

| Failure                                                                                     | Sentinel / response                      | Notes                                                       |
|:--------------------------------------------------------------------------------------------|:-----------------------------------------|:------------------------------------------------------------|
| `ctx==nil`                                                                                  | `ErrInvalidOption`                       | Before touching model or tensor                             |
| `ctx.Err()!=nil`                                                                            | `ctx.Err()`                              | Pre-canceled context takes precedence; polls continue later |
| Nil `*Model`, missing required matrix, missing positive-count `Data`                        | `ErrNilInput`                            | T=0 input may have nil Data                                 |
| Invalid B/T/D, mismatched feature count, bad matrix geometry, unrepresentable element count | `ErrShape`                               | Checked before allocation                                   |
| Nonnil `lengths` of wrong size, or values outside `[0,T]`                                   | `ErrInvalidOption`                       | `nil` mask means all T valid                                |
| `initial.ModelID!=Model.ID()`                                                               | `ErrStateVersion`                        | Before initial matrices are touched                         |
| Initial matrix absent                                                                       | `ErrNilInput`                            | Both Hidden and Cell are mandatory when `initial!=nil`      |
| Input/initial/weights contain NaN or ±Inf                                                   | `ErrNonFinite`                           | Padding is not exempt                                       |
| Finite operands produce unrepresentable affine or cell state                                | `ErrOverflow`                            | No saturated-Inf gate substitution                          |
| `Matrix.At` fails during import/state copy                                                  | `ErrShape` joined with lower-level cause | Both error identities remain inspectable with `errors.Is`   |

For `Import`, the validation path starts from `Hidden`, checked parameter
shapes, matrix presence and bias presence/length, then copies each weight
and checks all values. Import is atomic: a failed import returns nil Model.
For `InferChunk`, context and `previous` presence are checked before
delegating to `Infer`; a nil `previous` does **not** mean zero initial state.

`ctx.Err()` is checked before model/input validation, periodically while
scanning the full input, during active timesteps and long gate loops, and
before final publication. Cancellation may be observed at a poll boundary;
the contract is cooperative, not a hard real-time preemption guarantee.

A consumer should classify errors directly:

```go
switch {
case errors.Is(err, context.Canceled):
    // Retry only under an explicit source-offset/replay policy.
case errors.Is(err, lstm.ErrStateVersion):
    // Do not apply a checkpoint from different weights.
case errors.Is(err, lstm.ErrNonFinite):
    // Quarantine the entire input payload, including masked positions.
case errors.Is(err, lstm.ErrOverflow):
    // Reject the inference result and investigate numeric scaling/model weights.
case err != nil:
    // Handle shape, nil input, invalid option, or dependency errors.
}
```

The example assumes `err` and packages are already in scope. Never classify
sentinels by comparing their `.Error()` strings.

## 9. Executable Go scenario: fleet telemetry

Consider three machines uploading normalized measurements at a fixed sample
cadence. Raw physical values (bearing temperature and vibration) have already
been transformed using **frozen training-time scales** owned by the caller.
A rectangular request contains up to four observations per machine, but the
third machine has no valid new observations. The layer weights below are
**synthetic**, suitable for testing state/mask mechanics, not a trained
fault probability model.

```go
package main

import (
	"context"
	"fmt"
	"math"

	"github.com/lvlath/go/lstm"
	"github.com/lvlath/go/matrix"
)

func main() {
	// Four IFGO rows for H=1, each with D=2 input coefficients.
	inputWeights, err := matrix.NewDense(4, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := inputWeights.Fill([]float64{
		0.12, -0.08, // input gate
		0.06, 0.03, // forget gate
		0.30, 0.18, // candidate
		-0.04, 0.09, // output gate
	}); err != nil {
		fmt.Println(err)
		return
	}
	recurrentWeights, err := matrix.NewDense(4, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := recurrentWeights.Fill([]float64{0.02, 0.03, -0.01, 0.02}); err != nil {
		fmt.Println(err)
		return
	}
	model, err := lstm.Import(lstm.Weights{
		Input: inputWeights, Recurrent: recurrentWeights,
		Bias: []float64{0, 0.15, -0.05, 0}, Hidden: 1,
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	// [B=3,T=4,D=2]. The final row has only finite padding and zero valid steps.
	input := lstm.Tensor3{
		Batch: 3, Time: 4, Features: 2,
		Data: []float64{
			0.10, 0.20, 0.16, 0.21, 0, 0, 0, 0,
			0.35, 0.26, 0.41, 0.29, 0.49, 0.33, 0.57, 0.36,
			0, 0, 0, 0, 0, 0, 0, 0,
		},
	}
	result, err := model.InferSequence(context.Background(), input, []int{2, 4, 0}, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("batch=%d time=%d hidden=%d\n",
		result.Sequence.Batch, result.Sequence.Time, result.Sequence.Features)
	fmt.Printf("machine-1 padding zero=%t\n", result.Sequence.Data[2] == 0 && result.Sequence.Data[3] == 0)
	fmt.Printf("machine-3 final hidden has shape=%dx%d\n",
		result.Final.Hidden.Rows(), result.Final.Hidden.Cols())
	emptyHidden, err := result.Final.Hidden.At(2, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("machine-3 final hidden is zero=%t\n", math.Float64bits(emptyHidden) == 0)
	// Output:
	// batch=3 time=4 hidden=1
	// machine-1 padding zero=true
	// machine-3 final hidden has shape=3x1
	// machine-3 final hidden is zero=true
}
```

**Interpretation:** The model provides *recurrent features*, not automatic
machine-health labels. A production system must keep the scaler, feature
order, sampling cadence, model revision, checkpoint provenance, and downstream
threshold or supervised head under application control. For a richer four-device masked telemetry batch and streaming
scenarios, run the executable Go examples with:

```bash
go test ./lstm -run Example -v
```

The six package examples in [`lstm/example_test.go`](../lstm/example_test.go)
cover order-book microstructure, pump vibration streaming, four-device
telemetry masks, compressor comparison, deployment fingerprinting, and
quarantined invalid telemetry. Their `// Output:` blocks run under `go test`.

## 10. Independent oracles and PyTorch parity

The package has **three complementary** correctness checks; none replaces
the others:

1. **Analytical invariants:** zero weights with initial `c=2` produce
   `c1=1`, `h1=0.5*tanh(1)`; `c2=0.5`, `h2=0.5*tanh(0.5)`.
   A cell-candidate bias `atanh(0.5)` from zero state gives `c1=0.25`,
   `c2=0.375`. These establish gate and memory equations independently.
2. **Independent Go scalar oracle:** a separately coded recurrence checks
   a nontrivial multi-feature, multi-state batch against the public API.
   This is distinct from batch permutation/streaming consistency, which
   alone could preserve an identically wrong gate order.
3. **Frozen external PyTorch CPU double fixture:**
   [`lstm/pytorch_golden_test.go`](../lstm/pytorch_golden_test.go) contains
   numeric synthetic weights, input, expected outputs, final hidden/cell,
   and a canonical hash. The reference was generated with **PyTorch
   2.10.0+cpu**, `torch.float64`, one unidirectional layer,
   `batch_first=true`, `.eval()`, deterministic algorithms enabled,
   MKLDNN disabled and one CPU thread. The fixture's ModelID is:

   ```text
   d521b810acc028f8666cf7f6970f2a9b3f675710489d4c32be96fead620a1658
   ```

   Its conditioned **B=2, T=3, D=2, H=2** output and final states are
   compared at absolute tolerance `1e-12`. This is a *pinned numerical
   parity test*, not universal bit-identical execution across PyTorch
   versions, CPU dispatch choices, or host architectures.

No Python script, PyTorch library, arbitrary serializer or additional Go
module is needed to **run** or **test** `lstm`. The external provenance of
frozen reference numbers remains documented in the test itself; new
PyTorch versions must be compared deliberately rather than silently
regenerating and overwriting golden values.

## 11. Complexity and performance

Let `B` be batch size, `T` supplied time slots per row, `D` input width,
`H` hidden width, and `V = sum_b lengths[b]` (or `B*T` with no mask).

| Operation                                   | Main time                                   | Owned/extra space                          | Notes                                                                           |
|:--------------------------------------------|:--------------------------------------------|:-------------------------------------------|:--------------------------------------------------------------------------------|
| `Import`                                    | O(H(D+H))                                   | O(H(D+H))                                  | Validates/copies all `4H(D+H)` weights plus `4H` biases; hashes canonical bytes |
| `Infer` input scan and initial-state ingest | O(BTD + BH)                                 | O(BH)                                      | Scans all padding too; full numeric validation precedes execution               |
| Recurrent kernel (normal path)              | O(VH(D+H))                                  | O(H) transient                             | `4H` gate work vector is allocated once if any valid step exists                |
| Detached final publication                  | O(BH)                                       | O(BH)                                      | Two new `[B,H]` matrices, independent of input/state                            |
| Optional `ReturnSequence=true`              | O(BTH) zero-initialization and valid writes | O(BTH)                                     | All masked cells remain exactly zero                                            |
| Exceptional exact-rational rescue           | Operand-size-dependent                      | Additional transient `big.Rat` allocations | Only when float arithmetic/boundary requires classification                     |

The total usual time bound is `O(BTD + BH + VH(D+H) + BTH when sequence requested)`,
which is conventionally simplified to `O(BTH(D+H))` for an unmasked full
batch with positive `T`. The package checks integer size products before
`make`, but **does not promise to catch operating-system OOM** as a normal
error. It does not expose reusable mutable workspace because concurrent
model calls must remain isolated.

For benchmarks, see [`lstm/bench_test.go`](../lstm/bench_test.go), which
covers `B∈{1,16}`, `T∈{0,32,1024}`, `H∈{16,128}`, relevant `D`, and both
sequence-output policies. Use `go test -run '^$' -bench BenchmarkInfer -benchmem ./lstm`
from the module root. Timings depend on CPU, Go version, memory and data;
there is no universal throughput or zero-allocations-per-**call** promise.
The hot recurrent loop normally allocates **no storage per timestep**.

## 12. Common production pitfalls

| Mistake                                             | Why it breaks the contract                                             | Correct action                                                                  |
|:----------------------------------------------------|:-----------------------------------------------------------------------|:--------------------------------------------------------------------------------|
| Importing `i,f,o,g` without conversion              | Candidate and output activations are assigned wrong functions          | Convert to `i,f,g,o` externally and validate against a trusted golden fixture   |
| Passing only `bias_ih_l0`                           | Omits the recurrent bias contribution                                  | Supply `bias_ih_l0 + bias_hh_l0` explicitly, element by element                 |
| Treating `lengths` as a broadcast/global scalar     | Each batch row has its own valid prefix                                | Supply nil or exactly `B` local lengths                                         |
| Reusing total session lengths on each chunk         | Later chunks consume padded or nonexistent observations                | Compute `min(C,max(0,L_b-s))` per row/chunk                                     |
| Copying last valid hidden into padding              | Conceals missing time positions                                        | Read zero padded Sequence; use Final for last valid state                       |
| Skipping NaN checks on padding                      | Invalid payload can pass unnoticed                                     | Validate every supplied float, regardless of mask                               |
| Applying `sigmoid(Inf)` after affine overflow       | Hides arithmetic failure as a seemingly valid gate                     | Return `ErrOverflow`; don't saturate computed infinities                        |
| Holding a reference to mutable training weights     | Subsequent mutation changes deployed model behavior                    | Import once; use immutable model ID and independent results                     |
| Restoring a state solely by matching filename       | Filename does not bind actual coefficient bytes                        | Compare `State.ModelID` to the imported model's `ID()`                          |
| Treating ModelID as authentication                  | A hash alone provides no provenance or trusted source                  | Authenticate the complete artifact, feature schema and source cursor externally |
| Mistaking `Result.IsNil()` for numerical validation | It checks pointer presence only                                        | Inspect `err` first; use Result only on success                                 |
| Treating a hidden vector as a failure probability   | No calibrated output head is implemented                               | Apply a separately trained, validated head/policy in the caller                 |
| Claiming universal bitwise PyTorch compatibility    | CPU libraries and summation order may differ                           | Pin architecture/backend and test an explicit absolute tolerance                |
| Passing a typed-nil custom Matrix without `IsNil`   | Reflection is deliberately forbidden; its `At`/`Rows` must remain safe | Implement `IsNil() bool` or provide a nonnil valid Matrix                       |
| Assuming `ReturnSequence=false` means no allocation | Final matrices and workspace still exist                               | Benchmark the complete operation and choose the artifact you need               |

## 13. Verification and related documentation

Run the relevant package checks **after integrating the files into the
existing `github.com/lvlath/go` module**:

```bash
go test ./lstm -count=1
go test -race ./lstm -count=1
go test ./lstm -coverprofile=lstm.cover -count=1
go test ./...
go vet ./...
golangci-lint run
```

Tests live in the strict external `package lstm_test` and exercise the public
facade. `Result.IsNil` covers nil-pointer interface behavior, numerical
regressions cover `ErrOverflow` and unusual finite cancellations, masks
cover zero-length rows and nonfinite padding, chunk tests compare hidden
and cell bits, and the golden test anchors a real external reference.
Check the **current** coverage output rather than assuming an earlier
percentage still applies after edits.

Related material:

- [`lstm/doc.go`](../lstm/doc.go) - authoritative implemented package charter.
- [`lstm/api.go`](../lstm/api.go) - canonical entry point and convenient wrappers.
- [`lstm/example_test.go`](../lstm/example_test.go) - executable practical examples.
- [`TUTORIAL.md`](TUTORIAL.md#11t-lstm-foundations-frozen-ifgo-inference-and-masked-streaming) - first-use walkthrough.
- [`FAQ_&_TIPS.md`](FAQ_%26_TIPS.md#13s-lstm-faq-ifgo-inference-masking-and-streaming-replay) - troubleshooting and integration advice.
- [`lvlath_UES.md`](lvlath_UES.md) - shared contract and engineering requirements.
- [Official PyTorch LSTM reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) - external weight layout and mathematical recurrence.
