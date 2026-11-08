<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  File: docs/SERIES.md
  Package: github.com/lvlath/go/series
  Purpose: Mathematical and operational guide to exact timestamp grids,
           causal array transformations, numeric limits, and index windows.
  Contract status: Implemented API; synchronized with series public GoDoc.
  Scope: Grid validation, lag, differences, simple/log returns, complete windows.
  License: AGPL-3.0-only; see ../LICENSE.
-->

# Series: exact nanosecond grids and causal array primitives

> **A time index is not a clock, a missing grid slot is not a missing value, and an invalid warmup cell is not numeric zero.**

`github.com/lvlath/go/series` supplies small, deterministic operations that can be composed without inventing observations or changing their order. The package checks exact integer-nanosecond grids, transforms finite scalar sequences using available history, and enumerates complete index windows. It has **no dependencies beyond the Go standard library** and deliberately does not depend on `core`, `matrix`, or `dtw`.

This chapter teaches the implemented contract, not a proposed streaming engine. The authoritative callable surface is in [`series/api.go`](../series/api.go); the package charter is [`series/doc.go`](../series/doc.go); runnable output assertions are in [`series/example_test.go`](../series/example_test.go).

## Contents

1. [Domain boundaries and selecting a package](#1-domain-boundaries-and-selecting-a-package)
2. [Public API and result types](#2-public-api-and-result-types)
3. [Timestamp grids: exact integer law](#3-timestamp-grids-exact-integer-law)
4. [Causality and validity masks](#4-causality-and-validity-masks)
5. [Lag and finite differences](#5-lag-and-finite-differences)
6. [Simple and logarithmic returns](#6-simple-and-logarithmic-returns)
7. [Numerical stability and overflow](#7-numerical-stability-and-overflow)
8. [Complete half-open index windows](#8-complete-half-open-index-windows)
9. [Error classification and precedence](#9-error-classification-and-precedence)
10. [Determinism, ownership, and concurrency](#10-determinism-ownership-and-concurrency)
11. [Executable Go workflows](#11-executable-go-workflows)
12. [Complexity and memory](#12-complexity-and-memory)
13. [Tests, proofs, and review invariants](#13-tests-proofs-and-review-invariants)
14. [Pitfalls, composition, and non-goals](#14-pitfalls-composition-and-non-goals)

---

## 1. Domain boundaries and selecting a package

A sample series is a caller-ordered, nonempty `[]float64` with finite observations. **Input order is authoritative.** The package does not infer whether an index means a second, minute, trading bar, sensor event, or data-frame row. Only `Validate` accepts timestamps; those timestamps are signed `int64` nanoseconds with no timezone or epoch interpretation.

| Question                                               | Correct boundary                       | What it guarantees                                                                  |
|:-------------------------------------------------------|:---------------------------------------|:------------------------------------------------------------------------------------|
| Are timestamps increasing and on an exact cadence?     | `series.Validate`                      | Strict order, representable integer intervals, optional whole-slot gap reporting    |
| What was the observation `k` samples ago?              | `series.Lag`                           | Length-preserving lag and explicit warmup mask                                      |
| How much did an observation change over `k` positions? | `series.Difference`                    | Finite lag difference, no hidden scaling by elapsed time                            |
| What is the adjacent relative change?                  | `series.SimpleReturns`                 | Dimensionless simple return and checked denominator                                 |
| What is the adjacent logarithmic change?               | `series.LogReturns`                    | Positive-domain natural-log change without forming an extreme ratio                 |
| Which fixed-size sample-index windows are complete?    | `series.WindowRanges`                  | Ordered half-open index ranges without trailing partial windows                     |
| Do two sequences match despite local time stretching?  | `dtw`                                  | Alignment under an explicit DTW cost/window policy; **not** implemented by `series` |
| Do I need covariance, rolling statistics, or fitting?  | Other numerical/statistical components | These are **not** `series` operations                                               |

The essential separation is:

```text
caller supplies timestamps + observations
              │
              ▼
       series.Validate
  order / exact cadence / gap metadata
              │  (never inserts data)
              ▼
     original []float64
              │
   ┌──────────┴───────────┐
   ▼                      ▼
causal transforms    WindowRanges
Values + Valid       [Start, End)
   │                      │
   └─────────┬────────────┘
             ▼
    application-owned policy
  (skip bad windows, align, model,
   aggregate, or store as needed)
```

A successful `Validate` call does **not** automatically guarantee regular spacing when `AllowGaps` is true. It guarantees exact alignment to a lattice while explicitly reporting skipped lattice positions.

## 2. Public API and result types

The implemented signatures are deliberately small; there is no `Engine`, `Fit`, callback, context parameter, or hidden options factory.

```go
package series

type GridOptions struct {
    CadenceNS int64
    AllowGaps bool
}

type Gap struct {
    AfterIndex   int
    MissingCount int64
}

type ValidationResult struct {
    Samples int
    Gaps    []Gap
}

type TransformResult struct {
    Values []float64
    Valid  []bool
}

type Range struct {
    Start int
    End   int
}

func Validate(timesNS []int64, values []float64, opts GridOptions) (*ValidationResult, error)
func Lag(values []float64, lag int) (*TransformResult, error)
func Difference(values []float64, lag int) (*TransformResult, error)
func SimpleReturns(values []float64) (*TransformResult, error)
func LogReturns(values []float64) (*TransformResult, error)
func WindowRanges(length, width, stride int) ([]Range, error)
```

The block above is a **declaration reference**, not a file to paste into the package. Public GoDoc in [`types.go`](../series/types.go) explains each field individually.

| Public carrier     | Interpretation                                         | Ownership / invariant                                                  |
|:-------------------|:-------------------------------------------------------|:-----------------------------------------------------------------------|
| `GridOptions`      | Exact cadence and whole-gap policy                     | Passed by value; zero cadence means **order only**                     |
| `Gap`              | Missing lattice slots **after a supplied input index** | `AfterIndex` refers to the original index; `MissingCount > 0`          |
| `ValidationResult` | Number of accepted pairs and ordered gap records       | `Samples == len(values)`; `Gaps` **non-nil** on success                |
| `TransformResult`  | Computed values plus position-wise validity            | `len(Values) == len(Valid) == input length`; both detached             |
| `Range`            | Complete interval of sample indices                    | `Start` inclusive; `End` exclusive; emitted ranges have positive width |

All five public structures expose fields directly. Adding fields to such exported structures later can affect positional composite literals; callers should prefer named fields such as `series.GridOptions{CadenceNS: 10, AllowGaps: true}`.

### The zero-value policies are not guesses

- `GridOptions{}` means **strictly increasing and representable intervals only**. It never estimates a cadence.
- `GridOptions{CadenceNS: 10, AllowGaps: false}` requires every adjacent interval to be exactly 10 ns.
- `GridOptions{CadenceNS: 10, AllowGaps: true}` permits 20 ns, 30 ns, etc., but **not** 15 ns.
- `AllowGaps: true` with zero cadence changes no output: without a grid there is no well-defined count of missing slots.

## 3. Timestamp grids: exact integer law

Let input timestamps be `t[0], ..., t[n-1]`, measured in signed integer nanoseconds. Before any gap policy, each adjacent pair must satisfy:

![Strict timestamp-order condition](https://latex.codecogs.com/svg.image?t_i%3Et_%7Bi-1%7D%5Cquad%28i%3D1%2C%5Cldots%2Cn-1%29)

and the **mathematical** difference `dᵢ = tᵢ − tᵢ₋₁` must fit in signed `int64`:

```text
1 <= d_i <= math.MaxInt64
```

If `CadenceNS == 0`, this is the entire timestamp law. If `CadenceNS == c > 0`, additionally require exact integer divisibility and compute the number of absent slots:

![Exact-grid and missing-slot formula](https://latex.codecogs.com/svg.image?d_i%5Cbmod%20c%3D0%2C%5Cqquad%20m_i%3D%5Cfrac%7Bd_i%7D%7Bc%7D-1)

**No tolerance** is allowed: `dᵢ % c != 0` is off-grid and produces `ErrGap`, even when `AllowGaps` is true. For `mᵢ > 0`, `AllowGaps` decides whether the missing whole slots are rejected or recorded.

### Worked grid: one absent sensor sample

```text
Supplied index:  0      1      2             3      4
Timestamp (ns):  0     10     20            40     50
Grid slots:      ●------●------●------○------●------●
                                      ↑
                           slot at 30 ns is absent

ValidationResult{
    Samples: 5,
    Gaps: []Gap{{AfterIndex: 2, MissingCount: 1}},
}
```

There are **five supplied samples**, not six. `Gap.AfterIndex == 2` identifies the interval between original elements `2` and `3`; it does not refer to the missing timestamp or to an inserted index.

### Why signed subtraction must not precede overflow checking

For example, `t[0] = math.MinInt64` and `t[1] = 0` are strictly increasing, but the true difference is `2^63` ns. That number does **not** fit in signed `int64`. Naive signed subtraction wraps; it must not be interpreted as a negative delta or a valid grid.

The implementation first checks `next > prev`, then evaluates:

```go
span := uint64(next) - uint64(prev)
if span > uint64(math.MaxInt64) {
    return nil, ErrOverflow
}
delta := int64(span)
```

After order validation, unsigned subtraction produces the exact positive mathematical distance modulo `2^64`; its value is the real distance in the range `1` to `2^64 - 1`. Only after proving `span <= MaxInt64` is the cast safe. This check is required even if cadence is zero.

This fragment describes the **internal arithmetic**, not a new public callable API.

### Input-order and gap classification examples

| Adjacent signed timestamps | Cadence | Policy             | Classification                         |
|:---------------------------|--------:|:-------------------|:---------------------------------------|
| `[0, 10, 20]`              |     `0` | Either `AllowGaps` | Success, non-nil empty `Gaps`          |
| `[0, 10, 20]`              |    `10` | `false`            | Success, no missing slots              |
| `[0, 10, 30]`              |    `10` | `true`             | Success, one gap after index `1`       |
| `[0, 10, 30]`              |    `10` | `false`            | `ErrGap`                               |
| `[0, 10, 25]`              |    `10` | `true`             | `ErrGap`, nonmultiple interval         |
| `[5, 5]`                   |     `0` | Either             | `ErrOrder`, duplicate                  |
| `[8, 7]`                   |     `0` | Either             | `ErrOrder`, decreasing                 |
| `[MinInt64, 0]`            |     `0` | Either             | `ErrOverflow`                          |
| `[MinInt64, MaxInt64]`     |     `1` | Either             | `ErrOverflow`, before cadence division |

The package never sorts timestamps. Sorting would change the association between input timestamps and observations and would violate the index-preserving contract.

## 4. Causality and validity masks

A transform is **causal** if its result at index `i` uses only input positions `0` through `i`. Every transform in `series` has this property. It also preserves the input length, unlike a transform that trims its warmup prefix.

For a successful call on `x = (x₀,...,xₙ₋₁)` and a successful call on an extension `x' = (x₀,...,xₙ₋₁,xₙ,...)`, the prefix law is:

![Causal prefix invariance](https://latex.codecogs.com/svg.image?%5Cforall%20i%3Cn%3A%5Cquad%28y%27_i%2C%5C%20valid%27_i%29%3D%28y_i%2C%5C%20valid_i%29)

The qualification **both calls must succeed** matters: extending a series that currently ends in zero may make `SimpleReturns` reject the extended series because that previously terminal zero becomes a denominator. Similarly, extending `LogReturns` with a nonpositive observation makes the extended call fail rather than produce a prefix result. The package publishes no partial result on such an error.

### Values and Valid must travel together

```text
Input:                  100     105     105     110
Difference(lag=1):
    Values:              0       5       0       5
    Valid:             false   true    true    true
                         ↑               ↑
                      warmup          real zero difference
```

`Values[i] == 0` is **not** equivalent to `Valid[i] == false`. This distinction persists across larger compositions: a consumer must check the mask before using a transformed value. `Valid` does **not** encode missing timestamp slots, market closures, and domain errors. Gaps and valid-history masks describe different information.

### There is no implicit gap-aware transform

`Lag(values, 1)` means the preceding **supplied array element**, not the preceding 10-ns lattice position. If the timestamps are `[0, 10, 30]`, then the lag at supplied index `2` uses the value at supplied index `1` regardless of the missing 20-ns slot.

That behavior is deliberate and composable: the caller decides whether a gap invalidates a later observation pair, whether to segment data, or whether to reject the series by calling `Validate` with `AllowGaps: false`. The primitives themselves do not resample or inject synthetic zeros.

## 5. Lag and finite differences

### Lag: an index shift, not a duration

For a nonnegative integer `k`:

![Lag transform formula](https://latex.codecogs.com/svg.image?L_k%28x%29_i%3D%5Cbegin%7Bcases%7Dx_%7Bi-k%7D%2C%26i%5Cge%20k%5C%5C0%2C%26i%3Ck%5Cend%7Bcases%7D)

`Valid[i]` is true exactly for `i >= k`. In the invalid branch, `Values[i]` is exactly zero.

- `Lag(values, 0)` is an **identity of values** with all valid bits true, but returns new detached slices rather than aliasing the input.
- `Lag(values, k)` for `k >= len(values)` returns all zero/false cells **after validating every input** for finiteness.
- Negative `k` is `ErrInvalidOption`, even if `values` is nil.
- `k` is an index distance, not nanoseconds, not a duration, and not a number of grid slots when gaps exist.

### Difference: a positive-lag subtraction

For `k >= 1`:

![Finite-difference formula](https://latex.codecogs.com/svg.image?D_k%28x%29_i%3D%5Cbegin%7Bcases%7Dx_i-x_%7Bi-k%7D%2C%26i%5Cge%20k%5C%5C0%2C%26i%3Ck%5Cend%7Bcases%7D)

`Difference(values, 0)` is invalid; a zero-lag difference would be a constant zero sequence and is intentionally **not** the API's defined operation. For `k >= n`, all output positions remain invalid after whole-input validation. Every valid difference is checked to be finite; finite operands do not guarantee a finite subtraction.

**Overflow counterexample:** with values `[-math.MaxFloat64, math.MaxFloat64]` and lag `1`, the mathematical difference exceeds the maximum finite float64. The correct result is `(nil, ErrOverflow)`, not a `+Inf` valid cell.

### Why this is O(n), not O(n·k)

Each output cell reads at most two fixed input positions, with no search for a prior timestamp. Validation scans the full input once; output filling scans the usable suffix once. Both memory buffers are allocated once. Thus the running time and output memory are linear in the number of supplied observations, even when `k` is large.

## 6. Simple and logarithmic returns

Returns are adjacent-index transforms. They do not divide by elapsed nanoseconds, infer a market bar duration, or impute missing observations.

### SimpleReturns: one-step relative change

For `i >= 1`, with `x[i-1] != 0`:

![Simple return formula](https://latex.codecogs.com/svg.image?R_i%3D%5Cfrac%7Bx_i%7D%7Bx_%7Bi-1%7D%7D-1)

The result is dimensionless. `Values[0] == 0` and `Valid[0] == false`. Later entries are valid if all validation gates pass.

The domain rule is **specific to used denominators**: `values[0]` through `values[n-2]` must be nonzero. A zero at the last index is allowed because it is never used as a denominator for a subsequent pair in that call.

| Input       | Result / interpretation                                                              |
|:------------|:-------------------------------------------------------------------------------------|
| `[2, 4, 0]` | `[0, 1, -1]`, with `[false, true, true]` mask                                        |
| `[2, 0, 4]` | `ErrDomain` because the zero at index 1 becomes a denominator                        |
| `[5]`       | One zero/false warmup cell; no denominator exists                                    |
| `[-2, -4]`  | Computed return `1` is permitted mathematically; application meaning is not assigned |
| `[0]`       | One invalid warmup cell, **success**: no zero denominator was used                   |

The package deliberately does not impose financial sign conventions or assume that negative inputs encode valid prices. Such domain restrictions belong in a caller-specific layer.

### LogReturns: adjacent natural-log change

For `i >= 1`, with **all** observations strictly positive:

![Logarithmic return formula](https://latex.codecogs.com/svg.image?Q_i%3D%5Cln%28x_i%29-%5Cln%28x_%7Bi-1%7D%29)

This identity is defined in terms of natural logarithms, not `log10`. The first position remains zero/false. Unlike `SimpleReturns`, `LogReturns` rejects **any** input that is zero or negative, including the only observation in a singleton.

| Input             | Result / interpretation                                                 |
|:------------------|:------------------------------------------------------------------------|
| `[1, 2]`          | `log(2)` at index 1, finite and positive                                |
| `[2, 1]`          | `-log(2)` at index 1, finite and negative                               |
| `[2]`             | One zero/false warmup cell, success                                     |
| `[0]`             | `ErrDomain`, despite having no computed pair                            |
| `[1, -1]`         | `ErrDomain` after finite-input validation                               |
| `[1e-300, 1e300]` | Finite return near `600·ln(10)`; ratio-first calculation would overflow |

For positive real numbers, the identity `ln(a/b) = ln(a) - ln(b)` is valid. For finite `float64` values, computing `a/b` first can overflow or underflow even though the logarithmic difference is finite. The implementation therefore avoids that intermediate ratio.

## 7. Numerical stability and overflow

The source represents values in IEEE-754 `float64` and uses the standard library's `math` functions. Finite input validation is strict; there is no epsilon that turns an invalid value into an accepted one. No exact-real-arithmetic guarantee is claimed for return values.

### Close-pair branch: preserve small relative changes

The naive formula `curr/prev - 1` can lose a small nonzero return when `curr/prev` rounds to exactly `1`. For eligible close pairs, the implementation instead computes:

```text
delta    = curr - prev
relative = delta / prev
```

It uses that relative change only when `delta` and `relative` are finite and `abs(relative) < 0.5` (`stableRelativeChangeLimit`). Within this close same-sign region, Sterbenz's lemma explains why subtraction of nearby floating-point values avoids an additional rounding loss. The relative quotient still follows float64 division semantics; this is a stable branch, **not** a promise of arbitrary-precision output.

- `SimpleReturns` uses `relative` on the close branch; otherwise it evaluates `curr/prev - 1`.
- `LogReturns` uses `math.Log1p(relative)` on the close branch; otherwise it evaluates `math.Log(curr) - math.Log(prev)`.
- Both operations reject any nonfinite computed output with `ErrOverflow`.

### Extreme positive input: why a log difference matters

```text
prev = 1e-300
curr = 1e300

Incorrect intermediate:
    curr/prev = 1e600   -> +Inf in float64
    log(+Inf)           -> +Inf, losing a finite answer

Implemented distant-pair path:
    log(curr) - log(prev)
    ≈ 600 * ln(10)      -> finite
```

### Adjacent large float64 values: why log1p matters

Let `prev = 1e300` and `curr = math.Nextafter(prev, math.Inf(1))`. Both are distinct, positive, finite float64 values. Their ratio is so close to one that independently rounded logarithms may be equal; directly subtracting those logs can incorrectly yield zero. `math.Log1p((curr-prev)/prev)` preserves the observable small positive change for this case.

The implementation selects its branch deterministically from the operands and a fixed strict limit; it does **not** use an application tolerance, data-dependent model, or a fallback that changes the mathematical objective.

### Input nonfiniteness is different from arithmetic overflow

| Failure                                                   | Sentinel       | Example                                          |
|:----------------------------------------------------------|:---------------|:-------------------------------------------------|
| Nonfinite input observation                               | `ErrNonFinite` | `[]float64{1, math.NaN()}`                       |
| Computed nonfinite difference                             | `ErrOverflow`  | `MaxFloat64 - (-MaxFloat64)`                     |
| Computed nonfinite simple return                          | `ErrOverflow`  | Very large current / very small nonzero previous |
| Timestamp delta not representable as signed int64         | `ErrOverflow`  | `MinInt64 -> 0`                                  |
| Complete range array cannot be represented as `int` bytes | `ErrOverflow`  | Extreme `WindowRanges` dimensions                |

A success result never uses `+Inf` or NaN as a marker for missing data or failure. Invalid warmup cells are **zero/false**; errors return **nil results**.

## 8. Complete half-open index windows

`WindowRanges(length, width, stride)` returns only complete intervals `[Start, End)` with `End - Start == width`, where `Start` advances by `stride` from zero and `End <= length`.

For valid dimensions and `width <= length`, the count is:

![Complete-window count formula](https://latex.codecogs.com/svg.image?w%3D1%2B%5Cleft%5Clfloor%5Cfrac%7Blength-width%7D%7Bstride%7D%5Cright%5Crfloor)

For `width > length`, `w == 0` and the successful return is **empty but non-nil**. The count is computed using subtraction before division; every emitted `End` is bounded by `length`, and no unused final `Start + stride` is computed.

### Half-open layout

```text
Source indices:   0 1 2 3 4 5 6 7
width = 3
stride = 2

Window 0         [0-----3)        indices 0,1,2
Window 1             [2-----5)    indices 2,3,4
Window 2                 [4-----7) indices 4,5,6
No [6,9): it would extend beyond length 8.

Result: [{Start:0, End:3}, {Start:2, End:5}, {Start:4, End:7}]
```

The same rule allows disjoint windows: `WindowRanges(10, 2, 4)` returns `[0,2)`, `[4,6)`, `[8,10)`. It also allows windows that touch (`stride == width`) or overlap (`stride < width`). None of these modes require a new option.

### Index range is not a time range

Given `timesNS = [0, 10, 30]`, `Range{Start: 0, End: 2}` selects the first two **supplied samples**, not necessarily a duration of `2*CadenceNS` in more general inputs. `WindowRanges` does not call `Validate`. If a downstream operation requires uninterrupted wall-clock windows, the application must first enforce `AllowGaps: false` or independently filter windows crossing reported gaps.

### Allocation arithmetic and resource limits

The implementation checks `count > MaxInt / sizeof(Range)` before allocating the range slice. That catches a mathematically representable element count whose byte size cannot fit a Go `int` on the target architecture. The guard is **not** a memory budget: an extremely large but representable allocation can still fail due to insufficient memory. There is no allocation-budget option and no OOM recovery.

## 9. Error classification and precedence

All nine error identities belong to `series`; callers use `errors.Is(err, series.ErrXxx)`, not string comparison.

| Sentinel           | Trigger                                                                             |
|:-------------------|:------------------------------------------------------------------------------------|
| `ErrInvalidOption` | Negative cadence/lag/length, zero or negative width/stride, or `Difference` lag `0` |
| `ErrNilInput`      | Any required nil timestamp or values slice                                          |
| `ErrEmptyInput`    | Required non-nil empty unary input, or both equal-length empty inputs to `Validate` |
| `ErrShape`         | `Validate` timestamp/value length mismatch, including empty/nonempty mismatches     |
| `ErrNonFinite`     | NaN, +Inf, or -Inf in any observation, including warmup-only positions              |
| `ErrOrder`         | Equal or decreasing neighboring timestamps                                          |
| `ErrGap`           | Nonmultiple positive-cadence interval or missing whole slots when disallowed        |
| `ErrDomain`        | Used zero denominator in `SimpleReturns` or any nonpositive `LogReturns` input      |
| `ErrOverflow`      | Unrepresentable signed delta, computed nonfinite float, or range storage size       |

### Precedence is observable behavior

`Validate` applies:

```text
1. Negative CadenceNS?                         -> ErrInvalidOption
2. Either slice nil?                           -> ErrNilInput
3. Length mismatch?                           -> ErrShape
4. Both same-length empty?                    -> ErrEmptyInput
5. Any nonfinite observation?                 -> ErrNonFinite
6. For adjacent input pairs, in order:
   a. next <= previous?                       -> ErrOrder
   b. unsigned distance > MaxInt64?           -> ErrOverflow
   c. with positive cadence, nonmultiple?     -> ErrGap
   d. with positive cadence, missing slots?
      - AllowGaps == false                     -> ErrGap
      - AllowGaps == true                      -> append Gap
7. All checks pass                            -> ValidationResult
```

For `Lag` and `Difference`, option validation precedes input access. All unary transforms check nil, empty, and **every** nonfinite element before operation-specific domain or arithmetic checks. `SimpleReturns` validates all used denominators, and `LogReturns` validates all positive inputs, before evaluating any return. `WindowRanges` has only dimension and output-size checks.

| Call situation                                     | First result                            |
|:---------------------------------------------------|:----------------------------------------|
| `Lag(nil, -1)`                                     | `ErrInvalidOption`, not `ErrNilInput`   |
| `Difference(nil, 0)`                               | `ErrInvalidOption`, not `ErrNilInput`   |
| `Validate(nil, []float64{}, GridOptions{})`        | `ErrNilInput`, not `ErrShape`           |
| `Validate([]int64{}, []float64{1}, GridOptions{})` | `ErrShape`, not `ErrEmptyInput`         |
| Both allocated `Validate` slices empty             | `ErrEmptyInput`                         |
| `SimpleReturns([]float64{0, MaxFloat64, 1})`       | `ErrDomain` before arithmetic           |
| `LogReturns([]float64{0})`                         | `ErrDomain`, despite no pair to compute |
| `WindowRanges(0, 3, 1)`                            | Success: non-nil empty ranges           |

The public result is **nil on every error**; there is no successful-partial artifact, changed error identity, automatic retry, or fallback algorithm. Diagnostic wrapping in future versions must preserve sentinel identity.

## 10. Determinism, ownership, and concurrency

**Determinism.** Each function consumes the input in ascending index order, uses no maps or random seeds, and has a fixed branch policy. Gap records are ordered by increasing `AfterIndex`; windows by increasing `Start`. Float64 results are reproducible under the same runtime semantics and unchanged inputs; they are not promised to be bitwise identical across every platform or math-library implementation.

**Read-only input.** The package does not sort, mutate, or retain caller input. It does not take a snapshot at the start of a call: mutating a slice concurrently while it is being validated or transformed is unsupported.

**Detached output.** `ValidationResult.Gaps`, `TransformResult.Values`, `TransformResult.Valid`, and the returned `[]Range` are independently allocated output slices. Mutating them after a successful call does not mutate the source. Different successful calls do not share package-owned mutable working buffers.

**Concurrency boundary.** Calls may run simultaneously on inputs that remain immutable for the duration of those calls; returned outputs can be used independently. Mutating the same output concurrently still requires synchronization by the caller. There is no stateful model, accumulator, mutable cache, or streaming engine.

## 11. Executable Go workflows

The programs below are independently compilable examples using only the published package API. The package also contains six runnable examples in [`series/example_test.go`](../series/example_test.go).

### Workflow A: validate a telemetry grid without inventing samples

```go
package main

import (
	"errors"
	"fmt"

	"github.com/lvlath/go/series"
)

func main() {
	times := []int64{0, 10, 20, 40, 50}
	values := []float64{100, 105, 105, 110, 115}

	grid, err := series.Validate(times, values, series.GridOptions{
		CadenceNS: 10,
		AllowGaps: true,
	})
	if err != nil {
		fmt.Println("grid error:", err)
		return
	}

	fmt.Println("samples:", grid.Samples)
	for _, gap := range grid.Gaps {
		fmt.Printf("missing %d slot(s) after index %d\n", gap.MissingCount, gap.AfterIndex)
	}

	// The next call does not interpolate the missing 30-ns slot.
	delta, err := series.Difference(values, 1)
	if err != nil {
		fmt.Println("difference error:", err)
		return
	}
	fmt.Println("difference:", delta.Values)
	fmt.Println("valid:", delta.Valid)

	// A strict uninterrupted-grid policy would reject this input.
	_, err = series.Validate(times, values, series.GridOptions{CadenceNS: 10})
	fmt.Println("strict grid rejected:", errors.Is(err, series.ErrGap))
}
```

Expected output:

```text
samples: 5
missing 1 slot(s) after index 2
difference: [0 5 0 5 5]
valid: [false true true true true]
strict grid rejected: true
```

Notice that the fourth computed difference crosses a timestamp gap. This is intentional index arithmetic; if your domain forbids such a comparison, reject gaps or segment the data before transforming it.

### Workflow B: preserve masks across independent causal calculations

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/series"
)

func main() {
	values := []float64{2, 4, 4, 8, 0}

	lagged, err := series.Lag(values, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	differences, err := series.Difference(values, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	returns, err := series.SimpleReturns(values)
	if err != nil {
		fmt.Println(err)
		return
	}

	for i := range values {
		fmt.Printf("%d: lag=%g(%t), delta=%g(%t), return=%g(%t)\n",
			i,
			lagged.Values[i], lagged.Valid[i],
			differences.Values[i], differences.Valid[i],
			returns.Values[i], returns.Valid[i],
		)
	}
}
```

Expected output:

```text
0: lag=0(false), delta=0(false), return=0(false)
1: lag=0(false), delta=2(true), return=1(true)
2: lag=2(true), delta=0(true), return=0(true)
3: lag=4(true), delta=4(true), return=1(true)
4: lag=4(true), delta=-8(true), return=-1(true)
```

This dataset demonstrates three different zeros: an invalid lag warmup, a computed zero difference, and a computed zero return. Only the boolean masks distinguish them safely.

### Workflow C: complete windows over validated observations

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/series"
)

func main() {
	times := []int64{0, 10, 20, 30, 40, 50, 60, 70}
	observations := []float64{3, 5, 8, 13, 21, 34, 55, 89}

	_, err := series.Validate(times, observations, series.GridOptions{CadenceNS: 10})
	if err != nil {
		fmt.Println(err)
		return
	}

	windows, err := series.WindowRanges(len(observations), 3, 2)
	if err != nil {
		fmt.Println(err)
		return
	}

	for _, window := range windows {
		fmt.Printf("[%d,%d): %v\n", window.Start, window.End,
			observations[window.Start:window.End])
	}
}
```

Expected output:

```text
[0,3): [3 5 8]
[2,5): [8 13 21]
[4,7): [21 34 55]
```

There is no fourth partial window. The application slices its own original `observations` using the detached `Range` values; `series` itself does not copy or retain window contents.

### Workflow D: stable log returns for large dynamic ranges

```go
package main

import (
	"fmt"
	"math"

	"github.com/lvlath/go/series"
)

func main() {
	values := []float64{1e-300, 1e300, math.Nextafter(1e300, math.Inf(1))}
	result, err := series.LogReturns(values)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("large step finite: %t\n", !math.IsInf(result.Values[1], 0))
	fmt.Printf("small step positive: %t\n", result.Values[2] > 0)
	fmt.Println("valid:", result.Valid)
}
```

Expected output:

```text
large step finite: true
small step positive: true
valid: [false true true]
```

## 12. Complexity and memory

Define `n = len(values)`, `g = number of reported Gap entries`, and `w = number of complete windows`.

| Function        | Time   | Additional space | Reason                                                                            |
|:----------------|:-------|:-----------------|:----------------------------------------------------------------------------------|
| `Validate`      | `O(n)` | `O(g)`           | One finite-value scan, one adjacent-timestamp scan, gap records only when enabled |
| `Lag`           | `O(n)` | `O(n)`           | Full input validation plus detached values/mask allocation and shift              |
| `Difference`    | `O(n)` | `O(n)`           | Full validation plus at most `n - lag` checked differences                        |
| `SimpleReturns` | `O(n)` | `O(n)`           | Finite scan, denominator-domain scan, pair loop                                   |
| `LogReturns`    | `O(n)` | `O(n)`           | Finite scan, positive-domain scan, pair loop                                      |
| `WindowRanges`  | `O(w)` | `O(w)`           | O(1) count/size calculation and one emitted `Range` per complete window           |

The grid validator is **not** a timestamp index or a lookup structure. Transform kernels allocate two separate O(n) result slices; they do not allocate full pairwise matrices or retain a hidden working buffer. `WindowRanges` stores range metadata, not window samples.

## 13. Tests, proofs, and review invariants

The test suite lives in the external `package series_test`. Tests should use the public exported surface and package-owned sentinels; production methods must not be exposed solely to inspect a private branch.

### Invariants worth asserting exactly

- **Shape:** `len(result.Values) == len(result.Valid) == len(input)` on every successful transform.
- **Warmup:** every `!Valid[i]` position is exactly `Values[i] == 0`; valid zero results are permitted.
- **Prefix causality:** for two successful calls on a prefix and an extension, both values and mask bits agree at all original indices.
- **Grid order:** returned `Gap.AfterIndex` values are strictly increasing and every `MissingCount > 0`.
- **Grid integer arithmetic:** all deltas are representable and exact multiples under positive cadence, without float conversion or tolerance.
- **Range law:** every output has `End-Start == width`, `0 <= Start < End <= length`; starts are exactly stride-spaced.
- **Ownership:** mutations of returned buffers must not modify input or another result.
- **Sentinels:** `errors.Is` classifies errors; error-string matching is not a contract test.
- **No partial publication:** error calls return nil result, including a late overflow.

### Independent mathematical counterexamples

| Risk                                          | Counterexample                           | Correct classification or property    |
|:----------------------------------------------|:-----------------------------------------|:--------------------------------------|
| Signed subtraction wraps                      | `MinInt64 -> 0`                          | `ErrOverflow`                         |
| Duplicate timestamp                           | `[7, 7]`                                 | `ErrOrder`                            |
| Off-grid but gaps permitted                   | `[0, 15]`, cadence 10, `AllowGaps: true` | `ErrGap`                              |
| Finite operands overflow subtraction          | `[-MaxFloat64, MaxFloat64]`, lag 1       | `ErrOverflow`                         |
| Ratio-first log overflow                      | `[1e-300, 1e300]`                        | Finite log return                     |
| Close log pair lost by rounded logs           | `Nextafter(1e300, +Inf)`                 | Strictly positive small return        |
| Zero conflated with warmup                    | `[2, 2]` for `Difference(..., 1)`        | Valid zero at index 1                 |
| Storage overflows despite valid element count | `WindowRanges(MaxInt, 1, 1)`             | `ErrOverflow` before slice allocation |

For numeric oracle tests, treat each source float64 as the exact binary number it represents (for example using `math/big.Rat`), not as an exact copy of its printed decimal approximation. Floating-point comparisons require a justified magnitude-aware tolerance only where rounding is mathematically relevant. Never use epsilon to change integer grid acceptance, result masks, indices, or error identities.

### Cross-architecture caution

`int` and `unsafe.Sizeof(Range{})` are architecture-dependent. The implementation checks range allocation against the target `int` width. A 32-bit cross-compile is useful, but compiling is not the same as executing full tests on a 32-bit runtime. Runtime, Go version, OS/architecture, and invocation must accompany any benchmark or test claims.

## 14. Pitfalls, composition, and non-goals

| Tempting shortcut                                      | Why it is incorrect                                   | Correct action                                                 |
|:-------------------------------------------------------|:------------------------------------------------------|:---------------------------------------------------------------|
| Treat `CadenceNS == 0` as automatic cadence detection  | No cadence is inferred                                | Supply a positive explicit cadence when you need grid checks   |
| Treat `AllowGaps: true` as interpolation               | Only metadata is reported                             | Decide how to handle missing observations outside `series`     |
| Interpret lag `k` as elapsed nanoseconds               | It counts supplied indices                            | Validate cadence separately; reject or process gaps explicitly |
| Discard `TransformResult.Valid`                        | A zero can be a real result or a warmup placeholder   | Carry Values and Valid together                                |
| Reject negative simple-return inputs unconditionally   | Package math permits finite negative values           | Enforce application-level price/domain policy separately       |
| Accept zero as a log-return warmup input               | The **entire input** must be strictly positive        | Reject with `ErrDomain`                                        |
| Compute `math.Log(curr/prev)`                          | Intermediate ratio can overflow or underflow          | Use the published `LogReturns` function                        |
| Emit partial trailing windows                          | Changes the documented enumeration law                | Use only complete `[Start,End)` ranges                         |
| Assume checked byte size means allocation will succeed | OS memory availability is not guaranteed              | Add caller-side resource governance for very large workloads   |
| Use `series` as an online streaming engine             | No `Push`, `Reset`, `Snapshot`, or ring buffer exists | Maintain application buffers or define a separate reviewed API |

**Relationship to DTW.** `series` can validate input cadence, prepare differences/returns, or enumerate candidate index windows before an application calls `dtw`. It cannot decide a DTW warp path, cost, cutoff certificate, or search ranking. Validation with permitted gaps does not automatically make an input suitable for a model that requires evenly spaced observations. The caller owns this policy and must not pass masked warmup zeros to another algorithm as if they were meaningful samples.

**Relationship to matrix/statistics.** `series` exposes neither matrix storage nor rolling moments, normalization models, parameter fitting, covariance, inference, or streaming buffers. Composing it with other packages does not create implicit dependencies inside `series`.

**Non-goals.** Resampling, interpolation, timezone or trading-calendar treatment, timestamp reconstruction, market sessions, rolling statistics, probabilistic models, forward-looking features, streaming accumulators, and inference engines are intentionally outside this package. The small surface is a feature: each method solves exactly one reproducible mathematical question.

---

For the canonical Go signatures and per-symbol failure classes, consult [`series/api.go`](../series/api.go), [`series/types.go`](../series/types.go), and [`series/errors.go`](../series/errors.go). For executable examples and boundary regressions, see [`series/example_test.go`](../series/example_test.go), [`series/independent_counterexamples_test.go`](../series/independent_counterexamples_test.go), and [`series/numeric_regression_test.go`](../series/numeric_regression_test.go).
