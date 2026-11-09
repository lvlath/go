<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  File: docs/STATS.md
  Package: github.com/lvlath/go/stats

  Purpose:
    Mathematical, architectural, and operational guide to stable streaming
    moments, deterministic partition merging, and immutable fitted scaling.
    It links implemented API contracts to their numeric limits, error laws,
    ownership guarantees, proofs, and executable usage patterns.

  Contract status:
    - All signatures and policies below describe the implemented stats package.
    - Numeric correctness claims distinguish exact mean accumulation from
      compensated approximate second central moments.
    - No implicit imputation, holdout refitting, magic epsilon, or partial result
      publication is permitted.
    - Package GoDoc is authoritative for the public API and error precedence.

  Scope:
    - Welford updates and Chan-Golub-LeVeque merges.
    - Exact binary sum and scaled, compensated M2 storage.
    - Population/sample summaries, constant-feature policies, fitted scaling.
    - Matrix and series interoperability without import cycles.
    - Independent oracles, error classification, concurrency, and benchmarks.

  License:
    AGPL-3.0-only. See ../LICENSE.
-->

# Statistical Moments and Fitted Standardization

> **Package:** `github.com/lvlath/go/stats` · **Core:** Welford / Chan–Golub–LeVeque · **Numerics:** extended-range compensated central moments + exact binary mean sum · **Inference:** immutable, detached, train-only fitted scaling

`stats` is the numerical calibration boundary between **observations**, **partitioned reductions**, and **inference-time feature transformations**. It solves three related—but deliberately separate—problems: calculating stable moments from finite scalar values, merging independently accumulated partitions without replaying their data, and fitting per-column standardization parameters **only on training data**.

The package's distinguishing feature is not a shortcut formula. It preserves the semantics of the results when direct `float64` intermediates are more fragile than the final values. For example, a sample variance may be representable even though the unnormalized sum of squared deviations exceeds `math.MaxFloat64`.

For exact public signatures and receiver laws, begin with [`stats/doc.go`](../stats/doc.go). For running demonstrations, see [`stats/example_test.go`](../stats/example_test.go). This guide develops the mathematics, implementation boundaries, operational hazards, and verification rationale behind those contracts.

---

## Contents

1. [What and why: the seven governing laws](#1-what-and-why-the-seven-governing-laws)
2. [Domain scope and architecture](#2-domain-scope-and-architecture)
3. [Public API and result semantics](#3-public-api-and-result-semantics)
4. [Mathematics: Welford and Chan](#4-mathematics-welford-and-chan)
5. [Numeric implementation: scaled M2 and exact mean](#5-numeric-implementation-scaled-m2-and-exact-mean)
6. [Representability and adversarial examples](#6-representability-and-adversarial-examples)
7. [Fitted standardization and constant-feature policies](#7-fitted-standardization-and-constant-feature-policies)
8. [Validation order and sentinel error law](#8-validation-order-and-sentinel-error-law)
9. [Ownership, nilability, concurrency, and determinism](#9-ownership-nilability-concurrency-and-determinism)
10. [Algorithmic pipeline and pseudocode](#10-algorithmic-pipeline-and-pseudocode)
11. [ASCII diagrams: numerical and inference boundaries](#11-ascii-diagrams-numerical-and-inference-boundaries)
12. [Complete runnable Go scenarios](#12-complete-runnable-go-scenarios)
13. [Integration with matrix, series, and DTW](#13-integration-with-matrix-series-and-dtw)
14. [Tests, independent oracles, and benchmark regimes](#14-tests-independent-oracles-and-benchmark-regimes)
15. [Pitfalls and practical recipes](#15-pitfalls-and-practical-recipes)
16. [Implementation source map](#16-implementation-source-map)
17. [Further reading](#17-further-reading)

---

## 1. What and why: the seven governing laws

A statistical primitive may appear correct on a small dataset while returning zero variance for a genuinely variable high-offset signal, overflowing in a reduction that should succeed, or quietly incorporating a holdout batch into its normalization parameters. `stats` is engineered around explicit laws that rule out these failure modes.

### Law 1 — Centered moments, never cancellation-prone raw moments

The variance calculation accumulates centered second moments. It does **not** use the difference between two large, nearly equal raw terms:

$$\operatorname{Var}(X)=\mathbb E[X^2]-\bigl(\mathbb E[X]\bigr)^2\quad\text{(an unsafe floating-point evaluation).}$$

The mathematical identity is correct in real arithmetic; the naive evaluation is unsafe in finite precision. `Moments.Update` uses Welford's recurrence and `Moments.Merge` uses Chan's parallel formula.

### Law 2 — Intermediate magnitude is not the published result domain

The accumulator stores a scaled, compensated approximation to the unnormalized second central moment \(M_2\). Its exponent is not limited by the binary64 range. `Summary(ddof)` divides by \(n-\text{ddof}\) **before** converting the variance to `float64`; fitting takes the square root before converting a standard deviation.

### Law 3 — Precise mean accumulation has an independent numerical route

A fixed-width signed binary superaccumulator holds the **exact sum of the input `float64` values**. This protects the final mean against cancellation that can defeat even a well-centered recurrent estimate. The mean is rounded once at the public materialization boundary. It does not imply that the internal \(M_2\) is computed with exact rational arithmetic.

### Law 4 — Mutation is transactional on failure

`Update` and `Merge` validate first, build candidate states, and publish only after a successful transition. A rejected `NaN`, counter overflow, invalid self-merge, or nil input cannot change a committed count, extremum, compensated value, or exact sum.

### Law 5 — Model fitting and inference never share an adaptation loop

`FitStandardizer` reads a training matrix, computes one immutable parameter vector per column, and publishes a model. `Transform` and `InverseTransform` only **read that model**. No heldout statistics are computed or persisted as a replacement for fitted parameters.

### Law 6 — Zero spread is defined by the numerical kernel, not an epsilon

The fitting code checks the internal second moment for zero. It does not compare a rounded public variance with a made-up threshold. With `UnitScaleConstant`, the fitted mean remains intact and the scale is exactly one: a new value outside the training constant maps to its **difference** from the training mean.

### Law 7 — Stable contracts have explicit ownership and error classes

Every rejected operation returns an `errors.Is`-compatible sentinel or a wrapped matrix-origin error. Successful result snapshots are detached. Mutable accumulators are **not** concurrent safe; a fitted model supports simultaneous read-only inference on stable input matrices.

---

## 2. Domain scope and architecture

### 2.1. What the package computes

| Input object | Operation | Output | Key invariant |
|:--|:--|:--|:--|
| Finite scalar `float64` | `Moments.Update` | Updated mutable accumulator | One committed observation or unchanged state on error |
| Two independent `Moments` | `Moments.Merge` | Updated destination | Chan merge, source unmodified, checked `uint64` count |
| Accumulated state | `Moments.Summary(ddof)` | `*Summary` | Finite result, Count > ddof, detached snapshot |
| Finite non-nil `[]float64` | `Summarize` / wrappers | `*Summary` | Left-to-right stream without input copies |
| Nonempty training `matrix.Matrix` | `FitStandardizer` / wrappers | `*ScaleFitResult` | Each column fitted independently, parameters copied |
| Inference `matrix.Matrix` | `Standardizer.Transform` | New `matrix.Matrix` | No refit, output is a detached `*matrix.Dense` |
| Standardized `matrix.Matrix` | `Standardizer.InverseTransform` | New `matrix.Matrix` | Fused reconstruction using frozen training mean/scale |

A matrix is interpreted as **rows = observations, columns = features**. `stats` does not infer timestamp cadence, sample weights, missing-data flags, or business dimensions from the matrix shape.

### 2.2. Dependency boundaries

```text
       +------------------+      +-------------------------+
       | core             |      | series                  |
       | Nilable contract |      | Values + Valid / windows|
       +--------+---------+      +------------+------------+
                ^                             |
                |                             | caller explicitly filters
                |                             v
       +--------+---------+          []float64 or matrix.Matrix
       | matrix           |                    |
       | Matrix / Dense   |--------------------+
       | ValidateNotNil   |                    |
       +--------+---------+                    |
                ^                              |
                | import                       |
       +--------+------------------------------v---+
       | stats                                       |
       | Moments -> Merge -> Summary                 |
       | FitStandardizer -> immutable Standardizer   |
       | Transform / InverseTransform                |
       +---------------------------------------------+
                             |
                             | caller may forward standardized data
                             v
                  dtw.Align / dtw.AlignMatrix / ML
```

Actual dependency direction: `stats` imports `matrix` for `Matrix`/`Dense` operations and `core` for its `Nilable` assertion. `matrix` already depends on `core`; **`matrix` does not import `stats`**. `series` and `dtw` are integration partners at the **application** layer, not imports of `stats`.

### 2.3. Explicit non-goals

The package does not implement covariance/correlation, rolling deletion, quantiles, time-based weights, gap filling, imputation, training-split selection, parameter serialization, or automatic feature scaling inside DTW. `matrix` retains its own existing numerical operations; `stats` does not silently replace them.

A `Standardizer` has no public mutator or refit method. A zero-valued `Standardizer` is **unfitted**, not a useful identity transform.

---

## 3. Public API and result semantics

### 3.1. Canonical surface and wrappers

```go
// Scalar reductions and independently combinable accumulators.
func NewMoments() *Moments
func (m *Moments) Update(value float64) error
func (m *Moments) Merge(other *Moments) error
func (m *Moments) Summary(ddof int) (*Summary, error)
func Summarize(values []float64, ddof int) (*Summary, error)

// Explicit denominator wrappers; no second numerical kernel.
func SummarizePopulation(values []float64) (*Summary, error)
func SummarizeSample(values []float64) (*Summary, error)

// Column-wise training with explicit variance and constant-column policies.
func FitStandardizer(x matrix.Matrix, opts ScaleOptions) (*ScaleFitResult, error)
func FitPopulationStandardizer(x matrix.Matrix, policy ConstantPolicy) (*ScaleFitResult, error)
func FitSampleStandardizer(x matrix.Matrix, policy ConstantPolicy) (*ScaleFitResult, error)

// Immutable inference over matrices with the fitted feature count.
func (m *Standardizer) Transform(x matrix.Matrix) (matrix.Matrix, error)
func (m *Standardizer) InverseTransform(z matrix.Matrix) (matrix.Matrix, error)
```

These are real implemented signatures. `SummarizePopulation` delegates to `Summarize(values, 0)`, `SummarizeSample` to `Summarize(values, 1)`, and the two fit wrappers construct explicit `ScaleOptions` and delegate to `FitStandardizer`. They do **not** introduce alternative mathematics.

### 3.2. `Summary` — a finite materialized observation record

```go
type Summary struct {
    Count    uint64
    Mean     float64
    Variance float64
    Min      float64
    Max      float64
}
```

| Field | Meaning | Boundary |
|:--|:--|:--|
| `Count` | Exact number of successfully committed scalar observations | `uint64`, checked for overflow at each transition |
| `Mean` | Final arithmetic mean of original finite binary64 observations | Exact binary sum, final nearest-even conversion |
| `Variance` | Welford/Chan-based central variance divided by `Count - ddof` | Compensated approximate arithmetic; finite or error |
| `Min` | Minimum observed original value | Original observation comparison; not a normalized value |
| `Max` | Maximum observed original value | Original observation comparison; not a normalized value |

`Summary` has no retained accumulator pointer or slices. `Summary.IsNil()` reports pointer absence; it is not a test that `Count` or `Variance` is meaningful. The public `Variance` can equal zero either because the internal spread was zero **or** because a positive variance underflowed when converted to `float64`.

### 3.3. `Moments` — a bounded, mutable streaming state

The zero value is usable:

```go
var accumulator stats.Moments
err := accumulator.Update(17)
```

`NewMoments` is a convenience constructor, not a required initialization ritual. No public `Count()` or snapshot of private `M2` is exposed; use `Summary(ddof)` when enough observations have been committed.

The state consists of a checked count, compensated scaled center and `M2`, an exact fixed-width binary sum, and original-value extrema. These fields are private by design. Copying the value is not a synchronization mechanism; do not copy or merge a state concurrently with its mutation.

### 3.4. `ScaleOptions` and `ScaleFitResult`

```go
type ConstantPolicy uint8

const (
    RejectConstant ConstantPolicy = iota
    UnitScaleConstant
)

type ScaleOptions struct {
    DDOF           int
    ConstantPolicy ConstantPolicy
}

type ScaleFitResult struct {
    Model    *Standardizer
    Means    []float64
    Scales   []float64
    Constant []bool
}
```

`ScaleOptions{}` is valid and means **population variance + rejection of constant training features**. `ScaleFitResult.Means`, `.Scales`, and `.Constant` are independent, caller-owned slices, all indexed by the original training column order. `Model` retains **private copies** of these arrays. Mutating the public arrays does not modify future `Transform` outputs.

`Summary`, `ScaleFitResult`, and `Standardizer` all implement `core.Nilable` with a safe `IsNil() bool`; a non-nil but malformed result is *not* automatically treated as nil. Use nilability for pointer absence, not data validity.

### 3.5. Result publication

On success, returned result pointers are non-nil; `Transform`/`InverseTransform` return a new owning `*matrix.Dense` behind the `matrix.Matrix` interface. On **any** error, these operations return a nil result: no incomplete model or partially populated output is available for consumption.

---

## 4. Mathematics: Welford and Chan

### 4.1. Population and sample variance

For finite observations \(x_1,\ldots,x_n\), define the central-moment state:

$$\mu_n=\frac1n\sum_{i=1}^n x_i,\qquad M_{2,n}=\sum_{i=1}^n(x_i-\mu_n)^2.$$

![Central moments](https://latex.codecogs.com/svg.image?%5Cmu_n%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5En%20x_i%2C%5C%3BM_%7B2%2Cn%7D%3D%5Csum_%7Bi%3D1%7D%5En%28x_i-%5Cmu_n%29%5E2)

The materialized variance uses an explicit denominator:

$$\operatorname{Var}_{d}(x)=\frac{M_{2,n}}{n-d},\qquad d\in\{0,1\},\;n>d.$$

![Variance denominator](https://latex.codecogs.com/svg.image?%5Cmathrm%7BVar%7D_%7Bd%7D%28X%29%3D%5Cfrac%7BM_%7B2%2Cn%7D%7D%7Bn-d%7D%2C%5Cquad%20d%5Cin%5C%7B0%2C1%5C%7D)

When `ddof=0`, the batch is treated as the population. When `ddof=1`, the denominator is \(n-1\); for independent, identically distributed observations this is the usual unbiased estimator of population variance. The API does **not** infer sampling assumptions from the values.

### 4.2. Derivation of the Welford recurrence

Suppose the old state has size \(n\), mean \(\mu\), and central moment \(M_2\). Let \(\delta=x-\mu\) and \(n'=n+1\). Then the new mean is

$$\mu'=\frac{n\mu+x}{n'}=\mu+\frac{\delta}{n'}.$$

The new old-observation residuals shift by \(\mu-\mu'\), whose total cross term vanishes because \(\sum_i(x_i-\mu)=0\). Thus

$$\begin{aligned}M'_2&=M_2+n(\mu'-\mu)^2+(x-\mu')^2\\&=M_2+\delta^2\frac{n}{n+1}.\end{aligned}$$

![Welford update](https://latex.codecogs.com/svg.image?%5Cmu%5Cprime%3D%5Cmu%2B%5Cfrac%7B%5Cdelta%7D%7Bn%2B1%7D%2C%5Cquad%20M_2%5Cprime%3DM_2%2B%5Cfrac%7Bn%5Cdelta%5E2%7D%7Bn%2B1%7D)

In code, the first observation initializes `mean=x`, `M2=0`, `min=max=x`. Subsequent updates use the centered delta in extended-range arithmetic. This eliminates the large **raw-moment subtraction** responsible for classical cancellation. It does not make every floating-point operation exact.

### 4.3. Derivation of the Chan–Golub–LeVeque merge

Let partitions \(A,B\) have positive counts \(n_A,n_B\), means \(\mu_A,\mu_B\), and central sums \(M_{2,A},M_{2,B}\). Define \(\delta=\mu_B-\mu_A\) and \(n=n_A+n_B\).

The merged mean follows by weighting both partition means:

$$\mu=\mu_A+\delta\frac{n_B}{n}.$$

Each partition's residual squares can be shifted from its own mean to \(\mu\), producing the between-partition correction:

$$M_{2,A\cup B}=M_{2,A}+M_{2,B}+\delta^2\frac{n_A n_B}{n}.$$

![Chan merge](https://latex.codecogs.com/svg.image?M_%7B2%2CA%5Ccup%20B%7D%3DM_%7B2%2CA%7D%2BM_%7B2%2CB%7D%2B%5Cfrac%7Bn_A%20n_B%7D%7Bn_A%2Bn_B%7D%28%5Cmu_B-%5Cmu_A%29%5E2)

This formula combines partitions without replaying samples, but calculating `nA*nB` as `uint64` or `delta*delta` as an ordinary unscaled `float64` can overflow **before** the division. The actual kernel uses scaled count and delta representations to avoid that premature failure.

### 4.4. The non-associativity boundary

In exact arithmetic, disjoint partition grouping does not change the result. In floating arithmetic, a sequence of `Merge` operations is not guaranteed to produce bit-identical compensated `M2` for every different merge tree.

```text
             M(A + B + C + D)
                /      \
         Merge(A,B)   Merge(C,D)    fixed balanced tree
            /   \       /   \
           A     B     C     D

Other grouping:
  Merge(Merge(Merge(A,B),C),D)

Both implement Chan; rounded M2 may differ slightly.
```

For deterministic distributed computation, freeze the partition boundaries, the order **within** each partition, and the merge-tree topology. The final mean uses the exact binary sum, but the `M2` approximation still depends on the numerical history.

---

## 5. Numeric implementation: scaled M2 and exact mean

### 5.1. Why ordinary compensated `float64` is insufficient

Kahan summation can reduce rounding when accumulating representable terms, but it does not help if the term itself evaluates to `+Inf`:

```text
Finite input x:     1e154
Finite input y:    -1e154
Raw subtraction:    2e154                 (representable)
Squared delta:      4e308                 (not representable)
Desired M2/n:       may still be finite
```

The important distinction is **arithmetic-domain magnitude** versus **result-domain representability**. `ErrOverflow` is appropriate for an unrepresentable *published result*, not for an avoidable intermediate.

### 5.2. Compensated extended-exponent representation

The internal `scaledNumber` stores a number as

$$a=(h+\ell)\,2^e,$$

where `h` is the leading significand, `ℓ` is a small compensated residual, and `e` is a signed integer exponent independent of the IEEE-754 exponent field of a public `float64`.

![Scaled representation](https://latex.codecogs.com/svg.image?a%3D%28h%2B%5Cell%29%5C%2C2%5E%7Be%7D)

The implementation uses normalization, error-free-style two-sum residuals, `math.FMA` for product correction, and explicit division corrections. These operations improve precision and representable range, **but are floating-point approximations**, not proof of exact squared residuals.

`Moments.Update` computes \(\delta^2\,n/(n+1)\) in the scaled domain. `Merge` computes the between-partition cross term in the same domain, including counts above \(2^{53}\), where a direct conversion from `uint64` to `float64` would discard integer low bits.

### 5.3. Fixed-width exact binary sum for the mean

Every finite `float64` is an integer multiple of the smallest binary64 subnormal quantum:

$$x=k\cdot2^{-1074},\qquad k\in\mathbb Z.$$

![Binary64 quantum](https://latex.codecogs.com/svg.image?x%3Dk%5Ccdot%202%5E%7B-1074%7D%2C%5Cquad%20k%5Cin%5Cmathbb%7BZ%7D)

`exactSum` represents the signed sum of these integers using 34 little-endian `uint64` limbs (2176 bits). The width is sufficient for at most `math.MaxUint64` committed finite binary64 values, including extreme magnitudes and opposing signs.

`Update` adds the original input's decoded IEEE-754 significand and exponent into this fixed state. `Merge` adds the corresponding integer limbs. **Neither operation allocates a `big.Rat` per sample.** Only the final `Summary`/fit materialization converts the fixed sum to `big.Int`/`big.Rat`, divides by \(n\), and rounds the mean once to binary64.

Example of why this matters:

```text
Exact input sequence:
  +1e154
  +1
  -1e154

Raw float64 accumulation:
  (1e154 + 1) + (-1e154) -> 0

Exact binary sum:
  1e154 + 1 - 1e154 = 1

Correctly rounded mean:
  1/3 -> 0.33333333333333331
```

`exactSum` protects the **mean**, not the second moment. Internal Welford/Chan `M2` remains compensated approximate and should be verified using independent high-precision oracles.

### 5.4. Materialization boundaries

| Boundary | Internal operation | Public value |
|:--|:--|:--|
| Mean | Exact binary sum divided by exact count | One nearest-even `float64` rounding |
| Variance | Scaled approximate `M2 / (Count-ddof)` | Finite `float64` or `ErrOverflow` |
| Fitted scale | Scaled approximate `sqrt(M2 / (rows-DDOF))` | Strictly positive finite `float64` or `ErrOverflow` |
| Minimum/maximum | Compare original finite observations | Original observed `float64` |
| Transform | Finite difference/direct division or scaled fallback | Finite standardized `float64` or `ErrOverflow` |
| InverseTransform | `math.FMA(z, scale, mean)` | Finite reconstructed `float64` or `ErrOverflow` |

The package does not replace a positive but underflowed `Variance` with an epsilon, and it does not map an overflowed result to a successful infinite sentinel.

---

## 6. Representability and adversarial examples

### 6.1. Regression vectors and their purpose

| Input | `ddof` | Expected law | Threat model |
|:--|--:|:--|:--|
| `[1e12+1, 1e12+2, 1e12+3]` | 1 | `Variance == 1` | Large offset must not erase small spread |
| `[1e308, 1e308]` | 0 | `Mean == 1e308`, `Variance == 0` | Sum-of-inputs overflow must not corrupt mean |
| `[-1e154, 0, 1e154]` | 1 | Variance approximately `1e308` | Intermediate `M2` near `2e308` must not overflow |
| `[1e154, 1, -1e154]` | 0 | Mean rounds from exact `1/3` | Compensated center alone can lose small remaining sum |
| `[-math.MaxFloat64, math.MaxFloat64]` | 0 | `ErrOverflow` on `Summary` variance | Final variance is truly unrepresentable |
| Training constant `[5,5,5]`; holdout `[7]` | 0 | `UnitScaleConstant`: transformed value `2` | New value must not be forcibly zeroed |
| Training `[1,2]` | 0 | Mean `1.5`, scale `0.5` | Small training example for hand audit |

The first two rows are exactly checkable in binary64 for the stated observations. The extreme powers of ten are finite **binary64 approximations** to decimal literals, so comparison to an independent rational oracle should operate on their actual binary64 values, not their printed decimal spellings. Where appropriate, use a scale-aware error budget justified by the numerical kernel.

### 6.2. A representable variance with unrepresentable `M2`

For the symmetric three-point dataset \((-a,0,a)\), the mathematical mean is zero and

$$M_2=2a^2,\qquad\operatorname{Var}_{\text{sample}}=\frac{2a^2}{3-1}=a^2.$$

Set \(a\approx10^{154}\). The *unnormalized* \(M_2\approx2\times10^{308}\) is beyond the largest finite binary64 value; the *sample variance* \(\approx10^{308}\) is representable. The implementation deliberately preserves the scaled sum through the division.

### 6.3. A representable scale with unrepresentable variance

A fitted scale is a **standard deviation**, not a variance. For very large finite observations, it is possible that

$$\frac{M_2}{n-\text{DDOF}}>\text{MaxFloat64},\qquad\sqrt{\frac{M_2}{n-\text{DDOF}}}\le\text{MaxFloat64}.$$

Taking `math.Sqrt` of a `+Inf` variance would incorrectly refuse a valid scale. `FitStandardizer` instead calls the scaled square-root path **before** converting the result. Thus a `Summary` on the same data may return `ErrOverflow` while fitting may still succeed.

### 6.4. Underflow is not a constancy certificate

The public `Summary.Variance` is a rounded binary64 number. A positive mathematical spread can round to zero if its variance is below binary64's representable magnitude. Fitting classifies constancy against the internal scaled `M2` **before public variance rounding**. Do not interpret `summary.Variance == 0` as a universal equivalence to `fit.Constant[j] == true`.

### 6.5. Limitations of the accuracy claim

The sum used for `Mean` is exact over original finite binary64 inputs. Welford/Chan `M2` is *not* exact: it has compensated significand arithmetic and an extended exponent range. Its error depends on the data, update ordering, and merge tree. The existing adversarial tests provide empirical evidence, not a universal error theorem. Consumers requiring rigorous interval bounds, arbitrary-precision central moments, or reproducibility across arbitrary merge trees must use an explicitly different model.

---

## 7. Fitted standardization and constant-feature policies

### 7.1. Per-column training contract

Let \(X\in\mathbb R^{n\times d}\) be the training matrix. For each feature column \(j\), fitting computes

$$\mu_j=\frac1n\sum_{i=0}^{n-1}X_{ij},\qquad\sigma_j=\sqrt{\frac{M_{2,j}}{n-\text{DDOF}}}.$$

![Fitted scale](https://latex.codecogs.com/svg.image?%5Csigma_j%3D%5Csqrt%7B%5Cfrac%7BM_%7B2%2Cj%7D%7D%7Bn-%5Cmathrm%7BDDOF%7D%7D%7D)

Each feature is processed independently. Matrix rows are *observations*, not feature names or timestamps. The model stores only its fitted arrays; it does not retain the matrix or a per-row workspace.

### 7.2. Two constant-feature policies

| Training column | Policy | Fit outcome | Inference behavior |
|:--|:--|:--|:--|
| Internal `M2 == 0` | `RejectConstant` | `ErrZeroVariance`, nil result | No model published |
| Internal `M2 == 0` | `UnitScaleConstant` | Mean preserved; scale set to **1**; flag true | `z = x - trainingMean` |
| Internal `M2 > 0` | Either policy | Positive fitted scale; flag false | `z = (x-trainingMean)/scale` |

There is no epsilon cutoff. A column that varies by a small but representable amount must not be silently classified as constant. Also, a column *constant in training* may yield **nonzero** standardized values on new data.

### 7.3. Forward transformation without premature subtraction overflow

For a new observation \(x\) in feature \(j\), the intended affine map is

$$z_j=\frac{x_j-\mu_j}{\sigma_j}.$$

![Standardization](https://latex.codecogs.com/svg.image?z_j%3D%5Cfrac%7Bx_j-%5Cmu_j%7D%7B%5Csigma_j%7D)

The common path subtracts finite `float64` operands and divides directly. When `x-mean` becomes infinite even though both operands are finite, the implementation computes the centered ratio in the extended scaled domain, avoiding premature overflow. It does **not** unconditionally use `x/scale - mean/scale`, which can itself overflow or lose cancellation.

### 7.4. Inverse transformation with fused multiply-add

The inverse affine map is

$$x_j=z_j\sigma_j+\mu_j.$$

![Inverse transform](https://latex.codecogs.com/svg.image?x_j%3Dz_j%5Csigma_j%2B%5Cmu_j)

The implementation uses `math.FMA(z, scale, mean)`. A separate `z*scale` can overflow before an opposing `mean` restores a finite final value; FMA evaluates the combined operation with one final rounding. Inverse transform is *not* a promise of bit-identical round-trips for arbitrary data: forward rounding, fitted-scale rounding, and final FMA rounding still apply.

### 7.5. Input shape and zero-row inference

`FitStandardizer` requires \(n\ge1\), \(d\ge1\), and \(n>\text{DDOF}\). Inference requires `Cols()==d` but allows **zero rows**:

```text
Training:      4 x 3  -> Fit -> three learned means/scales
Heldout:       2 x 3  -> Transform -> detached 2 x 3
Empty batch:   0 x 3  -> Transform -> detached 0 x 3
Wrong shape:   2 x 2  -> Transform -> ErrShape; no matrix
```

`matrix.NewDense(0, d)` supports this structural result. A `0×d` inference matrix is not the same as a nil matrix, a `0×0` training dataset, or a model with zero fitted features.

---

## 8. Validation order and sentinel error law

### 8.1. Package-owned sentinels

All package-specific errors are exported `var ErrXxx = errors.New(...)` values. Classify with `errors.Is`; never parse human-readable `Error()` text.

| Sentinel | Precise originating condition | Partial publication |
|:--|:--|:--|
| `ErrInvalidOption` | Unsupported `ddof` or policy; self-merge | None |
| `ErrNilInput` | Nil accumulator receiver/source, nil slice, nil or conventionally typed-nil matrix, nil/unfitted model | None |
| `ErrEmptyInput` | Non-nil empty slice or zero-extent *training* matrix | None |
| `ErrShape` | Negative/unrepresentably large matrix dimensions; inference feature mismatch | None |
| `ErrNonFinite` | An input observation/cell is NaN or either infinity | None |
| `ErrInsufficientSamples` | Count ≤ ddof or training rows ≤ DDOF | None |
| `ErrZeroVariance` | Internal zero `M2` in a training column under `RejectConstant` | None |
| `ErrOverflow` | Count exceeds `MaxUint64`; published numeric mean, variance, scale, or transformed cell cannot be represented | None |

Underlying `matrix.Matrix.At`, `matrix.Matrix.Set`, or `matrix.NewDense` errors are preserved rather than rewritten into unrelated stats sentinels. Matrix nil validation specifically joins the matrix-origin nil error with `stats.ErrNilInput`, so `errors.Is` can recognize both layers.

### 8.2. Operation-specific precedence

A general principle such as “options before data” does **not** mean every function performs a global pre-scan. The implemented order matters:

| Operation | Validation and execution order |
|:--|:--|
| `Update` | nil receiver → nonfinite value → count overflow → candidate transition → commit |
| `Merge` | nil receiver → self-merge → nil source → checked count sum → empty identities or Chan transition → commit |
| `Summary` | ddof → nil receiver → insufficient count → finite materialization |
| `Summarize` | ddof → nil slice → empty slice → left-to-right `Update` scan → summary sufficiency/materialization |
| `FitStandardizer` | `ScaleOptions` → nil/typed-nil → matrix shape/product → empty shape → row-major `At`/finiteness scan → row sufficiency → column-wise mean/constant/scale → publication |
| `Transform` / `InverseTransform` | nil/unfitted model → matrix nil/shape → feature count → full row-major `At`/finiteness scan → `NewDense` → second row-major read/calculation/`Set` → publication |

**Important consequences:** an invalid `ddof` wins over a nil input in `Summarize`/`Summary`; a nonfinite value encountered by `Summarize` wins over insufficient count when the scan has not yet finished; a nonfinite training cell is checked before the sample-count sufficiency test. An `At` error in an earlier cell may occur before a later nonfinite value. During inference, data may be re-read and a custom matrix can return errors on its second pass; there is **no snapshot isolation**.

### 8.3. Examples of sentinel-safe code

```go
package main

import (
	"errors"
	"fmt"

	"github.com/lvlath/go/stats"
)

func main() {
	readings := []float64{2, 4, 6}
	summary, err := stats.SummarizeSample(readings)
	switch {
	case errors.Is(err, stats.ErrInsufficientSamples):
		fmt.Println("need more samples")
	case errors.Is(err, stats.ErrNonFinite):
		fmt.Println("nonfinite observation")
	case errors.Is(err, stats.ErrOverflow):
		fmt.Println("unrepresentable statistic")
	case err != nil:
		fmt.Println("other failure:", err)
	default:
		fmt.Printf("mean=%.0f variance=%.0f\n", summary.Mean, summary.Variance)
	}
	// Output: mean=4 variance=4
}
```

No error result may be treated as an implicitly useful partial `Summary` or partially fitted model.

---

## 9. Ownership, nilability, concurrency, and determinism

### 9.1. Ownership table

| Artifact | Storage and mutability | May caller mutate? | Concurrency rule |
|:--|:--|:--|:--|
| `Moments` | Bounded mutable state, no references to input samples | Via `Update` / `Merge` only | Shared mutations require synchronization |
| `Summary` | Independent scalar fields | Yes | Caller synchronizes concurrent writes |
| `ScaleFitResult.Means/Scales/Constant` | Detached mutable slices, separate from model storage | Yes | No effect on fitted model; synchronize shared slice writes |
| `ScaleFitResult.Model` | Pointer to immutable privately owned arrays | Do not mutate internals (not exported) | Parallel read-only inference supported |
| Training matrix | Caller-owned, progressively read during fitting | Not during the call | No concurrent mutation during fit |
| Inference matrix | Caller-owned, read twice during transformation | Not during the call | No concurrent mutation during transform |
| Output `matrix.Matrix` | New, detached `*matrix.Dense` | Yes, through `Set` | Concurrent writes need synchronization |

There is no hidden “input copy” at fit or inference. The input must remain stable while being scanned. Even if the fitted model is safe for concurrent use, this does not make an arbitrary custom `Matrix` safe for concurrent mutation.

### 9.2. Nilable convention

```go
var _ core.Nilable = (*stats.Summary)(nil)
var _ core.Nilable = (*stats.Standardizer)(nil)
var _ core.Nilable = (*stats.ScaleFitResult)(nil)
```

`IsNil()` is defined for these three pointer-backed artifacts and returns `receiver == nil`. A non-nil but invalid result, or an unfitted `Standardizer{}`, still returns `false`. The nil check does not certify mathematical validity.

The input `matrix.Matrix` interface can hold a typed-nil pointer. `stats` delegates that case to `matrix.ValidateNotNil`, which recognizes conforming `core.Nilable` implementations. A custom pointer-based matrix type should implement `IsNil()`; no reflection-based fallback is promised.

### 9.3. Deterministic ordering

`Summarize` visits indices ascending. `FitStandardizer` and both inference methods visit rows ascending and columns ascending within each row. Fitted feature publication follows ascending column order. No map iteration determines numeric order. For distributed moments, determinism additionally requires a **fixed external merge tree**.

### 9.4. Transactional accumulation, not thread-safe mutation

`Update` and `Merge` assemble a candidate and publish a complete new state only if validation succeeds. This is **error atomicity**, not a claim that simultaneous updates are linearizable. Use separate per-worker accumulators, wait for worker completion, and merge in a controlled order.

### 9.5. Immutable concurrent inference

The model keeps private `means`, `scales`, and `constant` arrays; inference methods allocate per-call output matrices. Two goroutines can call `Transform`/`InverseTransform` on the **same fitted model** with distinct stable inputs without sharing a mutable model workspace. Their outputs do not alias each other or the model's parameter slices.

---

## 10. Algorithmic pipeline and pseudocode

### 10.1. Stateful Welford update

```text
Update(m, x):
    if m == nil: return ErrNilInput
    if x is not finite: return ErrNonFinite
    if m.count == MaxUint64: return ErrOverflow

    candidate = valueCopy(m)
    if candidate.count == 0:
        initialize mean=x, M2=0, min=x, max=x, count=1
        exactSum.addFloat(x)
    else:
        n = scaledFromCount(m.count)
        nNext = scaledFromCount(m.count+1)
        delta = scaled(x) - m.center
        candidate.center = m.center + delta/nNext
        candidate.M2 = m.M2 + delta*delta*n/nNext
        candidate.exactSum.addFloat(x)
        candidate.count = m.count+1
        update original extrema

    m = candidate                     // the only publication point
    return nil
```

Time \(O(1)\) in the number of observations (fixed 34-word exact accumulator), auxiliary space \(O(1)\). No full sample history or per-sample `big.Rat` allocation.

### 10.2. Chan merge

```text
Merge(destination, source):
    reject nil destination / self-merge / nil source
    reject if source.count > MaxUint64-destination.count
    if source.count == 0: return nil
    if destination.count == 0:
        destination = valueCopy(source)
        return nil

    nA, nB = scaled counts
    n = scaled(destination.count + source.count)
    delta = source.center - destination.center
    candidate = valueCopy(destination)
    candidate.center = destination.center + delta*nB/n
    candidate.M2 = destination.M2 + source.M2 + delta*delta*nA*nB/n
    candidate.exactSum += source.exactSum
    candidate.count = destination.count + source.count
    candidate.min = min(destination.min, source.min)
    candidate.max = max(destination.max, source.max)
    destination = candidate            // source remains unchanged
    return nil
```

The cross term uses extended-range products and scaled counts; arithmetic cannot be naively reordered into a raw `uint64` multiplication.

### 10.3. Summary materialization

```text
Summary(m, ddof):
    validate ddof in {0,1}
    validate m and m.count > ddof
    denominator = scaledFromCount(m.count - ddof)
    if internal M2 has negative leading part: ErrOverflow
    mean = exactSum.meanFloat(m.count)
    variance = (scaledM2 / denominator).toFloat()
    publish *Summary{Count,Mean,Variance,Min,Max}
```

The distinction between the exact mean sum and approximate scaled variance must remain visible in both documentation and high-precision tests.

### 10.4. Fitting and inference stages

```text
FitStandardizer(train, opts):
    validate all options before reading the input matrix
    validate matrix presence, dimensions, and nonempty shape
    allocate one Moments per feature column
    for each row, then each column:
        read At(row,col), reject error / nonfinite, Update accumulator[col]
    check rows > DDOF
    for each feature in column order:
        materialize exact-binary-input mean
        if internal M2 == 0:
            RejectConstant -> ErrZeroVariance
            UnitScaleConstant -> scale=1, flag=true
        else:
            scale = sqrt(scaledM2/(rows-DDOF)) in extended arithmetic
    copy private arrays into immutable Standardizer
    publish ScaleFitResult with separate detached metadata arrays

Transform(model, batch):
    validate model, input matrix, feature count
    validate every input cell in row-major order
    allocate detached Dense of the same shape
    read input cells again and store (x-mean)/scale
    publish only when every cell succeeds

InverseTransform(model, z):
    same validation and two-pass input law
    store FMA(z,scale,mean) into a detached Dense
    publish only when every cell succeeds
```

**Complexity:** Fit \(O(nd)\) time and \(O(d)\) accumulator/model storage, where \(n\) is training rows and \(d\) features. Transform/inverse \(O(nd)\) time (two reads of each cell) and \(O(nd)\) output storage; no model-wide inference workspace is retained. Custom `Matrix.At` costs are additive and may change effective runtime.

---

## 11. ASCII diagrams: numerical and inference boundaries

### 11.1. Why M2 must survive until normalization

```text
             INPUT: [-1e154, 0, +1e154]       ddof = 1
                             |
                             v
                     Welford / Chan
                             |
                             v
                 M2 approx 2e308
                   (beyond MaxFloat64)
                             |
                    +--------+--------+
                    |                 |
                    v                 v
             Naive float64       Scaled M2
             materialization     retained in (hi,lo,e)
                    |                 |
                    v                 v
                 +Inf           divide by (3-1)
                    |                 |
                    X                 v
                              variance approx 1e308
                                      |
                                      v
                              finite Summary.Variance
```

### 11.2. Transactional merge and caller-controlled ordering

```text
                 Partition A                  Partition B
             (independent worker)         (independent worker)
                     |                           |
               Update stream               Update stream
                     |                           |
                     v                           v
                Moments A                    Moments B
                     |                           |
                     +------------+--------------+
                                  |
                             Merge(A,B)
                                  |
                         validation gates
                       nil / self / count
                                  |
                      +-----------+------------+
                      |                        |
                    error                    success
                      |                        |
                      v                        v
             A and B unchanged       candidate = A + B
                                               |
                                               v
                                         one commit to A
                                               |
                                               v
                                        B remains unchanged
```

### 11.3. Training / holdout separation

```text
     TRAINING MATRIX (n x d)
       [ row 0: f0 f1 ... fd ]
       [ row 1: f0 f1 ... fd ]
                 |
          FitStandardizer
                 |
        +--------v---------+
        | private means[d]  |  immutable after successful fit
        | private scales[d] |
        | constant mask[d]  |
        +--------+---------+
                 |
        +--------+--------------------------+
        |                                   |
        v                                   v
  HOLDOUT (m x d)                     PRODUCTION (k x d)
  Transform(...)                      Transform(...)
        |                                   |
        v                                   v
  detached (m x d)                     detached (k x d)

      NO arrows from holdout/production back to FitStandardizer.
      NO mutation of private fitted parameter arrays during inference.
```

### 11.4. Constant feature is not constant future data

```text
Training feature: [5, 5, 5]
    mean = 5
    M2   = 0

RejectConstant        -> ErrZeroVariance, no model
UnitScaleConstant     -> mean=5, scale=1, Constant=true

Heldout input: [5, 7, 3]
Transform:     [0, 2, -2]

The latter deviations are information, not values to be forced to zero.
```

---

## 12. Complete runnable Go scenarios

All examples below use the public API and check errors. `stats/example_test.go` contains five deterministic `// Output:` examples covering population/sample summaries, long-offset updates, extreme merging, holdout scaling, and fused inverse arithmetic. The standalone examples here explain adjacent integration decisions more fully.

### 12.1. Scenario A — sensor calibration on a large baseline

Use a streaming accumulator for ordered measurements. Materialize the *sample* result only after three observations exist:

```go
package main

import (
	"errors"
	"fmt"

	"github.com/lvlath/go/stats"
)

func main() {
	readings := []float64{1e12 + 1, 1e12 + 2, 1e12 + 3}
	var moments stats.Moments

	for _, value := range readings {
		if err := moments.Update(value); err != nil {
			fmt.Println("update error:", err)
			return
		}
	}

	summary, err := moments.Summary(1)
	if err != nil {
		fmt.Println("overflow:", errors.Is(err, stats.ErrOverflow))
		return
	}
	fmt.Printf("count=%d mean=%.0f sample variance=%.0f\n",
		summary.Count, summary.Mean, summary.Variance)

	// Output: count=3 mean=1000000000002 sample variance=1
}
```

**Interpretation:** the small spread remains observable despite the \(10^{12}\) offset. Input observations remain unchanged. For finite small batches that do not require streaming, `stats.SummarizeSample(readings)` is the corresponding convenience facade.

### 12.2. Scenario B — merge independent telemetry partitions

A parallel reduction should never share a concurrently mutable `Moments`. Here each partition is completed independently, then merged deterministically:

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/stats"
)

func main() {
	first := stats.NewMoments()
	second := stats.NewMoments()

	for _, x := range []float64{-1e154, 0} {
		if err := first.Update(x); err != nil {
			fmt.Println(err)
			return
		}
	}
	if err := second.Update(1e154); err != nil {
		fmt.Println(err)
		return
	}
	if err := first.Merge(second); err != nil {
		fmt.Println(err)
		return
	}

	summary, err := first.Summary(1)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("count=%d mean=%.0f sample variance=%.2e\n",
		summary.Count, summary.Mean, summary.Variance)

	// Output: count=3 mean=0 sample variance=1.00e+308
}
```

**Interpretation:** the first accumulator owns the merged state; `second` is unchanged. Reversing the grouping of many partitions need not reproduce the exact same compensated variance bits, so keep the partition order fixed in reproducibility-sensitive systems.

### 12.3. Scenario C — fit once, transform heldout values, invert

This scenario deliberately includes a training-constant feature. The fit uses the population denominator and accepts constant columns only because the caller explicitly selects `UnitScaleConstant`.

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

func main() {
	train, err := matrix.NewDense(4, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	training := [4][2]float64{{10, 5}, {12, 5}, {14, 5}, {16, 5}}
	for row, features := range training {
		for col, value := range features {
			if err := train.Set(row, col, value); err != nil {
				fmt.Println(err)
				return
			}
		}
	}

	fit, err := stats.FitStandardizer(train, stats.ScaleOptions{
		DDOF:           0,
		ConstantPolicy: stats.UnitScaleConstant,
	})
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	holdout, err := matrix.NewDense(1, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	for col, value := range []float64{18, 7} {
		if err := holdout.Set(0, col, value); err != nil {
			fmt.Println(err)
			return
		}
	}

	z, err := fit.Model.Transform(holdout)
	if err != nil {
		fmt.Println("transform:", err)
		return
	}
	delta, err := z.At(0, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	restored, err := fit.Model.InverseTransform(z)
	if err != nil {
		fmt.Println("inverse:", err)
		return
	}
	novel, err := restored.At(0, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("train mean=%.1f constant=%t heldout z=%.1f restored=%.1f\n",
		fit.Means[1], fit.Constant[1], delta, novel)

	// Output: train mean=5.0 constant=true heldout z=2.0 restored=7.0
}
```

**Interpretation:** training and inference are different operations. `Transform` does not refit on the `holdout` matrix. The nonconstant column is scaled using the same previously learned parameters as the constant column's unit-scale transformation.

### 12.4. Scenario D — respecting causal validity masks before statistics

A `series.TransformResult` stores `Values` and `Valid` in corresponding positions. A synthetic warmup zero must **not** be included in a statistical batch. The selection step belongs to the caller, because `stats.Summarize` accepts a plain slice and does not know series validity semantics:

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/series"
	"github.com/lvlath/go/stats"
)

func main() {
	difference, err := series.Difference([]float64{2, 4, 4, 8}, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	selected := make([]float64, 0, len(difference.Values))
	for index, value := range difference.Values {
		if difference.Valid[index] {
			selected = append(selected, value)
		}
	}

	summary, err := stats.SummarizePopulation(selected)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("valid count=%d mean=%.0f\n", summary.Count, summary.Mean)
	// Output: valid count=3 mean=2
}
```

**Interpretation:** the valid computed difference of zero is kept, while the first invalid warmup zero is discarded. If timestamps were supplied separately, `series.Validate` and explicit gap policy must precede a decision to treat consecutive supplied indices as consecutive time steps.

---

## 13. Integration with matrix, series, and DTW

### 13.1. `matrix.Matrix` vs `*matrix.Dense`

The public interface is:

```go
type Matrix interface {
     Rows() int
     Cols() int
     At(row, col int) (float64, error)
     Set(row, col int, value float64) error
     Clone() Matrix
 }
```

`FitStandardizer`, `Transform`, and `InverseTransform` accept the interface, so custom matrix implementations can be used. The current fit and transformation kernels call `At` through the interface; no undocumented access to `*matrix.Dense` private storage occurs. For ordinary dense inputs, `At` is O(1), but a custom wrapper may have additional access costs or return its own errors.

A matrix may be structurally valid with zero rows, but *fitting* explicitly requires positive rows and columns. Inference allows zero rows with a matching positive feature count. The returned output is a new `*matrix.Dense`, not a borrowed view.

### 13.2. `series` is not automatically a `stats` input adapter

`series` validates timestamps and produces length-preserving causal transforms with `Valid` masks. `stats` does not import `series` and cannot infer which zeros are placeholders, which values span timestamp gaps, or which indices should be excluded. Correct composition is:

```text
 timestamped observations
         |
         v
  series.Validate                 <- explicit cadence / gap policy
         |
         v
  series.Difference / Lag / Returns
         |
         v
  Values + Valid                  <- keep both together
         |
         v
  caller selects valid observations
         |
         v
  stats.Summarize / Moments.Update
```

`series.WindowRanges` returns half-open **index** partitions. If a window contains invalid warmup values or crosses missing timestamp slots, that is a caller policy decision, not a silent `stats` filtering rule.

### 13.3. `dtw.AlignMatrix` and comparable feature units

The multivariate DTW local cost uses squared L2 distance across matrix columns. Without external scaling, a feature measured in very large units can dominate a feature measured in small units. A common application pipeline is:

```text
 training matrix -> FitStandardizer -> immutable Model
                                           |
       +------------------+----------------+
       |                  |
 query matrix        candidate matrix
       |                  |
       v                  v
  Model.Transform    Model.Transform
       |                  |
       +--------+---------+
                |
                v
        dtw.AlignMatrix
```

No `stats` import is required inside DTW, and DTW does not learn or freeze scaling parameters on behalf of its caller. Decide whether standardized Euclidean geometry matches the real-world meaning of your features before applying this pipeline.

---

## 14. Tests, independent oracles, and benchmark regimes

### 14.1. External testing law

All package tests use `package stats_test`. `test_helper_test.go` centralizes `errors.Is` checks, nil-state guards, and independent numeric/reference assertions; production code exposes no `*TestOnly` escape hatch. The test suite includes validation, medium-scale calculations, large-offset and extreme-range cases, merge-count overflow, constant-policy semantics, aliasing, matrix error propagation, and concurrent model inference.

### 14.2. Independent rational oracles

To verify the **mathematics**, reference calculations must be independent of Welford/Chan production helpers. Each *actual* binary64 input is converted to an exact rational, the mean and central sum are calculated separately at high precision, and the final expected value is rounded to binary64. The test compares observed results against an explicitly justified magnitude-sensitive error budget for approximate `M2`—not against a second call to `stats.Summarize`.

```text
Production path:  float64 -> scaled Welford/Chan + exactSum -> Summary
                                      ||
                                      || compare
                                      \/
Independent path: float64 bits -> rational mean -> rational M2
                                      -> divide by denominator
                                      -> reference rounding
```

The exact mean accumulation permits especially strong cancellation assertions. Approximate variance assertions must not silently assume universal bit-for-bit parity across different merge trees.

### 14.3. Adversarial regimes worth preserving

1. Large offset with small spread, including offset signs and permutations.
2. Mixed exponents, subnormal positive increments, and alternating cancellation.
3. Intermediate `M2` beyond MaxFloat64 with finite final variance.
4. True **final** overflow, which must return `ErrOverflow`.
5. `uint64` partition counts above `2^53`, including merge near `MaxUint64`.
6. Empty source/destination merge identities and non-mutating error paths.
7. A training feature with tiny positive variance whose public squared variance would underflow.
8. The same immutable model transforming distinct heldout batches concurrently.
9. Wrong feature width, typed-nil matrices, injected `At` failures, and no partial output.

### 14.4. Benchmarks that isolate meaningful work

| Regime | Measured operation | Setup belongs outside `b.N` |
|:--|:--|:--|
| Scalar stream | `Moments.Update` per value | Prepared deterministic observation stream |
| Partition joining | `Moments.Merge` | Fully built independent source states and reset destination |
| Feature training | `FitStandardizer` | A valid `n×d` training matrix |
| Immutable inference | `Standardizer.Transform` | Fitted model and stable heldout matrix |
| Heavy input shapes | \(n=10^6\) and \(d\in\{1,16,128\}\) | Check actual memory/time budget before running |

Benchmark names and recorded results must state the real shape, CPU, Go toolchain, and whether setup includes allocations. Do not claim zero allocations for `Summary` or matrix output construction merely because scalar `Update` can avoid per-sample allocations. The complete repository gates remain `go test`, `go test -race`, `go vet`, and the configured linter; reported PASS results must come from actual executions.

---

## 15. Pitfalls and practical recipes

### 15.1. Incorrect approaches and their replacements

| Pitfall | Why it fails | Correct approach |
|:--|:--|:--|
| `meanSq - mean*mean` | Large nearly equal terms cancel | Welford/Chan centered `M2` |
| Raw `float64` accumulator for `M2` | Intermediate overflows before dividing by degrees of freedom | Keep scaled `M2` through final division |
| `float64(nA*nB)` | The integer product may overflow *before* float conversion | Multiply independently represented scaled counts |
| `math.Sqrt(summary.Variance)` for fitting | Public variance can overflow when scale remains finite | Fit from internal scaled `M2` |
| `summary.Variance <= 1e-5` means constant | A business epsilon erases genuine variation | Use internal zero `M2` classification and explicit policy |
| Force every constant-feature inference to zero | Novel heldout values lose their deviation | `UnitScaleConstant` with scale 1, retain training mean |
| Refit on test/production batch | Data leakage and inconsistent feature coordinates | Reuse fitted immutable `Model` |
| `z*scale + mean` in two operations | Product may overflow before cancellation | `math.FMA` inverse |
| Use `x/scale - mean/scale` unconditionally | Either quotient may overflow despite finite final result | Direct centered division, scaled fallback on overflow |
| Assume `Moments` mutation is goroutine-safe | Candidate commit is not a synchronization primitive | Per-worker accumulators, serialized merge |
| Mutate source matrix between fit/transform scans | Results are not a coherent snapshot | Hold input stable across the complete call |
| Feed `series.Values` without `Valid` | Warmup placeholder zeros contaminate statistics | Filter or segment explicitly using mask and timestamp policy |
| Interpret `ErrOverflow` as invalid observation | Overflow may occur only during result materialization | Distinguish `ErrNonFinite` from `ErrOverflow` |
| Use reflection or string-matched error text | Violates UES nil/error contracts | `core.Nilable`, `matrix.ValidateNotNil`, `errors.Is` |

### 15.2. Choosing the right facade

```text
Already have one finite slice?
    ├── Population convention -> SummarizePopulation(values)
    └── Sample convention     -> SummarizeSample(values)

Data arrives incrementally or from workers?
    ├── Per-worker -> Moments.Update
    ├── Join after completion -> Moments.Merge
    └── Report -> Moments.Summary(ddof)

Need stable coordinates for numeric features?
    ├── Training matrix -> FitStandardizer
    ├── Constant feature -> choose explicit ConstantPolicy
    ├── Heldout matrix -> fit.Model.Transform
    └── Reconstruction -> fit.Model.InverseTransform
```

### 15.3. Production deployment checklist

Before production use, ensure the input domain is finite; the variance denominator reflects the actual statistical question; a shared mutable accumulator is not updated concurrently; the partition merge order is stable; input matrix dimensions are consistent; the fit split is exclusively training data; parameter vectors are not mistaken for mutable model internals; and holdout/production results use the original frozen model. Store model provenance and feature order in the application layer.

---

## 16. Implementation source map

| File | Canonical responsibility |
|:--|:--|
| [`stats/doc.go`](../stats/doc.go) | Package charter, limits, and public contract laws |
| [`stats/api.go`](../stats/api.go) | `Summarize`, `FitStandardizer`, and explicit convenience wrappers |
| [`stats/types.go`](../stats/types.go) | Moments state, public result structures, constant policy, nilability |
| [`stats/errors.go`](../stats/errors.go) | Stable `ErrXxx` sentinel identities |
| [`stats/validators.go`](../stats/validators.go) | Finite, options, nil, shape, and matrix-cell validation |
| [`stats/numeric.go`](../stats/numeric.go) | Compensated extended-exponent arithmetic and scale materialization |
| [`stats/exactsum.go`](../stats/exactsum.go) | Fixed-width exact binary sum and correctly rounded mean |
| [`stats/moments.go`](../stats/moments.go) | Error-atomic Welford `Update` and Chan `Merge` |
| [`stats/summary.go`](../stats/summary.go) | Public `Summary` materialization and scalar batch kernel |
| [`stats/standardizer.go`](../stats/standardizer.go) | Column-wise fitting and immutable parameter publication |
| [`stats/transform.go`](../stats/transform.go) | Forward/inverse affine inference and `FMA` path |
| [`stats/example_test.go`](../stats/example_test.go) | Five deterministic executable usage examples |
| [`stats/moments_test.go`](../stats/moments_test.go) | Core scalar validations and numeric regression anchors |
| [`stats/precision_oracle_test.go`](../stats/precision_oracle_test.go) | Independent rational Welford/Chan adversarial oracles |
| [`stats/standardizer_precision_test.go`](../stats/standardizer_precision_test.go) | Fitted-scale and affine numeric reference comparisons |
| [`stats/transform_errors_test.go`](../stats/transform_errors_test.go) | Matrix-origin failure propagation and no partial inference |
| [`stats/test_helper_test.go`](../stats/test_helper_test.go) | Explicit external package helpers, no reflection-heavy comparison |
| [`stats/bench_test.go`](../stats/bench_test.go) | Structured stream/merge/fit/transform benchmark regimes |

No `series`- or `dtw`-level adapter is implemented inside the `stats` package. The end-to-end composition decisions above belong to callers and their domain policy.

---

## 17. Further reading

- [`stats/doc.go`](../stats/doc.go): authoritative package charter and numeric-error laws.
- [`stats/example_test.go`](../stats/example_test.go): five public-API examples executed by `go test`.
- [`MATRICES.md`](MATRICES.md): matrix ownership, zero-shape and representation semantics.
- [`SERIES.md`](SERIES.md): validity masks, exact timestamp grids, and causal preprocessing.
- [`DTW.md`](DTW.md): scalar/multivariate alignment and feature-scale considerations.
- [`TUTORIAL.md`](TUTORIAL.md): cross-package learning path.
- [`FAQ_&_TIPS.md`](FAQ_%26_TIPS.md): error interpretation and usage pitfalls.
- [`lvlath_UES.md`](lvlath_UES.md): determinism, documentation, error, aliasing and testing governance.

**Final rule:** numerical stability is not a license to hide approximation. Preserve the exact input semantics where possible, keep the approximate centered moment honest, report true unrepresentable outputs as errors, and never let inference change what training learned.
