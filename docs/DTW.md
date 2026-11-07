<!--
  lvlath - Repository Documentation

  Purpose:
    This document is the repository-level specification, tutorial, and contract map
    for lvlath/dtw. It explains Dynamic Time Warping as implemented by the package:
    scalar alignment, precomputed local-cost alignment, matrix-backed multivariate
    alignment, deterministic path reconstruction, numeric policy, sentinel errors,
    result ownership, and operational usage patterns.

  Contract status:
    - Public facade behavior described here is part of the public contract.
    - Window, slope-penalty, memory-mode, and cost-policy semantics described here
      are part of the public contract.
    - Result ownership and nilability semantics described here are part of the public contract.
    - Deterministic backtracking tie-break rules described here are part of the public contract.
    - Sentinel-error classification rules described here are part of the public contract.
    - Any incompatible change must be explicit, documented, and versioned.

  Scope:
    - Scalar DTW over []float64 through Align.
    - DTW over precomputed local-cost matrices through AlignCostMatrix.
    - Matrix-backed multivariate DTW through AlignMatrix.
    - Legacy tuple-returning DTW wrapper for compatibility.
    - Optional deterministic path reconstruction.
    - Optional accumulated-cost and local-cost matrix artifacts.
    - Bounded alignment and certified exact fixed-length top-K window search.

  License:
    The lvlath repository is licensed under AGPL-3.0-only. See LICENSE.
-->

# 8. Dynamic Time Warping (DTW)

> **Package:** `lvlath/dtw` | **Focus:** deterministic temporal alignment, numeric discipline, matrix-backed cost surfaces, result ownership

`lvlath/dtw` is the deterministic alignment kernel for ordered signals that share a structural pattern but do not move through time at the same pace. It is built for production use cases where a caller must understand not only the final distance, but also whether an admissible path exists, which phases stretched, which policy walls were active, and which diagnostic matrices were intentionally published.

The package is not a generic “compare two slices” snippet. It is a contract-driven Dynamic Time Warping implementation with explicit validation, sentinel-classified failures, stable path selection, and matrix-backed integration for multivariate time-series pipelines.

---

## 8.1. Header & Opening

`lvlath/dtw` exposes general alignment facades, two explicit search facades, an envelope diagnostic, and one legacy wrapper:

- `Align` compares scalar `[]float64` sequences.
- `AlignCostMatrix` compares a caller-provided finite non-negative local-cost matrix.
- `AlignMatrix` compares multivariate time series stored as `matrix.Matrix`, with rows as time steps and columns as features.
- `AlignBounded` classifies an exact scalar alignment against an inclusive cost limit.
- `Search` returns global exact top-K fixed-length windows under squared scalar local costs.
- `LBKeoghSquared` reports a rounded diagnostic envelope score, not a standalone pruning certificate.
- `DTW` preserves the old tuple-returning API and delegates to the same scalar recurrence.

All canonical facades return `*Result`, a detached artifact that records accumulated distance, reachability under policy, path-tracking state, effective options, and optional matrix surfaces.

A correct DTW consumer should ask four questions: did validation pass, is the final endpoint reachable, was the path requested and published, and are matrix artifacts needed for explanation or downstream analysis?

---

## 8.2. What & Why: The Laws of DTW

### The Five Laws of `lvlath/dtw`

1. **Determinism Law**

   The general forward dynamic-programming loop is row-major: `i` ascending, then `j` ascending. Matrix local costs are materialized in row-major order. The selected predecessor is recorded during forward recurrence with strict diagonal, then up, then left priority; backtracking follows those codes without cost subtraction. Equal-cost alignments therefore publish one stable representative optimal path.

2. **Numeric Discipline Law**

   Scalar samples are finite by default. Matrix inputs for `AlignMatrix` must pass finite validation. Local costs must always be finite and non-negative. `+Inf` is the public no-path distance, but a full accumulated matrix may also contain an overflowed reachable prefix. Internally those states are separately classified; final optimal overflow returns `ErrOverflow`, never a false no-path result. `NaN` is never a valid local or exact distance.

3. **Policy-Gated Warping Law**

   `WithWindow(-1)` disables the Sakoe-Chiba band. `WithWindow(0)` enforces strict diagonal alignment. `WithWindow(w)` for `w > 0` admits only cells satisfying `|i-j| <= w`. `WithSlopePenalty(p)` adds finite non-negative cost to vertical and horizontal steps.

4. **Result Ownership Law**

   Canonical facades return a `Result` artifact. Returned paths and matrices are detached and caller-owned. `PathOrError` distinguishes nil result, unreachable result, and path-not-tracked state. `Clone` copies result artifacts for independent downstream mutation.

5. **Sentinel-First Error Law**

   Callers classify failures with `errors.Is`. The package uses sentinel errors such as `ErrNilInput`, `ErrEmptyInput`, `ErrNaNInf`, `ErrNegativeCost`, `ErrInvalidWindow`, `ErrInvalidPenalty`, `ErrNoPath`, and `ErrPathNotTracked`. Matrix-backed adapters preserve relevant matrix errors through wrapping or joining.

---

## 8.3. Domain Scope & Non-Goals

### 8.3.1. What the Package Solves

`lvlath/dtw` solves deterministic alignment for scalar signals, externally computed local-cost surfaces, and multivariate time series. It computes accumulated DTW distance, reachability under policy, one deterministic optimal path when requested, optional accumulated-cost matrix, and optional local-cost matrix.

### 8.3.2. What the Package Does Not Do

The general alignment facades do not provide automatic normalization, smoothing, resampling, interpolation, imputation, streaming DTW, approximate DTW, all-pairs batch DTW, exhaustive optimal-path enumeration, classifier calibration, probability-to-cost conversion, hidden feature weighting, snapshot isolation for caller-owned inputs, concurrent mutation support, metric guarantees, or `O(1)` exact DTW memory. The separate `Search` facade implements independent optional population z-normalization for fixed-length squared-scalar windows only.

`AlignMatrix` uses squared L2 row distance. It does not normalize columns and does not take square roots. Feature scaling is caller responsibility.

---

## 8.4. Mathematical Formulation

### 8.4.1. Path Model

For inputs of lengths `n` and `m`, a DTW path is a monotone sequence of zero-based output coordinates:

$$ P = \big((i_0,j_0),(i_1,j_1),\ldots,(i_k,j_k)\big) $$

For a reachable published path, the endpoint law is:

$$ (i_0,j_0) = (0,0), \qquad (i_k,j_k) = (n-1,m-1) $$

Every step consumes one or both sequences:

$$ (i_{t+1}-i_t,\;j_{t+1}-j_t) \in \{(1,1),(1,0),(0,1)\} $$

A diagonal step consumes both sequences together. A vertical or horizontal step represents temporal stretching.

### 8.4.2. Accumulated DP Matrix

The implementation uses a prefix DP matrix with one extra row and one extra column:

$$ D \in \mathbb{R}^{(n+1)\times(m+1)} $$

Boundary conditions:

$$ D(0,0)=0 $$

$$ D(i,0)=+\infty \quad \text{for } i>0 $$

$$ D(0,j)=+\infty \quad \text{for } j>0 $$

For ordinary cells:

$$ D(i,j)=c(i,j)+\min\Big(D(i-1,j-1),\;D(i-1,j)+p,\;D(i,j-1)+p\Big) $$

where `c(i,j)` is the local alignment cost and `p` is `SlopePenalty`.

### 8.4.3. Objective Function

For a path `P`, the accumulated path cost is:

$$ Cost(P)=\sum_{(i,j)\in P} c(i,j) + p\cdot H(P) $$

where `H(P)` is the number of non-diagonal moves:

$$ H(P)=\left|\{t \mid (i_{t+1}-i_t,\;j_{t+1}-j_t)\in\{(1,0),(0,1)\}\}\right| $$

The DTW distance is:

$$ DTW(a,b)=\min_{P\in\mathcal{P}_{n,m,w}} Cost(P) $$

where `\mathcal{P}_{n,m,w}` is the set of monotone endpoint-valid paths admitted by the active window policy.

### 8.4.4. Cost Laws

For scalar `Align`, the default local cost is:

$$ c(i,j)=|a_{i-1}-b_{j-1}| $$

`WithSquaredCost` changes scalar local cost to:

$$ c(i,j)=(a_{i-1}-b_{j-1})^2 $$

For `AlignCostMatrix`, the caller supplies the local-cost surface directly:

$$ c(i,j)=C_{i-1,j-1} $$

For `AlignMatrix`, rows are time steps and columns are features. The local cost is squared L2 row distance:

$$ c(i,j)=\left\|X_{i-1}-Y_{j-1}\right\|_2^2 $$

Expanded over feature dimension `d`:

$$ c(i,j)=\sum_{\ell=1}^{d}\left(X_{i-1,\ell}-Y_{j-1,\ell}\right)^2 $$

The implementation subtracts corresponding coordinates first, then accumulates squared differences with scaling to protect representable results from intermediate overflow and cancellation. It does **not** use the unstable norm/dot-product identity `x²+y²−2xy` as its numeric kernel.

### 8.4.5. Window Policy

The admissible cell domain is:

$$ A_w=\{(i,j)\mid 1\le i\le n,\;1\le j\le m,\;w=-1\;\lor\;|i-j|\le w\} $$

Cells outside `A_w` are not usable predecessor targets and are published as `+Inf` in accumulated state when the matrix is stored.

Important implementation note: the current kernels still scan the rectangular DP grid. The window restricts admissible states and skips scalar local-cost evaluation outside the band, but the general `Align`/matrix DP still scans an `O(n*m)` grid. The specialized fixed-length `Search` scorer visits only admitted band cells, costing `O(L*b)` per candidate. `AlignMatrix` still constructs its complete local-cost surface before window gating.

### 8.4.6. Complexity by Phase

Let `n` be first length or row count, `m` be second length or row count, `d` be feature dimension for `AlignMatrix`, `k` be number of options, and `L` be returned path length.

| Phase                   | `Align`                                    | `AlignCostMatrix`                                    | `AlignMatrix`                                         |
|:------------------------|:-------------------------------------------|:-----------------------------------------------------|:------------------------------------------------------|
| Option assembly         | `O(k)` time, `O(1)` space                  | `O(k)` time, `O(1)` space                            | `O(k)` time, `O(1)` space                             |
| Input validation        | `O(n+m)` when finite validation is enabled | `O(n*m)` while materializing and validating costs    | matrix finite validation plus local-cost construction |
| Local-cost construction | inside DP for admissible scalar cells      | caller-provided, materialized to flat `O(n*m)` slice | `O(n*m*d)` scaled per-row costs       |
| DP recurrence           | `O(n*m)` time                              | `O(n*m)` time                                        | `O(n*m)` after cost construction                      |
| Distance-only DP memory | `O(m)` rolling rows                        | `O(m)` rolling rows plus flat local costs            | `O(m)` rolling rows plus local matrix                 |
| Full matrix/path memory | `O(n*m)`                                   | `O(n*m)`                                             | `O(n*m)`                                              |
| Backtracking            | `O(L)` time, `O(L)` space                  | `O(L)` time, `O(L)` space                            | `O(L)` time, `O(L)` space                             |
| Optional local artifact | `O(n*m)` when requested                    | detached `O(n*m)` copy when requested                | local matrix artifact `O(n*m)` when requested         |

---

## 8.5. Public API & Result Contract

### 8.5.1. Public Facades

```go
func Align(a, b []float64, opts ...Option) (*Result, error)

func AlignCostMatrix(local matrix.Matrix, opts ...Option) (*Result, error)

func AlignMatrix(x, y matrix.Matrix, opts ...Option) (*Result, error)

func AlignBounded(ctx context.Context, a, b []float64, limit float64, opts ...Option) (*BoundedResult, error)

func Search(ctx context.Context, query, data []float64, opts SearchOptions) (*SearchResult, error)

func LBKeoghSquared(query, lower, upper []float64) (float64, error)

func DTW(a, b []float64, legacy *Options) (dist float64, path []Coord, err error)
```

### 8.5.2. Public Domain Types

```go
type MemoryMode int

const (
	NoMemory MemoryMode = iota
	TwoRows
	FullMatrix
)

type Coord struct {
	I int
	J int
}

type Path []Coord

type CostFunc func(ai, bj float64) (float64, error)

type Result struct {
	Distance  float64
	Reachable bool

	Path        Path
	PathTracked bool

	Window       int
	SlopePenalty float64
	MemoryMode   MemoryMode

	Accumulated *matrix.Dense
	LocalCost   *matrix.Dense
}

type Options struct {
	Window       int
	SlopePenalty float64
	ReturnPath   bool
	MemoryMode   MemoryMode
}
```

### 8.5.3. Result Methods

```go
func (r *Result) IsNil() bool

func (r *Result) IsFinite() bool

func (r *Result) PathOrError() (Path, error)

func (r *Result) Clone() *Result
```

### 8.5.4. Result-Domain States

| State                                     | Return shape                              | Meaning                                                                    | Caller action                                            |
|:------------------------------------------|:------------------------------------------|:---------------------------------------------------------------------------|:---------------------------------------------------------|
| Reachable distance, path not requested    | `Result`, `nil`                           | Alignment exists; distance is authoritative; path is intentionally absent. | Use `Distance`, `Reachable`, and policy fields.          |
| Reachable distance, path requested        | `Result`, `nil`                           | Alignment exists; path is authoritative.                                   | Use `Path` or `PathOrError`.                             |
| No admissible path, path not requested    | `Result`, `nil`                           | Final cell is unreachable; `Distance=+Inf`, `Reachable=false`.             | Treat as blocked by policy.                              |
| No admissible path, path requested        | `Result`, `ErrNoPath`                     | Unreachable state is known; path cannot be published.                      | Use `errors.Is(err, ErrNoPath)` and inspect `Reachable`. |
| Terminal optimal cost overflow           | `nil`, `ErrOverflow` joined with `ErrNaNInf` | Reachable optimal cost cannot be represented.                         | Treat as numerical failure, not unreachable.                  |
| Invalid scalar input/options/matrix input | `nil`, `error`                            | No valid result artifact exists.                                           | Fix input/policy; classify with `errors.Is`.             |
| Path not tracked but requested later      | `PathOrError` returns `ErrPathNotTracked` | Valid distance result; path was not requested.                             | Re-run with `WithReturnPath(true)`.                      |
| Nil result helper call                    | `PathOrError` returns `ErrNilResult`      | Caller used a nil result pointer.                                          | Fix caller state.                                        |

---

## 8.6. Options, Numeric Policy & Error Law

### 8.6.1. Functional Options

```go
func WithWindow(w int) Option

func WithSlopePenalty(p float64) Option

func WithReturnPath(on bool) Option

func WithReturnAccumulated(on bool) Option

func WithReturnLocalCost(on bool) Option

func WithMemoryMode(mode MemoryMode) Option

func WithValidateFinite(on bool) Option

func WithCostFunc(fn CostFunc) Option

func WithAbsoluteCost() Option

func WithSquaredCost() Option
```

Options are runtime policies. They do not mutate input slices or input matrices. They shape admissibility, publication, and local scalar cost selection.

### 8.6.2. Numeric Policy

| Value           | Scalar input             | Local cost              | Accumulated DP state               |
|:----------------|:-------------------------|:------------------------|:-----------------------------------|
| finite float64  | allowed                  | allowed if non-negative | allowed                            |
| negative finite | allowed as scalar sample | rejected as local cost  | not expected from valid policy     |
| `NaN`           | rejected by default      | rejected                | never valid                        |
| `+Inf`          | rejected by default      | rejected                | structural no-path **or reachable overflow** inside accumulated artifacts; final overflow is error |
| `-Inf`          | rejected by default      | rejected                | never valid                        |

`WithValidateFinite(false)` only disables scalar input finite checks. It does not allow invalid local costs and does not weaken matrix input validation for `AlignMatrix`.

### 8.6.3. Sentinel Errors

```go
var (
	ErrNilInput        = errors.New("dtw: nil input sequence")
	ErrEmptyInput      = errors.New("dtw: input sequences must be non-empty")
	ErrNilOptions      = errors.New("dtw: nil options")
	ErrNilOption       = errors.New("dtw: nil option")
	ErrBadInput        = errors.New("dtw: invalid input")
	ErrInvalidWindow   = errors.New("dtw: invalid window")
	ErrInvalidPenalty  = errors.New("dtw: invalid slope penalty")
	ErrNaNInf          = errors.New("dtw: NaN or Inf encountered")
	ErrOverflow        = errors.New("dtw: optimal cost exceeds finite float64 range")
	ErrNoPath          = errors.New("dtw: no admissible warping path")
	ErrPathNeedsMatrix = errors.New("dtw: path tracking requires FullMatrix")
	ErrPathNotTracked  = errors.New("dtw: path was not tracked")
	ErrIncompletePath  = errors.New("dtw: path computation incomplete")
	ErrNilResult       = errors.New("dtw: nil result")
	ErrNilCostFunc     = errors.New("dtw: nil cost function")
	ErrNegativeCost    = errors.New("dtw: negative local cost")
	ErrInvalidOption   = errors.New("dtw: invalid search option")
	ErrShape           = errors.New("dtw: invalid search shape")
	ErrNonFinite       = errors.New("dtw: nonfinite search input")
	ErrZeroVariance    = errors.New("dtw: zero population variance")
	ErrBudget          = errors.New("dtw: search resource budget exceeded")
)
```

Rules:

- Use `errors.Is`.
- Do not parse error strings.
- Option failures occur before DP state allocation.
- Matrix-backed adapters preserve relevant matrix-layer errors.
- User errors returned by custom `CostFunc` are propagated.

### 8.6.4. Policy Effects

| Option                  | Changes input data? | Changes admissible cells? |   Changes local cost?    | Changes result publication? |
|:------------------------|:-------------------:|:-------------------------:|:------------------------:|:---------------------------:|
| `WithWindow`            |         No          |            Yes            |            No            |             No              |
| `WithSlopePenalty`      |         No          |            No             |            No            |             No              |
| `WithReturnPath`        |         No          |            No             |            No            |             Yes             |
| `WithReturnAccumulated` |         No          |            No             |            No            |             Yes             |
| `WithReturnLocalCost`   |         No          |            No             |            No            |             Yes             |
| `WithMemoryMode`        |         No          |            No             |            No            |       Storage policy        |
| `WithValidateFinite`    |         No          |            No             |            No            |      Validation policy      |
| `WithCostFunc`          |         No          |            No             | Yes, scalar `Align` only |             No              |
| `WithAbsoluteCost`      |         No          |            No             | Yes, scalar `Align` only |             No              |
| `WithSquaredCost`       |         No          |            No             | Yes, scalar `Align` only |             No              |

---

## 8.7. Algorithmic Architecture & Pseudocode

### 8.7.1. Shared Classified Recurrence

General scalar and precomputed-cost DTW use one classified predecessor recurrence. Each
DP cell has one of `dpUnreachable` (no structural path), `dpFinite` (finite optimal
prefix), or `dpOverflow` (reachable but unrepresentable prefix). A finite candidate
wins over overflow; strict finite minima preserve diagonal, then up, then left.

```text
STAGE 1  Validate existing API options/input in their established order.
STAGE 2  Allocate two rolling classified rows; allocate predecessors only for path.
         Full Accumulated and LocalCost artifacts are independent opt-in storage.
STAGE 3  If a full LocalCost artifact is requested, evaluate every n*m local cost
         exactly once, including cells outside the admitted warping window.
STAGE 4  For each ordinary admitted cell in row-major order:
           obtain its local cost from the cache or invoke callback once
           reject NaN, infinity and negative local costs
           diag = previous diagonal state
           up   = previous vertical state + slope penalty
           left = current horizontal state + slope penalty
           choose strict minimum (finite < overflow < unreachable)
           state = checked_add(chosen predecessor, local)
           save chosen predecessor code during forward computation when requested
           publish full accumulated cell if requested
STAGE 5  Classify final cell:
           unreachable -> Result{Distance:+Inf,Reachable:false}; ErrNoPath iff path
           overflow    -> nil, ErrOverflow joined with ErrNaNInf
           finite      -> Result{Distance:finite,Reachable:true}
STAGE 6  If path requested, follow recorded predecessor codes backwards and
         reverse once; NEVER re-evaluate the callback or subtract local cost.
```

### 8.7.2. Numeric Operations and Bounded Search

Finite inputs and local costs remain separate from overflow of their sums. Addition
uses the established float64 operation order, with a checked `MaxFloat64` boundary.
It is not an exact-real shortest-path oracle; deterministic strict comparison is
performed on the resulting floating states.

`AlignBounded` retains this general recurrence and then classifies the completed
result against an inclusive limit. `Search`, by contrast, uses a specialized equal-length
squared-local-cost scorer with two `O(b)` rows and actual admitted-column loops,
where `b=min(L,2r+1)` for nonnegative `r` or `L` when unconstrained. Both use the
same classified numeric and strict minimum primitives, not two conflicting DTW laws.

For the exact top-K window search, see [DTW_SEARCH.md](DTW_SEARCH.md) for the
independent oracle, certified bounds, memory budgets, cancellation, and workers.

---

## 8.8. ASCII Diagrams

### 8.8.1. Complex Policy-Banded DTW Lattice

This lattice models a 12-candle reference pattern against a 15-candle observed market window. The heavy path is the deterministic selected alignment. Dashed walls are policy-blocked cells outside `Window=4`.

```text
        Observed candles ───────────────────────────────────────────────▶
        L01  L02  L03  L04  L05  L06  L07  L08  L09  L10  L11  L12  L13  L14  L15
      ┏━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳┅┅┅┅┳┅┅┅┅┳┅┅┅┅┳┅┅┅┅┳┅┅┅┅┳┅┅┅┅┳┅┅┅┅┳┅┅┅┅┳┅┅┅┅┳┅┅┅┅┓
R01   ┃ ●  │ ○  │ ○  │ ○  │ ○  ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┃
R02   ┃ ○  │ ●━━│ ●  │ ○  │ ○  │ ○  ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┃
R03   ┃ ○  │ ○  │ ○  │ ●  │ ○  │ ○  │ ○  ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┃
R04   ┃ ○  │ ○  │ ○  │ ○  │ ●  │ ○  │ ○  │ ○  ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┃
R05   ┋ ## ┃ ○  │ ○  │ ○  │ ○  │ ●  │ ○  │ ○  │ ○  ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┃
R06   ┋ ## ┋ ## ┃ ○  │ ○  │ ○  │ ○  │ ●  │ ○  │ ○  │ ○  ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┃
R07   ┋ ## ┋ ## ┋ ## ┃ ○  │ ○  │ ○  │ ○  │ ●━━│ ●  │ ○  │ ○  ┋ ## ┋ ## ┋ ## ┋ ## ┃
R08   ┋ ## ┋ ## ┋ ## ┋ ## ┃ ○  │ ○  │ ○  │ ○  │ ○  │ ●  │ ○  │ ○  ┋ ## ┋ ## ┋ ## ┃
R09   ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┃ ○  │ ○  │ ○  │ ○  │ ○  │ ●  │ ○  │ ○  ┋ ## ┋ ## ┃
R10   ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┃ ○  │ ○  │ ○  │ ○  │ ○  │ ●  │ ○  │ ○  ┋ ## ┃
R11   ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┃ ○  │ ○  │ ○  │ ○  │ ○  │ ●  │ ○  │ ○  ┃
R12   ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┋ ## ┃ ○  │ ○  │ ○  │ ○  │ ○  │ ●━━│ ●  ┃
      ┗┅┅┅┅┻┅┅┅┅┻┅┅┅┅┻┅┅┅┅┻┅┅┅┅┻┅┅┅┅┻┅┅┅┅┻┅┅┅┅┻━━━━┻━━━━┻━━━━┻━━━━┻━━━━┻━━━━┻━━━━┛

Legend:
  ●  = selected path cell
  ○  = admissible non-path cell
  ## = blocked by window policy
  ━  = repeated-template or repeated-observed phase on the selected path

Interpretation:
  R02 aligns to both L02 and L03: the live accumulation phase is slower.
  R07 aligns to L08 and L09: the peak/cooling phase is stretched.
  R12 aligns to L14 and L15: the pullback lingers.
```

### 8.8.2. Boundary Matrix and Endpoint Law

```text
          Prefix/Observed axis ───────────────────────────────────▶

              ∅      B1       B2       B3      ...       Bm
          ╔══════╦════════╦════════╦════════╦════════╦════════╗
      ∅   ║  0   ║  +Inf  ║  +Inf  ║  +Inf  ║  ...   ║  +Inf  ║
          ╠══════╬════════╬════════╬════════╬════════╬════════╣
      A1  ║ +Inf ║  D11   ║  D12   ║  D13   ║  ...   ║  D1m   ║
      A2  ║ +Inf ║  D21   ║  D22   ║  D23   ║  ...   ║  D2m   ║
      A3  ║ +Inf ║  D31   ║  D32   ║  D33   ║  ...   ║  D3m   ║
     ...  ║ +Inf ║  ...   ║  ...   ║  ...   ║  ...   ║  ...   ║
      An  ║ +Inf ║  Dn1   ║  Dn2   ║  Dn3   ║  ...   ║  Dnm   ║
          ╚══════╩════════╩════════╩════════╩════════╩════════╝

Hard laws:
  D(0,0)       = 0
  D(i>0,0)    = +Inf
  D(0,j>0)    = +Inf
  Result.Path = zero-based coordinates over ordinary cells only

Reachable publication:
  first path point = {0,0}
  last path point  = {n-1,m-1}
```

### 8.8.3. Forward Predecessor Selection and Deterministic Witness

```text
                   Forward cell (i,j)
                           │
         ┌─────────────────┼──────────────────┐
         ▼                 ▼                  ▼
    D(i-1,j-1)        D(i-1,j)+p         D(i,j-1)+p
      diagonal          vertical             left
         │                 │                  │
         └─────────────────┼──────────────────┘
                           ▼
              strict minimum on classified
                accumulated predecessors
                           │
                           ▼
                 checked_add(best, local)
                           │
                           ▼
                  store predecessor code

Equal finite predecessor cost: diagonal -> up -> left.
Backtracking follows the saved code; it does not subtract local costs from D.
```

### 8.8.4. Strict Diagonal No-Path Cutoff

```text
A length = 6
B length = 9
Window   = 0

        B1   B2   B3   B4   B5   B6   B7   B8   B9
      ╔════╦════╦════╦════╦════╦════╦┅┅┅┅╦┅┅┅┅╦┅┅┅┅╗
A1    ║ ●  ║ ## ║ ## ║ ## ║ ## ║ ## ║ ## ║ ## ║ ## ║
A2    ║ ## ║ ●  ║ ## ║ ## ║ ## ║ ## ║ ## ║ ## ║ ## ║
A3    ║ ## ║ ## ║ ●  ║ ## ║ ## ║ ## ║ ## ║ ## ║ ## ║
A4    ║ ## ║ ## ║ ## ║ ●  ║ ## ║ ## ║ ## ║ ## ║ ## ║
A5    ║ ## ║ ## ║ ## ║ ## ║ ●  ║ ## ║ ## ║ ## ║ ## ║
A6    ║ ## ║ ## ║ ## ║ ## ║ ## ║ ●  ║ ## ║ ## ║ ✕  ║
      ╚════╩════╩════╩════╩════╩════╩┅┅┅┅╩┅┅┅┅╩┅┅┅┅╝

Required endpoint:
  A6/B9

Blocked because:
  |6 - 9| = 3 > 0

Result:
  Distance  = +Inf
  Reachable = false

If path tracking was requested:
  error satisfies errors.Is(err, ErrNoPath)
```

### 8.8.5. Matrix-Backed Local-Cost Pipeline

```text
Reference feature matrix X                 Observed feature matrix Y
rows = time steps                          rows = time steps
cols = feature dimensions                  cols = feature dimensions

┏━━━━━━━━━━━━━━━━━━━━━━┓                   ┏━━━━━━━━━━━━━━━━━━━━━━┓
┃ x00  x01  x02  ...   ┃                   ┃ y00  y01  y02  ...   ┃
┃ x10  x11  x12  ...   ┃                   ┃ y10  y11  y12  ...   ┃
┃ x20  x21  x22  ...   ┃                   ┃ y20  y21  y22  ...   ┃
┃ ...                  ┃                   ┃ ...                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━┛                   ┗━━━━━━━━━━━━━━━━━━━━━━┛
              │                                         │
              └──────────────┬──────────────────────────┘
                             ▼
          Local cost construction: C[i,j] = ||X_i - Y_j||²

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                         LocalCost C                       ┃
┣━━━━━━━━╦━━━━━━━━╦━━━━━━━━╦━━━━━━━━╦━━━━━━━━╦━━━━━━━━━━━━━━┫
┃        ║  Y0    ║  Y1    ║  Y2    ║  Y3    ║  ...         ┃
┣━━━━━━━━╬━━━━━━━━╬━━━━━━━━╬━━━━━━━━╬━━━━━━━━╬━━━━━━━━━━━━━━┫
┃ X0     ║ c00    ║ c01    ║ c02    ║ c03    ║              ┃
┃ X1     ║ c10    ║ c11    ║ c12    ║ c13    ║              ┃
┃ X2     ║ c20    ║ c21    ║ c22    ║ c23    ║              ┃
┃ ...    ║        ║        ║        ║        ║              ┃
┗━━━━━━━━╩━━━━━━━━╩━━━━━━━━╩━━━━━━━━╩━━━━━━━━╩━━━━━━━━━━━━━━┛
                             │
                             ▼
              DTW recurrence over finite non-negative C
                             │
                             ▼
        Result.Distance + Result.Path + optional matrix artifacts

Operational warning:
  no feature normalization happens inside AlignMatrix.
  if one column has larger scale, it dominates squared L2 costs.
```

---

## 8.9. Go Example Scenarios

These examples use deterministic production-style data. Each example follows the same pipeline: build input data, check construction errors, execute DTW, check algorithm errors, and consume `Result`.

### 8.9.1. Crypto OHLC Breakout Pattern Matching

This scenario compares a 12-candle reference breakout against a 15-candle live window. Each row is an OHLC vector. The path reveals where the live market stretched the template.

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

func main() {
	breakoutTemplate, err := matrix.NewPreparedDense(12, 4)
	if err != nil {
		fmt.Println("construct template:", err)
		return
	}

	err = breakoutTemplate.Fill([]float64{
		100.00, 100.40, 99.70, 100.10,
		100.10, 100.60, 99.90, 100.35,
		100.35, 101.00, 100.10, 100.85,
		100.85, 102.20, 100.70, 101.90,
		101.90, 104.30, 101.70, 103.80,
		103.80, 106.10, 103.20, 105.40,
		105.40, 106.40, 104.60, 105.90,
		105.90, 106.00, 104.80, 105.20,
		105.20, 105.60, 103.90, 104.30,
		104.30, 104.80, 103.20, 103.60,
		103.60, 104.10, 102.90, 103.20,
		103.20, 103.70, 102.70, 103.10,
	})
	if err != nil {
		fmt.Println("fill template:", err)
		return
	}

	liveWindow, err := matrix.NewPreparedDense(15, 4)
	if err != nil {
		fmt.Println("construct live window:", err)
		return
	}

	err = liveWindow.Fill([]float64{
		100.00, 100.30, 99.80, 100.05,
		100.05, 100.50, 99.90, 100.20,
		100.20, 100.80, 100.00, 100.55,
		100.55, 101.10, 100.30, 100.90,
		100.90, 102.40, 100.70, 102.00,
		102.00, 103.80, 101.70, 103.40,
		103.40, 105.50, 103.00, 104.80,
		104.80, 106.20, 104.30, 105.70,
		105.70, 106.50, 104.90, 105.80,
		105.80, 106.00, 104.70, 105.10,
		105.10, 105.50, 103.80, 104.40,
		104.40, 104.90, 103.30, 103.80,
		103.80, 104.20, 103.00, 103.30,
		103.30, 103.80, 102.80, 103.00,
		103.00, 103.50, 102.50, 102.90,
	})
	if err != nil {
		fmt.Println("fill live window:", err)
		return
	}

	res, err := dtw.AlignMatrix(
		breakoutTemplate,
		liveWindow,
		dtw.WithWindow(4),
		dtw.WithSlopePenalty(0.15),
		dtw.WithReturnPath(true),
		dtw.WithReturnAccumulated(true),
		dtw.WithReturnLocalCost(true),
	)
	if err != nil {
		fmt.Println("align:", err)
		return
	}

	fmt.Printf("distance=%.3f path-steps=%d\n", res.Distance, len(res.Path))
	fmt.Printf("anchors entry=%v ignition=%v peak=%v exit=%v\n", res.Path[0], res.Path[4], res.Path[7], res.Path[len(res.Path)-1])
	fmt.Printf("matrices accumulated=%dx%d local=%dx%d\n", res.Accumulated.Rows(), res.Accumulated.Cols(), res.LocalCost.Rows(), res.LocalCost.Cols())

	if res.Distance < 4.0 {
		fmt.Println("trade-filter=similar stretched breakout")
	} else {
		fmt.Println("trade-filter=reject pattern")
	}

	// Output:
	// distance=3.232 path-steps=15
	// anchors entry={0 0} ignition={3 4} peak={6 7} exit={11 14}
	// matrices accumulated=13x16 local=12x15
	// trade-filter=similar stretched breakout
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-DTW_Crypto_OHLC_Breakout_Pattern_Matching-blue?logo=go)](https://go.dev/play/p/e5CI7zuPTb0)


### 8.9.2. Voice Command Alignment over Acoustic Cost Surface

This scenario uses `AlignCostMatrix` because an upstream acoustic model already computed frame-to-frame costs. DTW consumes the cost surface directly and explains the stretched pronunciation.

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

func main() {
	localAcousticCost, err := matrix.NewPreparedDense(10, 14)
	if err != nil {
		fmt.Println("construct acoustic matrix:", err)
		return
	}

	err = localAcousticCost.Fill([]float64{
		0.03, 0.73, 1.31, 1.80, 2.38, 2.41, 2.90, 3.48, 4.06, 4.00, 4.58, 5.16, 5.10, 5.13,
		0.73, 0.02, 0.70, 1.28, 1.86, 1.80, 2.38, 2.96, 3.45, 3.48, 4.06, 4.55, 4.58, 4.61,
		1.31, 0.70, 0.04, 0.76, 1.25, 1.28, 1.86, 2.35, 2.93, 2.96, 3.45, 4.03, 4.06, 4.00,
		1.80, 1.28, 0.76, 0.01, 0.73, 0.76, 1.25, 1.83, 2.41, 2.35, 2.93, 3.51, 3.45, 3.48,
		2.38, 1.86, 1.25, 0.73, 0.02, 0.02, 0.73, 1.31, 1.80, 1.83, 2.41, 2.90, 2.93, 2.96,
		2.96, 2.35, 1.83, 1.31, 0.70, 0.73, 0.03, 0.70, 1.28, 1.31, 1.80, 2.38, 2.41, 2.35,
		3.45, 2.93, 2.41, 1.80, 1.28, 1.31, 0.70, 0.02, 0.76, 0.70, 1.28, 1.86, 1.80, 1.83,
		4.03, 3.51, 2.90, 2.38, 1.86, 1.80, 1.28, 0.76, 0.03, 0.04, 0.76, 1.25, 1.28, 1.31,
		4.61, 4.00, 3.48, 2.96, 2.35, 2.38, 1.86, 1.25, 0.73, 0.76, 0.02, 0.73, 0.76, 0.70,
		5.10, 4.58, 4.06, 3.45, 2.93, 2.96, 2.35, 1.83, 1.31, 1.25, 0.73, 0.02, 0.02, 0.03,
	})
	if err != nil {
		fmt.Println("fill acoustic matrix:", err)
		return
	}

	res, err := dtw.AlignCostMatrix(localAcousticCost, dtw.WithWindow(4), dtw.WithSlopePenalty(0.12), dtw.WithReturnPath(true), dtw.WithReturnLocalCost(true))
	if err != nil {
		fmt.Println("align:", err)
		return
	}

	stretchSteps := 0
	for i := 1; i < len(res.Path); i++ {
		if res.Path[i].I == res.Path[i-1].I || res.Path[i].J == res.Path[i-1].J {
			stretchSteps++
		}
	}

	fmt.Printf("command distance=%.2f\n", res.Distance)
	fmt.Printf("path-steps=%d stretch-steps=%d\n", len(res.Path), stretchSteps)
	fmt.Printf("anchors start=%v vault-core=%v end=%v\n", res.Path[0], res.Path[5], res.Path[len(res.Path)-1])
	fmt.Printf("local-shape=%dx%d\n", res.LocalCost.Rows(), res.LocalCost.Cols())

	if res.Distance < 1.0 && stretchSteps <= 4 {
		fmt.Println("decision=accepted with stretched pronunciation")
	} else {
		fmt.Println("decision=manual review")
	}

	// Output:
	// command distance=0.83
	// path-steps=14 stretch-steps=4
	// anchors start={0 0} vault-core={4 5} end={9 13}
	// local-shape=10x14
	// decision=accepted with stretched pronunciation
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-DTW_Voice_Command_Alignment-blue?logo=go)](https://go.dev/play/p/38atHI_TzoV)


### 8.9.3. Multivariate Vibration Signature Matching

This scenario compares a healthy spindle-impact reference signature with an observed signature containing an extra rise and peak frame.

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

func main() {
	referenceSignature, err := matrix.NewPreparedDense(14, 3)
	if err != nil {
		fmt.Println("construct reference:", err)
		return
	}

	err = referenceSignature.Fill([]float64{
		0.05, 0.10, 0.02,
		0.08, 0.13, 0.03,
		0.12, 0.18, 0.04,
		0.24, 0.32, 0.08,
		0.45, 0.55, 0.16,
		0.72, 0.80, 0.30,
		0.94, 0.88, 0.42,
		0.88, 0.70, 0.36,
		0.62, 0.48, 0.24,
		0.38, 0.31, 0.15,
		0.22, 0.22, 0.10,
		0.14, 0.18, 0.07,
		0.10, 0.15, 0.05,
		0.07, 0.12, 0.03,
	})
	if err != nil {
		fmt.Println("fill reference:", err)
		return
	}

	observedSignature, err := matrix.NewPreparedDense(17, 3)
	if err != nil {
		fmt.Println("construct observed:", err)
		return
	}

	err = observedSignature.Fill([]float64{
		0.04, 0.11, 0.02,
		0.07, 0.13, 0.03,
		0.09, 0.15, 0.03,
		0.13, 0.19, 0.05,
		0.22, 0.30, 0.08,
		0.33, 0.43, 0.11,
		0.47, 0.56, 0.17,
		0.70, 0.78, 0.29,
		0.90, 0.87, 0.41,
		0.96, 0.82, 0.43,
		0.87, 0.69, 0.35,
		0.63, 0.50, 0.25,
		0.39, 0.33, 0.16,
		0.24, 0.23, 0.11,
		0.15, 0.18, 0.07,
		0.10, 0.15, 0.05,
		0.07, 0.12, 0.03,
	})
	if err != nil {
		fmt.Println("fill observed:", err)
		return
	}

	res, err := dtw.AlignMatrix(referenceSignature, observedSignature, dtw.WithWindow(4), dtw.WithSlopePenalty(0.015), dtw.WithReturnPath(true), dtw.WithReturnLocalCost(true))
	if err != nil {
		fmt.Println("align:", err)
		return
	}

	stretchSteps := 0
	for i := 1; i < len(res.Path); i++ {
		if res.Path[i].I == res.Path[i-1].I || res.Path[i].J == res.Path[i-1].J {
			stretchSteps++
		}
	}

	fmt.Printf("vibration distance=%.4f\n", res.Distance)
	fmt.Printf("path-steps=%d stretch-steps=%d\n", len(res.Path), stretchSteps)
	fmt.Printf("anchors onset=%v peak=%v recovery=%v\n", res.Path[0], res.Path[8], res.Path[len(res.Path)-1])
	fmt.Printf("local-shape=%dx%d\n", res.LocalCost.Rows(), res.LocalCost.Cols())

	if res.Distance < 0.10 && stretchSteps <= 3 {
		fmt.Println("maintenance decision=signature match")
	} else {
		fmt.Println("maintenance decision=manual review")
	}

	// Output:
	// vibration distance=0.0776
	// path-steps=17 stretch-steps=3
	// anchors onset={0 0} peak={6 8} recovery={13 16}
	// local-shape=14x17
	// maintenance decision=signature match
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-DTW_Multivariate_Vibration_Signature-blue?logo=go)](https://go.dev/play/p/7zIwIMiHS1h)

---

## 8.10. Laws of Ownership, Concurrency & Partial Results

### 8.10.1. Ownership Law

`Result` is a detached artifact.

- `Result.Path` is caller-owned.
- `PathOrError` returns a copy of the path.
- `Result.Clone` copies path and Dense matrix artifacts.
- `Result.Accumulated` and `Result.LocalCost` are result-owned artifacts when present.
- Input slices and matrices are never mutated by the package.

### 8.10.2. Matrix Artifact Law

`Result.Accumulated` is the DP surface `D`. It is useful for heatmaps and diagnostics. It may contain `+Inf` both for structurally blocked and reachable-but-overflowed intermediate states; that numeric artifact alone does not encode the private class or saved predecessor codes.

`Result.LocalCost` is the **complete** local cost surface `C`, including cells outside `Window`. It is useful for checking feature scale, acoustic model behavior, embedding distances, and externally supplied cost surfaces. A pure deterministic callback is evaluated once per published cell.

Returned matrix artifacts are not live views into caller input.

### 8.10.3. Concurrency Law

The package is pure over caller-provided immutable inputs.

Concurrent independent calls are safe when each call receives inputs that are not mutated during execution. Concurrent mutation of input slices or matrices during `Align`, `AlignCostMatrix`, or `AlignMatrix` is unsupported and voids reproducibility.

Returned `Result` values are not internally synchronized. If callers mutate returned paths or matrices across goroutines, they must provide their own synchronization.

### 8.10.4. Partial-Result Law

Validation and option failures return `nil` result plus error. This prevents callers from accidentally consuming corrupted or uninitialized state.

No-path is different: it is a mathematically meaningful domain state. When the final cell is unreachable and path tracking is not requested, the package returns `Result{Distance:+Inf, Reachable:false}` with `nil` error. When path tracking is requested, the package returns the same unreachable result with `ErrNoPath`.

Backtracking failures such as `ErrIncompletePath` indicate that a finite forward DP state could not be reconstructed into a valid path. Callers must treat such results as failed executions and must not use `Path`.

### 8.10.5. Legacy Wrapper Law

`DTW(a, b, nil)` uses default legacy options and does not panic. Legacy `ReturnPath=true` requires `MemoryMode == FullMatrix`; otherwise `ErrPathNeedsMatrix` is returned. New APIs should use `Align` and `Result`.

---

## 8.11. Bounded Alignment and Exact Window Search

`AlignBounded(ctx,a,b,limit,opts...)` uses the normal scalar DTW metric and
published options; its result is `BoundExact` (finite distance and equal lower
bound), `BoundAboveLimit` (`Distance=NaN`, certified `LowerBound>limit`) or
`BoundUnreachable` (both distances `+Inf`). Final numerical overflow is still an
error. It currently computes the full alignment and does not promise early exit.

`Search(ctx,query,data,options)` ranks complete windows of length `len(query)`.
Each window uses squared scalar costs, no callbacks or penalties, and optional
independent population z normalization. `SearchOptions{}` is invalid; obtain
`DefaultSearchOptions()` before changing policy. `Radius=-1` means unconstrained,
`Radius=0` means strict diagonal. Excluded starts use half-open intervals.

When `Complete=true`, `CandidatesCompleted+CandidatesPruned=CandidatesTotal` and
Hits is the exact global top-K sorted `(SquaredDistance,Start,End)`. The machine
certificate may prune candidates strictly above the current finite kth score;
`LBKeoghSquared` alone is only a rounded diagnostic. Certification requires a
radius-derived envelope, finite local domain, a finite diagonal witness, and
monotone checked accumulation. If any condition is uncertain, the full scorer
runs. Equal lower bound and cutoff are never pruned.

When interrupted, Hits represents only **completed** candidates and is not a
certified global ranking. Search owns its result slices; workers have independent
scratch, and the configured cell budget never counts pruned candidate cells.
`MaxWorkingBytes` governs explicitly accounted workspace and may disable optional
envelope construction without altering the exact scorer. See
[DTW_SEARCH.md](DTW_SEARCH.md) for full details, proof, examples and benchmarks.

---

## 8.12. Pitfalls & Best Practices

### 8.12.1. Anti-Patterns

> **Do not treat `Window=0` as unconstrained.** It is strict diagonal alignment. Use `Window=-1` for no band.

> **Do not parse error strings.** Use `errors.Is`.

> **Do not request a path from rolling rows in the legacy API.** Path reconstruction needs accumulated state.

> **Do not pass probabilities directly as costs unless that scale is intentional.** Convert probability-like values outside DTW when needed.

> **Do not allow invalid local costs.** NaN, Inf, and negative costs are rejected.

> **Do not assume DTW distance is a metric.** DTW generally does not guarantee triangle inequality.

> **Do not rely on hidden normalization.** `AlignMatrix` does not normalize features.

> **Do not expect all optimal paths.** The package returns one deterministic representative optimal path.

> **Do not mutate input matrices while alignment is running.** The package does not snapshot caller-owned mutable inputs.

> **Do not document `NoMemory` as `O(1)` exact DTW.** It is a deprecated compatibility alias for rolling-row distance mode.

### 8.12.2. Best Practices

> **Choose the correct facade.** Use `Align` for scalar signals, `AlignCostMatrix` for external cost surfaces, and `AlignMatrix` for row-wise multivariate time series.

> **Tune window before penalty.** A window defines admissibility. A penalty prices stretching inside that admissible domain.

> **Use small positive slope penalties for real sensors and speech.** This keeps repeated frames possible but visible in the score.

> **Request expensive artifacts intentionally.** Path, accumulated matrix, and local matrix artifacts increase memory use.

> **Inspect path anchors.** Repeated `I` or `J` values identify the phases that stretched.

> **Normalize before `AlignMatrix` when units differ.** Feature scale directly affects squared L2 costs.

> **Use `PathOrError` in reusable code.** It separates nil result, no path, and path-not-tracked states.

> **Clone before independent mutation.** Use `Result.Clone` when multiple consumers will mutate artifacts independently.

> **Treat `Reachable=false` as a policy outcome.** It usually means the window/domain policy blocked the endpoint.

> **Keep custom `CostFunc` deterministic and cheap.** It runs inside the DP loop and participates directly in reproducibility and runtime.

---

**`lvlath/dtw`**: deterministic temporal alignment by contract, strict numeric policy by default, and matrix-backed result surfaces for explainable production analytics.

> Next: [9. Grid Graphs ->](GRID_GRAPH.md)
