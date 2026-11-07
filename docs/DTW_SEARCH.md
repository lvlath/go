<!--
  File: docs/DTW_SEARCH.md
  Package: github.com/lvlath/go/dtw
  Purpose: Exact fixed-length subsequence search and bounded-alignment contract.
  Contract: lvlath-spec-02-v1, implementation version exact-search-v0.1.
  Scope: new additive APIs; legacy DTW and matrix adapters retain their contracts.
  License: AGPL-3.0-only.
-->

# DTW bounded alignment and exact fixed-length window search

## 1. Scope, entry points, and non-goals

The existing `Align`, `AlignMatrix`, `AlignCostMatrix`, and legacy `DTW` remain the canonical general DTW entry points. The additive search API offers two separate operations:

```go
func AlignBounded(ctx context.Context, a, b []float64, limit float64,
    opts ...Option) (*BoundedResult, error)
func Search(ctx context.Context, query, data []float64,
    opts SearchOptions) (*SearchResult, error)
func LBKeoghSquared(query, lower, upper []float64) (float64, error)
```

`AlignBounded` computes **the existing configured DTW objective** and then classifies it against an inclusive cost limit. It currently runs the complete canonical recurrence; its name describes the *result contract*, not an early-abandonment speed guarantee. The preexisting scalar metric is absolute difference unless `WithSquaredCost` or `WithCostFunc` is supplied. Only the optional path is returned in `BoundedResult`; `WithReturnLocalCost(true)` and `WithReturnAccumulated(true)` are rejected with `ErrInvalidOption` rather than silently ignored.

`Search` is a different, narrower operation: fixed-length candidate windows, **squared scalar local cost**, no slope penalty, no callback, and no multivariate adapter. Query length L and candidate length are equal. Each candidate is evaluated with the same classified predecessor recurrence as the general kernel. The output is a globally exact top-K if and only if `Complete=true`.

This version does not implement open-ended subsequence alignment, SAX indexing, probabilistic candidates, approximate shortlists, or learned embeddings. It does implement machine-certified LB_Keogh pruning under a restricted squared-scalar float64 model; uncertain certificates leave the exact scorer unchanged.

## 2. Mathematically defined objective

For finite scalar samples, the fixed-length local cost is

![Squared scalar cost](https://latex.codecogs.com/svg.image?c%28i%2Cj%29%3D%28q_i-x_j%29%5E2).

A path starts at `(0,0)`, ends at `(L-1,L-1)`, and advances by `(1,1)`, `(1,0)`, or `(0,1)`. Only cells satisfying `r=-1` or `|i-j|<=r` are admitted. The recurrence is

![DTW recurrence](https://latex.codecogs.com/svg.image?D%28i%2Cj%29%3Dc%28i%2Cj%29%2B%5Cmin%5C%7BD%28i-1%2Cj-1%29%2CD%28i-1%2Cj%29%2CD%28i%2Cj-1%29%5C%7D).

The first consumed pair contributes its local cost exactly once. Unreachable prefixes are structurally absent, not numeric overflow. The private `dpCell` state holds one of `dpUnreachable`, `dpFinite`, or `dpOverflow`; checked addition saturates overflow branches into their class. A finite branch defeats an overflow candidate. Predecessors are selected by strict finite minimum in diagonal → up → left order. Both the general and search scorers reuse `choosePredecessor`, `addCost`, and checked binary64 addition.

**Numerical guarantee:** this is a deterministic rounded-float64 recurrence, not exact-real shortest-path optimization. For example, the result is not promised to equal a symbolic rational oracle for arbitrary near-overflow operands. The independent exhaustive oracle uses small exactly representable integer costs for bit-exact score comparisons; separate boundary tests exercise float64 range handling.

## 3. Search domain and canonical ranking

Given `query` length `L`, `data` length `N`, and `Stride=s`:

```text
raw candidate starts: 0, s, 2s, ... with start + L <= N
excluded starts:      union of user-supplied half-open ranges [Start,End)
admitted starts:      raw starts not excluded
candidate window:     [start,start+L) over the data samples
ranking key:          (exact squared distance, start, end), ascending
```

The exclusion-domain end is `N-L+1` when `N>=L`; otherwise only an empty exclusion set is accepted. Ranges are validated, copied, sorted, and coalesced. Nothing is silently clamped. Empty `[s,s)` ranges are permitted. When data is shorter than query, a valid request returns `Complete=true`, empty Hits, and zero candidates. A zero-value `SearchOptions` is intentionally invalid.

```go
opts := dtw.DefaultSearchOptions()
opts.TopK = 3
opts.Radius = 8
opts.Stride = 1
result, err := dtw.Search(ctx, query, series, opts)
if err != nil {
    // Result may be partial after candidate computation begins.
    // Do not consume result as a global ranking unless result != nil && result.Complete.
}
```

**Invariant:** when `Complete=true`, `CandidatesCompleted+CandidatesPruned=CandidatesTotal`. Certified pruning is optional and may yield `CandidatesPruned>0`; only candidates whose machine lower bound is **strictly greater** than the finite kth score are excluded. A candidate equal to cutoff is evaluated to preserve the earlier-start canonical tie. All other candidates are scored exactly. Thus global ranking does not rely on an uncertified floating comparison.

## 4. Inclusive bounded outcomes

| Status | Distance | LowerBound | Meaning |
|---|---:|---:|---|
| `BoundExact` | finite exact score | same exact score | Optimum is within inclusive limit |
| `BoundAboveLimit` | `NaN` | finite score strictly greater than limit | Full exact score was computed but deliberately not published in Distance |
| `BoundUnreachable` | `+Inf` | `+Inf` | Structural absence of a path under the configured window |

`limit` accepts finite nonnegative values and `+Inf`. Invalid values (`NaN`, `-Inf`, negative) return `ErrInvalidOption`; `distance==limit` is exact, not above-limit. Terminal overflow returns nil result and `ErrOverflow` joined with `ErrNaNInf`, irrespective of the limit. Existing `Align` no-path semantics are unchanged. The present bounded facade counts admitted cells after its complete recurrence and checks cancellation before/after execution, not between the legacy kernel's internal cells.

## 5. ZPopulation normalization, version 0.1

The caller explicitly selects `SearchZPopulation`; `SearchRaw` is the default. Query and **each complete candidate** are normalized independently using the population variance (denominator L):

![Z normalization](https://latex.codecogs.com/svg.image?z_i%3D%5Cfrac%7Bx_i-%5Cmu%7D%7B%5Csqrt%7B%5Cfrac%7B1%7D%7BL%7D%5Csum_k%28x_k-%5Cmu%29%5E2%7D%7D).

The implementation subtracts an anchor before scaling whenever finite differences permit it. If subtracting two finite samples overflows, it divides inputs by a common absolute scale before taking their difference. A forward Welford pass computes the variance of bounded differences; the result is transformed to z scores in private per-worker storage. There is no arbitrary epsilon for declaring variance zero.

An exactly constant query is an **input-validation** error, `ErrZeroVariance` with nil result. An exactly constant candidate encountered during computation is a runtime `ErrZeroVariance` with a partial `SearchResult` (`Complete=false`), including its start index in the error context but never raw sample values. Excluded data samples are nevertheless finite-validated before any worker executes.

This is one deterministic binary64 normalization policy, not an exact-real or high-precision statistics mode. Near-subnormal/infinite-range inputs that cannot be normalized finitely return `ErrNonFinite` joined with `ErrNaNInf`; invalid candidates are never silently skipped.

## 6. Band-only scorer and accounting

For `r>=0`, row `i` computes only `j=max(0,i-r)..min(L-1,i+r)`; there is **no full rectangular inner j-loop with a window if-statement**. Maximum width `b=min(L,2r+1)`; `r=-1` means `b=L`. Each worker retains two `dpCell` slices of width b and optionally a length-L normalized candidate buffer.

```text
for every admitted start:
    obtain worker-owned scratch
    optionally z-normalize the complete window
    for i = 0 .. L-1:
        rowStart = max(0, i-r)        # 0 if r==-1
        rowEnd   = min(L-1, i+r)      # L-1 if r==-1
        for j = rowStart .. rowEnd:
            check context; atomically reserve one MaxCells credit
            local = squared_difference(query[i], candidate[j])
            inspect only present diagonal, up, left classified states
            select strict minimum and extend by local
        swap two small row buffers
    publish finite terminal score or a numeric error
```

Without an admissible certificate (or when the optional envelope exceeds the byte budget), all admitted starts are evaluated independently. With C candidates, worst-case time is `O(C*L*b + C*log(TopK))`; normalization adds `O(C*L)`, and initial input scanning adds `O(N+L)`. Worker scratch is `O(W*b)` raw or `O(W*(b+L))` normalized; heap and copied exclusions add `O(K+R)`. No O(C) start-index list is materialized. No universal speedup is claimed: source validation, sorting, scheduling, and narrow-band behavior vary by workload.

`CellsVisited` counts every cell with a successfully reserved credit, including work on interrupted candidates. `MaxCells=0` means unlimited. A finite limit is enforced by compare-and-swap **before** each local-cost computation: the counter can never exceed MaxCells. Worker counts >1 use real parallel scoring with isolated workspaces; a successful complete ranking is scheduler-independent, while an interrupted partial ranking is not guaranteed identical across worker counts. For a reproducible partial prefix, use `Workers=1`.

`MaxWorkingBytes=0` disables the explicit buffer budget. The mandatory preflight includes copied exclusion ranges, the top-K heap, worker outcome channel, two classified band rows per worker, and normalized buffers. When memory permits, optional pruning also accounts for two length-L float envelope vectors, two transient length-L integer deques, and the envelope descriptor. If the optional bytes do not fit but mandatory scorer bytes do, the exact scorer runs without pruning. Products/sums are overflow-checked before buffer allocation. The budget is defined over **explicit backing buffers**, not Go allocator headers, goroutine stacks, runtime internal channel metadata, or total process RSS. A preflight insufficiency returns nil+`ErrBudget`; a MaxCells failure after starting workers returns partial+`ErrBudget`.

## 7. LB_Keogh: machine-certified pruning

`LBKeoghSquared(query,lower,upper)` remains a rounded **public diagnostic**.
It is not itself a proof that an arbitrary floating-point cutoff comparison
is safe, and it does not infer a radius. `Search` constructs a dedicated
query envelope from the scorer's *already normalized* query with the
admitted `Radius`; lower/upper positions use exactly the query indices that
a DTW path may align to the corresponding candidate index.

![Squared LB Keogh](https://latex.codecogs.com/svg.image?LB%3D%5Csum_i%5Cleft%28%5Cmax%5C%7B0%2CL_i-x_i%2Cx_i-U_i%5C%7D%5Cright%29%5E2)

**Proof for the actual float64 recurrence (not merely real arithmetic):**

1. Every valid monotone endpoint-to-endpoint path visits every candidate
   coordinate at least once, in increasing index order. For each such visit,
   the query coordinate lies inside that candidate position's radius-derived
   envelope.
2. When the candidate sample falls outside its envelope, subtraction with
   the nearest endpoint has no larger absolute rounded difference than
   subtraction against any admitted query sample. On nonnegative finite
   floats, squaring and `checkedNonnegativeAdd` are monotone in each operand.
3. Deleting all additional nonnegative visited-cell costs cannot increase
   the sequentially rounded result. Therefore sequentially accumulating
   one squared envelope deviation per candidate index in the **same checked
   binary64 operation model** cannot exceed any admitted path's finite cost.
4. Before any prune, the algorithm evaluates the largest possible absolute
   cross-pair differences from the query/candidate extrema. If their squared
   values are finite, every admitted local squared cost is finite too.
   It also checks that the diagonal path has a finite accumulated cost.
   Hence an optimal finite candidate exists and a skipped window cannot hide
   either an invalid local cost or an overflowing final optimum.
5. Only when the finite accumulated bound is **strictly greater** than the
   currently published finite kth exact score can the candidate be pruned.
   For any uncertain numeric or allocation condition, the candidate runs
   through the original banded scorer, preserving its exact/error contract.

The proof is specific to equal-length *squared-scalar*, nonnegative-local,
zero-penalty DTW with the documented IEEE-754 operations. It is **not** a
portable exact-real proof for arbitrary callbacks or other metrics. A
single `math.Nextafter(roundedBound,-Inf)` is not used as a substitute for
this proof. The certificate is O(L) per attempted candidate; preparing the
radius envelope with monotone min/max deques is O(L) once per Search.

### Execution sketch

```text
for every admitted window in worker scan order:
    normalize candidate if requested, propagating invalid-window errors
    if exact top-K heap is full and optional envelope was prepared:
        read atomic worst exact score (stale reads are conservative)
        if all candidate/query squared local costs must be finite AND
           the diagonal path has a finite accumulated score:
            accumulate radius-aligned squared envelope deviations using
            the same checked addition as the scorer
            if finite bound > finite worst exact score:
                emit certified-pruned outcome; do not reserve DP cells
                continue
    evaluate candidate using the exact banded DP scorer
coordinator:
    count completed scores and certified prunes separately
    update monotone atomic cutoff only after K completed exact hits
    publish canonical sorted Hits; Complete iff all starts were accounted
```

With multiple workers, the *set of pruned candidates* and `CellsVisited` may
change with scheduling because the kth threshold becomes available at
slightly different times. **Completed Hits are invariant**. A stale cutoff
can miss a pruning opportunity but cannot prune an admissible winner.

## 8. Errors, ownership, and operational rules

New sentinels `ErrInvalidOption`, `ErrShape`, `ErrNonFinite`, `ErrZeroVariance`, `ErrBudget` belong to `dtw`; existing `ErrNilInput`, `ErrEmptyInput`, `ErrNaNInf`, and `ErrOverflow` preserve identity. Classify via `errors.Is`, never error string parsing. Nonfinite input errors include both `ErrNonFinite` and `ErrNaNInf`; numeric accumulated or squared local overflow includes both `ErrOverflow` and `ErrNaNInf`. Errors contain indices but not sample payloads.

Search validation order is: non-nil/context cancellation → options → nil input → shape/exclusions → finite-input scan → normalization/domain → computation. Errors before candidate computation return nil result, except the zero-candidate success. After computation starts, cancellation, budget exhaustion, or candidate numerical errors return a partial `SearchResult` with `Complete=false`. The partial Hits are exact **only for completed candidates**. Neither input slice, exclusion slice, nor returned Hits is mutated or aliased across invocations. Concurrent mutation of query or data by a caller is unsupported; the result and copied configuration are not internally synchronized for caller mutation.

## 9. Examples and verification

The executable public API examples live in `dtw/example_search_test.go`. Two standalone scenarios cover the distinct usage laws: `examples/dtw_exact_window_search.go` (general exact retrieval) and `examples/dtw_certified_pruning.go` (one-worker certified pruning). Execute either with `go run ./examples/<filename>.go`; both use build constraints to avoid duplicate `main` symbols under `go test ./...`. The independent path-enumeration oracle lives in `dtw/search_oracle_test.go`; the new adversarial, budget, overflow, and cancellation checks are in `dtw/search_certified_test.go`, and `dtw/search_certified_scenario_test.go` validates the standalone output.

```bash
go test ./dtw -run 'TestSearch_|TestAlignBounded_|TestLBKeogh_' -count=1
go test -race ./dtw -count=1
go test ./... -count=1
go vet ./...
golangci-lint run
```

These commands describe the acceptance gate. A passing isolated harness with a minimal Matrix stand-in does not prove that the real module passes the same checks; run against the actual `go 1.23.4` checkout before accepting the integration SHA.
