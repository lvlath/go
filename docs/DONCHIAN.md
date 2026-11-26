<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  File: docs/DONCHIAN.md
  Package reference: github.com/lvlath/go/donchian
  Purpose: explain observed rolling ranges, exact midpoint arithmetic, and safe consumption.
  Contract status: operational guide; package GoDoc and source define the implemented API.
  Scope: batch windows, masks, errors, ownership, cancellation, examples, and verification.
  License: AGPL-3.0-only; see ../LICENSE.
-->

# Donchian: causal rolling range channels

Package: `github.com/lvlath/go/donchian`.

This is a batch-only numerical component, not a trading system. It computes observed
rolling extrema, their correctly rounded midpoint, and a shared readiness mask.
It does not implement calendars, orders, fills, Stream, Snapshot, or private
risk/execution dependencies. The repository module remains `github.com/lvlath/go`;
its declared minimum Go version is 1.23.4. Import the owning subpackage from
the single repository module.

## Scope and entry points

Choose the information set before selecting the call. A range drawn over a closed
bar and a range used to judge that bar are different features. The package offers
one kernel and three entry points; there is no implicit period or stateful API.

| Need                                          | Entry point                                     | Policy                                      |
|-----------------------------------------------|-------------------------------------------------|---------------------------------------------|
| Display the range including a finalized bar   | `ComputeIncludeCurrent(ctx, high, low, period)` | Current bar belongs to the range.           |
| Compare an observation against preceding bars | `ComputeExcludeCurrent(ctx, high, low, period)` | Current bar does not set its own threshold. |
| Apply an already stored configuration         | `Compute(ctx, high, low, opts)`                 | Uses `opts.ExcludeCurrent` exactly.         |

The named wrappers only construct `Options` and return the canonical result/error.
They perform no additional traversal, allocation buffer, or context polling. They
preserve every channel, error category, readiness index, tie decision, and bit.

### Navigation

- [Public API](#public-api), [windows and causality](#mathematics-and-information-sets), and [errors](#validation-and-errors).
- [Deque invariants](#monotonic-deque-algorithm), [midpoint proof](#correctly-rounded-midpoint), and [memory/ownership](#ownership-memory-and-cancellation).
- [Complete usage](#executable-usage), [composition and revision](#composition-revision-and-batch-boundaries), and [verification](#testing-and-verification).
- [Operational pitfalls](#pitfalls-and-integration-checklist) and [source references](#sources).

## What and why

Donchian channels summarize a price range in the same units as the input. A
previous-bar range can serve as a threshold for a separately defined current-bar
comparison. A current-inclusive range can serve as a finalized-bar overlay.
The algorithm does not assess the economic value or profitability of either use.

`series` can prepare and validate a time grid before this call. A separately
computed `atr` result can supply a noise scale to application-owned features.
Neither package is imported here. The caller must align masks, information
availability, units and decision times when combining results.

Production imports are exactly `context`, `errors`, `fmt`, and `math`.

## Public API

```go
type Options struct {
	Period         int
	ExcludeCurrent bool
}

type Result struct {
	Upper  []float64
	Middle []float64
	Lower  []float64
	Valid  []bool
}

func Compute(ctx context.Context, high, low []float64, opts Options) (*Result, error)
func ComputeIncludeCurrent(ctx context.Context, high, low []float64, period int) (*Result, error)
func ComputeExcludeCurrent(ctx context.Context, high, low []float64, period int) (*Result, error)
func (r *Result) IsNil() bool
```

The declarations above are a reference excerpt, not a standalone program.
All fields and methods have package GoDoc. The only other exported names are the
seven stable sentinel variables: `ErrNilContext`, `ErrNilInput`, `ErrInvalidOption`,
`ErrShape`, `ErrNonFinite`, `ErrBar`, and `ErrOverflow`.

`Options{}` is invalid. `Period` is an observation count, not a count of calendar
days. Every positive `int` period is valid, including `math.MaxInt`.

The successful result always has `len(Upper)==len(Middle)==len(Lower)==len(Valid)==n`.
Upper and Lower are selected input values, not reconstructed endpoints. Middle
has the same units. `Valid` applies to **all three** boundaries, unlike ATR's
ATR-only mask. There is no `Width`, `Period`, `SourceID`, or stored `Options` field;
keep the actual configuration and source identity in an application-owned envelope.

A zero-value or manually edited `Result` is not automatically a validated channel.
`IsNil` remains safe and tests only the pointer, even when a non-nil result has
empty, mismatched, or caller-modified fields. No helper validates those edits.

## Mathematics and information sets

For bar index `t` and period `P`, the two windows are:

![Window policies](https://latex.codecogs.com/svg.image?W_t%5E%7Binc%7D%3D%5Bt-P%2B1%2Ct%5D%2C%5Cquad%20W_t%5E%7Bexc%7D%3D%5Bt-P%2Ct-1%5D)

The bounds and midpoint are:

![Channel definitions](https://latex.codecogs.com/svg.image?U_t%3D%5Cmax_%7Bj%5Cin%20W_t%7DH_j%2C%5Cquad%20L_t%3D%5Cmin_%7Bj%5Cin%20W_t%7D%5Cell_j%2C%5Cquad%20M_t%3D%5Coperatorname%7BRN%7D_%7B64%7D%5Cleft%28%5Cfrac%7BU_t%2BL_t%7D%7B2%7D%5Cright%29)

The formula images are documentation resources only. Execution and tests are
fully offline; no image, embedded fixture or external service is loaded by Go.

| Policy          | Complete window at `t` | First valid index | Short-history condition |
|-----------------|------------------------|-------------------|-------------------------|
| Include-current | `[t-P+1, t]`           | `P-1`             | `n < P`                 |
| Exclude-current | `[t-P, t-1]`           | `P`               | `n <= P`                |

For a successful call, readiness is exactly the corresponding index comparison;
zero-valued complete windows are still ready. P=1 selects the current H/L in
include mode and the preceding H/L in exclude mode; exclude row zero is masked.

For `P=2`, `high=[2,4,3]`, `low=[1,2,0]`:

| Mode    | Valid                | Ready rows `(Upper, Middle, Lower)`  |
|---------|----------------------|--------------------------------------|
| Include | `[false,true,true]`  | row 1: `(4,2.5,1)`; row 2: `(4,2,0)` |
| Exclude | `[false,false,true]` | row 2: `(4,2.5,1)`                   |

On the same admissible history, exclude row `t` equals include row `t-1` bit for
bit whenever those rows are ready. This identity concerns numerical rows, not
an excuse to bypass validation or discard the final supplied input.

Include-current is causal **after** the current bar becomes available. It is not a
previous-bar breakout threshold. For a consistent candle, `close[t] <= high[t] <=
Upper[t]` in that mode, so a strict current-close upper breakout cannot occur.
Exclude-current removes the current bar from the threshold's information set.
Changing the current bar therefore cannot change its own exclude-current bounds.
Changing a later valid bar cannot change any earlier numerical result in either
mode. Prefix invariance is tested on admissible histories.

Neither mode proves that a historical revision was available in the past, or
licenses execution at a close that was not yet known. The caller governs bar
finality, revisions, missing observations, calendars and fill assumptions. Batch
validation still examines all supplied bars, including excluded or masked bars;
an invalid later input rejects the entire batch rather than returning a prefix.

## Validation and errors

Precedence is:

```text
context -> options -> shape -> full-input finite validation
        -> first invalid finite candle -> checked allocation -> kernel
        -> final context check -> publication
```

At each input index, high is checked before low for finiteness. The first finite
geometry violation is remembered. A later non-finite value takes precedence over
that remembered geometry error. This needs one linear pass and constant space.

| Condition                                           | Classification and result                              |
|-----------------------------------------------------|--------------------------------------------------------|
| `ctx == nil`                                        | `nil, ErrNilContext`                                   |
| Context already canceled or expired                 | `nil, ctx.Err()` before input checks                   |
| `Period < 1`                                        | `nil, ErrInvalidOption`                                |
| Length mismatch, both slices non-nil                | `nil, ErrShape`                                        |
| Nil slice paired with a nonempty slice              | `nil, errors.Join(ErrShape, ErrNilInput)`              |
| Both lengths zero, including nil/empty combinations | Non-nil success with four non-nil empty slices         |
| Any NaN or infinity                                 | `nil, ErrNonFinite`, including wholly masked histories |
| Any finite `high[i] < low[i]`                       | `nil, ErrBar` after complete finite validation         |
| Packed float-output byte count cannot fit an `int`  | `nil, ErrOverflow`                                     |
| Cancellation observed after work starts             | `nil, ctx.Err()`; no partial result                    |

Negative prices and either sign of zero are legal. No epsilon, clamping, sorting,
interpolation, or NaN-to-zero repair is performed. Compare errors using
`errors.Is`, never message strings. Nil/nonempty can match two sentinels; choose
which classification an application wishes to prioritize in its diagnostics.

`IsNil` is exactly `return r == nil`. It does not test readiness or slice length.
A typed nil *Result works through `interface { IsNil() bool }` without reflection.
A typed-nil or otherwise misbehaving custom `context.Context` implementation is
outside the supported context contract; the library does not recover its methods.

## Monotonic-deque algorithm

The upper ring stores increasing indices with strictly decreasing high values.
The lower ring stores increasing indices with strictly increasing low values.
On insertion, remove upper tails with value `<=` the new high and lower tails
with value `>=` the new low. Equal values therefore retain the **latest** index.
This is observable for ties between positive and negative zero, even though the
numeric extrema compare equal.

Each ring has exactly `P` slots when a complete output is possible. Head, tail and
size distinguish full and empty rings without a sentinel slot. Expiration always
precedes insertion; at most `P-1` retained indices can precede the new index.
No ring grows, copies itself, or retains a view into caller-owned input indices.

```text
validate all input
allocate zeroed result
if every row is warmup:
    skip rings
else:
    allocate upper and lower rings of capacity P
    for t = 0 .. n-1:
        poll context periodically
        if exclude-current and t >= P:
            write extrema from the existing rings to row t
        expire ring heads with index < t-(P-1)
        insert t into both rings, retaining latest equal extrema
        if include-current and t >= P-1:
            write extrema from the updated rings to row t
check context immediately before returning result
```

```text
Previous retained bars       Current t                 Future
[t-P ............. t-1]      [high[t], low[t]]          [t+1 ...]
           |                         |
           +--> exclude output       |
           |                         |
           +-- expire oldest -- insert current --> include output
```

An index enters each ring exactly once and leaves it at most once, through either
its head or its tail. Across the whole history, each ring therefore performs at
most `n` insertions and `n` removals. Comparisons that end a tail-removal loop add
at most one per insertion. This proves total O(n) work, not worst-case O(1) per
bar. A single bar can trigger O(P) removals, so those long bursts also poll context.

## Correctly rounded midpoint

The exact mean lies between the finite bounds:

![Finite convex mean](https://latex.codecogs.com/svg.image?L_t%5Cle%5Cfrac%7BU_t%2BL_t%7D%7B2%7D%5Cle%20U_t)

Consequently, numeric overflow of the final midpoint is not a valid failure case.
However, both `(a+b)/2` and `a+(b-a)/2` can overflow in intermediate operations.
Unconditionally using `a/2+b/2` loses small representable results: when both
operands equal the smallest positive subnormal, it produces zero instead of that
subnormal.

The production kernel partitions the finite binary64 domain:

1. Exact opposite operands, including both zero signs: return canonical `+0`.
2. Both magnitudes at most `MaxFloat64/2`: form the sum and then halve it. The sum
   cannot overflow. If the final halving is in the subnormal region, the sum is
   exactly representable in units of the minimum subnormal. Otherwise halving
   the rounded sum is exact and commutes with nearest-even rounding of the mean.
3. At least one magnitude exceeds that threshold: halve both operands when both
   magnitudes are at least `2*MinNormal`. Both halves are exact normal values and
   their sum needs only one rounding.
4. In the large-magnitude case with a tiny operand below `2*MinNormal`, leave the
   tiny operand unscaled. It is far below half an ULP of the huge operand's exact
   half and cannot affect rounding, whether halved or not. This avoids a needless
   underflow without changing the correctly rounded answer.

Explicit `float64` conversions keep the required rounding boundaries. There is
no arbitrary-precision production fallback and no dependence on period or history
shape. The test oracle adds exact `big.Rat` representations, divides by two, then
converts once to float64. All comparisons are bitwise, including subnormal and
signed-zero cases. Display formatting is not an oracle and cannot replace bit
and readiness comparisons.

## Ownership, memory and cancellation

The three price channels occupy one non-overlapping float buffer of length `3*n`.
Full slice expressions cap each channel at length `n`; appending one channel cannot
overwrite another. The bool mask has its own allocation. Inputs are never aliased
or mutated, and different calls have separate results. Keeping just one price
channel alive retains the shared three-channel backing allocation; copy that
channel when independent long-term retention matters.

The packed float allocation is checked before multiplication: `n <= MaxInt/24`.
The numerical payload is `24*n` bytes plus `n` bool elements for the mask,
excluding headers, alignment, and allocator effects.
The two rings need `2*P` integers when ready output is possible, stored in separate
allocations without calculating an overflowing `P+1` or combined ring length.
When `P>n`, or `P==n` in exclude mode, **no rings are allocated**. Output storage is
still O(n); this is not a zero-allocation batch API.

Total complexity is O(n) time, O(n) output memory, and O(min(P,n)) workspace.
Allocation-size checks do not reserve RAM or promise recoverable runtime OOM for
arbitrarily large actual histories. A giant period with a small input does not
cause a giant allocation.

Context is observed before other validation, at least every 1024 input bars,
additionally every 1024 removals within one long deque insertion, after validation,
and immediately before result publication. Errors discard all private work. Concurrent cancellation
and an input failure have no prescribed winner after entry. Cancellation immediately
after the final successful check cannot retroactively withdraw a returned result.
Independent calls are safe on shared read-only input. Concurrent input mutation or
writes to the same returned result remain the caller's responsibility.

The polling counts apply separately to the validation scan, kernel scan, and a
single insertion's removal loop. They do not promise one poll for every 1024
aggregate operations, hard real-time response, or interruption inside `make`.
Copying a `Result` struct copies slice headers, not their elements; explicitly
copy all four channels for an independently mutable artifact.

## Executable usage

### Complete previous-range comparison

This deliberately small synthetic example isolates window alignment. Every close
is finite, has the same length as H/L, and lies within its own candle; those are
application preconditions because Donchian does not receive `close`.

<!-- donchian-example: guide-prior-range -->
```go
package main

import (
	"context"
	"fmt"

	"github.com/lvlath/go/donchian"
)

func main() {
	const period = 3 // Application policy, not a library default.
	// Ordered, synthetic observations in one consistent price unit.
	high := []float64{102, 104, 103, 108, 107, 109}
	low := []float64{99, 101, 100, 104, 102, 105}
	close := []float64{101, 103, 102, 107, 104, 108}

	// The current close must not help define the range being tested.
	channel, err := donchian.ComputeExcludeCurrent(context.Background(), high, low, period)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	for i, ready := range channel.Valid {
		// Warmup zeros are absent measurements, not zero-priced thresholds.
		if !ready {
			continue
		}
		// Equality is not a strict upper-range breakout.
		above := close[i] > channel.Upper[i]
		fmt.Printf("bar=%d prior=[%.0f,%.0f] close=%.0f above=%t\n",
			i, channel.Lower[i], channel.Upper[i], close[i], above)
	}
}
```

Output:

```text
bar=3 prior=[99,104] close=107 above=true
bar=4 prior=[100,108] close=104 above=false
bar=5 prior=[100,108] close=108 above=false
```

The program computes comparisons, not orders, positions, fills, costs, or returns.
No transaction is assumed executable at the reported close. For a current-bar
visual overlay, select `ComputeIncludeCurrent` explicitly instead.

### Five executable operational scenarios

The authoritative programs and exact output blocks live in
[`donchian/example_test.go`](../donchian/example_test.go). Run them offline:

```sh
go test ./donchian -run '^Example' -count=1 -v
```

| Example                                              | Pipeline and caller-visible result                                                                                                   |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `ExampleComputeIncludeCurrent_marketRangeReport`     | Frozen OHLC -> current 20-observation channel -> readiness/width gate -> close-position report.                                      |
| `ExampleComputeExcludeCurrent_twentySessionAudit`    | Frozen OHLC -> previous 20-observation range -> upper/lower close-event ledger; a later open is observational context, never a fill. |
| `ExampleCompute_multiPeriodRangeMonitor`             | Previous 10/50 ranges -> joint readiness -> finite widths and positive slow width -> explicitly selected compression feature.        |
| `ExampleComputeExcludeCurrent_negativeSpreadMonitor` | Synchronous paired quotes -> observed signed spreads -> per-bar sampled extrema -> previous-range excursion monitor.                 |
| `ExampleCompute_feedQuarantine`                      | Keep a last accepted artifact -> reject a deliberately corrupted batch -> reload clean data -> recompute before replacement.         |

The source fixture in [`test_helper_test.go`](../donchian/test_helper_test.go)
records 58 IBM OHLC observations dated 2026-09-01 through 2026-11-20 and identifies
an Alpha Vantage daily demo response. This description records the fixture's
provenance, not independent exchange verification, live availability, or an
instruction to refresh it. Its source URL, supplied precision, and oldest-first
order remain attached to the literal. The negative-spread fixture is engineered;
the quarantine example deliberately inserts NaN into its own copy. Neither is
misrepresented as an observed market incident.

## Composition, revision, and batch boundaries

### Prepare the observation grid outside the kernel

H/L positions are observation indices. A period of 20 means 20 admitted rows,
not 20 elapsed days. Preserve aligned date, close, and source-revision columns;
reject or handle missing observations under an explicit upstream policy. `series`
can check an appropriate grid, but Donchian neither imports it nor infers an
exchange calendar. Holiday-separated sessions are not a uniform 24-hour grid
merely because a vendor labels them daily. Do not silently sort, interpolate,
or compress missing observations to make a calculation run.

### Combine a prior channel with an independently defined scale

Donchian gives observed boundaries. ATR gives a gap-aware smoothed range.
For a comparison made after bar `t` against **pre-bar** information, the channel
must be exclude-current and a prior scale is ATR at `t-1`. Require `t>0`,
`channel.Valid[t]`, and `atrResult.Valid[t-1]` before reading that denominator.
For periods P and Q on the same fully aligned input, the earliest joint readiness
is index `max(P,Q)`. ATR at `t` is also causal after bar close, but incorporates
the current shock; it defines a different, post-bar-normalized feature.

A ready zero ATR is a legitimate measurement but an unusable divisor for an
ordinary ratio. Check finiteness/positivity of the scale, finiteness of any
subtraction, and finiteness of the quotient. Even finite positive operands can
produce an infinite quotient. Skip, classify, or use a justified alternative
representation according to application policy; never invent a universal epsilon.
Donchian's ErrOverflow does not classify arithmetic performed later by a caller.

```text
admit ordered H/L and aligned metadata
               |
       +-------+--------------------+
       |                            |
Donchian exclude at t          separate ATR at t-1
       |                            |
       +---- masks + unit + arithmetic checks ----+
                                                   |
                                      caller-owned feature/review
                                      (not an order or a probability)
```

### Reconstruct spread extrema from synchronized observations

For aligned samples A and B, first compute each admitted spread `A[j]-B[j]`,
then aggregate the spreads into sampled highs/lows. Independently observed leg
extrema need not occur at the same instant: `max(A)-max(B)` is not the maximum
spread. `max(A)-min(B)` is only an envelope without paired timing, not an observed
spread high. Subtraction itself needs an overflow policy. Sampled extrema are
not proof of continuous intrabar extrema when observations are sparse.

### Correct historical bars without inventing streaming state

Changing admitted H/L at index k can affect only ready rows in these intervals:

| Mode    | Potentially affected output indices                        |
|---------|------------------------------------------------------------|
| Include | `[k, k+P-1]`, intersected with ready indices and `[0,n-1]` |
| Exclude | `[k+1, k+P]`, intersected with ready indices and `[0,n-1]` |

These are mathematical bounds, not permission to compute `k+P` with an unchecked
machine integer. The public API recomputes a supplied batch; it exposes no
incremental update, mutable deque, Stream, or Snapshot. Replacing a bar by an
invalid one rejects the whole new batch, despite the finite numerical influence
horizon for admissible corrections.

Independent suffix processing needs context observations: retain up to P-1
preceding bars for include mode, or P preceding bars for exclude mode; compute
on overlap plus suffix and consume only suffix rows at the corrected offset.
Do not concatenate outputs from cold, disjoint batches: that restarts warmup at
every boundary. Choose overlap with bounded differences, not overflow-prone
`start+P`. Saved envelopes should include source/revision, units, ordered row
mapping, Period, ExcludeCurrent, and package version; `Result` contains none of
that provenance. Save all masks with their payloads.

## Testing and verification

All tests use `package donchian_test` and public API calls. Shared assertions,
fixed fixtures, and independent window/rational helpers live in
`test_helper_test.go`; `oracle_test.go` contains the public-API differential tests.
There are no test-only public exports or embedded data.

Coverage includes both goldens, P=1, first readiness, all-empty combinations,
negative prices, future/current spikes, every admissible prefix, equal extrema,
signed-zero latest ties, maximal-length monotonic rings, wrap-around, MaxInt
periods, all validation classes reachable with small inputs, midpoint extremes,
normal/subnormal ties across all normal exponent fields, exhaustive small histories,
wrapper parity and error-category sets, shifted-mode equality, correction influence
horizons, exactly representable affine fixtures, ownership,
independent concurrent calls and deterministic cancellation at every observed poll.
Fuzz targets cover public shape/period/price boundaries and binary64 midpoint pairs.

The enormous packed-allocation rejection cannot be safely exercised with a small,
legal slice. It is not exposed through a test-only hook or a forged slice. Report
its uncovered statement honestly instead of fabricating numerical overflow.

Run these gates in the actual repository with its accepted toolchain:

```sh
gofmt -l donchian/*.go
go test ./donchian
go test -race ./donchian
go test ./...
go test -race ./...
go vet ./...
golangci-lint run
govulncheck ./...
go test ./donchian -coverprofile=donchian.cover
go tool cover -func=donchian.cover
go test ./donchian -run '^$' -fuzz '^FuzzDonchian_PublicBoundaries$' -fuzztime=10s -parallel=2
go test ./donchian -run '^$' -fuzz '^FuzzDonchian_Midpoint$' -fuzztime=10s -parallel=2
go test ./donchian -run '^$' -bench '^BenchmarkCompute' -benchmem
```

Benchmarks cover `n=1000,100000,1000000`, `P=14,64,512`, both window modes,
and the MaxInt-period short-history path. Additional benchmarks separate full-ring
monotone histories, equal-value eviction, repeated tail-removal bursts, and wrapper
cost. Input generation is outside the timed region. Record toolchain, OS,
architecture, CPU, GOMAXPROCS, and measurement duration.
Timings from shared infrastructure are observations, not performance guarantees.
Record actual runs separately from intended gates: an isolated package test is
not a repository-wide CI result, and an earlier coverage percentage is not a new measurement.
Require at least 92% statement coverage overall and per executable production file;
files without instrumented statements are N/A. Line/statement coverage is not exhaustive
state-space verification. Do not add test-only public exports or lint suppressions
to hide an unreachable legal-input allocation guard.

## Pitfalls and integration checklist

Never read warmup zeros as valid prices, use an include-current threshold as a
previous-bar range, treat n observations as n calendar days, or interpret a signal
close as an executable fill. Never silently compress missing bars or use revised
history without an explicit availability policy.

A finite midpoint does **not** imply a representable width: `MaxFloat64 -
(-MaxFloat64)` overflows. Widths, ratios and ATR normalization are deliberately
outside the API. A zero slow width also needs an application-defined ratio policy.
The squeeze example uses bounded, nonzero-width fixtures; it is not a general
safe-width function.

Positive scaling or translation can introduce their own binary64 rounding.
Metamorphic tests use exactly representable fixtures; general transformed inputs
are checked against their own rational window oracle, not a universal epsilon.
Do not change precision, equal-value eviction, zero-sign rules or window ordering
as a performance optimization without rerunning the independent oracle suite.

## Sources

The implemented contract is defined by the current repository files:

- [Package charter](../donchian/doc.go), [canonical facade and wrappers](../donchian/api.go),
  [Options/Result](../donchian/types.go), and [error categories](../donchian/errors.go).
- [Window kernel](../donchian/compute.go), [ring invariants](../donchian/deque.go),
  [midpoint implementation](../donchian/midpoint.go), and [validators](../donchian/validators.go).
- [Independent oracles and fixture provenance](../donchian/test_helper_test.go),
  [regression checks](../donchian/regression_test.go), and [wrapper checks](../donchian/api_test.go).
- [Go floating-point operators and explicit rounding](https://go.dev/ref/spec#Floating_point_operators).
- [Go rational-to-float conversion](https://pkg.go.dev/math/big#Rat.Float64).

The numerical proof above explains this kernel's finite-domain partition. The
independent rational oracle is not a copy of its floating-point branches. Read
[Tutorial](TUTORIAL.md#11y-donchian-foundations-observed-ranges-and-information-sets)
and [Donchian FAQ](FAQ_%26_TIPS.md#13x-donchian-faq-windows-errors-and-range-safety)
for selection and troubleshooting; neither adds a new numerical policy.
