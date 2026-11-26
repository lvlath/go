<!--
SPDX-License-Identifier: AGPL-3.0-only
Copyright (C) 2025-2026 katalvlaran
File: docs/ATR.md
Package: github.com/lvlath/go/atr
Purpose: operational and mathematical guide for exact True Range and Wilder ATR.
Contract status: operational reference for the public atr API and atr/wilder-rn/v1 continuation.
Scope: finite signed candle prices, aligned batch results, transactional streams.
License: AGPL-3.0-only.
-->

# ATR: exact True Range and Wilder smoothing

`atr` measures candle ranges in price units. It has no required imports from other
lvlath packages. It accepts finite negative prices and does not infer calendars,
bar completion, missing observations, trade sizes, money risk, or stop levels.

The package belongs to the single `github.com/lvlath/go` module. Its repository
minimum remains Go 1.23.4. There is no nested module and no new external dependency.
The canonical guide path is `docs/ATR.md`. Keep links case-correct on case-sensitive
filesystems; do not maintain a second case-only copy.

## 1. Public API and ownership

```go
func Compute(ctx context.Context, high, low, close []float64, opts Options) (*Result, error)
func ComputePeriod(ctx context.Context, high, low, close []float64, period int) (*Result, error)
func ComputeDefault(ctx context.Context, high, low, close []float64) (*Result, error)
func New(opts Options) (*Stream, error)
func NewPeriod(period int) (*Stream, error)
func NewDefault() (*Stream, error)
func (s *Stream) Push(ctx context.Context, high, low, close float64) (Value, error)
func (s *Stream) Reset()
func (s *Stream) Snapshot() Snapshot
func (s *Stream) Restore(state Snapshot) error
func (r *Result) IsNil() bool
```

`Options` has one field, `Period int`. All positive `int` values, including
`MaxInt`, are valid; zero is not an implicit default. Period counts accepted
observations, not days. The stream retains its configuration by value.

### Choose the smallest explicit facade

| Need                                    | Public entry point                                            | Policy and cost                                           |
|-----------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------|
| Configuration stored by the application | `Compute(..., Options{Period: p})`, `New(Options{Period: p})` | Canonical validation and numerical transition.            |
| A known observation horizon             | `ComputePeriod(..., p)`, `NewPeriod(p)`                       | Direct delegation with the exact supplied integer period. |
| Explicitly opt into ATR14               | `ComputeDefault(...)`, `NewDefault()`                         | Delegation with exported `DefaultPeriod = 14`.            |

`Options{}` remains invalid. A named default is an explicit call-site choice, not
missing-configuration recovery. Every wrapper preserves result ownership, error
identity, validation precedence, and the same seed and rounding law. No wrapper
selects EMA, first-value seeding, a moving finite window, or a percentage result.
A period is an observation count: fourteen daily records and fourteen one-second
records have different physical horizons. No period-specific buffer is allocated.

`Result` owns three equal-length, non-aliasing channels:

| Field           | Meaning                                                      |
|-----------------|--------------------------------------------------------------|
| `TR []float64`  | Required finite nonnegative True Range at every input index. |
| `ATR []float64` | Wilder measurement; exactly positive zero before readiness.  |
| `Valid []bool`  | Readiness of ATR only, beginning at index `Period-1`.        |

`Value` contains the same three fields as scalars. On a failed Push the returned
value is exactly `Value{}` and no state changes. On any failed Compute the result
is nil; no partially useful result is published. Successful empty input returns
an allocated Result with three non-nil empty slices. A short history `n < Period`
is successful: TR is populated, ATR is positive zero, and every mask is false.

Results never retain or alias the input slices. The caller owns the returned
channels and may mutate them after return. Copying a Result struct only copies
slice headers; those copies share their arrays. A Value and Snapshot are detached
values. SeedSum is an immutable Go string, not an exposed mutable accumulator.

`IsNil` is exactly a pointer-presence test, not a length or readiness predicate:

```go
func (r *Result) IsNil() bool { return r == nil }
```

## 2. Candle domain and the first observation

All three prices must be finite; then `low <= close <= high` must hold. Zero,
negative prices, both zero signs, and subnormals are admitted. No epsilon repairs
inverted or out-of-range candles. Non-finite input gives `ErrNonFinite`; finite
invalid geometry gives `ErrBar` before numerical range arithmetic.

The first accepted bar has no previous close. Its range is:

![First true range](https://latex.codecogs.com/svg.image?%5Cmathrm%7BTR%7D_0%3D%5Coperatorname%7BRN%7D_%7B64%7D%28H_0-L_0%29)

Plain-text equivalent: `TR[0] = RN64(high[0] - low[0])`.

For a later bar:

![True range](https://latex.codecogs.com/svg.image?%5Cmathrm%7BTR%7D_t%3D%5Coperatorname%7BRN%7D_%7B64%7D%5Cleft%28%5Cmax%5Cleft%28H_t-L_t%2C%7CH_t-C_%7Bt-1%7D%7C%2C%7CL_t-C_%7Bt-1%7D%7C%5Cright%29%5Cright%29)

RN64 denotes binary64 round-to-nearest, ties-to-even. Operands in the exact
expression are the represented input floats, not their original decimal text.
Every newly computed zero range or ATR is canonical positive zero.

### Why the single-subtraction kernel is exact

For a valid interval `[L,H]`, consider the prior close C.

* When C is inside the interval, both distances to C are at most H-L.
* When C is above H, C-L is at least both H-L and C-H.
* When C is below L, H-C is at least both H-L and L-C.

Therefore the exact maximum of the three candidates is
`max(H,C) - min(L,C)`. Endpoint selection introduces no rounding. Subtracting
those two binary64 endpoints once gives the required rounded exact maximum.
The independent test oracle deliberately uses the three rational differences,
not this envelope implementation.

At count zero the endpoint comparisons are skipped, even though zero-valued
storage physically exists in the struct. That storage is not a fabricated
historical observation. This distinction is visible for a first negative-price
bar: H=-1, L=-3, C=-2 gives TR=2, not a distance to an invented zero close.

A gap is measured from the last **successfully accepted close**, never from the
last rejected bar, current open, current close, or a future anchor. Reset clears
that prior close. Splitting a history without restoring a snapshot starts a new
first-bar calculation; it is not equivalent to continuing the original series.

### Overflow means overflow of the required rounded output

If TR rounds to infinity, the operation returns `ErrOverflow`, even when a
hypothetical smoothed ATR could be finite. TR itself is an obligatory output.
For example, H=MaxFloat64, L=-MaxFloat64, C=0 fails.

A mathematical result slightly above MaxFloat64 can still round to MaxFloat64.
That is not an error. Tests distinguish the finite rounding shoulder from the
ties-to-even overflow threshold. A representable result is never rejected just
because an unneeded intermediate formula would overflow.

## 3. Seed and exact Wilder recurrence

The first ready value is the exact mean of the first P **already rounded TRs**:

![ATR seed](https://latex.codecogs.com/svg.image?%5Cmathrm%7BATR%7D_%7BP-1%7D%3D%5Coperatorname%7BRN%7D_%7B64%7D%5Cleft%28%5Cfrac%7B%5Csum_%7Bi%3D0%7D%5E%7BP-1%7D%5Cmathrm%7BTR%7D_i%7D%7BP%7D%5Cright%29)

For every subsequent bar:

![Wilder recurrence](https://latex.codecogs.com/svg.image?%5Cmathrm%7BATR%7D_t%3D%5Coperatorname%7BRN%7D_%7B64%7D%5Cleft%28%5Cfrac%7B%28P-1%29%5Cmathrm%7BATR%7D_%7Bt-1%7D%2B%5Cmathrm%7BTR%7D_t%7D%7BP%7D%5Cright%29)

Plain-text equivalents:

```text
seed = RN64(exactSum(roundedTR[0:P]) / P)
next = RN64(((P-1) * previousRoundedATR + roundedTR) / P)
```

This is a sequentially rounded recurrence. It is not a single unrounded formula
for the complete historical series. Neither `alpha=2/(P+1)` nor a separately
rounded `alpha=1/P` implements this contract. Even `previous + (TR-previous)/P`
can double-round incorrectly: previous=1, TR=0, P=3 must produce float bits
`0x3fe5555555555555`; the floating delta expression can be one ULP higher.

For P=1, the kernel assigns ATR directly from the computed TR. Every position is
ready and both channels have identical bits, including their positive zeros.
No subtraction of a stale ATR or residual seed can disturb this identity.

### Fixed exact integer representation

Every finite nonnegative binary64 is an integer multiple of `u = 2^-1074`.
`exactSum` stores that integer in 34 little-endian uint64 limbs (2176 bits).

```text
MaxBinary64Units = (2^53 - 1) << 2045
bitLength(MaxBinary64Units) = 2098
bitLength(MaxBinary64Units * any uint64 count) <= 2162 < 2176
```

This bound covers every seed and every nonnegative weighted affine numerator.
During warmup Count<P and P<=MaxInt; mature weights sum exactly to P. No history-
length allocation, arbitrary-precision heap object, or float64 numerator is
needed. The production package does not import math/big; the oracle does.

Dividing the exact integer numerator by P yields an integer quotient and a
remainder. The converter retains the leading 53 bits, guard/sticky information,
and the division remainder. It increments precisely for a greater-than-half
residue or a half residue with an odd retained significand. For subnormals, the
remainder is compared directly with its complement `P-remainder`, avoiding an
overflowing `2*remainder`. This includes ties rounding to positive zero and carry
from a full significand into the next exponent.

The seed is an average of finite nonnegative values. A mature exact step is a
convex combination of previous ATR and TR. Both exact results lie between finite
operands, proving that the final rounded ATR cannot overflow. An unnecessarily
formed floating numerator could overflow; this implementation never forms it.

## 4. Canonical transition and cancellation

```text
Compute
  context -> options -> shape
  full validation scan in ascending index order
  checked output size -> context -> allocate owning channels
  numerical scan: advance(private stream) -> aligned outputs
  final context check -> publish Result

Push
  context -> receiver -> finite domain -> candle geometry
  copy state -> advance(candidate) -> final context check -> commit candidate

advance
  reject exhausted counter
  compute first/prior-close TR
  P=1: assign TR
  else warming: exact-add TR; at count P compute seed and clear accumulator
  else ready: exact affine Wilder step
  update count and previous close
```

Both surfaces delegate to the same `advance` transition; validation and
publication policy differ, not the mathematics. There are two linear batch scans:
one validation pass and one numerical pass. Describing the complete operation as
a single scan would conceal the validation-first requirement.

Context is checked before any input processing, at least every 1024 elements in
each long batch pass, before output allocation, and immediately before returning
a result. Push checks context initially and immediately before committing its
candidate. A detected failure discards the candidate in its entirety.

The final successful context check is a publication boundary, not a scheduler
barrier: cancellation immediately after that check can race with success. When
cancellation and a new input failure occur concurrently after the call starts,
the first detected failure is returned. There is no promised ordering between
those concurrent events. Custom context implementations remain responsible for
their own behavior and allocations; the package does not recover from their
panics or enforce real-time scheduling deadlines.

The complete batch validation scan can reject an invalid future row before
performing any arithmetic. This is not numerical lookahead: on valid histories,
every output depends only on its accepted prefix. Prefix tests protect that law.

## 5. Errors, nil values, and precedence

Use `errors.Is`, never string matching. Error strings are diagnostic only.

| Category           | Reachable condition                                                                                         |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| `ErrNilContext`    | A nil context passed to Compute or Push.                                                                    |
| `ctx.Err()`        | Cancellation/deadline observed by a context check.                                                          |
| `ErrInvalidOption` | Period<1 in New or Compute.                                                                                 |
| `ErrNilInput`      | Nil Stream receiver; or nil input channel with another nonempty channel.                                    |
| `ErrShape`         | Other unequal input channel lengths.                                                                        |
| `ErrNonFinite`     | NaN or either infinity in any input price.                                                                  |
| `ErrBar`           | Finite candle with low>high or close outside [low,high].                                                    |
| `ErrOverflow`      | Required TR rounds to infinity, counter increment cannot fit, or checked output byte arithmetic cannot fit. |
| `ErrStateVersion`  | Unconstructed stream or invalid snapshot/schema/configuration/phase.                                        |

Compute precedence is context, options, shape, then per-index finite checks and
geometry checks in increasing index order, then calculation. It is **not** a
whole-array non-finite sweep followed by a whole-array geometry sweep. An earlier
bad candle wins over a later non-finite candle. Because validation precedes all
range calculations, a later invalid candle wins over an earlier overflowing TR.

A mixed nil/nonempty input is classified as `ErrNilInput`; this is distinct from
a non-nil empty channel in an otherwise nonempty, unequal-length input. All channels with zero length, whether nil or allocated empty,
represent the same successful empty history. A non-nil empty channel accompanying
nonempty channels instead gives `ErrShape`.

Push precedence is context, receiver, finite domain, geometry, numerical transition,
final context, publication. Restore has no context parameter and validates its
receiver first. Nil/zero behavior is fixed:

| Operation                 | Nil receiver | Unconstructed `Stream{}` |
|---------------------------|--------------|--------------------------|
| Push, after context check | ErrNilInput  | ErrStateVersion          |
| Restore                   | ErrNilInput  | ErrStateVersion          |
| Reset                     | No-op        | No-op                    |
| Snapshot                  | Snapshot{}   | Snapshot{}               |

A context must be a valid Context implementation, not an interface containing a
broken typed-nil implementation. No reflection is used to inspect implementation
internals. Construct streams with New, NewPeriod, or NewDefault instead of treating their
zero value as implicitly configured.

## 6. Snapshot ABI and exact continuation

The complete wire-facing value is:

```go
type Snapshot struct {
    Version       uint32
    Algorithm     string
    Options       Options
    Count         uint64
    PreviousClose float64
    Current       float64
    SeedSum       string
}
```

Version is exactly 1; Algorithm is exactly `atr/wilder-rn/v1`. Restore only accepts
a stream constructed by New with exactly matching Options. No seed policy or
smoothing mode can silently change during restore.

| Phase     | PreviousClose                            | Current                               | SeedSum                                              |
|-----------|------------------------------------------|---------------------------------------|------------------------------------------------------|
| Count=0   | Zero; either sign is accepted by Restore | Exactly +0                            | `"0"`                                                |
| 0<Count<P | Last accepted close, finite, either sign | Exactly +0                            | Exact nonnegative sum of rounded seed TRs in units u |
| Count>=P  | Last accepted close, finite, either sign | Finite and >=0; mature -0 is accepted | `"0"`                                                |

New and Reset create positive-zero empty fields. Generated numeric outputs use
positive zero. Restore deliberately preserves a valid mature negative-zero
Current instead of strengthening the specified `finite && >=0` domain. Warmup
Current is stricter because it is a masked output payload. The next successful
calculation returns canonical positive-zero outputs when mathematically zero.

SeedSum must use the complete ASCII grammar `0|[1-9][0-9]*` and have at most 660
bytes. Signs, whitespace, leading zeros, Unicode digits, and exponent notation
are rejected. The fixed parser checks capacity while consuming digits and then
checks `sum <= Count*MaxBinary64Units`. Oversized strings are rejected before
parsing or extra allocation. Mature phase accepts only the single-byte `"0"`.

This protects Restore's work and storage. It does **not** prevent a caller's JSON
parser from allocating an enormous incoming string before Restore is called;
the surrounding service must independently enforce message-size limits.

### Why warmup restore is exact

After k accepted seed bars the invariant is
`SeedSum = sum(TR[i]/u, i=0..k-1)`. Integer addition preserves that invariant
without rounding. Canonical decimal encoding and parsing preserve the exact
integer. Restore also preserves k, Period, and the previous close. The next TR,
next exact seed integer, and first ready mean therefore match the uninterrupted
stream. In maturity the saved Current is the already rounded prior ATR, which is
precisely the operand needed by the recurrence. Induction over future accepted
bars proves bitwise continuation parity.

### Structural integrity is not authentication

Validation detects invalid metadata, non-finite scalars, wrong phase, malformed
sum syntax, and an impossible magnitude bound. It cannot prove that an otherwise
valid finite PreviousClose, Count, or seed came from the claimed historical bars.
A plausible alternative history can have a valid alternative snapshot.

Authenticate the envelope externally when provenance matters, including schema,
algorithm, options, count, both floats, seed, instrument identity, and source
cursor. Persist all float bits and uint64 counts losslessly. Typed Go JSON
round-tripping is tested; passing Count through an untyped binary64 JSON number
in another system can lose integers above 2^53. A string or integer-preserving
codec is required at such boundaries. Snapshot is not an event-deduplication or
source-offset protocol.

Count=MaxUint64 is a structurally valid mature checkpoint. Its next Push fails
with ErrOverflow and preserves state. Reset explicitly starts a new history.

## 7. Practical use

A complete minimal batch example:

```go
package main

import (
    "context"
    "fmt"

    "github.com/lvlath/go/atr"
)

func main() {
    result, err := atr.Compute(context.Background(),
        []float64{10, 13, 12}, []float64{8, 11, 10}, []float64{9, 12, 11},
        atr.Options{Period: 2})
    if err != nil {
        fmt.Println("ATR error:", err)
        return
    }
    fmt.Println(result.TR)
    fmt.Println(result.ATR)
    fmt.Println(result.Valid)
}
```

Output:

```text
[2 4 2]
[0 3 2.5]
[false true true]
```

The six canonical executable pipelines live in
[`atr/example_test.go`](../atr/example_test.go). Shared fixture parsing lives in
[`atr/test_helper_test.go`](../atr/test_helper_test.go), outside production code.

| Scenario                      | Operational question                                                           | Protected boundary                                                        |
|-------------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Historical range audit        | When is ATR14 ready, and which range includes a prior-close gap?               | Date alignment, warmup TR, exact seed.                                    |
| Prior-baseline monitoring     | Is today's range unusually large relative to yesterday's ready measurement?    | Current shock never increases its own comparison baseline.                |
| Hot failover                  | Can a standby resume after five of fourteen seed observations?                 | Exact SeedSum plus source cursor, typed serialization, suffix bit parity. |
| Quarantine and retry          | Can a bad close be corrected without skipping or double-counting a source row? | ErrBar, complete rollback, no cursor advance on error.                    |
| Signed spread review          | Do zero crossings and negative quotes remain ordinary numeric inputs?          | Signed prices, gap measurement, no close normalization.                   |
| Flat and period-one histories | Is a ready zero different from unavailable warmup?                             | Valid independent of value; ATR and TR bit identity at P=1.               |

The first four use a frozen vendor demo response for IBM. The last two are
**engineered stress fixtures**, not historical quotes. Neither category is an
execution strategy, profitability claim, or calibrated alert threshold.

### Source provenance and offline replay

The CSV [`atr/testdata/ibm_daily_20261026_20261120.csv`](../atr/testdata/ibm_daily_20261026_20261120.csv)
contains twenty HLC records dated 2026-10-26 through 2026-11-20, in ascending
source order. The source is the public Alpha Vantage `TIME_SERIES_DAILY` demo
response for `symbol=IBM`, inspected on 2026-11-26. Its metadata reports
`Last Refreshed: 2026-11-20` and `Time Zone: US/Eastern`. The HLC decimal texts are
retained at their published precision; Go parses them once into binary64.
Open and volume are deliberately omitted because ATR consumes neither.

Source: [Alpha Vantage daily demo response](https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo).
This is a frozen **vendor demo capture**, not an independently verified exchange
tape, a live-price service, or proof of when a historical revision was available.
No test makes an HTTP request. Do not overwrite this fixture with a changing API
response; add a separately identified revision when the source changes.

The standalone [`examples/atr/main.go`](../examples/atr/main.go) admits a CSV of
at most 64 KiB, checks strict date order, validates all candles before reporting,
serializes a typed checkpoint after five bars, restores a standby, and checks
its suffix against an uninterrupted primary. The SHA-256 input fingerprint binds
the local replay to the same bytes; it is **not** checkpoint authentication.
The demonstration materializes its bounded input; the library Stream itself
retains no history. Output writer errors stop the report and may leave a printed
prefix; that is I/O behavior, not the library's all-or-nothing batch contract.

```sh
go test ./atr -run '^Example' -count=1 -v
go run ./examples/atr < atr/testdata/ibm_daily_20261026_20261120.csv
```

The source fixture first becomes ready on 2026-11-12. Its final printed ATR14 is
`6.904444`; printed decimals are for reporting, while tests compare output bits
against an independent rational oracle. Date labels and numeric source values
are provenance, not evidence of predictive validity.

External consumers may combine this measurement with another package's EMA to
build Keltner bands, but this package neither imports EMA nor computes bands.
A service normalizing a breakout by ATR must first check `Valid && ATR>0` and
independently validate the quotient's representability. A valid zero is not a
missing observation and must never silently become a denominator.

NATR, position sizing, stop placement, risk budgets, contract multipliers, FX,
commissions, and profitability are outside this API. No executable example
places an order or claims a trading guarantee.

### Compose measurements without crossing responsibility boundaries

```text
closed source row + instrument/revision/cursor
                   |
          caller admission / ordering
                   |
     atr.NewPeriod -> Push -> TR + ATR + Valid
                   |                 |
     checkpoint + next cursor     consumer policy
                   |                 |
          authenticated storage   prior-baseline review
```

A stream is causal with respect to its accepted prefix, but the caller must still
establish that each bar was complete and available at decision time. If reviewing
the current range against a pre-event baseline, compare `TR[t]` against a ready
`ATR[t-1]`, not against the already updated `ATR[t]`. A zero prior ATR requires an
explicit consumer policy. The example multiplier is illustrative, not estimated
or calibrated by this package.

For an accepted historical correction at index k, restore a checkpoint **before**
k and replay the corrected suffix. Appending the correction as a new bar changes
both Count and PreviousClose and does not undo the old observation. Likewise,
switching instruments, adjustment conventions, bar duration, or source revisions
requires a new/recomputed history; matching Options alone does not establish
source compatibility.

An EMA centerline and an ATR width may be joined at the same index only after
both readiness masks are true. EMA alpha=2/(P+1) cannot replace the Wilder
alpha=1/P contract. Combining series does not imply an import dependency:
`atr` imports only `context`, `errors`, `math`, and `math/bits`.

Application-owned arithmetic still needs its own overflow checks. Multiplying a
finite ATR by a channel multiplier or converting it into currency is not covered
by the indicator's finite-result guarantee. ATR is neither variance nor a
confidence interval, and observed zeros must not silently become denominators.


## 8. Complexity, allocation, and concurrency

Let W=34 limbs, D<=660 decimal bytes, and n be the input length.

| Operation   | Time                        | Additional storage / ownership                           |
|-------------|-----------------------------|----------------------------------------------------------|
| New / Reset | O(W), fixed                 | O(W) bounded stream object; Reset allocates nothing      |
| Push        | O(W), hence O(1) in n and P | Fixed local candidate; zero internal heap allocation     |
| Compute     | O(n*W), hence O(n)          | O(n+W); three owned output arrays and one Result         |
| Snapshot    | O(D*W), bounded             | Fixed work buffers plus the returned string when nonzero |
| Restore     | O(D*W), bounded             | O(W); no heap allocation in parsing/validation           |

Zero-allocation tests cover warmup, seed completion, and maturity for multiple
periods, not just the period-one shortcut. Custom contexts can allocate outside
the package; zero kernel allocation is not a hard-real-time guarantee. Runtime
out-of-memory termination is not converted to an error. Checked byte arithmetic
prevents arithmetic wrap, not exhaustion of physical memory.

One Stream has exactly one owner. Concurrent Push/Reset/Restore/Snapshot on the
same instance is unsupported without external synchronization. There are no
internal locks or hidden background workers. Independent streams and batch calls
can be used in parallel; callers must not concurrently mutate input slices.

## 9. Tests, oracle, benchmarks, and verification

All tests use `package atr_test`. Helpers and exact oracles live in
`test_helper_test.go`; there are no production test hooks or export bridges.
The rational oracle computes three exact range candidates, rounds TR, then
independently computes the exact seed or affine rational result. Comparisons are
bitwise, not tolerance-based. Randomized fixtures use reproducible seeds.

Coverage includes the golden vector, gaps in both directions, signed prices,
P1 identity, flat zero ranges, all finite/non-finite domain gates, overflow
thresholds, subnormal ties, large integer periods, ownership, prefix invariance,
cancellation checkpoints, counter overflow, snapshot phases, JSON round trips,
malformed fields, rollback, independent concurrent owners, and zero allocations.
Two public-boundary fuzz targets protect numeric processing and snapshot parsing.
Fuzzing is additional evidence, not a proof over all inputs.

Use the repository's accepted Go version and unchanged root module:

```sh
gofmt -w atr examples/atr
go test ./atr
go test -race ./atr
go vet ./...
go test ./...
go test -race ./...
golangci-lint run
govulncheck ./...
go test ./atr -coverprofile=atr.coverage.out
go tool cover -func=atr.coverage.out
go test ./atr -run '^$' -fuzz '^FuzzATR_PublicBoundaries$' -fuzztime=30s
go test ./atr -run '^$' -fuzz '^FuzzSnapshot_PublicBoundaries$' -fuzztime=30s
go test ./atr -run '^$' -bench . -benchmem
```

Batch benchmarks use n=1e3, 1e5, 1e6 crossed with P=14,64,512. Stream benchmarks
separate mature constant-range updates, changing gaps, and warmup. Inputs and initial warmup are outside the
measured loop. Report Go version, OS/architecture, CPU, ns/op, B/op, and allocs/op;
never promote measurements from a different toolchain into repository CI results.
A verification report should identify the actual toolchain, checked tree, and missing gates.

## 10. Pitfalls and source boundaries

Do not substitute another library's ATR without matching TR0, seed, rounding,
mask alignment, and smoothing. TradingView's official description provides the
three TR candidates and describes RMA as its default, but does not establish
binary64 bit parity with this package's exact-affine contract.

Do not silently interpolate or reorder observations, infer availability from an
index, repair candle geometry, or carry state across an instrument/configuration
change. Corrected historical bars require explicit recomputation from an earlier
valid checkpoint. The library cannot determine whether a supplied bar was closed
or later revised.

The package charter, public GoDoc, and executable examples define the behavior
of the selected source revision. Useful primary references:

- [Go language specification: numeric types and floating-point operations](https://go.dev/ref/spec).
- [Go math/big: independent rational oracle operations](https://pkg.go.dev/math/big).
- [TradingView: ATR definition and smoothing modes](https://www.tradingview.com/support/solutions/43000501823-average-true-range-atr/).

The implementation, not a third-party chart, fixes `atr/wilder-rn/v1` continuation
semantics. A change to those semantics requires a new algorithm identifier.
