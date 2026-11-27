<!--
File: docs/DRAWDOWN.md
Package: github.com/lvlath/go/drawdown
Purpose: sampled equity drawdown and exact witness audit
Contract status: Usage guide; package doc.go and current Go source define the public API.
Scope: observed equity batch, exact arithmetic, witness, ownership, validation, NAV admission and risk reporting.
SPDX-License-Identifier: AGPL-3.0-only
Copyright (C) 2025-2026 katalvlaran
-->
# Drawdown — sampled equity risk audit

The package is standalone pure Go under `github.com/lvlath/go` and uses no
other lvlath package. See the selected module's `go.mod` for the required Go version.

## Contents

- [What and why](#what-and-why)
- [Public API](#public-api)
- [Input admission and NAV](#input-admission-and-nav)
- [Mathematical contract](#mathematical-contract)
- [Exact arithmetic](#exact-arithmetic)
- [Algorithm](#algorithm)
- [Errors and ownership](#errors-and-ownership)
- [Go example](#go-example)
- [Operational scenarios](#operational-scenarios)
- [Pitfalls and composition](#pitfalls-and-composition)
- [Verification and benchmarks](#verification-and-benchmarks)
- [Further reading](#further-reading)

## What and why

`drawdown.Compute(ctx, equity)` produces the positive relative underwater
profile of a chronological equity history and one exact maximum drawdown
witness. It does not estimate intrabar extrema, infer cashflows, reconstruct
positions or promise strategy performance. Feed externally prepared NAV if
cashflows are present. Record the currency, observation schedule and revision
policy alongside the result in the evaluation artifact.

### Which question is being answered?

| Metric                | Question                                                                                 | Source                                      |
|-----------------------|------------------------------------------------------------------------------------------|---------------------------------------------|
| Current sampled depth | How far below the running peak is the latest observed mark?                              | `Depth[len(Depth)-1]`                       |
| Historical maximum    | What was the deepest observed relative loss over this history?                           | `Maximum`                                   |
| Exact witness         | Which earliest peak/trough pair proves the maximum and when was its peak first regained? | `PeakIndex`, `TroughIndex`, `RecoveryIndex` |
| Terminal return       | How much did admitted NAV change relative to its origin?                                 | Caller calculation, not drawdown            |

Two strategies can have the same terminal return and very different path risk.
A recovered account can have current depth zero but a nonzero historical MDD.

## Public API

```go
type Result struct {
    Depth         []float64
    Maximum       float64
    PeakIndex     int
    TroughIndex   int
    RecoveryIndex int
}

func (r *Result) IsNil() bool
func Compute(ctx context.Context, equity []float64) (*Result, error)
func ComputeMaximum(ctx context.Context, equity []float64) (float64, error)
func ComputeCurrentDepth(ctx context.Context, equity []float64) (float64, error)
```

These are interface declarations, not a standalone Go program. `Compute` is the
sole canonical computation. The scalar wrappers project `Maximum` or the last
`Depth` element after a successful full batch; they discard the other evidence.
They do not introduce different arithmetic, error precedence, parameters or
recovery policy. Each still takes O(n) time and allocates O(n) transient output.
If a report needs several fields, run `Compute` once instead of two scalar calls.
On failure, scalar wrappers return +0 with the unchanged error. Check err first:
that zero is not a risk measurement. There is no Options, Stream or Snapshot.

## Input admission and NAV

The caller supplies chronological observed marks in a single economic unit,
including a strictly positive origin. Retain mark IDs, timestamps, valuation
policy and revision identity outside the algorithm. Preserve original ordering;
missing marks are not silently removed, interpolated or filled by this package.
A timestamp grid is neither a proof of completeness nor a model of intrabar equity.

A capital subscription increases assets without necessarily increasing per-unit
investment performance; a redemption may decrease assets without an investment
loss. For investment risk, an external fund administrator or ledger must prepare
cashflow-neutral NAV. Under a simplified unitized ledger, subscriptions/redemptions
at the preceding NAV change the unit count, and the next marked balance divided
by its post-flow units gives NAV per unit. Real dealing price, valuation, fees,
rounding and reconciliation policies must be established by the caller. The
`ExampleCompute` executable demonstrates this distinction with engineered ledger
records: NAV MDD 18.18% versus unadjusted balance MDD 29.09%.

`series` may check grids and compute returns for other metrics, but its return
channels are not drawdown equity. SessionLevels, ATR, Donchian and VWAP measure
prices or ranges, not capital. Caller-side composition is not an import hierarchy.

## Mathematical contract

![Depth](https://latex.codecogs.com/svg.image?D_t%3D%5Cfrac%7BP_t-E_t%7D%7BP_t%7D%2C%5Cquad%20P_t%3D%5Cmax_%7Bs%5Cle%20t%7DE_s)

The first observation is finite and strictly positive; later observations are
finite and nonnegative. Every depth is the nearest-even binary64 rounding of
the exact ratio of the supplied binary64 values. Zero depth is positive zero.
There is no warmup, mask, signed loss or missing observation convention.

Witness selection compares exact residual ratios, not rounded depths:

![Compare](https://latex.codecogs.com/svg.image?D_A%3ED_B%5Ciff%20E_AP_B%3CE_BP_A)

Equal exact ratios keep the earliest trough. The peak is the earliest occurrence
of its running maximum, even across previously recovered episodes. Recovery is
the first index strictly after the selected trough with equity at least that
peak, or -1. When there is no drawdown, all three indices are zero. Maximum
copies Depth[TroughIndex] bit for bit. Rounded Maximum=1 does not prove zero
capital: tiny positive residuals can round to the same depth as total loss.

### Witness and episode semantics

For `[100,120,90,120,80]`, the equal 120 peak at index 3 does not replace the
first occurrence at index 1. The later trough at index 4 has the larger exact
loss and remains unrecovered. A recovery before index 4 is irrelevant to that
witness. For `[100,50,100,50,100]`, the equal exact depth keeps trough 1 and
recovery 2, not the later episode.

```text
index       0      1      2      3      4
Equity    100    120     90    120     80
Peak      100    120    120    120    120
Depth       0      0   1/4       0    1/3
                   ^                    ^
              PeakIndex            TroughIndex
RecoveryIndex = -1 (no subsequent observation reaches 120)
```

Distinct exact depths may have identical rounded bits. For
`[1, 0x1p-55, 2, 0x1p-55, 2]`, both losses publish as 1, yet the later residual
ratio is smaller; the correct witness is 2/3/4. Check the original equity at
TroughIndex when actual zero capital matters.

## Exact arithmetic

### Near the peak

For E >= P/2, Sterbenz subtraction is exact, followed by a single correctly
rounded division. For `[1, math.Nextafter(1,0)]`, the exact depth is `0x1p-53`.
A strict decrease between positive binary64 operands has relative separation
at least 2^-53, so no positive depth can round to zero. Do not replace the
formula with `1-E/P`: it can lose precision by rounding the residual ratio first.
The simple subtraction/division expression is not enough for all deep losses,
because the intermediate difference may round before division. For example,
`[3, math.Nextafter(3,0)]` has exact depth `2^-51/3`, not the rounded-complement
result of `1-E/P`. At the subnormal boundary, the subtraction remains exact
on the integer subnormal grid even if the half-peak comparison rounded.

### Deep loss on the integer grid

Otherwise, for positive E, D lies in (0.5,1). Zero E gives depth exactly 1.
Let N=2^53. Compute the nearest-even integer
rounding of N*E/P by integer quotient and remainder, then subtract that integer
from N. Since N is even, rounding commutes with this integer complement,
including ties. Multiplication by 2^-53 is exact. No rounded floating ratio
is subtracted from 1. Exponent gaps above 54 give residual below half a grid
unit; gap 54 needs only a mantissa comparison. More precisely, with
`P=mP*2^eP`, `E=mE*2^eE` and 53-bit normalized mantissas, the residual grid
coordinate is `mE*2^(53-gap)/mP`, where `gap=eP-eE >= 1`.

- `gap > 54`: coordinate < 1/2; the residual rounds to zero.
- `gap == 54`: coordinate is `mE/(2*mP)`; it rounds to one only if `mE > mP`.
- `1 <= gap <= 53`: form a two-word integer numerator, divide by `mP`, and
  round quotient upward iff twice the remainder exceeds the divisor, or if it
  equals the divisor and the quotient is odd.

The numerator requires at most 105 bits (a 53-bit mantissa shifted by at most
52). Its high word is less than 2^42, while the divisor is at least 2^52.
`bits.Div64` is therefore safe; the remainder doubled fits within 54 bits.
The integer complement is <=2^53, exactly convertible to float64. Scaling
by the exact step 2^-53 publishes one correctly rounded result.

### Exact product ordering

For witness comparison, equal denominators reduce to an equity comparison and
zero residuals are ordered explicitly. Otherwise normalized significands are
53-bit integers, including subnormal operands. Products lie in [2^104,2^106)
and have 105 or 106 bits. Exponent gaps of two or
more decide ordering directly; otherwise align by at most one bit in two
64-bit words. The exact strict comparison retains ties without epsilon.
All exponents fit signed `int`, and all shifted products fit 107 bits, including
on 32-bit targets. Production never needs arbitrary-precision allocation.
The tests use `math/big.Rat` independently as an oracle, not as a second kernel.

## Algorithm

```text
validate context, length, first observation
allocate zeroed detached Depth
peak = 0; witness = (0,0,0)
for i = 1..n-1:
    poll cancellation at stride boundaries
    validate finite value, then sign
    if equity[i] >= equity[peak]:
        update peak only on strict increase
        continue
    Depth[i] = correctly rounded exact depth
    if no prior trough or exact candidate depth > exact best depth:
        witness.peak = peak; witness.trough = i
Maximum = Depth[witness.trough]
if witness.trough != 0:
    recovery = -1
    for j = trough+1..n-1:
        poll cancellation at suffix-relative stride boundaries
        if equity[j] >= equity[witness.peak]: recovery = j; break
check context; publish result
```

```text
observations --> validate + running peak --> Depth + exact witness
                                                |
                                                v
                                      recovery suffix scan
                                                |
                                                v
                                       final context check
                                                |
                                                v
                                          owning Result
```

Time O(n), at most two passes, auxiliary space O(1), output O(n). No
per-observation heap allocation on successful calls. Input values are checked
in the private first pass; errors discard unpublished output. Independent calls
may share read-only input, but concurrent mutation is unsupported. A late invalid
mark after an earlier recovered total loss still invalidates the full batch.
A scalar projection is not a cheaper summary-only scan.

## Errors and ownership

| Condition                      | Classification  |
|--------------------------------|-----------------|
| nil context interface          | ErrNilContext   |
| cancellation/deadline          | ctx.Err()       |
| nil or empty history           | ErrInsufficient |
| NaN or either infinity         | ErrNonFinite    |
| finite initial <=0 or later <0 | ErrEquity       |

Context precedes shape, then input errors are detected in ascending index order,
with finiteness before domain at each index. Concurrent cancellation and input
failure follow first detection. Poll at least every 1024 observations and
immediately before publication. All errors return nil Result; use errors.Is.
No input mutation or retention; caller owns the returned Depth. IsNil checks
pointer nilability, not validity of a manually constructed zero-value Result.
A typed-nil `*Result` implements the structural `core.Nilable` interface without
importing core. Copying Result by value aliases the Depth slice; clone the slice
explicitly for independent edits. Caller modifications can invalidate the
relationship between published fields. No input, output or shared state is
retained by the package.

Context must be a usable non-nil interface. Typed-nil or misbehaving custom
contexts are outside the contract. Polling is a bound on observations, not
hard real-time latency; allocation is not interruptible. Cancellation observed
after the final successful context check cannot retract publication. The fixed
numeric domain excludes arithmetic overflow of a valid depth; normal runtime
memory exhaustion is not converted to a package sentinel.

## Go example

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/lvlath/go/drawdown"
)

func main() {
    r, err := drawdown.Compute(context.Background(), []float64{100,120,90,120,80})
    if err != nil { log.Fatal(err) }
    fmt.Println(r.Maximum, r.PeakIndex, r.TroughIndex, r.RecoveryIndex)
    // 0.3333333333333333 1 4 -1
}
```

## Operational scenarios

All five [executable examples](../drawdown/example_test.go) use offline
engineered fixtures. They demonstrate operational workflows rather than claim
historical market performance.

| Example                         | Workflow and checked evidence                                                                                              |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `ExampleCompute`                | Unitized fund ledger: subscriptions/redemptions, raw-balance distortion, NAV witness IDs and final NAV.                    |
| `ExampleCompute_backtest`       | 500 weekday marks; +52% terminal return, 25% MDD, caller review threshold, exact peak/trough/recovery dates and intervals. |
| `ExampleCompute_trailingLimit`  | Selected available prefixes, inclusive rounded 10% review threshold, recovery and historical MDD retained.                 |
| `ExampleCompute_crisisRecovery` | 2008-style slow and 2020-style fast synthetic stress paths, full underwater profile counts and an unrecovered cut.         |
| `ExampleCompute_precision`      | Adjacent binary64 values, rounded-depth collisions, exact witness and true zero capital on extension.                      |

Run the independent [standalone audit](../examples/drawdown/main.go) with
`go run ./examples/drawdown`. It prints the observed profile and mark-identified
witness. For an admitted external history, the command also accepts CSV on
standard input:

```sh
go run ./examples/drawdown -stdin < nav.csv
```

```csv
timestamp,nav
2025-01-01T00:00:00Z,100
2025-01-02T00:00:00Z,50
2025-01-03T00:00:00Z,100
```

The command rejects non-increasing RFC3339 timestamps or malformed records,
then delegates numeric admission to Compute. It does not verify exchange
coverage, unit identity, revision provenance or cashflow adjustment. Report I/O
is not transactional, and a blocked input read is not interruptible by the
context alone. The tutorial includes a complete minimal report program.

## Pitfalls and composition

- Depth is prefix invariant; the final witness and recovery are retrospective.
  Do not use the final summary as a feature at an earlier decision time.
- For a trailing sampled limit, inspect the last Depth, not the historical MDD.
  A comparison against a rounded depth is a rounded threshold policy, not an
  exact rational threshold. Repeated batch calls are not a streaming engine.
- RecoveryIndex-PeakIndex measures observation intervals only for a recovered
  positive drawdown;
  it is not calendar time. Use external timestamps and handle -1 explicitly.
- Exact positive scaling preserves ratios when the scaled inputs are exact;
  arbitrary rounded currency conversions need not preserve ties.
- Zero equity later is valid. Negative equity is outside this contract.
- Do not compare sampled equity MDD directly with intrabar tester MDD without
  aligning sampling and execution assumptions.
- Calmar and Ulcer Index can consume the relevant summary/profile externally;
  Sharpe and Sortino require return statistics, not MDD alone.


## Verification and benchmarks

Use the module toolchain and the actual repository's configured lint/CI gates.
The external tests exercise the exact oracle, bit-level adjacent/half-grid cases,
subnormal grid, exact ties at different peaks, rounded collisions, late invalid
rows, prefix/scale laws, independent ownership, deterministic cancellation,
wrapper parity and public-boundary fuzzing. No embedded fixtures or test-only
production exports are needed. Independent `big.Rat` tests intentionally rescan
small prefixes rather than reuse the production comparator or rounding.

```sh
gofmt -w drawdown examples/drawdown
go test ./drawdown -count=1
go test -race ./drawdown
go test ./drawdown -run '^Example' -count=1 -v
go test ./drawdown -count=1 -covermode=atomic -coverprofile=drawdown.cover
go tool cover -func=drawdown.cover
go test ./drawdown -run '^$' -fuzz '^FuzzDrawdown_PublicBoundary$' -fuzztime=10s
go test ./drawdown -run '^$' -bench '^BenchmarkCompute$' -benchmem
go test ./...
go test -race ./...
go vet ./...
golangci-lint run
govulncheck ./...
```

Require at least 92% statement coverage overall and per executable production
file; files without instrumentable statements are N/A. Coverage alone is not a
proof of exactness. The shallow and deep benchmark regimes each run at
1,000, 100,000 and 1,000,000 observations, including allocation, validation and
recovery. The fixed-width kernel has O(1) workspace; report ns/op, B/op,
allocs/op, Go/OS/CPU and the actual command. Keep fixture generation outside the
measurement. Short runs are smoke evidence, not statistically established speedups.

## Further reading

- [Package charter](../drawdown/doc.go), [facades](../drawdown/api.go), [exact comparison](../drawdown/witness.go) and [kernel](../drawdown/compute.go).
- [Tutorial](TUTORIAL.md#11ab-drawdown-foundations-sampled-equity-risk-and-recovery) and [FAQ](FAQ_%26_TIPS.md#13aa-drawdown-faq-exact-witness-and-nav-boundaries).
- [SessionLevels](SESSIONLEVELS.md), [Donchian](DONCHIAN.md), [VWAP](VWAP.md), [ATR](ATR.md): separate price-feature inputs, not equity adapters.
- [Series](SERIES.md) and [Stats](STATS.md): independent caller-side grid, returns and statistics tools.
- [TradingView intrabar drawdown](https://www.tradingview.com/support/solutions/43000681690/): a distinct sampling/execution convention; not a data dependency.
- [Universal Engineering Standard](lvlath_UES.md): verification and contract governance.
