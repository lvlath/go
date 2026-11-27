<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  File: docs/SESSIONLEVELS.md
  Package reference: sessionlevels
  Purpose: Teach previous completed session reference levels and their safe composition with aligned market-data features.
  Contract status: Learning and usage guide; package doc.go and source define the public contract.
  Scope: Explicit session manifests, exact high/low/final-close selection, readiness, causality, replay, ownership, and verification.
  License: AGPL-3.0-only. See ../LICENSE.
-->

# SessionLevels: previous completed session reference levels

`github.com/lvlath/go/sessionlevels` selects the high, low, and final close of the
strictly preceding completed session. Every bar of a session with a completed
predecessor receives that previous summary. The current session's developing or final prices never
enter its own reference levels.

This is a pure batch algorithm, not a trading bot. Session calendars, availability,
revisions, missing-bar evidence, orders, fills, and profitability belong to callers.
The module is `github.com/lvlath/go`, with the repository's Go 1.23.4 minimum.
There is no nested module and no dependency on another lvlath package.

## Contents

- [What and why](#what-and-why)
- [Public API](#public-api)
- [Segmentation and availability](#segmentation-and-availability)
- [Mathematical contract](#mathematical-contract)
- [Algorithm and proof of causality](#algorithm-and-proof-of-causality)
- [Validation and error precedence](#validation-and-error-precedence)
- [Ownership and resource bounds](#ownership-and-resource-bounds)
- [Context behavior](#context-behavior)
- [Minimal executable usage](#minimal-executable-usage)
- [Operational scenarios](#operational-scenarios)
- [Composition and correction replay](#composition-and-correction-replay)
- [Verification and benchmarks](#verification-and-benchmarks)
- [Pitfalls and integration boundaries](#pitfalls-and-integration-boundaries)
- [Further reading](#further-reading)

## What and why

A fixed previous-session triple is useful when a risk monitor, feature pipeline,
opening-range audit or historical replay must compare new observations against
reference levels that do not move with the current session's prices. PDH/PDL/PDC
mean previous trading-day high, low and last close; PWH/PWL/PWC refer to the
previous trading week. These labels are meaningful only under the caller's
chosen session policy. A rolling P-bar maximum is not a previous-day high;
an anchored price-volume mean is not a previous close.

The first interval intentionally publishes no measurement. For empty histories,
three non-nil empty price channels and an empty manifest succeed. Nil price
channels are invalid. A successful masked or empty result is not an error.

## Public API

The following declarations summarize the public API; they are not a standalone
Go program. See the complete usage example below.

```go
type Session struct {
    Start    int
    End      int
    Complete bool
}

type Options struct{}

type Result struct {
    PreviousHigh  []float64
    PreviousLow   []float64
    PreviousClose []float64
    Valid         []bool
}

func (r *Result) IsNil() bool

func Compute(
    ctx context.Context,
    high, low, closePrices []float64,
    sessions []Session,
    opts Options,
) (*Result, error)
```

Convenience facades (public signatures; the types above are declared by the
package, not by callers):

```go
func ComputeDefault(ctx context.Context, high, low, closePrices []float64,
    sessions []Session) (*Result, error)
func ComputeDaily(ctx context.Context, high, low, closePrices []float64,
    daySessions []Session) (*Result, error)
func ComputeWeekly(ctx context.Context, high, low, closePrices []float64,
    weekSessions []Session) (*Result, error)
```

| Use case                            | Entry point                             | Caller supplies                               |
|-------------------------------------|-----------------------------------------|-----------------------------------------------|
| Canonical explicit batch            | `Compute(ctx,h,l,c,sessions,Options{})` | A complete observed-index partition           |
| Same batch without options argument | `ComputeDefault(ctx,h,l,c,sessions)`    | The same partition                            |
| Previous trading-day H/L/C          | `ComputeDaily(ctx,h,l,c,daySessions)`   | Day-session intervals, including any live cut |
| Previous trading-week H/L/C         | `ComputeWeekly(ctx,h,l,c,weekSessions)` | Independently prepared weekly intervals       |

All wrappers call `Compute` with `Options{}` and return its result and error
unchanged. They neither inspect time nor validate whether a supplied interval
really represents a day or week. The same slices and manifest give identical
results; using different manifests changes the question, not the kernel.
There is no rolling period, calendar option, current-session mode, Stream or
Snapshot. No implicit previous session is retained between calls.

The six exported error identities are `ErrNilContext`, `ErrNilInput`, `ErrShape`,
`ErrSession`, `ErrBar`, and `ErrNonFinite`. They are variables created with
`errors.New`, not Go constants. Use `errors.Is`; diagnostic text is not a protocol.
`Options{}` is the only configuration, and selection of finite prices needs no
arithmetic overflow category. The result exposes `IsNil`, not a readiness helper:
it returns true exactly when the receiver pointer is nil.

## Segmentation and availability

A `Session` denotes `[Start, End)` in the supplied arrays, not in a global database.
For nonempty input, the first start is zero, every end equals the next start, each
interval is nonempty and within the arrays, and the final end equals `n`. These
conditions prove a strictly increasing, gap-free and overlap-free *index*
partition. They do not prove that a market-data feed contains every expected
observation. Indices refer to the provided slice, not timestamps; verify the
source manifest before mapping it to indices. A left-truncated interval must not
be declared complete merely because it begins at array index zero.

Only the last interval may have `Complete=false`. In particular, an incomplete
session followed by another incomplete session is invalid too. A one-session input
may be complete or incomplete; either way, all rows are masked because no preceding
session is available. A final incomplete session receives the full previous
completed session's summary at every available bar.

`Complete` asserts both finality and accepted coverage under the caller's policy.
The package can validate indices, not whether an exchange bar is missing. One call
must use one instrument, venue, unit convention, adjustment basis, and session policy.
The caller keeps that metadata with its result; Result itself does not store it.

A zero-trade day must not be represented by a fake zero candle or an empty interval.
The choice between "previous trading session" and "previous calendar day" belongs
upstream. A supplied manifest may reflect DST, holidays, half-days, auctions, or
clearing breaks without any change in the kernel.

## Mathematical contract

Let session `j-1` occupy `[a,b)` and session `j` occupy `[u,v)`. For every ready
output index `i` in `[u,v)`, the selected levels are:

![Previous session selection](https://latex.codecogs.com/svg.image?H_%7Bprev%7D%3D%5Cmax_%7Bi%5Cin%5Ba%2Cb%29%7DH_i%2C%5Cquad%20L_%7Bprev%7D%3D%5Cmin_%7Bi%5Cin%5Ba%2Cb%29%7DL_i%2C%5Cquad%20C_%7Bprev%7D%3DC_%7Bb-1%7D)

In plain notation, `PreviousHigh[i] = max(high[a:b])`,
`PreviousLow[i] = min(low[a:b])`, and `PreviousClose[i] = closePrices[b-1]`.
The extrema are selected from the represented inputs, not rounded results of
arithmetic. `Valid[i]` is true exactly when the containing session has a
completed predecessor.

The first session has `Valid=false` and all three payloads are exactly positive
binary64 zero. Later sessions have `Valid=true`, including real zero-valued levels.
An incomplete final session does not introduce an additional warmup.

Every input candle must satisfy:

![Admitted candle](https://latex.codecogs.com/svg.image?L_i%5Cle%20C_i%5Cle%20H_i%2C%5Cquad%20H_i%2CL_i%2CC_i%5Cin%5Cmathbb%7BR%7D_%7Bfinite%7D)

A valid `PreviousClose` is the last bar's close, not an extreme, midpoint, average,
settlement-price substitute, or forward-filled value.

### Tie-breaking and binary64 selection

The kernel seeds each candidate from its first actual high and low. Strict `>`
and `<` comparisons retain the earliest equal extremum. Equal positive and
negative zero compare equal, so the first occurrence's bits are retained. The final
close is copied bit for bit, including its zero sign.

There is no arithmetic on prices. Negative prices, the largest finite magnitudes,
and subnormals are selected without rounding, cancellation, or intermediate
arithmetic overflow. Derived widths or differences in consumer code can still
overflow (for example, subtracting -MaxFloat64 from MaxFloat64); a ready mask
is not a guarantee that a downstream ratio is defined.
The package does not initialize extrema at zero, average
prices, use epsilon, clamp values, or substitute zeros for invalid data.

## Algorithm and proof of causality

```text
reject nil context; return any already observed ctx.Err()
validate non-nil price channels, equal lengths, and the full session partition
    poll context at each 1024-descriptor checkpoint
for each bar in increasing index order:
    poll context at each 1024-bar checkpoint
    reject immediately on NaN/Inf in high, low, or close
    remember the first finite invalid candle, if any
reject the remembered candle only if the entire finite scan succeeds
check context; allocate zeroed independent output channels

hasPrevious = false
for session in ascending input order:
    currentHigh = high[session.Start]
    currentLow = low[session.Start]
    for i in [session.Start, session.End):
        poll context at the global bar checkpoints
        if hasPrevious:
            output[i] = previousHigh, previousLow, previousClose, true
        update private currentHigh/currentLow with strict comparisons
    if session.Complete:
        previousHigh = currentHigh
        previousLow = currentLow
        previousClose = closePrices[session.End - 1]
        hasPrevious = true

check context immediately before publication
return the complete result
```

```text
session 0 prices ----> private summary 0 ----> publish throughout session 1
session 1 prices ----> private summary 1 ----> publish throughout session 2
session 2 incomplete -> private candidate --> discard at end of batch
       |
       +-- its published levels still come exclusively from summary 1
```

Initially no summary exists, so the entire first interval remains masked. Suppose
that before session `j`, the saved scalars describe exactly session `j-1`. They are
not modified inside session `j`'s bar loop, so every published row there uses only
that preceding summary. A complete session commits its own summary after the loop,
establishing the induction hypothesis for `j+1`. Validation disallows an earlier
incomplete session, so no session can accidentally skip an unfinished predecessor.

Consequently, an admissible mutation confined to session `j` cannot change outputs
in sessions `0..j`. With unchanged segmentation, it can affect only session `j+1`,
not all later sessions. Oracle and prefix comparisons check the selected bits.
A mutation that introduces NaN or an invalid candle instead rejects the entire
batch; numerical prefix invariance does not promise success for an invalid extension.

## Validation and error precedence

| Stage                   | Condition                                                               | Result               |
|-------------------------|-------------------------------------------------------------------------|----------------------|
| Context                 | Nil interface                                                           | `nil, ErrNilContext` |
| Context                 | Already canceled or expired                                             | `nil, ctx.Err()`     |
| Nil inputs              | Any H/L/C slice is nil                                                  | `nil, ErrNilInput`   |
| Shape                   | H/L/C lengths differ                                                    | `nil, ErrShape`      |
| Empty/session structure | Nonempty sessions with zero bars                                        | `nil, ErrSession`    |
| Session structure       | Missing, empty, inverted, out-of-range, gapped or overlapping intervals | `nil, ErrSession`    |
| Session completion      | Any non-final interval is incomplete                                    | `nil, ErrSession`    |
| Numeric input           | NaN or either infinity anywhere                                         | `nil, ErrNonFinite`  |
| Candle domain           | A finite row violates `low <= close <= high`                            | `nil, ErrBar`        |
| Computation/publication | Observed cancellation                                                   | `nil, ctx.Err()`     |

After structure admission, the numeric validator visits indices ascending, checking
high, low, and close. It remembers the first finite bad candle instead of returning
immediately: a non-finite value at a later index has the explicitly required global
priority. The algorithm validates masked bars and the final incomplete session too.

An input error and cancellation racing after entry have no deterministic winner.
Whichever is observed first may be returned. No error includes a partial Result.

A useful nil/empty distinction:

| H/L/C inputs                          | Sessions     | Outcome with active context                 |
|---------------------------------------|--------------|---------------------------------------------|
| Any nil price slice                   | Any          | `ErrNilInput`, nil result                   |
| Three non-nil empty slices            | nil or empty | Non-nil Result, four non-nil empty channels |
| Three non-nil empty slices            | Nonempty     | `ErrSession`, nil result                    |
| Three present slices, unequal lengths | Any          | `ErrShape`, nil result                      |
| Nonempty equal-length prices          | nil or empty | `ErrSession`, nil result                    |

Unlike some other packages, missing prices are not classified simultaneously as
`ErrShape`. Check the owning package's validation law rather than sharing an
error-precedence table across indicators.

## Ownership and resource bounds

Nonempty calls allocate three separate `[]float64` arrays, one `[]bool` array, and
the result carrier. Outputs do not alias inputs, one another, or other calls. Each
channel has capacity equal to length, so appending to one cannot overwrite another.
Successful empty channels are non-nil with zero length and capacity; they contain
no mutable storage. The package neither mutates nor retains caller data.

Copying a Result value copies slice headers, not their contents. Copy individual
slices before independent mutation. Independent calls may share stable read-only
input; caller writes during Compute are unsupported. A result may be shared for
reading, but concurrent writes require external synchronization.

The whole API, including validation and zeroing, takes `O(n+s)` time. The kernel
visits every bar exactly once. Session validation costs `O(s)` and numeric validation
`O(n)`. These are additional scans, not an inaccurate claim that the entire facade
reads each input only once. Output space is `O(n)`; auxiliary scalar state is `O(1)`.
There are no intermediate matrices, per-session summary arrays, or hot-loop allocations.

The owning API is not allocation-free. Its nonempty allocation layout is four
output buffers plus the result carrier, independent of `s`. The actual allocation
count is a toolchain/workload measurement; use `-benchmem` to reproduce it.
For large inputs the numeric payload alone is `24*n` bytes, plus `n` bool
entries, slice headers, the carrier and allocator overhead (Go implementation
sizes apply). These figures describe storage, not a checked memory budget.
The facade wrappers add no output buffers or validation passes. No packed `3*n`
or `24*n` allocation length is calculated. Each float allocation uses an existing float-slice length,
which already witnesses representability of that individual buffer size. This
avoids an unreachable size-guard branch and an invented overflow error.

Representable sizes do not guarantee sufficient physical memory. The package
neither reserves RAM nor attempts recovery from runtime memory exhaustion.

## Context behavior

Compute polls at entry, every 1024 session descriptors in structural validation,
every 1024 bar indices in numeric validation and in the kernel, before output
allocation, and immediately before publication. The bounds are per scan, not
universal counts of machine operations or wall-clock guarantees.

Runtime allocation is not interruptible. Cancellation just after the final check
may follow a successful return. A valid usable context implementation is required;
a typed-nil or misbehaving custom implementation is not recognized with reflection
or recovered after a failure. Tests cancel a real context at each observed checkpoint,
including late computation and the final publication check, without sleeps or
call-stack inspection.

## Minimal executable usage

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/lvlath/go/sessionlevels"
)

func main() {
	high := []float64{2, 4, 9, 8}
	low := []float64{1, 2, 6, 5}
	closePrices := []float64{1.5, 3, 7, 6}
	sessions := []sessionlevels.Session{
		{Start: 0, End: 2, Complete: true},
		{Start: 2, End: 4, Complete: false},
	}
	result, err := sessionlevels.Compute(
		context.Background(), high, low, closePrices, sessions, sessionlevels.Options{},
	)
	if err != nil {
		log.Fatal(err)
	}
	for i := range result.Valid {
		if result.Valid[i] {
			fmt.Printf("bar=%d H=%.0f L=%.0f C=%.0f\n", i,
				result.PreviousHigh[i], result.PreviousLow[i], result.PreviousClose[i])
		}
	}
}
```

Output:

```text
bar=2 H=4 L=1 C=3
bar=3 H=4 L=1 C=3
```

For an error-aware service, use `errors.Is(err, sessionlevels.ErrSession)` or
`errors.Is(err, context.Canceled)` rather than matching text. A false Valid entry
is a successful absence of a measurement, not an error.

## Operational scenarios

All five [executable package examples](../sessionlevels/example_test.go) use
reproducible engineered market-data case studies, not independently audited
exchange observations, current prices, or execution records. Reusable fixtures
are defined in [test_helper_test.go](../sessionlevels/test_helper_test.go).

| Regime  | Executable example                                 | Evidence and operational meaning                                                                                                                                    |
|---------|----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Medium  | `ExampleComputeDaily_nyseOpeningRiskGate`          | 390 prior regular-session minute bars and three finalized next-session bars. PDH=103, PDL=97, PDC=101; a close at 104 crosses PDH, the following 105 remains above. |
| Medium+ | `ExampleComputeWeekly_holidayAdjustedTreasuryWeek` | Four and five daily observations in completed weeks, then two in a live week. Previous-week H/L/C are 115/100/113 and 121/110/120 respectively.                     |
| Medium+ | `ExampleComputeDefault_liveReplayAudit`            | Cuts at 391, 392 and 393 preserve the reference. A late high changed to 1000 cannot change its own session's output bits.                                           |
| Special | `ExampleCompute_signedCalendarOvernightSpread`     | Signed synchronized spread observations: prior H/L/C = -2/-10/-4, ready. Negative prices are not missing values.                                                    |
| Special | `ExampleCompute_feedIntegrityGate`                 | Overlap is rejected before numeric validation; after structural repair, later NaN beats earlier finite geometry failure. Both return no result.                     |

The prior-day case uses a synthetic NYSE regular-hours manifest (09:30-16:00),
not an embedded exchange calendar. Its crossing predicate detects a transition
relative to the same fixed PDH on adjacent closes; it is not a persistent
first-cross-of-day latch or an assumed fill. Such a latch and execution policy
belong to the consumer. The weekly and overnight cases use supplied intervals,
not seven-day bins or UTC-day splitting. Spread candles must originate from
synchronized spread observations, not independent leg highs and lows.

Run package examples and the [standalone offline scenario](../examples/sessionlevels/main.go)
from the repository root:

```sh
go test ./sessionlevels -run '^Example' -count=1 -v
go run ./examples/sessionlevels
```

## Composition and correction replay

The diagrams below describe caller-side data flow, not Go import dependencies.
The package's production imports are only `context`, `errors`, `fmt`, and `math`.

```text
feed + venue/instrument policy + as-of revision evidence
                       |
             caller admits finalized H/L/C
             and resolves explicit sessions
                       |
        +--------------+--------------+
        |                             |
 day partition                  week partition
        |                             |
 ComputeDaily                   ComputeWeekly
        |                             |
 PDH/PDL/PDC + Valid            PWH/PWL/PWC + Valid
        +--------------+--------------+
                       |
       caller aligns row identity, availability, units and masks
                       |
           alert / model feature / review (not an order fill)
```

| Package                          | Information set at index i                     | Consumer caution                                                                     |
|----------------------------------|------------------------------------------------|--------------------------------------------------------------------------------------|
| `sessionlevels`                  | Strictly previous completed explicit session   | Constant levels within the current interval; first session masked                    |
| `donchian.ComputeExcludeCurrent` | Previous P observed bars, excluding i          | Window slides each bar and can cross session boundaries; not PDH                     |
| `donchian.ComputeIncludeCurrent` | P observed bars including i                    | Requires current bar finality; not a prior threshold                                 |
| `vwap.ComputeDefault`            | Weighted prefix through i since explicit reset | Current-inclusive, ready after positive weight; not a prior-session close            |
| `atr.ComputePeriod`              | Gap-aware TR and smoothed ranges through i     | For a scale available before i use separately ready ATR[i-1] and verify availability |

A daily and weekly partition may both cover the same intraday observations. Do
not align a daily-bar weekly result with minute indices by matching lengths or
array offsets: either use the same row domain in both calls, or explicitly map
weekly data to minute observations using source availability. SessionLevels
requires no graph/matrix storage and does not import `series`, ATR, Donchian or
VWAP. `series` can help check cadence on caller-selected domains, but it cannot
supply a market calendar or prove full session coverage.

For a live cut, retain complete preceding sessions, clip the last interval's End
to the supplied length, and set its Complete flag false if truncated. A call
containing only the new current session correctly has no predecessor. To recover
its prior levels, include the complete previous interval and rebase all indices;
never reconstruct levels by calling this API on a truncated predecessor.

With unchanged segmentation and accepted geometry, a correction within session j
can affect only j+1's three level channels, never j itself or j+2. A change to the
manifest or availability policy may affect more of the history; rerun admission
and recompute. Publish the new artifact only after success, and retain source
identity, session policy, as-of revision, and row mapping beside it. The package
has no incremental persistence or snapshot ABI. Repeated full-prefix calls cost
the sum of their individual O(n+s) workloads; they are not a constant-time stream.

A prior threshold is available before a current bar only if the source finalized
and delivered its predecessor in time. A close-based crossing is observed after
that close. Do not label it as a guaranteed trade at that close or an earlier
price. Derived distances and normalizations need independent finiteness and
nonzero-denominator checks; a valid zero may be a genuine level.

## Verification and benchmarks

All tests use `package sessionlevels_test` and public calls. There are no test-only
exports, internal-package test files, reflection, unsafe input construction,
embedded fixtures, panic/recovery calls, or lint suppressions.

The suite protects the exact golden vector; current-session future mutations;
revision horizons; every admissible prefix; single-session warmup; true final close;
invalid coverage and early incomplete flags; nil/empty/shape precedence; all non-finite
channels; finite invalid bars; negative prices; signed-zero ties; extreme finite and
subnormal values; nilable interfaces; ownership; input aliasing; read-only concurrent
calls; cancellation; bounded-allocation behavior; facade parity (including error and
context identity); and seeded plus fuzzed oracle parity.

The independent oracle locates the containing session and directly re-extracts the
entire preceding interval for every output row. It uses reverse traversal and
non-strict comparisons to independently obtain the same earliest-equal tie rule.
Its cost is `O(n*s + sum(m[j]*m[j-1]))`, worst-case `O(n^2)`. Calling this generally
`O(n*s)` would be false: with two half-length sessions, the repeated previous-range
scan already costs quadratic time. Production remains `O(n+s)`.

```sh
gofmt -w sessionlevels/*.go examples/sessionlevels/*.go
go test -count=1 -covermode=atomic -coverprofile=sessionlevels.cover ./sessionlevels
go tool cover -func=sessionlevels.cover
go test -race ./sessionlevels
go test ./sessionlevels -run '^$' -fuzz '^FuzzCompute$' -fuzztime=30s
go vet ./...
go test ./...
go test -race ./...
golangci-lint run
govulncheck ./...
go test ./sessionlevels -run '^$' -bench '^BenchmarkCompute$' -benchmem
```

Require at least 92% statement coverage overall and for each executable production
file. `doc.go` and sentinel-only `errors.go` have no instrumented executable statements;
report them as N/A, not as fabricated 100% measurements. Local Go coverage is not
proof of a completed remote Codecov upload or of every boolean branch outcome.

Benchmarks use `n=1000,100000,1000000` and synthetic session lengths
`1,14,64,512,n`, producing different session counts. These widths are workload
parameters used to construct explicit intervals, not public period options.
Fixture generation is excluded; validation, zeroing, allocation, and computation
are included. Record Go version, OS, architecture, CPU, command, ns/op, B/op, and
allocs/op. Compare optimizations only after oracle parity; short runs are smoke
measurements rather than statistically established performance claims.

## Pitfalls and integration boundaries

Never replace an exchange session with a UTC day or a fixed number of minutes.
Never feed End beyond the current history, infer finality from length, repair gaps,
sort sessions, backfill missing observations, or reuse a result across a changed
unit/session policy without explicit recomputation.

Do not publish a current candidate before its session ends. Do not interpret a
masked zero as a price, use IsNil as a readiness predicate, or substitute a high/low
for the final close. Do not assume a false final Complete invalidates preceding
levels, or that a true first Complete removes warmup.

When combining these levels with ATR or Donchian, align features to the caller's
information-availability boundary. A threshold known before the current bar is
not permission to use a later close as an earlier fill. The package cannot prove
historical feed availability or correct revisions. All such evidence stays with
the upstream manifest and downstream execution logic.

## Further reading

- [Package charter](../sessionlevels/doc.go) and [canonical API](../sessionlevels/api.go): implemented public laws and facades.
- [Tutorial: previous completed session levels](TUTORIAL.md#11aa-sessionlevels-foundations-previous-completed-session-levels): dual daily/weekly manifests over a shared input domain.
- [SessionLevels FAQ](FAQ_%26_TIPS.md#13z-sessionlevels-faq-manifests-readiness-and-causality): failure diagnosis and replay boundaries.
- [Series](SERIES.md), [Donchian](DONCHIAN.md), [ATR](ATR.md), [VWAP](VWAP.md): separate feature contracts for caller-side composition.
- [TradingView session concepts](https://www.tradingview.com/pine-script-docs/concepts/sessions/): background on session policies, not a data or calendar dependency.
- [Universal Engineering Standard](lvlath_UES.md): verification and contract governance.
