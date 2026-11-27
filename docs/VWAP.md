<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  File: docs/VWAP.md
  Package reference: github.com/lvlath/go/vwap
  Purpose: explain exact anchored price-volume means, readiness, and recoverable operation.
  Contract status: operational guide; vwap/doc.go and the public source define API behavior.
  Scope: explicit segmentation, represented-binary64 arithmetic, errors, ownership,
    cancellation, bounded snapshots, causal admission, examples, and Go-native verification.
  License: AGPL-3.0-only; see ../LICENSE.
-->

# VWAP: exact volume-weighted means with explicit anchors

## Contents

- [What and why](#what-and-why)
- [Price meaning and compatible units](#price-meaning-and-compatible-units)
- [Public API](#public-api)
- [Mathematical contract](#mathematical-contract)
- [Canonical transition and batch structure](#canonical-transition-and-batch-structure)
- [Anchors, zero-volume phases, and state transitions](#anchors-zero-volume-phases-and-state-transitions)
- [Validation and error precedence](#validation-and-error-precedence)
- [Transactionality and context](#transactionality-and-context)
- [Snapshot/Restore and trust boundaries](#snapshotrestore-and-trust-boundaries)
- [Ownership, nil safety, and concurrency](#ownership-nil-safety-and-concurrency)
- [Lifecycle and nil behavior](#lifecycle-and-nil-behavior)
- [Complete Go example](#complete-go-example)
- [Executable scenarios and data provenance](#executable-scenarios-and-data-provenance)
- [Composition, corrections, and application boundaries](#composition-corrections-and-application-boundaries)
- [Tests and independent oracles](#tests-and-independent-oracles)
- [Complexity, allocation, and benchmarks](#complexity-allocation-and-benchmarks)
- [Verification commands and integration](#verification-commands-and-integration)
- [Pitfalls and non-goals](#pitfalls-and-non-goals)
- [Contract clarifications and file routing](#contract-clarifications-and-file-routing)
- [References](#references)

## What and why

`github.com/lvlath/go/vwap` computes a cumulative price-volume average within
explicitly marked segments. Session VWAP and anchored VWAP use exactly the same
calculation. The difference is the reset schedule supplied by the application.

This package is an arithmetic processor, not a trading strategy, order-book
reconstructor, execution simulator, calendar, or rolling-window estimator.
It has no mandatory imports from other lvlath packages. Production imports are
only `context`, `errors`, `math`, and `math/bits`.

The [package charter](../vwap/doc.go), [public facade](../vwap/api.go),
[types](../vwap/types.go), and [sentinels](../vwap/errors.go) describe the
implemented behavior. The continuation identifiers are `Version=1` and
`Algorithm="vwap/exact-weighted/v1"`. The single module is `github.com/lvlath/go`;
read [go.mod](../go.mod) for its minimum toolchain (Go 1.23.4).
There is no nested module or package-wide mutable numerical policy.

## Price meaning and compatible units

Exact arithmetic does not repair an inappropriate observation definition.
Declare the instrument, price units, weight units, observation universe, and
anchor policy before admitting records:

| Supplied price and weight                                        | Meaning of the resulting benchmark      | What it does not establish                                               |
|------------------------------------------------------------------|-----------------------------------------|--------------------------------------------------------------------------|
| Every admitted trade price and its executed base quantity        | VWAP of that admitted trade tape        | Completeness of a venue/consolidated feed or best execution              |
| Bar close or caller-computed HLC3 and bar volume                 | Volume-weighted **bar-price proxy**     | Exact VWAP of the trades hidden inside each bar                          |
| A paired spread and the corresponding completed matched quantity | Volume-weighted signed paired basis     | Difference of independently weighted leg benchmarks                      |
| Each child-order fill price and executed quantity                | Average execution price for those fills | Fees, impact attribution, unfilled quantity, or implementation shortfall |

A quote update with zero weight may be a useful admitted observation, but is not
an executed trade. A cumulative session-volume counter is not a per-observation
weight: its reset, revision, and differencing rules belong in the source adapter.
Never mix shares, contracts, or quote-currency turnover without an explicitly
justified conversion. Negative weights are not a cancellation/correction API.

Prices and weights are already represented binary64 numbers. Input conversion,
price-proxy arithmetic, currency conversion, rounding to exchange ticks, and
record selection occur outside this package. In particular, the package cannot
recover an overflowing or previously rounded HLC3/spread supplied by an adapter.

## Public API

```go
type Options struct{}

type Result struct {
    Values []float64
    Valid  []bool
}

type Value struct {
    Value float64
    Valid bool
}

type Snapshot struct {
    Version     uint32
    Algorithm   string
    Options     Options
    Count       uint64
    Started     bool
    VolumeSum   string
    WeightedSum string
}

func Compute(ctx context.Context, price, volume []float64, reset []bool, opts Options) (*Result, error)
func ComputeDefault(ctx context.Context, price, volume []float64, reset []bool) (*Result, error)
func New(opts Options) (*Stream, error)
func NewDefault() (*Stream, error)
func (r *Result) IsNil() bool
func (s *Stream) Push(ctx context.Context, price, volume float64, reset bool) (Value, error)
func (s *Stream) Reset()
func (s *Stream) Snapshot() Snapshot
func (s *Stream) Restore(state Snapshot) error
```

`Stream` has private fields and must be constructed with `New` or `NewDefault`.
`Options{}` has one possible value; there is no period, seed policy, price selector,
calendar, tolerance, or configurable rounding mode.

| Caller need                                               | Entry point                                 | Delegation / ownership                             |
|-----------------------------------------------------------|---------------------------------------------|----------------------------------------------------|
| An aligned history with an explicit configuration value   | `Compute(ctx, price, volume, reset, opts)`  | Owns both returned channels; all errors return nil |
| The same history without spelling the empty options value | `ComputeDefault(ctx, price, volume, reset)` | Exactly `Compute(..., Options{})`                  |
| A recoverable single-owner processor                      | `New(opts)`                                 | Fresh, unstarted `*Stream`                         |
| The same processor with the only current configuration    | `NewDefault()`                              | Exactly `New(Options{})`                           |
| One accepted observation and its current benchmark        | `stream.Push(ctx, price, volume, reset)`    | Scalar value; commit only on success               |
| Explicit discard, save, or replacement of continuation    | `Reset`, `Snapshot`, `Restore`              | Preserve the lifecycle and trust boundaries below  |

The two default wrappers add no validation, context polls, arrays, anchor flags,
or alternative arithmetic. They do not discard `Valid`. Their errors and
publication behavior are exactly those of their canonical calls. `New` retains
an error return but no options value can be rejected in this version.

## Mathematical contract

For a segment beginning at index `a`, positive cumulative volume defines:

![VWAP](https://latex.codecogs.com/svg.image?VWAP_%7Ba%3At%7D%3D%5Cfrac%7B%5Csum_%7Bi%3Da%7D%5Et%20p_i%20v_i%7D%7B%5Csum_%7Bi%3Da%7D%5Et%20v_i%7D)

The formula is evaluated over the **supplied binary64 values**. Decimal prices
that existed before conversion to `float64` cannot be reconstructed.
No HLC3, close, midquote, or other price proxy is selected internally.

Let `P_i = p_i * 2^1074` and `V_i = v_i * 2^1074`, interpreted as exact integers.
Every finite binary64 value is representable in this form. The stored integers are
`V = sum(V_i)` and `W = sum(P_i * V_i)`.

![Canonical rounding](https://latex.codecogs.com/svg.image?%5Cmathrm%7BVWAP%7D%3D%5Coperatorname%7BRN%7D_%7B64%7D%5Cleft%28%5Cfrac%7BW%7D%7BV%7D%5Ccdot2%5E%7B-1074%7D%5Cright%29)

`RN64` means round to the nearest binary64 value, with ties to even. No price-volume
product is first rounded to binary64. No running float sum or approximate mean is
stored. Exact zero is canonical `+0`; a negative, nonzero exact average may round
to `-0` through legitimate underflow. Neither is an error.

### Why finite means do not suffer product overflow

For `V > 0`, define nonnegative weights `alpha_i = v_i / sum(v_j)`. These weights
sum to one. Multiplying each inequality `min(p) <= p_i <= max(p)` by `alpha_i`
and adding gives `min(p) <= sum(alpha_i*p_i) <= max(p)`. Extrema need only include
observations with strictly positive weight. Thus the exact mean is finite and
bounded by finite admitted prices. Correct rounding cannot cross beyond the
finite endpoint `MaxFloat64` when the exact value is already at or below it.

The implementation never forms the dangerous floating expression `price*volume`.
For `price=volume=1e308`, the integer numerator may represent a quantity around
`1e616`, but the final division still returns exactly the supplied `1e308` bits.
Likewise, prices `[1e308, 1, -1e308]` with unit weights retain the exact residual
and return `RN64(1/3)`, not zero.

### Fixed storage proof

The largest finite positive binary64 integer in subnormal units is
`Umax = (2^53 - 1) << 2045`, requiring 2098 bits. With accepted segment count
`C <= 2^64 - 1`:

![Accumulator bounds](https://latex.codecogs.com/svg.image?0%5Cle%20V%5Cle%20C%20U_%7Bmax%7D%3C2%5E%7B2162%7D%2C%5Cqquad%7CW%7C%5Cle%20U_%7Bmax%7D%20V%3C2%5E%7B4260%7D)

Each magnitude is held in a value-type array of 67 `uint64` limbs: 4288 bits.
Using the same magnitude type for volume and weighted sums keeps one auditable
arithmetic implementation; it does not allocate according to history length.
Zero leading words are removed from the active length, but storage remains fixed.
The sign of `W` is stored separately, with exact cancellation normalizing zero.
The rounding workspace requires at most 4261 bits; the decimal parser may briefly
need 4264 bits immediately before rejecting an out-of-range next digit.

### One rounding, including subnormal and halfway cases

For `W != 0`, let `A = abs(W)` and choose
`shift = max(0, floor(log2(A/V)) - 52)`. This selects the binary64 output lattice
without using a floating logarithm. Set `D = V << shift` and compute exact integer
`q, r = divmod(A, D)`.

Increment `q` exactly when `2*r > D`, or when `2*r == D` and `q` is odd.
The final magnitude is `q * 2^(shift-1074)`. Here `q <= 2^53`, so converting the
rounded integer to float and scaling by the exact power of two introduces no
second rounding. Apply the sign of `W` afterward, retaining negative underflow.

The quotient kernel normalizes the denominator, estimates the single-limb quotient
from two numerator limbs, corrects an overestimate using exact multiplication and
subtraction, then compares the full remainder. Normalization bounds the initial
estimate to at most two above the exact floor quotient. The quotient is less than
`2^53` before rounding, so `bits.Div64` always has a nonzero divisor and a high
numerator limb strictly below that divisor. There is no zero-divide or quotient
panic path for admitted public state.

## Canonical transition and batch structure

The following pseudocode describes the existing transition, not another public
entry point. `integerUnits` denotes the exact binary64 decomposition and `RN64`
the exact-ratio rounding described above.

```text
Push(ctx, price, volume, reset):
    validate context, constructed receiver, first anchor, then the current row
    reject a non-reset row when Count is already MaxUint64
    candidate := value-copy of current state
    if reset:
        clear candidate.Count, both exact sums, and cached Value
    candidate.Started := true
    candidate.Count := candidate.Count + 1
    if volume != 0:
        candidate.V := candidate.V + integerUnits(volume)
        candidate.W := candidate.W + integerUnits(price) * integerUnits(volume)
        candidate.Value := {RN64((candidate.W / candidate.V) * 2^-1074), true}
    poll context immediately before commit
    replace receiver with candidate; return candidate.Value

Compute(ctx, price[], volume[], reset[], opts):
    validate context, three lengths, and the first anchor when nonempty
    validate all rows in index order; poll during and after this pass
    check output-byte arithmetic; allocate owning Values and Valid
    start one local constructed state
    visit each row in order, using the same accepted-row transition as Push
    store the returned value and mask at the original index
    poll before publication; return the complete Result
```

There is one numerical transition and one exact accumulator. Compute performs
**two linear scans** (validation, then numerical evaluation), not one total pass.
The accepted-row kernel omits redundant public validation after the full batch
scan. It also needs no per-row count-overflow check in batch: the slice length
fits `int`, and no segment can exceed that many observations. This optimization
does not change stream/batch numerical results or publish partial batches.

## Anchors, zero-volume phases, and state transitions

For any nonempty batch, `reset[0]` must be true. After `New` or `Reset`, the first
successful `Push` must pass `reset=true`. A failed first call does not start a
stream. `reset=true` clears the preceding segment **before** the current accepted
row is included.

Before the first strictly positive segment volume, output is `Value{Value: 0,
Valid: false}`, with a positive-zero payload. Once positive volume has arrived,
zero volume retains the preceding VWAP bits and `Valid=true`. Every accepted row,
including a zero-volume row, increments the segment count.

```text
New / Reset
     |
     v
EMPTY: Started=false, Count=0, V=0, W=0
     |
     | first accepted Push MUST have reset=true
     +------------------------+
     | volume=0               | volume>0
     v                        v
ACTIVE MASKED -------------> READY
Started=true   volume>0       Started=true
Count>=1, V=0                 Count>=1, V>0
Value=+0, Valid=false         RN64 mean, Valid=true
     ^                        |
     | accepted reset=true,   | zero volume: retain value,
     | volume=0               | increment Count
     +------------------------+

Any invalid/canceled Push or invalid Restore: remain in the prior state.
Accepted reset=true with positive volume: a new READY segment at Count=1.
Explicit Reset(): return to EMPTY.
```

### An explicit anchor is not proof against future information

The library never searches future prices for a favorable anchor. Every output
uses only the admitted input prefix and its reset flags. However, a caller could
still compute those flags retrospectively. This API has no timestamps or
`AvailableAt` fields and cannot authenticate that external decision.

Session calendars, event availability, bar finality, pivot confirmation, ordering,
missing observations, source corrections, and revision replay belong to the
application. An event-confirmed pivot is available only after confirmation.
Selecting the final high of a session as an earlier anchor is not made causal by
setting a Boolean flag.

The synthetic 10:00:00 execution-window boundary in the examples is a supplied
scenario, not an exchange calendar or an assertion about a venue's opening time.
The news example anchors at the first admitted record at or after its separate
availability timestamp; it does not rewrite already published event-time history.
Keep an application's pending-reset latch pending until `Push` succeeds.

## Validation and error precedence

Compute validates rows in increasing input-index order. Its effective order is:

```text
context -> Options domain (vacuous in v1) -> lengths -> first anchor
        -> rows in index order [finiteness -> volume domain]
        -> allocation arithmetic -> exact computation -> final context poll
```

Consequently, a finite negative volume at row 0 is reported before a NaN at row 1.
Within one row, any NaN/Inf wins over negative volume. This is not a global
whole-array nonfinite-priority scan. No numerical processing starts until the
complete input validation pass has succeeded.

| Condition                                     | Error/result                                      |
|-----------------------------------------------|---------------------------------------------------|
| Nil context                                   | `ErrNilContext`                                   |
| Canceled/deadline-expired context             | The original `ctx.Err()`                          |
| Unequal lengths                               | `ErrShape`                                        |
| A nil channel mixed with nonempty input       | Both `ErrShape` and `ErrNilInput` via `errors.Is` |
| Nonempty input without first reset            | `ErrAnchor`                                       |
| NaN or either infinity in price/volume        | `ErrNonFinite`, even on a zero-volume row         |
| Finite negative volume                        | `ErrVolume`                                       |
| Negative finite price                         | Accepted                                          |
| Either sign of zero volume                    | Accepted as zero                                  |
| Count already `MaxUint64`, next non-reset row | `ErrOverflow`                                     |
| Output byte-size arithmetic cannot fit        | `ErrOverflow`                                     |
| Invalid snapshot or unconstructed stream      | `ErrStateVersion`                                 |
| Nil stream for Push/Restore                   | `ErrNilInput`                                     |
| Three empty channels, including nil           | Non-nil Result, non-nil empty Values/Valid        |

The exported error identities are `ErrNilContext`, `ErrNilInput`, `ErrShape`,
`ErrAnchor`, `ErrNonFinite`, `ErrVolume`, `ErrOverflow`, and `ErrStateVersion`.
The empty options domain has no option-error category. Successful empty input
still requires a non-nil active context, including the final publication check.

Push has its own precise order:

```text
context -> receiver presence/construction -> first anchor -> current-row finite checks
        -> volume domain -> non-reset Count exhaustion -> local candidate -> final context -> commit
```

Restore has no context parameter. It checks receiver, version/algorithm/options,
Started/Count consistency, the volume string, the weighted string, then joint
magnitude bounds before reconstructing and committing a candidate. A malformed
snapshot on a nil receiver is `ErrNilInput`; on `Stream{}` it is `ErrStateVersion`.
Error text or the name "StateVersion" does not narrow that sentinel to Version alone.

A nil slice has length zero, so after successful equal-length shape validation
there cannot be a separate nil/nonempty branch. Dual sentinel classification
preserves both applicable categories without changing shape-before-anchor order.

All batch errors return `nil, error`. All Push errors return `Value{}, error`.
Classify errors with `errors.Is`; error text is not the machine protocol.

## Transactionality and context

`Push` validates first, copies the complete fixed-size state, applies any reset
and exact accumulation to that copy, then polls context immediately before assigning
it to the receiver. Rejected rows do not alter Count, Started, either exact sum,
or the cached Value. A bad `Push(reset=true)` cannot erase the old segment.

`Restore` similarly parses and validates a temporary candidate before one final
assignment. `Reset` itself is an explicit unconditional operation: it cannot be
rolled back together with a later separate Push. Use `Push(..., true)` to obtain
transactional reset-plus-observation semantics.

Batch validation and computation each poll context at row zero and at most 1024
rows apart. There is a check after validation and before final result publication.
Push checks at entry and just before commit. A cancellation after the final
successful poll does not retroactively invalidate publication. Concurrent input
errors and cancellation have no deterministic ordering beyond whichever is
observed first. Allocation itself is not a cancellable operation.

## Snapshot/Restore and trust boundaries

A constructed, never-started stream has Version 1, the fixed Algorithm string,
`Count=0`, `Started=false`, and both sum strings equal to `"0"`. A nil or
unconstructed receiver instead returns the zero `Snapshot{}`.

Restore requires a destination constructed by `New` with the same Options.
The complete integer-string grammar is `0|-?[1-9][0-9]*`, with a nonnegative
VolumeSum. There may be no plus sign, whitespace, Unicode digit, leading zero,
negative zero, decimal separator, exponent, radix prefix, or underscore.
Each field has a 1300-byte limit, checked before parsing. Valid strings are ASCII,
so byte count is also their character count.

Validation checks:

* `Started == (Count != 0)` and both fixed schema identifiers.
* Volume bit length at most 2162; weighted magnitude bit length at most 4260.
* `VolumeSum <= Count*Umax` and `abs(WeightedSum) <= VolumeSum*Umax`.

The inequalities imply zero sums for Count zero and zero WeightedSum for zero
VolumeSum. Parsing uses fixed arrays and bounded per-digit checks; even malformed
maximum-length data does not trigger arbitrary-size integer allocation. Restore
reconstructs the cached rounded output from exact sums, rather than trusting an
externally supplied approximate mean.

These checks are sufficient for the requested structural continuation contract,
not proof that a real history generated the sums. Protect persistence with an
application-level authentication envelope containing source identity, revision,
session/anchor policy, and event availability. Authentication is not part of this
package and cannot be inferred from a successful Restore.

`Count` is `uint64`. A typed Go JSON round trip preserves it; decoding it through
an arbitrary JSON-number-to-float pipeline can lose information. The sum fields
are strings precisely to retain every integer bit. No rounded float sum should
replace them. Snapshot fields and returned strings do not alias mutable stream
storage. Saving, restoring, and continuing across zero-volume rows preserves both
phase and output bits.

## Ownership, nil safety, and concurrency

Compute reads caller slices without mutation and returns independently owned
Values and Valid allocations. Mutating an output cannot change an input, another
result, or a stream. Copying Result itself copies slice headers and is shallow.
`Result.IsNil()` returns exactly `r == nil`, not an empty/history/readiness test.

One Stream has one owner; concurrent operations on the same instance, including
Snapshot, need external synchronization. Independent streams and batch calls can
run concurrently. The implementation has no shared mutable global state.
A value copy of a constructed Stream is detached because its state is fixed-size
value storage without slice/pointer aliases.

`Reset(nil)` and Reset on Stream{} are no-ops. Snapshot on either returns the zero
Snapshot. Push/Restore reject nil with ErrNilInput and Stream{} with ErrStateVersion.
For Push, context validation precedes receiver validation.

## Lifecycle and nil behavior

| Receiver/history state                       | Push / Restore                                                      | Reset                          | Snapshot                                                |
|----------------------------------------------|---------------------------------------------------------------------|--------------------------------|---------------------------------------------------------|
| Nil `*Stream`                                | `ErrNilInput`, after context checks for Push                        | No-op                          | `Snapshot{}`                                            |
| Unconstructed `Stream{}`                     | `ErrStateVersion`, after context checks for Push                    | No-op; does not construct      | `Snapshot{}`                                            |
| Newly constructed or explicitly reset stream | First accepted Push requires reset=true; Restore may replace state  | Preserves construction/options | Versioned empty state with sum strings `"0"`            |
| Started, no positive weight                  | Successful Push can remain masked while Count grows                 | Returns to unstarted phase     | Versioned started state with zero sums                  |
| Started, positive weight                     | Zero weight retains output bits; positive weight updates exact sums | Returns to unstarted phase     | Exact sums sufficient to reconstruct readiness and mean |
| Count at MaxUint64                           | Non-reset Push fails; valid reset Push starts Count=1               | Clears the segment             | Legal bounded continuation state                        |

`Count` measures accepted observations in the **current** segment, not elapsed
sessions, positive trades, source offset, or unique transport messages. It cannot
be used for duplicate suppression. Snapshot has no Current/Valid field: Restore
reconstructs those values from the exact sums. Snapshot itself does not advance Count.

## Complete Go example

This complete phase demonstration uses a masked anchor, positive trades, and a
zero-weight observation. Save it as a Go source file in an environment that can
import the repository module. Do not add another package main file to the repository
root merely to run a documentation example.

<!-- vwap-example: guide-zero-volume -->
```go
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package main

import (
	"context"
	"fmt"

	"github.com/lvlath/go/vwap"
)

// main distinguishes a zero-volume anchor from a ready unchanged benchmark.
// Each accepted row counts even when it contributes no weight.
// Implementation:
//   - Stage 1: construct a stream and provide one explicit initial reset.
//   - Stage 2: admit finite records in order through Push.
//   - Stage 3: print readiness beside each returned scalar.
//
// Behavior highlights:
//   - The last record keeps the preceding ready value without adding weight.
//
// Inputs:
//   - Four fixed price-volume observations in consistent units.
//
// Returns:
//   - Deterministic stdout with two-decimal formatting and a readiness flag.
//
// Errors:
//   - Constructor or Push errors stop processing before result consumption.
//
// Determinism:
//   - Fixed row order and no current-time or external-data dependency.
//
// Complexity:
//   - O(n*L) time and O(L) stream state, with fixed L=67; the tape is caller data.
//
// Notes:
//   - This is a small phase-law demonstration, not an execution model.
//
// AI-Hints:
//   - Check the error before Valid; Value{} also accompanies a rejected Push.
func main() {
	// Construction is required; Stream{} is not an initialized processor.
	stream, err := vwap.NewDefault()
	if err != nil {
		fmt.Println("construct:", err)
		return
	}
	// Only the first record creates a segment boundary in this example.
	rows := []struct {
		price, volume float64
		reset         bool
	}{{100, 0, true}, {102, 3, false}, {98, 1, false}, {105, 0, false}}
	ctx := context.Background() // No deadline is needed for this fixed demonstration.
	for _, row := range rows {
		// Failed input must not become a successful undefined observation.
		value, pushErr := stream.Push(ctx, row.price, row.volume, row.reset)
		if pushErr != nil {
			fmt.Println("push:", pushErr)
			return
		}
		fmt.Printf("%.2f %t\n", value.Value, value.Valid)
	}
}
```

Output:

```text
0.00 false
102.00 true
101.00 true
101.00 true
```

For the complete twelve-record typed-checkpoint workflow, including a failed
reset, valid retry, separate source offset, and per-row bit comparison, read
[Tutorial: VWAP foundations](TUTORIAL.md#11z-vwap-foundations-exact-weights-explicit-anchors-and-replay).
The source-backed package examples run directly with:

```sh
go test ./vwap -run '^Example' -count=1 -v
```

## Executable scenarios and data provenance

The five [package examples](../vwap/example_test.go) are complete
build -> calculate -> consume workflows. Their inputs live in
[test_helper_test.go](../vwap/test_helper_test.go), not external data files.

| Example name                               | Operational question                                                           | Important boundary                                                                  |
|--------------------------------------------|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `ExampleComputeDefault_vendorBarBenchmark` | Report a twelve-bar close-volume benchmark                                     | Frozen vendor-demo prices, not tick VWAP or live quotes                             |
| `ExampleCompute_executionQuality`          | Compare six buy fills with twelve market observations in the same window       | Separate market/fill weights; fees and impact attribution excluded                  |
| `ExampleStream_Push_availableAtAnchor`     | Start a news-relative mean only after delivery                                 | Pending anchor is consumed only on successful admission; initial weight may be zero |
| `ExampleStream_Restore_failoverCursor`     | Restart before positive volume, reject a corrupt reset, and skip one duplicate | Source cursor is external; in-memory JSON is not durable exactly-once storage       |
| `ExampleComputeDefault_matchedSpreadBasis` | Measure signed basis from twelve synchronized paired records                   | Completed matched quantities, not unrelated leg volumes or percentage division      |

The helper records provenance for the **IBM 2026-08-07 15:00..15:55 US/Eastern**
five-minute close/volume subset of an Alpha Vantage public demo response. The
source comment records inspection on 2026-11-27 and provider metadata
`Last Refreshed=2026-08-07 19:55:00`. Those are fixture/source labels, not a
current-price claim, independent exchange audit, or inferred availability time.
This guide does not refresh that fixture from the provider. The other four
scenarios are explicitly engineered tapes; deliberate corruptions are not
market observations. Every `Output` block is checked by `go test`.

## Composition, corrections, and application boundaries

Keep the following layers distinct:

```text
source identity + units + ordering + revision + availability
                         |
          application admission and explicit reset flags
                         |
      vwap.Compute / Stream.Push -> Value(s) + Valid
                         |
         separately checked consumer arithmetic / reporting

Snapshot + source cursor + anchor policy + revision
                         |
       application-owned atomic persistence and authentication
```

**With series.** Time-grid checks or availability-aware admission may precede
VWAP; neither package is imported or invoked implicitly. Align masks and row
identities explicitly. A missing record is not automatically a zero-volume row,
and dropping genuine zero-volume observations changes Count and source alignment.

**With ATR and Donchian.** VWAP is a weighted price level; ATR is a range scale;
Donchian is an observed high/low boundary. Align the same instrument, units,
revisions, and admitted bars without replacing one algorithm's input law with
another's. A current-inclusive VWAP is available after the current accepted
observation. A pre-observation comparison needs an already available prefix; a
new reset segment has no such prior VWAP. Likewise, using ATR[t-1] rather than
ATR[t] is an explicit information-set choice, not a convenience index shift.

**Consumer arithmetic.** Read error, then Valid, then the value. A ready zero
or signed benchmark is not a safe positive-price basis-point denominator.
Even two finite outputs may have an overflowing difference. Check domain and
finiteness of differences, products, and quotients in application calculations.
Subtracting rounded means is not a new exact-sum diagnostic promised by Result.
The execution example is a descriptive fill-price comparison, not a complete
implementation-shortfall, fee, impact, or profitability model.

**Chunking.** Independent Compute calls require their own first anchor and do
not continue a preceding segment. For exact continuation, retain a Stream or
Restore a matching checkpoint, then Push only the unread suffix. Overlapping
replay without external cursor control double-counts observations. Independent
known segments may be processed separately; no public Merge method exists.

**Corrections.** A correction to price or weight at index k can change every
later prefix of that segment. With unchanged admissible reset flags it cannot
change values at or after the next reset, which discards the earlier segment.
Replay from a verified checkpoint **before** the corrected record, or from a
known earlier segment origin; there is no inverse/decrement API. Changed reset
flags, event availability, or record insertion/deletion require re-evaluating
segmentation and source offsets, not merely applying the old boundary horizon.
An invalid corrected record rejects a whole batch; prefix dependence does not
imply partial publication. Keep an existing artifact until replacement succeeds.

**Persistence.** Atomically store state, source identity/revision, cursor, and
anchor-policy identity in the storage layer. Authenticate the envelope outside
VWAP. Bound untrusted transport/JSON size before decoding: Restore bounds its
integer parsing, not memory already allocated by an external decoder. Use typed
uint64 decoding or a lossless representation; never route Count through float64.

## Tests and independent oracles

All tests are in `package vwap_test`. Reusable fixture generation and oracles are
in `test_helper_test.go`. No production helper is exported for tests.

The numerical oracle converts each finite input directly to `math/big.Rat`,
multiplies and adds rationally, then uses Rat.Float64. It does not decode the input
into production limbs or call the production quotient helper. Snapshot oracles
use independent big integers and canonical decimal reformatting. Assertions compare
binary64 bits, masks, counts, errors.Is identities, and complete snapshots exactly.

Coverage includes golden cases, signed prices, huge products, tiny
subnormal products, exact cancellation, ties to even, carries, normal/subnormal
boundaries, negative underflow, zero-volume resets, shape/anchor/domain precedence,
Count exhaustion, failed reset/restore atomicity, deterministic cancellation in
both batch passes and at publication, ownership, independently parallel streams,
and round-trip restart at every prefix of a representative segmented history.

Metamorphic tests preserve final means under permutation within a segment, not
every prefix. Common volume scaling is tested only for exactly representable
power-of-two rescalings that preserve the relative binary64 weights. Prefix tests
verify absence of dependence on appended future observations.

Additional regressions cover all 2047 finite exponent fields, significand/sign
variants, exact-ratio halfway lattices through bounded snapshots, final-count
arithmetic, negative-zero continuation, wrapper/error/context-poll parity, and
independent scenario-prefix checks. Default wrappers must preserve ownership and
error precedence as well as numerical bits.

Fuzz targets exercise raw binary64 public input and malformed Snapshot boundaries.
They do not recover from panics; any such failure is a test failure. Short fuzz runs
are useful evidence, not a proof over every input bit pattern.

## Complexity, allocation, and benchmarks

Let `L=67` limbs and `D<=1300` decimal bytes; both are protocol constants.

| Operation | Time                                | Auxiliary/returned storage             |
|-----------|-------------------------------------|----------------------------------------|
| New       | O(L)                                | One fixed-size Stream object           |
| Push      | O(L), strict O(1) in history length | O(L) fixed temporary value state       |
| Compute   | O(n*L), plus O(n) validation        | O(L) workspace plus O(n) owning output |
| Reset     | O(L)                                | No retained history                    |
| Snapshot  | O(D*L)                              | Fixed workspace plus returned strings  |
| Restore   | O(D*L)                              | O(L) fixed workspace                   |

The source avoids internal heap allocation in Push and Restore; allocation tests
measure that property on the executing toolchain. Custom context methods can still
allocate. Snapshot may allocate its returned nonzero strings. Compute allocates
the Result and two channels; input generation is not part of benchmark timing.

Benchmarks cover `n=1000`, `100000`, and `1000000`, with explicit segment lengths
14, 64, and 512. These lengths are reset cadence, **not** a window option: this API
has no rolling-window period. Also benchmark Push and restoring extreme sums.
Record Go version, GOOS/GOARCH, CPU, ns/op, B/op, allocs/op, repetitions, and workload.
A single benchmark observation is not a stable production latency guarantee.

The allocation-size guard rejects arithmetic overflow, not actual RAM shortage.
No runtime out-of-memory recovery, wall-clock deadline bound, or real-time scheduling
guarantee is promised.

## Verification commands and integration

Run from the existing repository root, retaining its go.mod:

```sh
gofmt -w vwap/*.go
go test -count=1 ./vwap
go test -race -count=1 ./vwap
go test -count=1 ./...
go test -race -count=1 ./...
go vet ./...
golangci-lint run
govulncheck ./...
# Generate the profile before requesting the separate Go-only gate.
# Do not inherit a stale gate request while producing its replacement input.
unset LVLATH_VWAP_COVERPROFILE
go test ./vwap -count=1 -covermode=atomic -coverprofile=vwap.coverage.out
LVLATH_VWAP_COVERPROFILE="$PWD/vwap.coverage.out" \
  go test ./vwap -run '^TestVWAP_CoverageThreshold$' -count=1 -v
go test -run '^$' -fuzz '^FuzzVWAP_PublicBoundaries$' -fuzztime=10s ./vwap
go test -run '^$' -fuzz '^FuzzSnapshot_PublicBoundary$' -fuzztime=10s ./vwap
go test -run '^$' -bench . -benchmem ./vwap
go test ./vwap -run '^Example' -count=1 -v
```

The [Go-only coverage gate](../vwap/coverage_test.go) requires at least 92%
statement coverage overall and per executable production file. Normal tests do
not require a profile: this one gate skips when `LVLATH_VWAP_COVERPROFILE` is unset.
An explicitly supplied missing/malformed profile fails. Use the same checkout for
both commands and an absolute path because package tests run in the package directory.
The gate validates profile structure and file inventory, not cryptographic provenance
or proof that every control-flow path was exercised. No Python tool or new module
is needed. `doc.go` and `errors.go` contain no instrumented
statements and are reported as N/A, not a fabricated 100%. The legal-input output
allocation overflow guard cannot be exercised by fabricating a huge slice through
unsafe code; no such test-only bypass is introduced.

These are commands to execute on the selected checkout, not a claim of successful
CI. Record actual toolchain, command scope, results, and missing tools separately.
Package-only tests are not a substitute for the repository-wide gates.

## Pitfalls and non-goals

Do not omit the first anchor, silently reset on an input error, discard a finite
negative price, accept an invalid quote because volume is zero, or equate valid
zero with undefined. Do not save only a rounded VWAP for restart, or treat canonical
Snapshot strings as authenticated provenance. Do not replace exact products by
floating products, a delta-mean recurrence, compensation with dropped terms, or an
unbounded allocation-based fallback while keeping the same continuation identifier.

A supplied bar HLC3 is a bar price proxy, not a trade-by-trade VWAP. A futures basket
must keep instruments' volume and price units separate unless the application has
explicitly defined a meaningful transformed series. Source alignment with ATR,
Donchian, or series remains outside this package. Historical corrections require
explicit recomputation/replay. There is no profit, fill, liquidity-discovery, or
transaction-cost-estimation guarantee implied by a numerically correct mean.

## Contract clarifications and file routing

The import path is `github.com/lvlath/go/vwap`. Use the single uppercase guide
path `docs/VWAP.md`; do not maintain a divergent case-only duplicate. `Options{}`
is the entire current options domain. Session and anchored use cases differ only
in caller-supplied resets, not hidden modes. Batch checks are per-index, and mixed
nil/nonempty shape errors preserve both applicable sentinel identities.

Start with [README](../README.md#vwap-exact-weighted-prices-with-explicit-segment-origins),
follow [the tutorial](TUTORIAL.md#11z-vwap-foundations-exact-weights-explicit-anchors-and-replay),
and consult [VWAP FAQ](FAQ_%26_TIPS.md#13y-vwap-faq-anchors-exact-means-and-continuation)
for failure diagnosis. The [package charter](../vwap/doc.go) is the formal
package-level contract; [AGENTS](../AGENTS.md#vwap) and [llms.txt](../llms.txt)
provide contributor and machine-readable navigation, not alternative APIs.

## References

Read the implemented [API](../vwap/api.go), [types](../vwap/types.go),
[error categories](../vwap/errors.go), [exact accumulator](../vwap/accumulator.go),
and [snapshot codec](../vwap/snapshot.go) together with their public GoDoc.
The [engineering standard](lvlath_UES.md) governs ownership, determinism,
error identity, executable examples, and documentation consistency.

Standard-library semantics are documented in the official
[Go numeric types specification](https://go.dev/ref/spec#Numeric_types),
[math.Frexp](https://pkg.go.dev/math#Frexp), [math/bits](https://pkg.go.dev/math/bits),
[math/big.Rat](https://pkg.go.dev/math/big#Rat), and
[testing examples](https://pkg.go.dev/testing#hdr-Examples).
The numerical oracle and frozen-data provenance are in
[test_helper_test.go](../vwap/test_helper_test.go); neither live fetching nor a
production big-number dependency is required.
