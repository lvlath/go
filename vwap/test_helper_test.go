// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap_test owns procedural fixtures and the independent rational oracle.
// This file keeps fixtures and helpers independent of production-private state.
package vwap_test

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"io/fs"
	"math"
	"math/big"
	"math/rand"
	"slices"
	"strconv"
	"strings"
	"testing"
	"testing/fstest"
	"time"

	"github.com/lvlath/go/vwap"
)

// newStream constructs a fresh public Stream or fails the calling test.
// A failed constructor cannot silently turn a numerical test into a nil-receiver test.
//
// Implementation:
//   - Stage 1: mark the testing helper frame
//   - Stage 2: call vwap.New with explicit empty Options
//   - Stage 3: require a non-nil successful stream.
//
// Behavior highlights:
//   - A failed constructor cannot silently turn a numerical test into a nil-receiver test.
//
// Inputs:
//   - tb: the owning test or benchmark.
//
// Returns:
//   - *vwap.Stream: a fresh independent constructed instance.
//
// Errors:
//   - Fatal test failure on constructor error or a nil success result.
//
// Determinism:
//   - The empty configuration and initial state are fixed.
//
// Complexity:
//   - O(L) time and storage for the stream, L=67.
//
// Notes:
//   - This helper is for known-valid setup, not constructor-rejection tests.
//
// AI-Hints:
//   - Keep constructor error checks outside benchmark timers.
func newStream(tb testing.TB) *vwap.Stream {
	tb.Helper()
	stream, err := vwap.New(vwap.Options{})
	if err != nil || stream == nil {
		tb.Fatalf("New: stream=%v error=%v", stream, err)
	}
	return stream
}

// mustPush accepts one known-valid row through the public streaming boundary.
// Every successful fixture step is checked rather than discarding an unexpected error.
//
// Implementation:
//   - Stage 1: mark the helper frame
//   - Stage 2: call Push with an explicit background context
//   - Stage 3: fail immediately on any error.
//
// Behavior highlights:
//   - Every successful fixture step is checked rather than discarding an unexpected error.
//
// Inputs:
//   - stream: a constructed stream; price/volume/reset: a deliberately valid observation.
//
// Returns:
//   - vwap.Value: exactly the returned scalar and validity flag.
//
// Errors:
//   - Fatal test failure on an unexpected Push error.
//
// Determinism:
//   - The supplied observation is consumed once in caller order.
//
// Complexity:
//   - O(L) time and workspace, L=67; test diagnostics occur only on failure.
//
// Notes:
//   - Use direct Push for cancellation and rejection tests.
//
// AI-Hints:
//   - Do not use this helper to hide the error identity expected by a validation test.
func mustPush(tb testing.TB, stream *vwap.Stream, price, volume float64, reset bool) vwap.Value {
	tb.Helper()
	value, err := stream.Push(context.Background(), price, volume, reset)
	if err != nil {
		tb.Fatalf("Push(%g,%g,%t): %v", price, volume, reset, err)
	}
	return value
}

// mustCompute computes a known-valid batch and requires a non-nil successful result.
// Shape, domain, and anchor error tests call the public facade directly instead.
//
// Implementation:
//   - Stage 1: mark the helper frame
//   - Stage 2: invoke Compute with explicit context/options
//   - Stage 3: reject an error or nil success.
//
// Behavior highlights:
//   - Shape, domain, and anchor error tests call the public facade directly instead.
//
// Inputs:
//   - price, volume, reset: aligned valid fixture channels, read-only during the call.
//
// Returns:
//   - *vwap.Result: owning aligned channels from the public facade.
//
// Errors:
//   - Fatal test failure on an unexpected error or missing result.
//
// Determinism:
//   - Input order is unchanged and no fixture channel is rewritten.
//
// Complexity:
//   - O(n*L) time, O(n+L) storage, L=67.
//
// Notes:
//   - The helper does not decide numerical correctness; independent assertions follow.
//
// AI-Hints:
//   - Do not interpret successful construction of a result as oracle parity.
func mustCompute(tb testing.TB, price, volume []float64, reset []bool) *vwap.Result {
	tb.Helper()
	result, err := vwap.Compute(context.Background(), price, volume, reset, vwap.Options{})
	if err != nil || result == nil {
		tb.Fatalf("Compute: result=%v error=%v", result, err)
	}
	return result
}

// requireError checks a public error category using errors.Is.
// It protects stable sentinel identity without depending on diagnostic spelling.
//
// Implementation:
//   - Stage 1: mark the helper frame
//   - Stage 2: evaluate errors.Is
//   - Stage 3: fail with both errors when classification differs.
//
// Behavior highlights:
//   - It protects stable sentinel identity without depending on diagnostic spelling.
//
// Inputs:
//   - err: observed error; target: expected sentinel or context error.
//
// Returns:
//   - No value; the test continues only when classification matches.
//
// Errors:
//   - Fatal test failure on a missing or incorrect error identity.
//
// Determinism:
//   - No substring, formatting, or map traversal participates in classification.
//
// Complexity:
//   - O(d) time, O(d) worst-case error-tree traversal, d=wrapping depth/size.
//
// Notes:
//   - A joined error may intentionally match more than one category.
//
// AI-Hints:
//   - Keep separate errors.Is assertions when both ErrShape and ErrNilInput must be preserved.
func requireError(tb testing.TB, err, target error) {
	tb.Helper()
	if !errors.Is(err, target) {
		tb.Fatalf("error=%v; expected errors.Is(...,%v)", err, target)
	}
}

// requireValue compares readiness and all binary64 payload bits.
// Signed underflow zeros and masked positive zeros must not be conflated by numeric equality.
//
// Implementation:
//   - Stage 1: mark the helper frame
//   - Stage 2: compare Valid and Float64bits
//   - Stage 3: print both values and encodings on mismatch.
//
// Behavior highlights:
//   - Signed underflow zeros and masked positive zeros must not be conflated by numeric equality.
//
// Inputs:
//   - got, want: observed and independently expected public Value carriers.
//
// Returns:
//   - No value; the assertion admits only bit-identical outputs.
//
// Errors:
//   - Fatal test failure on either mask or bit mismatch.
//
// Determinism:
//   - All 64 bits are compared without tolerance or rounding.
//
// Complexity:
//   - O(1) time and space on success.
//
// Notes:
//   - Expected values must come from a mathematical identity or independent oracle.
//
// AI-Hints:
//   - Do not replace this assertion with an epsilon or plain float equality.
func requireValue(tb testing.TB, got, want vwap.Value) {
	tb.Helper()
	if got.Valid != want.Valid || math.Float64bits(got.Value) != math.Float64bits(want.Value) {
		tb.Fatalf("value=(%g,%t,0x%016x), want=(%g,%t,0x%016x)",
			got.Value, got.Valid, math.Float64bits(got.Value),
			want.Value, want.Valid, math.Float64bits(want.Value))
	}
}

// requireResult checks an owning batch carrier against per-prefix expected Values.
// Shape is checked before any indexed output read, including empty histories.
//
// Implementation:
//   - Stage 1: mark the helper frame
//   - Stage 2: require both output lengths and a non-nil result
//   - Stage 3: compare every prefix with requireValue.
//
// Behavior highlights:
//   - Shape is checked before any indexed output read, including empty histories.
//
// Inputs:
//   - result: public batch result; want: independently expected aligned prefix outputs.
//
// Returns:
//   - No value; each prefix and readiness bit must agree.
//
// Errors:
//   - Fatal test failure on shape, nil success, or numeric mismatch.
//
// Determinism:
//   - Prefixes are checked in ascending input index order.
//
// Complexity:
//   - O(n) time, O(1) auxiliary space.
//
// Notes:
//   - This checks values and dimensions; ownership tests mutate arrays separately.
//
// AI-Hints:
//   - Do not accept matching final means as proof that every prefix or reset is correct.
func requireResult(tb testing.TB, result *vwap.Result, want []vwap.Value) {
	tb.Helper()
	if result == nil || len(result.Values) != len(want) || len(result.Valid) != len(want) {
		tb.Fatalf("wrong result shape: result=%v wanted length=%d", result, len(want))
	}
	for i := range want {
		requireValue(tb, vwap.Value{Value: result.Values[i], Valid: result.Valid[i]}, want[i])
	}
}

// rationalOracle recomputes every prefix with exact rational arithmetic independent of production limbs.
// It preserves exact binary64 products and performs one rational-to-binary64 rounding per ready row.
//
// Implementation:
//   - Stage 1: create owning outputs and zero rational sums
//   - Stage 2: apply resets before exact rational multiply/add
//   - Stage 3: round each positive-weight rational with Rat.Float64.
//
// Behavior highlights:
//   - It preserves exact binary64 products and performs one rational-to-binary64 rounding per ready row.
//
// Inputs:
//   - price/volume/reset: aligned finite valid inputs with nonnegative weights and an explicit first anchor.
//
// Returns:
//   - []vwap.Value: owning independently rounded prefix values and masks.
//
// Errors:
//   - No error return; invalid-domain tests must not send NaN/Inf into this admitted-input oracle.
//
// Determinism:
//   - Rows are scanned ascending; exact rational zero becomes positive zero.
//
// Complexity:
//   - O(n*A(B)) time and O(n+B) space, B=big-integer size and A its rational-operation cost.
//
// Notes:
//   - math/big is used only in tests; it is deliberately not a production helper.
//
// AI-Hints:
//   - Never replace exact Rat multiplication with new(big.Rat).SetFloat64(price*volume).
func rationalOracle(price, volume []float64, reset []bool) []vwap.Value {
	out := make([]vwap.Value, len(price))
	var weighted, total big.Rat
	for i := range price {
		if reset[i] {
			weighted.SetInt64(0)
			total.SetInt64(0)
		}
		p := new(big.Rat).SetFloat64(price[i])
		v := new(big.Rat).SetFloat64(volume[i])
		weighted.Add(&weighted, new(big.Rat).Mul(p, v))
		total.Add(&total, v)
		if total.Sign() != 0 {
			q := new(big.Rat).Quo(&weighted, &total)
			value, _ := q.Float64()
			out[i] = vwap.Value{Value: value, Valid: true}
		}
	}
	return out
}

// units converts a finite binary64 value to exact integer multiples of 2^-1074.
// Rational scaling avoids copying production bit decomposition into expected checkpoint strings.
//
// Implementation:
//   - Stage 1: construct the integer scale 2^1074
//   - Stage 2: multiply the exact rational input by that scale
//   - Stage 3: copy the resulting integer numerator.
//
// Behavior highlights:
//   - Rational scaling avoids copying production bit decomposition into expected checkpoint strings.
//
// Inputs:
//   - value: finite binary64 with any sign.
//
// Returns:
//   - *big.Int: owning exact lattice integer.
//
// Errors:
//   - No error return; the caller guarantees a finite input.
//
// Determinism:
//   - The lattice quotient is an integer for every admitted binary64 value.
//
// Complexity:
//   - O(A(B)) time and O(B) space, B bounded by binary64 units.
//
// Notes:
//   - The denominator is exactly one after scaling.
//
// AI-Hints:
//   - Keep this oracle independent of Frexp, significand masks, and fixed-word production code.
func units(value float64) *big.Int {
	scale := new(big.Int).Lsh(big.NewInt(1), 1074)
	rational := new(big.Rat).Mul(new(big.Rat).SetFloat64(value), new(big.Rat).SetInt(scale))
	return new(big.Int).Set(rational.Num())
}

// oracleSums constructs exact public checkpoint sums with arbitrary-precision test integers.
// The numerator is a sum of exact products in 2^-2148 units, not rounded float products.
//
// Implementation:
//   - Stage 1: start zero totals
//   - Stage 2: convert each input through rational units and add its exact integer product
//   - Stage 3: serialize both totals canonically.
//
// Behavior highlights:
//   - The numerator is a sum of exact products in 2^-2148 units, not rounded float products.
//
// Inputs:
//   - price, volume: one aligned valid segment with nonnegative volume.
//
// Returns:
//   - Two strings: volume units and signed weighted-product units.
//
// Errors:
//   - No error return; segment boundaries are handled by the caller.
//
// Determinism:
//   - Ascending input order; exact final sums are also permutation-invariant within a segment.
//
// Complexity:
//   - O(n*M(B)) time and O(B) live arithmetic storage plus output strings.
//
// Notes:
//   - This helper intentionally has no implicit reset policy.
//
// AI-Hints:
//   - Do not compare checkpoint strings to an approximation derived from the displayed VWAP.
func oracleSums(price, volume []float64) (string, string) {
	var weighted, total big.Int
	for i := range price {
		v := units(volume[i])
		total.Add(&total, v)
		weighted.Add(&weighted, new(big.Int).Mul(units(price[i]), v))
	}
	return total.String(), weighted.String()
}

// countedContext closes its Done channel on a deterministic Err poll.
// The test owns it exclusively; it obeys stable cancellation once Err becomes non-nil.
type countedContext struct {
	context.Context
	calls    int
	cancelAt int
	done     chan struct{}
	canceled bool
}

// newCountedContext creates a single-owner cancellation instrument with a known Err poll boundary.
// Cancellation timing depends on logical calls rather than scheduler delays.
//
// Implementation:
//   - Stage 1: use a background parent
//   - Stage 2: allocate an open Done channel
//   - Stage 3: record the poll count at which cancellation will latch.
//
// Behavior highlights:
//   - Cancellation timing depends on logical calls rather than scheduler delays.
//
// Inputs:
//   - cancelAt: positive target number of Err calls.
//
// Returns:
//   - *countedContext: independent poll counter and cancellation channel.
//
// Errors:
//   - None for the positive counts supplied by tests.
//
// Determinism:
//   - The same Err-call sequence closes Done at the same point.
//
// Complexity:
//   - O(1) time and storage.
//
// Notes:
//   - The helper is intentionally restricted to nonconcurrent tests.
//
// AI-Hints:
//   - Do not share this instrument across goroutines; ordinary context implementations are required for concurrent tests.
func newCountedContext(cancelAt int) *countedContext {
	return &countedContext{Context: context.Background(), cancelAt: cancelAt, done: make(chan struct{})}
}

// Done returns the instrumented cancellation channel without advancing the poll counter.
// It stays open until Err reaches the configured logical cancellation boundary.
//
// Implementation:
//   - Stage 1: read the exclusively owned channel field
//   - Stage 2: do not trigger cancellation
//   - Stage 3: return the same channel.
//
// Behavior highlights:
//   - It stays open until Err reaches the configured logical cancellation boundary.
//
// Inputs:
//   - c: a countedContext constructed by newCountedContext.
//
// Returns:
//   - <-chan struct{}: stable channel closed exactly once by Err.
//
// Errors:
//   - None under the constructed-context precondition.
//
// Determinism:
//   - Reading Done has no side effects on the scheduled poll boundary.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - No worker goroutine or wall-clock timeout is created.
//
// AI-Hints:
//   - Do not poll Err here; doing so would shift the observed cancellation stage.
func (c *countedContext) Done() <-chan struct{} { return c.done }

// Err advances one logical poll and latches cancellation at the configured boundary.
// After cancellation it always returns context.Canceled and never closes Done again.
//
// Implementation:
//   - Stage 1: increment the poll counter
//   - Stage 2: close Done only on the first threshold crossing
//   - Stage 3: return the latched cancellation status.
//
// Behavior highlights:
//   - After cancellation it always returns context.Canceled and never closes Done again.
//
// Inputs:
//   - c: exclusively owned constructed countedContext.
//
// Returns:
//   - error: nil before the threshold, context.Canceled afterward.
//
// Errors:
//   - This is the context error surface itself, not a test assertion.
//
// Determinism:
//   - The positive cancelAt count completely determines the first non-nil call.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - Called synchronously by the code under test; there is no scheduler race.
//
// AI-Hints:
//   - Do not reset the counter after cancellation or return nil again.
func (c *countedContext) Err() error {
	c.calls++
	if !c.canceled && c.calls >= c.cancelAt {
		close(c.done)
		c.canceled = true
	}
	if c.canceled {
		return context.Canceled
	}
	return nil
}

// randomFinite generates a deterministic finite binary64 encoding from a test RNG.
// It samples extreme exponents and subnormals rather than concentrating near ordinary prices.
//
// Implementation:
//   - Stage 1: read one uint64
//   - Stage 2: clear one exponent bit only for the all-ones exponent
//   - Stage 3: reinterpret the remaining finite encoding.
//
// Behavior highlights:
//   - It samples extreme exponents and subnormals rather than concentrating near ordinary prices.
//
// Inputs:
//   - rng: caller-owned seeded test generator.
//
// Returns:
//   - float64: finite value with either sign and possibly signed zero.
//
// Errors:
//   - None for a constructed RNG.
//
// Determinism:
//   - One RNG word is consumed for each returned value.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - The adjusted distribution is a stress distribution, not a statistical model of prices.
//
// AI-Hints:
//   - Do not use this noncryptographic generator for production randomness or security.
func randomFinite(rng *rand.Rand) float64 {
	word := rng.Uint64()
	// Clear an exponent bit only for all-ones exponent, retaining finite extremes.
	if word>>52&0x7ff == 0x7ff {
		word &^= uint64(1) << 52
	}
	return math.Float64frombits(word)
}

// randomHistory builds a reproducible multi-segment full-range arithmetic stress history.
// Explicit zero-price/zero-volume rows exercise masked and ready transitions among extreme values.
//
// Implementation:
//   - Stage 1: create a locally seeded RNG and owning channels
//   - Stage 2: fill finite signed prices/nonnegative weights with selected zeros
//   - Stage 3: write explicit reset flags including the first row.
//
// Behavior highlights:
//   - Explicit zero-price/zero-volume rows exercise masked and ready transitions among extreme values.
//
// Inputs:
//   - seed: fixed RNG seed; n: nonnegative row count.
//
// Returns:
//   - Three owning, aligned slices of finite valid observations.
//
// Errors:
//   - None for the nonnegative test sizes supplied.
//
// Determinism:
//   - The seed and row index fix every value and reset.
//
// Complexity:
//   - O(n) time and space before oracle or benchmark execution.
//
// Notes:
//   - Values are engineered stress inputs, not real market observations.
//
// AI-Hints:
//   - Do not change the seed in response to a failing mathematical assertion.
func randomHistory(seed int64, n int) ([]float64, []float64, []bool) {
	rng := rand.New(rand.NewSource(seed))
	price := make([]float64, n)
	volume := make([]float64, n)
	reset := make([]bool, n)
	for i := range price {
		price[i] = randomFinite(rng)
		volume[i] = math.Abs(randomFinite(rng))
		if i%11 == 0 {
			volume[i] = 0
		}
		if i%17 == 0 {
			price[i] = 0
		}
		reset[i] = i == 0 || i%37 == 0
	}
	return price, volume, reset
}

// benchmarkHistory prepares moderate-scale batch input with a declared segment cadence.
// The cadence is an explicit reset pattern, not a rolling VWAP period.
//
// Implementation:
//   - Stage 1: allocate three n-row channels
//   - Stage 2: fill fixed binary-exact price/volume patterns
//   - Stage 3: reset every segment rows starting at zero.
//
// Behavior highlights:
//   - The cadence is an explicit reset pattern, not a rolling VWAP period.
//
// Inputs:
//   - n: nonnegative size; segment: positive reset cadence.
//
// Returns:
//   - Owning price, volume, and reset slices.
//
// Errors:
//   - None for the valid benchmark sizes and cadences.
//
// Determinism:
//   - All values depend solely on their index, without RNG or current time.
//
// Complexity:
//   - O(n) time and space, deliberately outside the benchmark timer.
//
// Notes:
//   - Zero weights and nonzero weights both occur in the generated history.
//
// AI-Hints:
//   - Do not move fixture generation inside the measured b.N loop.
func benchmarkHistory(n, segment int) ([]float64, []float64, []bool) {
	price := make([]float64, n)
	volume := make([]float64, n)
	reset := make([]bool, n)
	for i := range price {
		price[i] = 100 + float64(i%71)/8
		volume[i] = float64(i % 31)
		reset[i] = i%segment == 0
	}
	return price, volume, reset
}

// verifyParity compares both public execution surfaces with an independent exact prefix oracle.
// It catches batch/stream agreement on the same wrong answer by checking both against rational arithmetic.
//
// Implementation:
//   - Stage 1: calculate oracle prefixes
//   - Stage 2: compare the full Compute result
//   - Stage 3: replay the same rows through a new Stream and compare each Value.
//
// Behavior highlights:
//   - It catches batch/stream agreement on the same wrong answer by checking both against rational arithmetic.
//
// Inputs:
//   - tb and aligned valid price/volume/reset channels.
//
// Returns:
//   - No value; both execution surfaces must satisfy every expected prefix.
//
// Errors:
//   - Fatal test failure from setup, public errors, or a mismatched output bit/mask.
//
// Determinism:
//   - Every comparison follows the original row order and explicit reset sequence.
//
// Complexity:
//   - O(n*A(B)+n*L) time, O(n+B+L) space; oracle work is test-only.
//
// Notes:
//   - Inputs must remain immutable through both public traversals.
//
// AI-Hints:
//   - Do not replace the independent oracle with the first surface being compared.
func verifyParity(tb testing.TB, price, volume []float64, reset []bool) {
	tb.Helper()
	want := rationalOracle(price, volume, reset)
	requireResult(tb, mustCompute(tb, price, volume, reset), want)
	stream := newStream(tb)
	for i := range price {
		requireValue(tb, mustPush(tb, stream, price[i], volume[i], reset[i]), want[i])
	}
}

// requireUnchanged checks the entire public checkpoint after a rejected operation.
// Count, phase, exact volume, and signed numerator must all remain identical, not just the displayed mean.
//
// Implementation:
//   - Stage 1: mark the helper frame
//   - Stage 2: take a detached Snapshot
//   - Stage 3: compare all comparable public fields.
//
// Behavior highlights:
//   - Count, phase, exact volume, and signed numerator must all remain identical, not just the displayed mean.
//
// Inputs:
//   - stream: receiver after rejection; before: previously captured snapshot.
//
// Returns:
//   - No value; complete public continuation equality is required.
//
// Errors:
//   - Fatal test failure if any checkpoint field differs.
//
// Determinism:
//   - Snapshot fields are compared directly without reflection or formatting heuristics.
//
// Complexity:
//   - O(D*L) time and O(D+L) storage for snapshot encoding.
//
// Notes:
//   - This tests externally visible continuation; cached-value parity is tested through Push as well.
//
// AI-Hints:
//   - A mean that appears unchanged does not prove that an erroneous reset left its exact sums intact.
func requireUnchanged(tb testing.TB, stream *vwap.Stream, before vwap.Snapshot) {
	tb.Helper()
	if after := stream.Snapshot(); after != before {
		tb.Fatalf("state changed after rejection:\nbefore=%+v\nafter=%+v", before, after)
	}
}

// snapshotFromInts builds a public snapshot directly from arbitrary-precision integer sums.
// It can express admissible boundary states and intentionally invalid tamper cases without private access.
//
// Implementation:
//   - Stage 1: use the fixed v1 schema
//   - Stage 2: derive Started from Count
//   - Stage 3: format both supplied integers canonically.
//
// Behavior highlights:
//   - It can express admissible boundary states and intentionally invalid tamper cases without private access.
//
// Inputs:
//   - count: uint64; volume, weighted: non-nil integer totals in their public lattice units.
//
// Returns:
//   - vwap.Snapshot: detached strings and scalar metadata.
//
// Errors:
//   - No error return; Restore or the independent oracle decides admission.
//
// Determinism:
//   - Integer decimal formatting is canonical and independent of machine endianness.
//
// Complexity:
//   - O(D*A(B)) formatting time and O(D) output space for bounded inputs.
//
// Notes:
//   - Creating a snapshot does not assert that it represents real historical trades.
//
// AI-Hints:
//   - Never expose production-private arithmetic solely to manufacture these public test states.
func snapshotFromInts(count uint64, volume, weighted *big.Int) vwap.Snapshot {
	return vwap.Snapshot{
		Version: 1, Algorithm: "vwap/exact-weighted/v1", Count: count, Started: count != 0,
		VolumeSum: volume.String(), WeightedSum: weighted.String(),
	}
}

// finite classifies a floating-point test value without using production validators.
// It separates invalid-domain fuzz inputs from exact-oracle inputs.
//
// Implementation:
//   - Stage 1: check NaN
//   - Stage 2: check either infinity
//   - Stage 3: admit all other encodings.
//
// Behavior highlights:
//   - It separates invalid-domain fuzz inputs from exact-oracle inputs.
//
// Inputs:
//   - value: arbitrary float64.
//
// Returns:
//   - bool: true exactly for a finite input, including either zero sign.
//
// Errors:
//   - None.
//
// Determinism:
//   - Classification depends only on the input encoding.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - Signedness and volume nonnegativity are separate checks.
//
// AI-Hints:
//   - Do not let NaN enter big.Rat.SetFloat64 in an admitted-input oracle.
func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// expiredContext creates an already expired deadline without waiting for the scheduler.
// It makes DeadlineExceeded precedence reproducible across machines.
//
// Implementation:
//   - Stage 1: use a background parent
//   - Stage 2: select the Unix epoch as an elapsed deadline
//   - Stage 3: return the context and its cleanup function.
//
// Behavior highlights:
//   - It makes DeadlineExceeded precedence reproducible across machines.
//
// Inputs:
//   - None.
//
// Returns:
//   - context.Context and context.CancelFunc owned by the test.
//
// Errors:
//   - Err returns context.DeadlineExceeded immediately under the test environment clock.
//
// Determinism:
//   - The deadline is fixed in the distant past, not a short relative timeout.
//
// Complexity:
//   - O(1) time and storage.
//
// Notes:
//   - The caller still invokes cancel to release any context resources.
//
// AI-Hints:
//   - Never replace an already expired context with a sleep-based race.
func expiredContext() (context.Context, context.CancelFunc) {
	return context.WithDeadline(context.Background(), time.Unix(0, 0))
}

// validSnapshotOracle independently checks the full structural continuation contract.
// Unbounded test integers and canonical reformatting avoid reusing the production decimal parser.
//
// Implementation:
//   - Stage 1: validate schema, phase, and bounded lengths
//   - Stage 2: parse and reformat both sums canonically
//   - Stage 3: check sign, bit lengths, and exact cross-field bounds.
//
// Behavior highlights:
//   - Unbounded test integers and canonical reformatting avoid reusing the production decimal parser.
//
// Inputs:
//   - state: any public Snapshot, including deliberately malformed fields.
//
// Returns:
//   - bool: true exactly for the structural conditions tested here.
//
// Errors:
//   - None; malformed input is classified false.
//
// Determinism:
//   - Field order is explicit; every inequality uses exact test integer arithmetic.
//
// Complexity:
//   - O(D*A(B)) time and O(B+D) space, with D<=1300 per admitted string.
//
// Notes:
//   - Options has one value; structural validity is not trade provenance.
//
// AI-Hints:
//   - Do not infer authenticity or realizability of an actual trade history from these inequalities.
func validSnapshotOracle(state vwap.Snapshot) bool {
	if state.Version != 1 || state.Algorithm != "vwap/exact-weighted/v1" ||
		state.Started != (state.Count != 0) {
		return false
	}
	if len(state.VolumeSum) == 0 || len(state.VolumeSum) > 1300 ||
		len(state.WeightedSum) == 0 || len(state.WeightedSum) > 1300 {
		return false
	}
	volume, vok := new(big.Int).SetString(state.VolumeSum, 10)
	weighted, wok := new(big.Int).SetString(state.WeightedSum, 10)
	if !vok || !wok || volume.String() != state.VolumeSum || weighted.String() != state.WeightedSum {
		return false
	}
	if volume.Sign() < 0 || volume.BitLen() > 2162 || weighted.BitLen() > 4260 {
		return false
	}
	unitMax := units(math.MaxFloat64)
	limitV := new(big.Int).Mul(new(big.Int).SetUint64(state.Count), unitMax)
	limitW := new(big.Int).Mul(volume, unitMax)
	return volume.Cmp(limitV) <= 0 && new(big.Int).Abs(weighted).Cmp(limitW) <= 0
}

// historyFixture owns procedural test inputs; callers may mutate their fresh copy.
type historyFixture struct {
	price, volume []float64
	reset         []bool
}

// zeroSeedFixture builds fresh deterministic inputs for zero-volume seeding and an optional last-row reset.
// Every returned slice belongs to its caller; no external fixture file is used.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for zero-volume seeding and an optional last-row reset.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func zeroSeedFixture(resetLast bool) historyFixture {
	return historyFixture{[]float64{10, 20, 30}, []float64{0, 2, 1}, []bool{true, false, resetLast}}
}

// cancellationFixture builds fresh deterministic inputs for low-order precision surviving opposite extreme signed prices.
// Every returned slice belongs to its caller; no external fixture file is used.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for low-order precision surviving opposite extreme signed prices.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func cancellationFixture() historyFixture {
	return historyFixture{[]float64{1e308, 1, -1e308}, []float64{1, 1, 1}, []bool{true, false, false}}
}

// zeroVolumeFixture builds fresh deterministic inputs for undefined prefixes across explicit resets and both zero volume signs.
// Every returned slice belongs to its caller; no external fixture file is used.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for undefined prefixes across explicit resets and both zero volume signs.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func zeroVolumeFixture() historyFixture {
	return historyFixture{
		[]float64{math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64, math.Copysign(0, -1), 32},
		[]float64{0, math.Copysign(0, -1), 0, 0, 0},
		[]bool{true, false, true, false, false},
	}
}

// phaseFixture builds fresh deterministic inputs for masked, ready, zero-weight, and reset phase transitions.
// Every returned slice belongs to its caller; no external fixture file is used.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for masked, ready, zero-weight, and reset phase transitions.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func phaseFixture() historyFixture {
	return historyFixture{
		[]float64{3, 4, 9, 15, -12, -7, -1, -2},
		[]float64{0, 0, 2, 0, 0, 0, 3, 0},
		[]bool{true, false, false, false, true, false, false, false},
	}
}

// restartFixture builds fresh deterministic inputs for restart at every split across zero weights, resets, signed cancellation, and subnormals.
// Every returned slice belongs to its caller; no external fixture file is used.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for restart at every split across zero weights, resets, signed cancellation, and subnormals.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func restartFixture() historyFixture {
	return historyFixture{
		[]float64{10, 12, -2, 13, 0, -17, 4, 1e308, -1e308, math.SmallestNonzeroFloat64},
		[]float64{0, 0, 3, 0, 0, 2, 7, 1, 1, math.SmallestNonzeroFloat64},
		[]bool{true, false, false, true, false, false, false, false, false, false},
	}
}

// scaleFixture builds fresh deterministic inputs for invariance under exactly representable common positive weight scaling.
// Every returned slice belongs to its caller; no external fixture file is used.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for invariance under exactly representable common positive weight scaling.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func scaleFixture() historyFixture {
	return historyFixture{
		[]float64{-18, 4, 64, -2, 8, 0}, []float64{1, 3, 2, 4, 8, 0},
		[]bool{true, false, false, false, false, false},
	}
}

// roundingFixtures builds fresh deterministic inputs for halfway cases, normal/subnormal boundaries, and signed zero outcomes.
// Every returned slice belongs to its caller; no external fixture file is used.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for halfway cases, normal/subnormal boundaries, and signed zero outcomes.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func roundingFixtures() []historyFixture {
	tiny := math.SmallestNonzeroFloat64
	boundary := math.Ldexp(1, -1022)
	pairs := [][]float64{
		{0, tiny}, {-tiny, 0}, {tiny, 2 * tiny}, {2 * tiny, 3 * tiny},
		{math.Nextafter(boundary, 0), boundary}, {boundary, math.Nextafter(boundary, math.Inf(1))},
		{1, math.Nextafter(1, 2)}, {math.Nextafter(1, 2), math.Nextafter(math.Nextafter(1, 2), 2)},
		{math.Nextafter(2, 0), 2}, {-math.MaxFloat64, math.MaxFloat64},
		{-1, 1}, {0, 0}, {math.Copysign(0, -1), math.Copysign(0, -1)},
	}
	fixtures := make([]historyFixture, 0, len(pairs)+2)
	for _, price := range pairs {
		fixtures = append(fixtures, historyFixture{price, []float64{1, 1}, []bool{true, false}})
	}
	return append(fixtures,
		historyFixture{[]float64{tiny, -tiny, 0, tiny}, []float64{tiny, tiny, math.MaxFloat64, tiny}, []bool{true, false, false, false}},
		historyFixture{[]float64{-tiny, 0}, []float64{tiny, math.MaxFloat64}, []bool{true, false}},
	)
}

// coverageProfileEnv opts into the two-command coverage gate without adding a runtime tool.
const coverageProfileEnv = "LVLATH_VWAP_COVERPROFILE"

const (
	// coveragePrefix admits only records for this package from a repository-wide Go profile.
	coveragePrefix = "github.com/lvlath/go/vwap/"
	// coverageMinimum is inclusive and applies independently to each executable source file.
	coverageMinimum = uint64(92)
	// coveragePercentScale makes threshold checks exact rather than display-rounded.
	coveragePercentScale = uint64(100)
	// maxCoverageStatements leaves multiplication headroom for exact percentage comparisons.
	maxCoverageStatements = ^uint64(0) / coveragePercentScale
)

// coverageCounts records covered and instrumented statements, never execution frequencies.
// A block with multiple executions contributes its statement count exactly once.
type coverageCounts struct {
	covered    uint64
	statements uint64
}

// meetsThreshold checks the inclusive coverage floor without floating-point rounding.
// A displayed 92.00 percent must not let a true sub-threshold fraction pass.
//
// Implementation:
//   - Stage 1: require a nonzero statement denominator
//   - Stage 2: compare covered*100 with minimum*statements
//   - Stage 3: return the exact comparison.
//
// Behavior highlights:
//   - A displayed 92.00 percent must not let a true sub-threshold fraction pass.
//
// Inputs:
//   - c: counts bounded by maxCoverageStatements and covered<=statements.
//
// Returns:
//   - bool: true exactly when the inclusive minimum is reached.
//
// Errors:
//   - None; parsing and total validation establish arithmetic headroom.
//
// Determinism:
//   - Only unsigned integer multiplication/comparison determines the gate.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - Zero statements are N/A at file reporting level, not a successful percentage.
//
// AI-Hints:
//   - Do not use percent(), formatted strings, or an epsilon for admission.
func (c coverageCounts) meetsThreshold() bool {
	return c.statements != 0 && c.covered*coveragePercentScale >= coverageMinimum*c.statements
}

// percent calculates a human-readable statement percentage for test logs.
// This display value is never used to decide whether a file meets the threshold.
//
// Implementation:
//   - Stage 1: handle an empty denominator
//   - Stage 2: convert bounded counts to float64
//   - Stage 3: return the scaled ratio.
//
// Behavior highlights:
//   - This display value is never used to decide whether a file meets the threshold.
//
// Inputs:
//   - c: nonnegative covered and total statement counts.
//
// Returns:
//   - float64: display percentage, or zero for an empty total.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fixed arithmetic affects display only, not exact gate behavior.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - Large counts may round for display without changing gate correctness.
//
// AI-Hints:
//   - All admission decisions must call meetsThreshold instead.
func (c coverageCounts) percent() float64 {
	if c.statements == 0 {
		return 0
	}
	return float64(c.covered) * float64(coveragePercentScale) / float64(c.statements)
}

// readCoverageProfile reads a standard set/count/atomic Go coverage profile.
// It rejects duplicated blocks and malformed records before accumulating per-file counts.
//
// Implementation:
//   - Stage 1: validate the mode header and record grammar.
//   - Stage 2: filter the exact VWAP module path and count each unique block once.
//   - Stage 3: return the detached counts or reject profiles with no VWAP statements.
//
// Behavior highlights:
//   - Other packages are ignored, but an empty/foreign-only report is not successful evidence.
//
// Inputs:
//   - input: a readable coverage stream; no market data is loaded.
//
// Returns:
//   - A map keyed by production basename; iteration order is never used for public output.
//
// Errors:
//   - Diagnostic errors for malformed, negative, overflowing, duplicated, or absent records.
//
// Determinism:
//   - Input records are processed in order; each block contributes at most once.
//
// Complexity:
//   - Time O(B), Space O(B), B=profile bytes/blocks, with Scanner's bounded line buffer.
//
// Notes:
//   - Only counts within maxCoverageStatements are admitted to preserve exact threshold math.
//
// AI-Hints:
//   - A hit count is not a statement count; count-mode repetition must not inflate coverage.
func readCoverageProfile(input io.Reader) (map[string]coverageCounts, error) {
	scanner := bufio.NewScanner(input)
	if !scanner.Scan() {
		return nil, fmt.Errorf("coverage profile has no mode header")
	}
	switch scanner.Text() {
	case "mode: set", "mode: count", "mode: atomic":
	default:
		return nil, fmt.Errorf("invalid coverage mode")
	}
	report := make(map[string]coverageCounts)
	seen := make(map[string]bool)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 3 {
			return nil, fmt.Errorf("coverage record must have three fields")
		}
		location := fields[0]
		colon := strings.LastIndexByte(location, ':')
		if colon < 0 || !validCoverageRange(location[colon+1:]) {
			return nil, fmt.Errorf("invalid coverage location %q", location)
		}
		statements, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil || statements > maxCoverageStatements {
			return nil, fmt.Errorf("invalid statement count %q", fields[1])
		}
		hits, err := strconv.ParseUint(fields[2], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid execution count %q", fields[2])
		}
		if !strings.HasPrefix(location[:colon], coveragePrefix) {
			continue
		}
		name := strings.TrimPrefix(location[:colon], coveragePrefix)
		if strings.Contains(name, "/") || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			return nil, fmt.Errorf("invalid VWAP source filename %q", name)
		}
		if seen[location] {
			return nil, fmt.Errorf("duplicate coverage block %q", location)
		}
		seen[location] = true
		counts := report[name]
		if counts.statements > maxCoverageStatements-statements {
			return nil, fmt.Errorf("coverage statement count overflows safe percentage range")
		}
		counts.statements += statements
		if hits != 0 {
			counts.covered += statements
		}
		report[name] = counts
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(report) == 0 {
		return nil, fmt.Errorf("profile contains no VWAP records")
	}
	return report, nil
}

// validCoverageRange validates a canonical source interval in a Go coverage record.
// A reversed or malformed line/column range cannot masquerade as valid coverage evidence.
//
// Implementation:
//   - Stage 1: split four unsigned positions
//   - Stage 2: require positive values and exact delimiter spelling
//   - Stage 3: require end not before start.
//
// Behavior highlights:
//   - A reversed or malformed line/column range cannot masquerade as valid coverage evidence.
//
// Inputs:
//   - text: candidate line.column,line.column record suffix.
//
// Returns:
//   - bool: true only for the canonical nondecreasing interval grammar.
//
// Errors:
//   - None; parse errors return false.
//
// Determinism:
//   - Canonical reformatting excludes alternative numeric spellings or misplaced separators.
//
// Complexity:
//   - O(len(text)) time and storage.
//
// Notes:
//   - The function validates syntax, not the existence of these source lines.
//
// AI-Hints:
//   - Freshness is a CI responsibility; a syntactically valid interval cannot authenticate a profile.
func validCoverageRange(text string) bool {
	parts := strings.FieldsFunc(text, func(r rune) bool { return r == '.' || r == ',' })
	if len(parts) != 4 || strings.Count(text, ".") != 2 || strings.Count(text, ",") != 1 {
		return false
	}
	var positions [4]uint64
	for i, part := range parts {
		value, err := strconv.ParseUint(part, 10, 64)
		if err != nil || value == 0 {
			return false
		}
		positions[i] = value
	}
	// Reformatting excludes misplaced delimiters or repeated separators.
	canonical := fmt.Sprintf("%d.%d,%d.%d", positions[0], positions[1], positions[2], positions[3])
	return canonical == text && (positions[2] > positions[0] || positions[2] == positions[0] && positions[3] >= positions[1])
}

// coverageSource identifies whether a production file has an executable function body.
// Declaration-only files remain explicitly N/A rather than receiving fabricated coverage.
type coverageSource struct {
	name       string
	executable bool
}

// coverageSourceFiles discovers source files eligible for per-file statement coverage.
// Function bodies are inspected instead of exempting a hardcoded list of filenames.
//
// Implementation:
//   - Stage 1: glob Go sources in sorted order
//   - Stage 2: ignore external test files and parse production ASTs
//   - Stage 3: mark nonempty function/literal bodies executable.
//
// Behavior highlights:
//   - Function bodies are inspected instead of exempting a hardcoded list of filenames.
//
// Inputs:
//   - source: readable filesystem rooted at the package directory.
//
// Returns:
//   - []coverageSource: sorted filename/executable inventory.
//
// Errors:
//   - Filesystem/parser errors or a directory with no production Go files.
//
// Determinism:
//   - fs.Glob order controls the resulting inventory and diagnostics.
//
// Complexity:
//   - O(source bytes) time and space.
//
// Notes:
//   - Declaration-only files remain N/A; no percentage is invented for them.
//
// AI-Hints:
//   - Do not hide a new executable file by excluding its name from the coverage gate.
func coverageSourceFiles(source fs.FS) ([]coverageSource, error) {
	names, err := fs.Glob(source, "*.go")
	if err != nil {
		return nil, err
	}
	files := make([]coverageSource, 0, len(names))
	for _, name := range names {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		contents, err := fs.ReadFile(source, name)
		if err != nil {
			return nil, err
		}
		file, err := parser.ParseFile(token.NewFileSet(), name, contents, 0)
		if err != nil {
			return nil, err
		}
		executable := false
		ast.Inspect(file, func(node ast.Node) bool {
			switch item := node.(type) {
			case *ast.FuncDecl:
				executable = executable || item.Body != nil && len(item.Body.List) != 0
			case *ast.FuncLit:
				executable = executable || len(item.Body.List) != 0
			}
			return true
		})
		files = append(files, coverageSource{name: name, executable: executable})
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no production Go sources found")
	}
	// fs.Glob returns sorted names, giving stable diagnostics across runs.
	return files, nil
}

// validateCoverageFiles matches parsed profile totals to the current source inventory.
// It rejects missing executable files and profile basenames no longer present in the checkout.
//
// Implementation:
//   - Stage 1: traverse the sorted source inventory
//   - Stage 2: accumulate checked statement counts
//   - Stage 3: sort profile names and reject unknown sources.
//
// Behavior highlights:
//   - It rejects missing executable files and profile basenames no longer present in the checkout.
//
// Inputs:
//   - report: parsed per-file totals; files: current production source inventory.
//
// Returns:
//   - coverageCounts: checked package total, or a diagnostic error.
//
// Errors:
//   - Missing executable coverage, covered>total, oversized totals, or an unknown profile basename.
//
// Determinism:
//   - All diagnostic selection follows sorted source/name order, never map iteration order.
//
// Complexity:
//   - O(F log F) time and O(F) space, F=number of production files.
//
// Notes:
//   - Same-name stale profiles cannot be authenticated; regenerate the profile immediately before the gate.
//
// AI-Hints:
//   - Do not let one well-covered file conceal a missing file or a per-file failure.
func validateCoverageFiles(report map[string]coverageCounts, files []coverageSource) (coverageCounts, error) {
	var total coverageCounts
	known := make(map[string]bool, len(files))
	for _, file := range files {
		known[file.name] = true
		counts := report[file.name]
		if file.executable && counts.statements == 0 {
			return total, fmt.Errorf("missing executable-file coverage: %s", file.name)
		}
		if counts.covered > counts.statements || counts.statements > maxCoverageStatements-total.statements {
			return total, fmt.Errorf("invalid or overflowing statement totals")
		}
		total.covered += counts.covered
		total.statements += counts.statements
	}
	// Sort before checking extra names so multiple stale records have stable diagnostics.
	names := make([]string, 0, len(report))
	for name := range report {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		if !known[name] {
			return total, fmt.Errorf("profile source is not in this checkout: %s", name)
		}
	}
	return total, nil
}

// coverageProfileCase is one procedural parser fixture and its expected statement counts.
type coverageProfileCase struct {
	name, text string
	bad        bool
	want       coverageCounts
}

// coverageProfileFixtures constructs valid and adversarial evidence entirely in Go.
// No generated profile must be checked into the repository for these parser tests.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for valid coverage modes and adversarial parser records.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func coverageProfileFixtures() []coverageProfileCase {
	row := coveragePrefix + "api.go:1.1,2.2 "
	return []coverageProfileCase{
		{"set", "mode: set\n" + row + "25 1\n", false, coverageCounts{25, 25}},
		{"count", "mode: count\n" + row + "25 100\n", false, coverageCounts{25, 25}},
		{"atomic-uncovered", "mode: atomic\n" + row + "25 0\n", false, coverageCounts{0, 25}},
		{"empty", "", true, coverageCounts{}},
		{"mode", "mode: unknown\n" + row + "1 1\n", true, coverageCounts{}},
		{"no-records", "mode: set\n", true, coverageCounts{}},
		{"foreign-only", "mode: set\nother/pkg.go:1.1,2.2 1 1\n", true, coverageCounts{}},
		{"fields", "mode: set\n" + row + "1\n", true, coverageCounts{}},
		{"negative-count", "mode: set\n" + row + "-1 1\n", true, coverageCounts{}},
		{"negative-hits", "mode: set\n" + row + "1 -1\n", true, coverageCounts{}},
		{"overflow", "mode: set\n" + row + "18446744073709551615 1\n", true, coverageCounts{}},
		{"invalid-range", "mode: set\n" + coveragePrefix + "api.go:2.3,1.1 1 1\n", true, coverageCounts{}},
		{"duplicate", "mode: set\n" + row + "1 1\n" + row + "1 1\n", true, coverageCounts{}},
		{"test-file", "mode: set\n" + coveragePrefix + "api_test.go:1.1,2.2 1 1\n", true, coverageCounts{}},
		{"scanner-limit", "mode: set\n" + strings.Repeat("x", 70000), true, coverageCounts{}},
	}
}

// coverageSourceFixture supplies one executable source, one declaration, and an ignored test.
// All files are in-memory strings owned by the standard-library test filesystem.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for executable source discovery with declaration-only and test-file controls.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func coverageSourceFixture() fs.FS {
	return fstest.MapFS{
		"api.go":          &fstest.MapFile{Data: []byte("package vwap\nfunc f() int { return 1 }\n")},
		"types.go":        &fstest.MapFile{Data: []byte("package vwap\ntype Result struct{}\n")},
		"ignored_test.go": &fstest.MapFile{Data: []byte("not parsed")},
	}
}

// wrapperCase supplies one public default/canonical facade parity scenario.
// ctx and all slice channels belong to the test; neither invocation may mutate them.
type wrapperCase struct {
	name          string
	ctx           context.Context
	price, volume []float64
	reset         []bool
	want          error
}

// wrapperFixtures covers successful, empty, shape, anchor, context, and numerical boundaries.
// Cancellation is fixed before use, not scheduled by a wall-clock timeout.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for default/canonical facade error, cancellation, and success parity.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func wrapperFixtures() []wrapperCase {
	ctx := context.Background()
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	fixture := restartFixture()
	return []wrapperCase{
		{"restart", ctx, fixture.price, fixture.volume, fixture.reset, nil},
		{"nil-empty", ctx, nil, nil, nil, nil},
		{"allocated-empty", ctx, []float64{}, []float64{}, []bool{}, nil},
		{"nil-context", nil, nil, nil, nil, vwap.ErrNilContext},
		{"canceled", canceled, nil, nil, nil, context.Canceled},
		{"shape", ctx, []float64{1}, []float64{}, []bool{true}, vwap.ErrShape},
		{"nil-channel", ctx, []float64{1}, nil, []bool{true}, vwap.ErrNilInput},
		{"anchor", ctx, []float64{math.NaN()}, []float64{-1}, []bool{false}, vwap.ErrAnchor},
		{"nonfinite", ctx, []float64{math.Inf(1)}, []float64{0}, []bool{true}, vwap.ErrNonFinite},
		{"negative-volume", ctx, []float64{-10}, []float64{-1}, []bool{true}, vwap.ErrVolume},
		{"per-row-order", ctx, []float64{1, math.NaN()}, []float64{-1, 1}, []bool{true, false}, vwap.ErrVolume},
	}
}

// quotientStateFixture builds structurally valid rounding states around an exact half-denominator remainder.
// Its independent integer numerator and rational oracle exercise normalized division through Restore.
//
// Implementation:
//   - Stage 1: choose a denominator with a controlled leading-bit position
//   - Stage 2: form quotient plus half remainder and offset
//   - Stage 3: apply scale/sign and independently round the physical rational.
//
// Behavior highlights:
//   - Its independent integer numerator and rational oracle exercise normalized division through Restore.
//
// Inputs:
//   - leading: 0..63; scale: tested binary lattice shift; quotient/offset: small bounded integers; negative: numerator sign.
//
// Returns:
//   - vwap.Snapshot and float64: canonical exact state and independent expected rounded value.
//
// Errors:
//   - None for the bounded parameters enumerated by the regression test.
//
// Determinism:
//   - Every case is determined by the five explicit parameters.
//
// Complexity:
//   - O(A(B)) arithmetic time, O(B+D) storage, bounded by the tested checkpoint sizes.
//
// Notes:
//   - The snapshot is a structural arithmetic fixture, not claimed historical market data.
//
// AI-Hints:
//   - Keep remainder construction independent from production quotient estimation and tie correction.
func quotientStateFixture(leading, scale uint, quotient, offset int64, negative bool) (vwap.Snapshot, float64) {
	denominator := new(big.Int).Lsh(big.NewInt(1), 128+leading)
	denominator.Add(denominator, new(big.Int).Lsh(big.NewInt(1), 63))
	denominator.Add(denominator, big.NewInt(18)) // Even, so half-way remainders exist exactly.
	remainder := new(big.Int).Rsh(new(big.Int).Set(denominator), 1)
	remainder.Add(remainder, big.NewInt(offset))
	numerator := new(big.Int).Mul(denominator, big.NewInt(quotient))
	numerator.Add(numerator, remainder)
	numerator.Lsh(numerator, scale)
	if negative {
		numerator.Neg(numerator)
	}
	state := snapshotFromInts(^uint64(0)-1, denominator, numerator)
	physicalDenominator := new(big.Int).Lsh(new(big.Int).Set(denominator), 1074)
	expected, _ := new(big.Rat).SetFrac(numerator, physicalDenominator).Float64()
	return state, expected
}

// vendorBar is one frozen close/volume record from the provider's public demonstration response.
// label preserves the provider's US/Eastern bar label; it is not an inferred availability timestamp.
type vendorBar struct {
	label  string
	price  float64
	volume float64
}

// vendorIBMClosingHour returns the twelve 2026-08-07 15:00..15:55 IBM demo bars in ascending order.
// Source: https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo
// The response inspected on 2026-11-27 identifies Last Refreshed=2026-08-07 19:55:00, US/Eastern.
// These are frozen vendor-demo observations, not live quotes or independently exchange-audited trades.
// Close is used deliberately: close*bar-volume is only a bar-price benchmark, not tick VWAP.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for a reproducible close-volume reporting benchmark from frozen vendor-demo bars.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func vendorIBMClosingHour() []vendorBar {
	return []vendorBar{
		{"15:00", 236.4600, 48847}, {"15:05", 236.5409, 53105},
		{"15:10", 236.4300, 66378}, {"15:15", 236.3400, 29011},
		{"15:20", 236.4550, 34613}, {"15:25", 236.8950, 78705},
		{"15:30", 236.5900, 45408}, {"15:35", 237.2650, 57650},
		{"15:40", 236.8950, 52567}, {"15:45", 236.5700, 71097},
		{"15:50", 236.4400, 101225}, {"15:55", 237.2100, 446739},
	}
}

// executionFixture returns a complete engineered market tape and a separate buy-fill tape.
// Both tapes refer to the same explicitly chosen 10:00:00..10:01:00 window, symbol, currency, and share units.
// These records model a test exchange, not actual customer orders or verified historical executions.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for same-window market and buy-fill benchmarks with separate execution weights.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func executionFixture() (historyFixture, historyFixture) {
	market := historyFixture{
		price:  []float64{100, 100.05, 100.10, 100.15, 100.10, 100.20, 100.25, 100.20, 100.30, 100.25, 100.35, 100.40},
		volume: []float64{200, 300, 150, 400, 250, 100, 350, 200, 300, 150, 250, 400},
		reset:  []bool{true, false, false, false, false, false, false, false, false, false, false, false},
	}
	fills := historyFixture{
		price:  []float64{100.08, 100.14, 100.18, 100.24, 100.29, 100.34},
		volume: []float64{100, 150, 100, 200, 150, 100},
		reset:  []bool{true, false, false, false, false, false},
	}
	return market, fills
}

// availableObservation carries application admission time separately from the numeric input.
// The VWAP package receives only the price, volume, and explicit reset decision.
type availableObservation struct {
	availableAt   time.Time
	price, volume float64
}

// newsAvailabilityFixture models a synthetic publication received 1200 ms after its event time.
// Observations before delivery cannot be retrospectively relabeled as an online available-at anchor.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for causal anchoring at information delivery rather than retrospective event time.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func newsAvailabilityFixture() (time.Time, []availableObservation) {
	eventTime := time.Date(2026, time.August, 10, 14, 0, 0, 0, time.UTC)
	availableAt := eventTime.Add(1200 * time.Millisecond)
	offsets := []time.Duration{-1000, -500, 0, 250, 750, 1200, 1500, 2000, 2500, 3000, 3500, 4000}
	prices := []float64{100, 100.10, 100.20, 100.15, 100.30, 100.50, 100.60, 100.70, 100.65, 100.80, 100.75, 100.90}
	volumes := []float64{200, 100, 500, 100, 200, 0, 100, 200, 0, 150, 250, 300}
	rows := make([]availableObservation, len(prices))
	for i := range prices {
		rows[i] = availableObservation{eventTime.Add(offsets[i] * time.Millisecond), prices[i], volumes[i]}
	}
	return availableAt, rows
}

// replayObservation attaches a transport sequence to an engineered numeric observation.
// Sequence belongs to the source adapter, not to the internal segment Count.
type replayObservation struct {
	sequence      uint64
	price, volume float64
	reset         bool
}

// replayFixture creates a twelve-record session with an initially empty volume seed.
// Delivery of sequence 2 is duplicated by the example; the canonical tape itself has no duplicate.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for source-cursor replay around a zero-volume checkpoint and invalid reset.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func replayFixture() []replayObservation {
	return []replayObservation{
		{1, 100, 0, true}, {2, 100.10, 0, false}, {3, 100.25, 100, false},
		{4, 100.50, 200, false}, {5, 100.25, 100, false}, {6, 101, 100, false},
		{7, 100.75, 300, false}, {8, 101.25, 200, false}, {9, 101, 0, false},
		{10, 101.50, 100, false}, {11, 101.25, 100, false}, {12, 101.75, 200, false},
	}
}

// restartEnvelope associates an exact numerical checkpoint with an application source cursor.
// A real storage transaction must persist both fields together and authenticate the record.
type restartEnvelope struct {
	State        vwap.Snapshot
	LastSequence uint64
}

// pairedSpreadRow describes prices observed for one simultaneous pair and its matched quantity.
// Independent leg volumes are intentionally absent: they are not valid spread weights.
type pairedSpreadRow struct {
	near, far float64
	matched   float64
}

// matchedSpreadFixture supplies twelve engineered paired-lot observations with signed prices.
// The numerical unit is price per matched pair, not a ratio of unrelated leg volumes.
//
// Implementation:
//   - Stage 1: specify fixed records or bounded procedural arithmetic for signed price-basis averaging of synchronized completed matched pairs.
//   - Stage 2: construct fresh scalar records and owning slice channels where applicable.
//   - Stage 3: return the data without invoking production-private code.
//
// Behavior highlights:
//   - No network, current-time read, embed directive, or checked-in binary fixture is required.
//
// Inputs:
//   - Explicit function parameters only; no ambient market or application state.
//
// Returns:
//   - Fresh scenario data; numerical expectations are checked separately through an independent oracle.
//
// Errors:
//   - No error return; the fixed fixture construction uses admitted scalar operations.
//
// Determinism:
//   - Record order and constants are fixed; changing external services cannot refresh these inputs.
//
// Complexity:
//   - Time and space O(n) in the fixed number of returned records, including owning arrays.
//
// Notes:
//   - A fixture is test/example input, not evidence that a live trading system can reproduce its outcome.
//
// AI-Hints:
//   - Preserve the stated provenance and scenario meaning; do not silently relabel engineered records as market history.
func matchedSpreadFixture() []pairedSpreadRow {
	return []pairedSpreadRow{
		{94, 100, 0}, {96, 100, 3}, {98, 100, 4}, {100, 100, 0},
		{102, 100, 4}, {104, 100, 3}, {106, 100, 2}, {104, 100, 0},
		{102, 100, 4}, {100, 100, 3}, {98, 100, 2}, {96, 100, 3},
	}
}
