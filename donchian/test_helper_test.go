// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian_test centralizes reusable assertions, independent oracles, and fixed fixtures.
// This file uses only exported production operations and preserves explicit fixture ownership.
// AI-Hints: Keep math/big test-only and never label engineered stress data as observed market history.
package donchian_test

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"math/big"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/donchian"
)

// benchmarkResult keeps returned owning artifacts alive across measured iterations.
// Benchmarks are serial; concurrent production calls never access this test variable.
var benchmarkResult *donchian.Result

// The structural assertion checks nilability without adding a production dependency.
var _ interface{ IsNil() bool } = (*donchian.Result)(nil)

const (
	// priceFloatBytes is the fixed binary64 byte width in the procedural fuzz codec.
	priceFloatBytes = 8
	// pricePairBytes stores one high followed by one low, both little-endian binary64.
	pricePairBytes = 2 * priceFloatBytes
	// maxFuzzBars bounds independent O(n*P) oracle work for untrusted fuzz byte streams.
	maxFuzzBars = 64
)

// mustCompute runs canonical Compute and requires an owning successful result.
// It does not derive expectations or replace the public algorithm.
//
// Implementation:
//   - Stage 1: mark the assertion as a helper.
//   - Stage 2: call Compute with an active background context.
//   - Stage 3: fail on error or nil and return the result.
//
// Behavior highlights:
//   - The returned artifact is the actual public result, never a reconstructed expectation.
//
// Inputs:
//   - t: owning test/benchmark harness; high/low/opts: arguments forwarded unchanged.
//
// Returns:
//   - *donchian.Result: the caller-owned successful artifact.
//
// Errors:
//   - Unexpected errors or nil success fail the owning harness.
//
// Determinism:
//   - Forwards the supplied slices and options unchanged; Compute owns ordering.
//
// Complexity:
//   - O(n) public execution and O(n) output storage.
//
// Notes:
//   - This helper is not used to assert expected errors.
//
// AI-Hints:
//   - Never construct expected values by calling this helper twice instead of an independent oracle.
func mustCompute(t testing.TB, high, low []float64, opts donchian.Options) *donchian.Result {
	t.Helper()
	result, err := donchian.Compute(context.Background(), high, low, opts)
	if err != nil {
		t.Fatalf("Compute: %v", err)
	}
	if result == nil {
		t.Fatal("successful Compute returned nil")
	}
	return result
}

// assertFloats compares float slices by their binary64 representations.
// It detects signed-zero differences and does not hide ULP errors behind a tolerance.
//
// Implementation:
//   - Stage 1: mark the helper and compare lengths.
//   - Stage 2: scan matching positions ascending.
//   - Stage 3: report the first unequal payload and both bit patterns.
//
// Behavior highlights:
//   - NaN payload bits and negative zero are compared exactly, not by float equality.
//
// Inputs:
//   - t: owning harness; got/want: immutable slices.
//
// Returns:
//   - No value; a mismatch fails the harness.
//
// Errors:
//   - Length or bit differences are assertion failures.
//
// Determinism:
//   - Compares lengths first, then bit patterns from index zero upward.
//
// Complexity:
//   - O(n) time and O(1) extra space.
//
// Notes:
//   - Nil and empty float slices have equal lengths here; assertResult separately enforces non-nil channels.
//
// AI-Hints:
//   - Do not use numerical equality where the zero sign is observable.
func assertFloats(t testing.TB, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("index %d: got %g (%016x), want %g (%016x)",
				i, got[i], math.Float64bits(got[i]), want[i], math.Float64bits(want[i]))
		}
	}
}

// assertResult checks a result against a separately defined complete artifact.
// All numeric channels, readiness bits, and masked positive-zero payloads are defended.
//
// Implementation:
//   - Stage 1: reject nil results or nil successful channels.
//   - Stage 2: compare each numeric slice and the mask length.
//   - Stage 3: check every mask bit and all masked payload signs.
//
// Behavior highlights:
//   - Success requires non-nil slices even when every channel has length zero.
//
// Inputs:
//   - t: owning harness; got/want: complete aligned results.
//
// Returns:
//   - No value; a contract mismatch fails the harness.
//
// Errors:
//   - Nil channels, length mismatch, or payload differences are failures.
//
// Determinism:
//   - Checks Upper, Middle, Lower, then ascending mask positions.
//
// Complexity:
//   - O(n) time and O(1) extra space.
//
// Notes:
//   - Expected artifacts are supplied by literals or the independent direct-window oracle.
//
// AI-Hints:
//   - Never infer validity from a nonzero boundary.
func assertResult(t testing.TB, got, want *donchian.Result) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatal("nil result in comparison")
	}
	if got.Upper == nil || got.Middle == nil || got.Lower == nil || got.Valid == nil {
		t.Fatal("successful output contains a nil channel")
	}
	assertFloats(t, got.Upper, want.Upper)
	assertFloats(t, got.Middle, want.Middle)
	assertFloats(t, got.Lower, want.Lower)
	if len(got.Valid) != len(want.Valid) {
		t.Fatal("mask lengths differ")
	}
	for i, valid := range want.Valid {
		if got.Valid[i] != valid {
			t.Fatalf("Valid[%d]: got %t, want %t", i, got.Valid[i], valid)
		}
		if !valid && (math.Float64bits(got.Upper[i]) != 0 ||
			math.Float64bits(got.Middle[i]) != 0 || math.Float64bits(got.Lower[i]) != 0) {
			t.Fatalf("masked row %d is not positive zero", i)
		}
	}
}

// assertPrefix compares a successful prefix with the corresponding whole-history rows.
// The views are test-local and never change either result.
//
// Implementation:
//   - Stage 1: mark the helper and read prefix length.
//   - Stage 2: form bounded whole-history views.
//   - Stage 3: delegate exact channel/mask comparison.
//
// Behavior highlights:
//   - The whole-history views are read-only and cannot hide a changed warmup prefix.
//
// Inputs:
//   - whole: at least as long as prefix; t: owning harness.
//
// Returns:
//   - No value; changed historical output fails the harness.
//
// Errors:
//   - The caller establishes slice bounds before this internal assertion.
//
// Determinism:
//   - Compares the first len(prefix.Valid) rows in original positional order.
//
// Complexity:
//   - O(k) time for prefix length k, O(1) space.
//
// Notes:
//   - Both arguments must already be successful owning results.
//
// AI-Hints:
//   - Checking only the last row misses earlier look-ahead contamination.
func assertPrefix(t testing.TB, whole, prefix *donchian.Result) {
	t.Helper()
	n := len(prefix.Valid)
	assertResult(t, &donchian.Result{
		Upper: whole.Upper[:n], Middle: whole.Middle[:n],
		Lower: whole.Lower[:n], Valid: whole.Valid[:n],
	}, prefix)
}

// syntheticBars builds a deterministic signed oscillating candle sequence.
// Quarter-unit centers and widths remain exact for the bounded test workloads.
//
// Implementation:
//   - Stage 1: allocate equally sized owning high and low slices.
//   - Stage 2: compute the fixed modular center and positive width.
//   - Stage 3: store valid signed candle pairs.
//
// Behavior highlights:
//   - The sequence alternates signed levels and widths to exercise both deque tails.
//
// Inputs:
//   - n: nonnegative bounded fixture length chosen by the suite.
//
// Returns:
//   - Two fresh finite slices with high >= low.
//
// Errors:
//   - None for suite-provided sizes.
//
// Determinism:
//   - Index-based modular centers and widths use no random or clock state.
//
// Complexity:
//   - O(n) time and O(n) output space.
//
// Notes:
//   - This is engineered data, not a market record.
//
// AI-Hints:
//   - Do not label generated price histories as observed exchange data.
func syntheticBars(n int) ([]float64, []float64) {
	high, low := make([]float64, n), make([]float64, n)
	for i := range high {
		center := float64((i*37)%101-50) * 0.25
		width := float64(i%7+1) * 0.25
		high[i], low[i] = center+width, center-width
	}
	return high, low
}

// monotoneBars builds a strictly monotonic finite history in either direction.
// One deque retains a long frontier while the other repeatedly drops its tail.
//
// Implementation:
//   - Stage 1: allocate equally sized channels.
//   - Stage 2: choose the sign of each increasing index.
//   - Stage 3: offset the center by one unit for high and low.
//
// Behavior highlights:
//   - Ascending and descending cases exchange which queue retains the full window.
//
// Inputs:
//   - n: bounded test length; descending: reverses the direction.
//
// Returns:
//   - Two fresh monotonic valid price channels.
//
// Errors:
//   - None for bounded suite sizes.
//
// Determinism:
//   - Centers follow increasing integer indices, with an explicitly chosen sign.
//
// Complexity:
//   - O(n) time and O(n) space.
//
// Notes:
//   - The sequence targets queue occupancy, not realistic market calibration.
//
// AI-Hints:
//   - Keep the signed direction explicit when interpreting benchmark names.
func monotoneBars(n int, descending bool) ([]float64, []float64) {
	high, low := make([]float64, n), make([]float64, n)
	for i := range high {
		center := float64(i)
		if descending {
			center = -center
		}
		high[i], low[i] = center+1, center-1
	}
	return high, low
}

// burstBars builds a long monotonic frontier followed by one dominating extremum.
// The final insertion forces internal cancellation polling inside a removal burst.
//
// Implementation:
//   - Stage 1: allocate valid zero-based signed ranges.
//   - Stage 2: build the selected monotonic candidate frontier.
//   - Stage 3: replace the last high or low with a dominating value.
//
// Behavior highlights:
//   - The targeted queue performs thousands of removals in one insertion.
//
// Inputs:
//   - n: positive bounded size; upperBurst: selects the queue to drain.
//
// Returns:
//   - Owning high/low channels for a controlled large tail burst.
//
// Errors:
//   - None under the positive-length fixture precondition.
//
// Determinism:
//   - A fixed monotone prefix is followed by one selected extreme at the last index.
//
// Complexity:
//   - O(n) time and O(n) space.
//
// Notes:
//   - Use a period large enough to retain thousands of candidates before the last insertion.
//
// AI-Hints:
//   - A random history rarely exercises a long single-bar removal loop.
func burstBars(n int, upperBurst bool) ([]float64, []float64) {
	high, low := make([]float64, n), make([]float64, n)
	for i := range high {
		if upperBurst {
			high[i] = float64(n - i)
		} else {
			low[i] = -float64(n - i)
		}
	}
	if upperBurst {
		high[n-1] = 1e6
	} else {
		low[n-1] = -1e6
	}
	return high, low
}

// pollCancelContext observes Err calls and cancels a real child at an exact observation count.
// It makes cancellation paths reproducible without replacing context error semantics.
//
// Implementation:
//   - Stage 1: newPollContext constructs the child and fixes the trigger.
//   - Stage 2: Err increments an atomic counter and triggers cancel at the selected count.
//   - Stage 3: the embedded context supplies sticky Err and Done behavior.
//
// Behavior highlights:
//   - Trigger zero only measures; a positive trigger cancels once reached.
//
// Inputs:
//   - Constructed by newPollContext; callers must not mutate at during use.
//
// Returns:
//   - A context.Context-compatible probe with an independently inspectable counter.
//
// Errors:
//   - Err returns the child context error rather than a synthetic test sentinel.
//
// Determinism:
//   - Atomic counting orders observations; sequential callers reach the same trigger.
//
// Complexity:
//   - O(1) storage and O(1) work per Err observation.
//
// Notes:
//   - The owner releases the real child with cancel even on successful paths.
//
// AI-Hints:
//   - Do not replace this probe with sleeps or a context whose cancellation can disappear.
type pollCancelContext struct {
	// Context supplies the real cancellation state and all non-Err methods.
	context.Context
	// cancel releases the child both at the trigger and when the test is finished.
	cancel context.CancelFunc
	// at is the immutable trigger; zero means count observations without canceling.
	at int64
	// calls counts Err observations without racing concurrent context users.
	calls atomic.Int64
}

// newPollContext creates a deterministic cancellation probe backed by a real context.
// The test chooses an Err observation rather than relying on scheduler timing.
//
// Implementation:
//   - Stage 1: create a child cancellation context.
//   - Stage 2: store its cancellation function and immutable trigger.
//   - Stage 3: return a fresh atomic observation counter.
//
// Behavior highlights:
//   - A real child context supplies Done and sticky cancellation semantics.
//
// Inputs:
//   - at: positive target poll; zero observes without triggering cancellation.
//
// Returns:
//   - *pollCancelContext: fresh test context whose cancel must be called by its owner.
//
// Errors:
//   - None; the child context uses context.Background.
//
// Determinism:
//   - The immutable at value selects an Err observation count; zero never triggers.
//
// Complexity:
//   - O(1) time and bounded context storage.
//
// Notes:
//   - The probe is test-only and is never accepted as part of the production public surface.
//
// AI-Hints:
//   - Call cancel on successful probe paths as well as interrupted paths.
func newPollContext(at int64) *pollCancelContext {
	ctx, cancel := context.WithCancel(context.Background())
	return &pollCancelContext{Context: ctx, cancel: cancel, at: at}
}

// Err observes cancellation and optionally triggers it on a designated poll.
// All non-Err context methods retain the real child context semantics.
//
// Implementation:
//   - Stage 1: increment the atomic observation counter.
//   - Stage 2: cancel the child exactly when the designated count is reached.
//   - Stage 3: return the child context error unchanged.
//
// Behavior highlights:
//   - After the child is canceled, later observations retain context.Canceled.
//
// Inputs:
//   - c: initialized test probe with immutable trigger and real cancellation function.
//
// Returns:
//   - error: nil before cancellation; the actual child context error afterward.
//
// Errors:
//   - context.Canceled is produced by the real child, not a fabricated string.
//
// Determinism:
//   - Atomic increments order observations; the triggering call cancels before reading Err.
//
// Complexity:
//   - O(1) time and O(1) space.
//
// Notes:
//   - Atomic accounting preserves safety of concurrent context method calls.
//
// AI-Hints:
//   - Do not schedule cancellation with Sleep; it would make branch coverage flaky.
func (c *pollCancelContext) Err() error {
	if c.calls.Add(1) == c.at {
		c.cancel()
	}
	return c.Context.Err()
}

// encodeBars encodes fixture price bits as high/low little-endian pairs.
// The fuzz corpus is generated in Go without external data files or embedding.
//
// Implementation:
//   - Stage 1: allocate the exact paired byte length.
//   - Stage 2: encode each high bit pattern.
//   - Stage 3: encode its matching low immediately afterward.
//
// Behavior highlights:
//   - Arbitrary input bit patterns, including non-finite seeds, are retained losslessly.
//
// Inputs:
//   - high/low: equal-length bounded fixture channels.
//
// Returns:
//   - []byte: fresh binary seed preserving NaN payloads and zero signs.
//
// Errors:
//   - None under the equal-length fixture precondition.
//
// Determinism:
//   - Rows are encoded ascending, with high before low in little-endian byte order.
//
// Complexity:
//   - O(n) time and O(n) space.
//
// Notes:
//   - This is a test codec, not a public serialization format.
//
// AI-Hints:
//   - Do not stringify floats or round seed values before fuzzing.
func encodeBars(high, low []float64) []byte {
	data := make([]byte, pricePairBytes*len(high))
	for i := range high {
		binary.LittleEndian.PutUint64(data[pricePairBytes*i:], math.Float64bits(high[i]))
		binary.LittleEndian.PutUint64(data[pricePairBytes*i+priceFloatBytes:], math.Float64bits(low[i]))
	}
	return data
}

// decodeBars decodes at most maxFuzzBars complete price pairs.
// Trailing bytes are deliberately ignored while arbitrary bit patterns remain unmodified.
//
// Implementation:
//   - Stage 1: count complete pairs and cap oracle work.
//   - Stage 2: allocate equal-length result channels.
//   - Stage 3: decode each high and low without finite normalization.
//
// Behavior highlights:
//   - Only the number of records is bounded; price bits are never repaired.
//
// Inputs:
//   - data: arbitrary fuzz bytes, including empty or partial records.
//
// Returns:
//   - Two fresh possibly non-finite channels for public admission tests.
//
// Errors:
//   - None; the public API classifies the resulting domain.
//
// Determinism:
//   - Complete records are decoded ascending; the same trailing bytes are always ignored.
//
// Complexity:
//   - O(min(len(data)/pricePairBytes,maxFuzzBars)) time and space.
//
// Notes:
//   - The pair-count bound prevents an untrusted fuzz input from creating quadratic oracle explosions.
//
// AI-Hints:
//   - Do not sanitize decoded NaN/Inf before the public validation call.
func decodeBars(data []byte) ([]float64, []float64) {
	n := len(data) / pricePairBytes
	if n > maxFuzzBars {
		n = maxFuzzBars
	}
	high, low := make([]float64, n), make([]float64, n)
	for i := range high {
		high[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[pricePairBytes*i:]))
		low[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[pricePairBytes*i+priceFloatBytes:]))
	}
	return high, low
}

// expectedInputError independently classifies non-context public input failures.
// It uses separate finite and geometry passes, not the production deferred-error implementation.
//
// Implementation:
//   - Stage 1: reject a nonpositive period and unequal channel lengths.
//   - Stage 2: scan all observations for non-finite values.
//   - Stage 3: then scan candle geometry.
//
// Behavior highlights:
//   - Two independent passes enforce global non-finite precedence without deferred-error state.
//
// Inputs:
//   - high/low/period: fuzz arguments; context is independently known active.
//
// Returns:
//   - error: expected sentinel or joined shape/nil categories, otherwise nil.
//
// Errors:
//   - Only declared input categories; this bounded helper does not simulate resource exhaustion.
//
// Determinism:
//   - Options and shape precede a complete finite scan, then an ascending geometry scan.
//
// Complexity:
//   - O(n) time and O(1) auxiliary space.
//
// Notes:
//   - The separate passes intentionally differ from the production one-pass validator.
//
// AI-Hints:
//   - Do not call production validators or compare error text to generate expectations.
func expectedInputError(high, low []float64, period int) error {
	if period < 1 {
		return donchian.ErrInvalidOption
	}
	if len(high) != len(low) {
		if high == nil || low == nil {
			return errors.Join(donchian.ErrShape, donchian.ErrNilInput)
		}
		return donchian.ErrShape
	}
	for i := range high {
		if math.IsNaN(high[i]) || math.IsNaN(low[i]) || math.IsInf(high[i], 0) || math.IsInf(low[i], 0) {
			return donchian.ErrNonFinite
		}
	}
	for i := range high {
		if high[i] < low[i] {
			return donchian.ErrBar
		}
	}
	return nil
}

// exactOracle recomputes each complete window by direct scanning.
// Exact rational midpoint rounding is independent of the production deque and arithmetic.
//
// Implementation:
//   - Stage 1: allocate owning zeroed result channels.
//   - Stage 2: derive each window directly from period and mode.
//   - Stage 3: scan every eligible index and round the exact boundary mean once.
//
// Behavior highlights:
//   - Each output window is reconstructed independently, with no retained deque state.
//
// Inputs:
//   - high/low/opts: already admitted small test data with Period >= 1.
//
// Returns:
//   - *donchian.Result: independent mathematical expectation.
//
// Errors:
//   - None; invalid public inputs are tested before oracle invocation.
//
// Determinism:
//   - Window rows are scanned ascending; >= and <= retain the last equal extremum.
//
// Complexity:
//   - O(n*P) time for ready windows and O(n) result space.
//
// Notes:
//   - This expensive path exists only in external tests.
//
// AI-Hints:
//   - Never optimize this oracle into the production monotonic-deque implementation.
func exactOracle(high, low []float64, opts donchian.Options) *donchian.Result {
	n := len(high)
	result := &donchian.Result{
		Upper: make([]float64, n), Middle: make([]float64, n),
		Lower: make([]float64, n), Valid: make([]bool, n),
	}
	for i := range high {
		end := i + 1
		if opts.ExcludeCurrent {
			end = i
		}
		if end < opts.Period {
			continue
		}
		start := end - opts.Period
		upper, lower := high[start], low[start]
		for j := start + 1; j < end; j++ {
			if high[j] >= upper {
				upper = high[j]
			}
			if low[j] <= lower {
				lower = low[j]
			}
		}
		result.Upper[i], result.Lower[i] = upper, lower
		result.Middle[i] = exactMidpoint(upper, lower)
		result.Valid[i] = true
	}
	return result
}

// exactMidpoint rounds the exact rational mean of two finite binary64 values.
// It supplies a mathematical expectation without calling the production midpoint.
//
// Implementation:
//   - Stage 1: convert both finite floats to exact rational values.
//   - Stage 2: add and divide by the integer two exactly.
//   - Stage 3: perform one final rational-to-binary64 conversion.
//
// Behavior highlights:
//   - An overflowing floating sum is never formed, even for two maximal finite bounds.
//
// Inputs:
//   - upper/lower: finite binary64 operands.
//
// Returns:
//   - float64: nearest representable mean with ties to even.
//
// Errors:
//   - None on the finite-input domain.
//
// Determinism:
//   - Exact rational addition and division precede the sole nearest-even conversion.
//
// Complexity:
//   - Bounded binary64-domain rational work; test allocations are allowed.
//
// Notes:
//   - Exact rational zero converts to positive zero; tiny negative means retain their rounded sign.
//
// AI-Hints:
//   - Do not pre-round a floating sum before constructing the rational expectation.
func exactMidpoint(upper, lower float64) float64 {
	var a, b, sum, two big.Rat
	a.SetFloat64(upper)
	b.SetFloat64(lower)
	sum.Add(&a, &b)
	two.SetInt64(2)
	sum.Quo(&sum, &two)
	value, _ := sum.Float64()
	return value
}

// exampleCandle retains the dated OHLC record used by the example applications.
// Only high and low are passed to Donchian; timing and trade interpretation remain outside it.
//
// Implementation:
//   - Stage 1: instantiate the frozen vendor records.
//   - Stage 2: candleChannels detaches the two numerical inputs in original order.
//   - Stage 3: the caller consumes dates, closes, and later opens alongside the result.
//
// Behavior highlights:
//   - No generated calendar or inferred missing observation is stored.
//
// Inputs:
//   - Vendor-supplied date and four prices in consistent supplied units.
//
// Returns:
//   - A value record; fixture calls return fresh slices of these records.
//
// Errors:
//   - Fixture consistency is checked by TestDonchian_MarketFixtureIntegrity.
//
// Determinism:
//   - Record order is explicitly oldest-first; no implicit sort is performed.
//
// Complexity:
//   - O(1) storage per record, O(n) for the fixture slice.
//
// Notes:
//   - A later supplied open is an observation, not a simulated execution price.
//
// AI-Hints:
//   - Do not mix adjusted closes with differently adjusted highs and lows.
type exampleCandle struct {
	date  string  // ISO date from the vendor response, not a generated trading calendar.
	open  float64 // First supplied daily price; used only as a later observed quote.
	high  float64 // Supplied daily maximum in the same units as low and close.
	low   float64 // Supplied daily minimum; never inferred from an adjusted close.
	close float64 // Supplied daily closing observation, not an assumed executable fill.
}

// ibmDailyCandles returns a fresh frozen vendor demonstration history.
// The 58 original OHLC observations retain provenance and supplied precision.
//
// Source: Alpha Vantage TIME_SERIES_DAILY demo, symbol IBM, last refreshed 2026-11-20.
// URL: https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo
// Observed on 2026-11-26; retained interval: 2026-09-01 through 2026-11-20 inclusive.
// The vendor supplies newest-first rows; this literal stores them oldest-first.
// No rounding, interpolation, corporate-action adjustment, or missing-date filling is applied.
// This is a vendor demonstration snapshot, not an independently audited exchange feed.
// It is not a claim that 2026-11-20 is the latest available market session.
// Each call owns its slice so examples and regression tests cannot corrupt one another.
//
// Implementation:
//   - Stage 1: instantiate the oldest-first literal records.
//   - Stage 2: retain dates and all four supplied price columns.
//   - Stage 3: return a caller-owned fixture slice.
//
// Behavior highlights:
//   - Each invocation allocates independent fixture storage; tests may mutate their own copy.
//
// Inputs:
//   - None; values are pinned in the source instead of fetched during a test.
//
// Returns:
//   - []exampleCandle: independent mutable copy per call.
//
// Errors:
//   - None; consistency is checked by TestDonchian_MarketFixtureIntegrity.
//
// Determinism:
//   - The literal is oldest-first and returns identical values on every call.
//
// Complexity:
//   - O(n) time and O(n) space for n=58.
//
// Notes:
//   - The source URL and retained interval above identify the frozen vendor response.
//
// AI-Hints:
//   - Do not silently refresh a frozen example fixture or claim it is a live quote.
func ibmDailyCandles() []exampleCandle {
	return []exampleCandle{
		{"2026-09-01", 231.0000, 233.0900, 229.9000, 231.4000},
		{"2026-09-02", 230.5450, 232.8800, 229.1000, 231.7000},
		{"2026-09-03", 234.3800, 238.2900, 232.8150, 234.7100},
		{"2026-09-04", 233.3300, 236.1700, 231.6800, 234.8900},
		{"2026-09-08", 232.9500, 233.4000, 230.2000, 232.0900},
		{"2026-09-09", 232.6500, 241.5000, 230.3600, 239.9400},
		{"2026-09-10", 238.1800, 242.3100, 231.8300, 234.0200},
		{"2026-09-11", 235.7800, 243.6300, 235.0000, 243.2900},
		{"2026-09-14", 249.4100, 251.3700, 244.1201, 249.0900},
		{"2026-09-15", 246.1900, 251.8150, 246.1500, 248.3700},
		{"2026-09-16", 240.5000, 243.3399, 235.4100, 237.4900},
		{"2026-09-17", 238.5800, 243.3650, 237.3500, 237.7500},
		{"2026-09-18", 237.8500, 237.8900, 229.3800, 229.5500},
		{"2026-09-21", 231.9900, 234.7000, 228.3500, 231.9300},
		{"2026-09-22", 234.4700, 236.2000, 230.0800, 231.3800},
		{"2026-09-23", 233.4700, 237.8600, 232.5300, 232.7600},
		{"2026-09-24", 232.0000, 232.6168, 226.0100, 227.0600},
		{"2026-09-25", 226.1500, 228.9900, 225.1900, 225.5100},
		{"2026-09-28", 222.5000, 223.1356, 219.2300, 220.6700},
		{"2026-09-29", 220.8650, 223.1200, 219.9100, 219.9900},
		{"2026-09-30", 220.7000, 222.3300, 218.6000, 219.9300},
		{"2026-10-01", 230.7600, 234.2700, 224.5000, 225.6200},
		{"2026-10-02", 225.5100, 225.9150, 221.6300, 222.6400},
		{"2026-10-05", 222.3000, 222.8800, 220.3000, 221.5800},
		{"2026-10-06", 222.7100, 224.5100, 221.0000, 221.2900},
		{"2026-10-07", 221.9000, 222.4100, 218.7900, 220.5100},
		{"2026-10-08", 220.1500, 227.3050, 217.7950, 226.6100},
		{"2026-10-09", 225.8700, 228.7600, 224.0501, 227.1300},
		{"2026-10-12", 228.3100, 229.2850, 224.7700, 227.0200},
		{"2026-10-13", 228.2150, 233.5000, 227.2770, 230.7800},
		{"2026-10-14", 231.1300, 233.1400, 227.4900, 227.8800},
		{"2026-10-15", 226.3600, 234.1400, 226.0100, 234.1200},
		{"2026-10-16", 236.0500, 236.9800, 230.2500, 232.1400},
		{"2026-10-19", 230.6800, 238.3800, 229.2965, 236.3900},
		{"2026-10-20", 236.0800, 242.5250, 236.0000, 241.9200},
		{"2026-10-21", 241.9150, 251.7500, 240.6200, 251.4600},
		{"2026-10-22", 241.2450, 256.3000, 240.0000, 252.1100},
		{"2026-10-23", 249.7300, 254.7100, 247.0300, 253.5600},
		{"2026-10-26", 253.0100, 256.0000, 250.8000, 253.4400},
		{"2026-10-27", 251.9325, 253.1800, 249.1250, 250.2500},
		{"2026-10-28", 248.4900, 253.8500, 247.6700, 253.2200},
		{"2026-10-29", 249.2700, 258.3300, 247.8001, 258.2800},
		{"2026-10-30", 258.2800, 259.5900, 254.7105, 257.8100},
		{"2026-11-02", 258.1200, 259.8700, 252.3600, 252.5100},
		{"2026-11-03", 249.5600, 250.9500, 244.6100, 250.0400},
		{"2026-11-04", 255.0000, 258.5899, 253.5100, 254.3200},
		{"2026-11-05", 255.9200, 256.1750, 245.7200, 247.1100},
		{"2026-11-06", 242.9200, 246.3400, 240.3500, 245.0000},
		{"2026-11-09", 246.0000, 248.8100, 245.1957, 246.1100},
		{"2026-11-10", 244.3800, 246.0000, 242.7000, 244.5300},
		{"2026-11-11", 244.7200, 249.3100, 242.8800, 248.2200},
		{"2026-11-12", 248.3000, 248.7700, 244.3700, 244.6700},
		{"2026-11-13", 246.6300, 247.8500, 244.8300, 247.1300},
		{"2026-11-16", 246.9500, 247.0700, 241.3400, 242.5100},
		{"2026-11-17", 239.5300, 240.9985, 230.6400, 231.0900},
		{"2026-11-18", 230.4500, 232.8450, 228.4000, 228.5300},
		{"2026-11-19", 229.2200, 232.7700, 222.7437, 223.8800},
		{"2026-11-20", 224.5950, 233.0000, 223.3900, 224.4400},
	}
}

// candleChannels extracts detached high and low channels from dated candles.
// Dates, opens, and closes remain application metadata rather than numerical-package inputs.
//
// candleChannels copies the two input columns without retaining the candle slice.
// Caller-owned date/open/close metadata stays outside the numerical package.
//
// Implementation:
//   - Stage 1: allocate two length-matched channels.
//   - Stage 2: copy each original high and low in order.
//   - Stage 3: return the detached price slices.
//
// Behavior highlights:
//   - The extracted channels do not share their backing storage with fixture records.
//
// Inputs:
//   - candles: read-only ordered example observations.
//
// Returns:
//   - Two fresh slices with the same positional alignment.
//
// Errors:
//   - None; the public API performs numerical admission afterward.
//
// Determinism:
//   - Copies high and low from each supplied record in unchanged index order.
//
// Complexity:
//   - O(n) time and O(n) space.
//
// Notes:
//   - No calendar normalization, resampling, or implicit sorting is applied.
//
// AI-Hints:
//   - Do not accidentally combine an adjusted close with differently adjusted highs and lows.
func candleChannels(candles []exampleCandle) ([]float64, []float64) {
	high, low := make([]float64, len(candles)), make([]float64, len(candles))
	for i, candle := range candles {
		high[i], low[i] = candle.high, candle.low
	}
	return high, low
}

// assertErrorCategories compares every supported public failure category without string matching.
// A partial category comparison could miss a lost or newly introduced joined sentinel.
//
// Implementation:
//   - Stage 1: check error presence.
//   - Stage 2: visit the fixed sentinel/context classification list.
//   - Stage 3: compare each errors.Is result.
//
// Behavior highlights:
//   - Both missing expected identities and spurious joined identities fail the comparison.
//
// Inputs:
//   - t: owning harness; got/want: errors whose category sets are expected equal.
//
// Returns:
//   - No value; unequal category membership fails the harness.
//
// Errors:
//   - Classification differences are assertion failures, not production errors.
//
// Determinism:
//   - Visits the seven sentinels, context.Canceled, then context.DeadlineExceeded.
//
// Complexity:
//   - O(k*d) for k fixed categories and bounded wrapping depth d.
//
// Notes:
//   - Expected joined errors need not share object identity with returned joined errors.
//
// AI-Hints:
//   - Never compare two independently allocated errors.Join values by pointer equality.
func assertErrorCategories(t testing.TB, got, want error) {
	t.Helper()
	if (got == nil) != (want == nil) {
		t.Fatalf("error presence: got %v, want %v", got, want)
	}
	for _, category := range []error{
		donchian.ErrNilContext, donchian.ErrNilInput, donchian.ErrInvalidOption,
		donchian.ErrShape, donchian.ErrNonFinite, donchian.ErrBar, donchian.ErrOverflow,
		context.Canceled, context.DeadlineExceeded,
	} {
		if errors.Is(got, category) != errors.Is(want, category) {
			t.Fatalf("category %v: got %v, want %v", category, got, want)
		}
	}
}

// modeFacade binds an explicit public wrapper to its expected canonical window policy.
// Tests use this description to prove equivalence without accessing private kernels.
//
// Implementation:
//   - Stage 1: modeFacades selects each exported function.
//   - Stage 2: tests construct independent Options from exclude.
//   - Stage 3: compare errors, results, polling, and ownership across both paths.
//
// Behavior highlights:
//   - The descriptor contains no production hooks or mutable package-level overrides.
//
// Inputs:
//   - name: case label; exclude: expected mode; call: actual public wrapper.
//
// Returns:
//   - One test-case description, not an alternative implementation.
//
// Errors:
//   - call returns the unchanged public error contract.
//
// Determinism:
//   - modeFacades returns include before exclude; all descriptor fields remain fixed.
//
// Complexity:
//   - O(1) descriptor storage; execution cost is that of the selected public wrapper.
//
// Notes:
//   - The function-value dispatch cost belongs to the benchmark harness when used there.
//
// AI-Hints:
//   - Never point call at a test reimplementation; that would stop testing the public API.
type modeFacade struct {
	// name distinguishes subtest and benchmark output without encoding error semantics.
	name string
	// exclude fixes the Options value used by the independent canonical comparison.
	exclude bool
	// call is a real exported wrapper, not a test replacement of the numerical kernel.
	call func(context.Context, []float64, []float64, int) (*donchian.Result, error)
}

// modeFacades describes both explicit public window-mode wrappers.
// Descriptors select real exported functions rather than replacing a private kernel.
//
// Implementation:
//   - Stage 1: create the include descriptor.
//   - Stage 2: create the exclude descriptor.
//   - Stage 3: return a fresh ordered descriptor slice.
//
// Behavior highlights:
//   - The same descriptor drives wrapper execution and independently constructed Options.
//
// Inputs:
//   - None; the two policies are fixed by the public API.
//
// Returns:
//   - []modeFacade: include followed by exclude.
//
// Errors:
//   - None.
//
// Determinism:
//   - Include-current precedes exclude-current in the returned descriptor slice.
//
// Complexity:
//   - O(1) time and bounded descriptor storage.
//
// Notes:
//   - No mutable production globals or test-only exported symbols are involved.
//
// AI-Hints:
//   - Use the exclude flag only to build independent canonical expectations.
func modeFacades() []modeFacade {
	return []modeFacade{
		{"include", false, donchian.ComputeIncludeCurrent},
		{"exclude", true, donchian.ComputeExcludeCurrent},
	}
}

// syntheticSpreadBars builds sampled signed spreads from simultaneous two-leg observations.
// Subtraction occurs before aggregation, preventing fictitious cross-leg extrema.
//
// These are engineered risk-monitor inputs in USD, not a recorded vendor history.
// Each row contains three simultaneous A/B observations; subtraction precedes min/max.
//
// Implementation:
//   - Stage 1: load engineered synchronized quarter-dollar marks.
//   - Stage 2: subtract each simultaneous pair and update actual spread extrema.
//   - Stage 3: retain the last sampled spread as the bar close.
//
// Behavior highlights:
//   - Signed extrema are seeded from the first actual pair, never from artificial zero.
//
// Inputs:
//   - None; fourteen fixed illustrative bars each contain three simultaneous pairs.
//
// Returns:
//   - Fresh high, low, and close slices describing sampled spreads.
//
// Errors:
//   - None; the exact bounded marks satisfy finite candle geometry.
//
// Determinism:
//   - Processes bars and their three simultaneous pairs in their literal order.
//
// Complexity:
//   - O(n*k) time for k=3 and O(n) space.
//
// Notes:
//   - These are marked synthetic examples, not exchange records or bid/ask executions.
//
// AI-Hints:
//   - max(A)-max(B) is generally not max(A-B); preserve pair synchronization.
func syntheticSpreadBars() (high, low, closes []float64) {
	const observationsPerBar = 3
	// Quarter-dollar marks make every spread and midpoint exact in binary64.
	// Both legs move inside each bar; subtracting their separate highs gives the wrong range.
	marks := [][observationsPerBar][2]float64{
		{{100, 102}, {99.75, 101.5}, {100.75, 103}},
		{{100.5, 102.5}, {100.25, 102}, {101.25, 103.5}},
		{{101, 103}, {100.75, 102.5}, {101.75, 104}},
		{{101.5, 103.5}, {101.25, 103}, {102.25, 104.5}},
		{{102, 104}, {101.75, 103.5}, {102.75, 105}},
		{{102.5, 104.5}, {102.25, 104}, {103.25, 105.5}},
		{{103, 105}, {102.75, 104.5}, {103.75, 106}},
		{{103.5, 105.5}, {103.25, 105}, {104.25, 106.5}},
		{{104, 106}, {103.75, 105.5}, {104.75, 107}},
		{{104.5, 106.5}, {104.25, 106}, {105.25, 107.5}},
		{{105, 107}, {104.75, 106.5}, {105.75, 108}},
		{{105.5, 107.5}, {105.25, 107}, {106.25, 108.5}},
		{{107.75, 109}, {107.5, 109}, {108, 109}},
		{{108.5, 110}, {108.25, 110}, {108.75, 110}},
	}
	high, low, closes = make([]float64, len(marks)), make([]float64, len(marks)), make([]float64, len(marks))
	for i, samples := range marks {
		// Seed from a real sample; zero is not a valid initial extremum for signed prices.
		high[i], low[i] = samples[0][0]-samples[0][1], samples[0][0]-samples[0][1]
		for _, simultaneous := range samples {
			spread := simultaneous[0] - simultaneous[1]
			if spread > high[i] {
				high[i] = spread
			}
			if spread < low[i] {
				low[i] = spread
			}
			// Input sample order defines the closing mark of the engineered bar.
			closes[i] = spread
		}
	}
	return high, low, closes
}

// burstCycleBars builds repeated dominating-extremum bursts for queue benchmarks.
// Long candidate lifetimes alternate with large single-insertion removal costs.
//
// Implementation:
//   - Stage 1: allocate signed candle channels.
//   - Stage 2: generate monotonic runs of period observations.
//   - Stage 3: insert a dominating value at each cycle boundary.
//
// Behavior highlights:
//   - Repeated runs measure sustained amortized behavior instead of one isolated tail event.
//
// Inputs:
//   - n: bounded length; period: positive bounded run size; upperBurst: queue direction.
//
// Returns:
//   - Fresh valid high/low channels for repeated amortized-work measurements.
//
// Errors:
//   - None for the benchmark-selected bounds.
//
// Determinism:
//   - Index modulo period+1 fixes run length and every dominating insertion.
//
// Complexity:
//   - O(n) generation time and O(n) storage outside benchmark timing.
//
// Notes:
//   - period+1 is safe for the fixed benchmark period; this is not a public arbitrary-int constructor.
//
// AI-Hints:
//   - Do not include generation in the timed call or confuse one-burst cost with per-bar worst-case O(1).
func burstCycleBars(n, period int, upperBurst bool) ([]float64, []float64) {
	high, low := make([]float64, n), make([]float64, n)
	for i := range high {
		position := i % (period + 1)
		candidate := float64(period - position)
		if position == period {
			candidate = float64(period + 1)
		}
		if upperBurst {
			high[i] = candidate
		} else {
			low[i] = -candidate
		}
	}
	return high, low
}
