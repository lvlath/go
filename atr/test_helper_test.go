// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr_test uses only exported ATR boundaries and independent oracles.
// AI-Hints: never delegate oracle arithmetic to a production helper or compare error text.
package atr_test

import (
	"context"
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/atr"
)

// candle is a numeric HLC fixture independent of production storage.
// Positional literals use high, low, then close consistently in the test suite.
type candle struct {
	high  float64 // Upper interval endpoint; no positivity assumption.
	low   float64 // Lower interval endpoint in the same units.
	close float64 // Last price within the interval for valid fixtures.
}

// nilable is the structural result-presence convention, without an extra core import.
// A typed nil pointer must remain safe when invoked through this interface.
type nilable interface{ IsNil() bool }

var (
	benchmarkResult *atr.Result
	benchmarkValue  atr.Value
)

// mustErrorIs classifies an expected error using the public sentinel chain.
// Work is bounded by the error chain length; package sentinels have constant depth.
//
// Implementation:
//   - Stage 1: got and want errors; testing.TB owner.
//   - Stage 2: report mismatch without comparing diagnostic text.
//   - Stage 3: No returned value; the test continues exactly when errors.Is(got,want).
//
// Behavior highlights:
//   - Work is bounded by the error chain length; package sentinels have constant depth.
//
// Inputs:
//   - got and want errors; testing.TB owner.
//
// Returns:
//   - No returned value; the test continues exactly when errors.Is(got,want).
//
// Errors:
//   - A mismatched error category fails the owning test; text is only diagnostic.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(1) for these package sentinels.
//
// Notes:
//   - Work is bounded by the error chain length; package sentinels have constant depth.
//
// AI-Hints:
//   - Do not replace errors.Is with string matching.
func mustErrorIs(t testing.TB, got, want error) {
	t.Helper()
	if !errors.Is(got, want) {
		t.Fatalf("error: got %v; want category %v", got, want)
	}
}

// mustFloatBits compares two binary64 bit patterns including zero signs.
// The caller decides whether a zero sign is contractual; this helper never erases it.
//
// Implementation:
//   - Stage 1: got and want represented floats.
//   - Stage 2: report both numerical values and hexadecimal bits.
//   - Stage 3: No returned value; equal binary64 bits are required.
//
// Behavior highlights:
//   - The caller decides whether a zero sign is contractual; this helper never erases it.
//
// Inputs:
//   - got and want represented floats.
//
// Returns:
//   - No returned value; equal binary64 bits are required.
//
// Errors:
//   - Mismatched bits fail with numeric and hexadecimal diagnostics.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(1).
//
// Notes:
//   - The caller decides whether a zero sign is contractual; this helper never erases it.
//
// AI-Hints:
//   - Approximate equality would hide the rounding contract.
func mustFloatBits(t testing.TB, got, want float64) {
	t.Helper()
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("float bits: got %.17g [%016x]; want %.17g [%016x]", got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

// mustValue compares both scalar outputs and the independent readiness flag.
// A warming Value still carries a measured TR.
//
// Implementation:
//   - Stage 1: detached atr.Value values.
//   - Stage 2: check TR bits, ATR bits, then Valid.
//   - Stage 3: No returned value; TR bits, ATR bits, and Valid must all match.
//
// Behavior highlights:
//   - A warming Value still carries a measured TR.
//
// Inputs:
//   - detached atr.Value values.
//
// Returns:
//   - No returned value; TR bits, ATR bits, and Valid must all match.
//
// Errors:
//   - Any scalar or mask mismatch fails the owning test.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(1).
//
// Notes:
//   - A warming Value still carries a measured TR.
//
// AI-Hints:
//   - Do not skip TR when Valid is false.
func mustValue(t testing.TB, got, want atr.Value) {
	t.Helper()
	mustFloatBits(t, got.TR, want.TR)
	mustFloatBits(t, got.ATR, want.ATR)
	if got.Valid != want.Valid {
		t.Fatalf("readiness: got %v; want %v", got.Valid, want.Valid)
	}
}

// mustSnapshot compares all checkpoint fields without reflection.
// Both arguments are detached values; no serialization is performed here.
//
// Implementation:
//   - Stage 1: two complete atr.Snapshot values.
//   - Stage 2: check schema/options/count/seed, then both float bit patterns.
//   - Stage 3: No returned value; schema, options, count, seed text, and scalar bits must agree.
//
// Behavior highlights:
//   - Both arguments are detached values; no serialization is performed here.
//
// Inputs:
//   - two complete atr.Snapshot values.
//
// Returns:
//   - No returned value; schema, options, count, seed text, and scalar bits must agree.
//
// Errors:
//   - A metadata, decimal, or float-bit mismatch fails the owning test.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(D), D<=660 for valid snapshots.
//
// Notes:
//   - Both arguments are detached values; no serialization is performed here.
//
// AI-Hints:
//   - Comparing only Current misses corrupted continuation.
func mustSnapshot(t testing.TB, got, want atr.Snapshot) {
	t.Helper()
	if got.Version != want.Version || got.Algorithm != want.Algorithm ||
		got.Options != want.Options || got.Count != want.Count || got.SeedSum != want.SeedSum {
		t.Fatalf("snapshot metadata/seed mismatch:\ngot  %+v\nwant %+v", got, want)
	}
	mustFloatBits(t, got.PreviousClose, want.PreviousClose)
	mustFloatBits(t, got.Current, want.Current)
}

// mustResult compares complete aligned result artifacts bit for bit.
// The helper reads arrays without mutation and reports the earliest differing index.
//
// Implementation:
//   - Stage 1: two nonnil owning results with aligned channels.
//   - Stage 2: check presence and lengths before indexed payload comparisons.
//   - Stage 3: No returned value; all three channels must match at every index.
//
// Behavior highlights:
//   - The helper reads arrays without mutation and reports the earliest differing index.
//
// Inputs:
//   - two nonnil owning results with aligned channels.
//
// Returns:
//   - No returned value; all three channels must match at every index.
//
// Errors:
//   - Nil results, shape differences, payload or readiness mismatches fail the test.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(n).
//
// Notes:
//   - The helper reads arrays without mutation and reports the earliest differing index.
//
// AI-Hints:
//   - Keep index-specific diagnostics; equal summaries do not imply equal histories.
func mustResult(t testing.TB, got, want *atr.Result) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatal("result unexpectedly nil")
	}
	if len(got.TR) != len(want.TR) || len(got.ATR) != len(want.ATR) || len(got.Valid) != len(want.Valid) {
		t.Fatalf("result lengths: got %d/%d/%d; want %d/%d/%d", len(got.TR), len(got.ATR), len(got.Valid), len(want.TR), len(want.ATR), len(want.Valid))
	}
	for i := range got.TR {
		if math.Float64bits(got.TR[i]) != math.Float64bits(want.TR[i]) ||
			math.Float64bits(got.ATR[i]) != math.Float64bits(want.ATR[i]) || got.Valid[i] != want.Valid[i] {
			t.Fatalf("index %d: got TR=%g [%x] ATR=%g [%x] valid=%v; want TR=%g [%x] ATR=%g [%x] valid=%v", i,
				got.TR[i], math.Float64bits(got.TR[i]), got.ATR[i], math.Float64bits(got.ATR[i]), got.Valid[i],
				want.TR[i], math.Float64bits(want.TR[i]), want.ATR[i], math.Float64bits(want.ATR[i]), want.Valid[i])
		}
	}
}

// newStream constructs a configured stream or fails the owning test.
// Use atr.New directly to test invalid construction.
//
// Implementation:
//   - Stage 1: positive period and testing.TB.
//   - Stage 2: call atr.New and reject any unexpected error.
//   - Stage 3: An independent initialized *atr.Stream with the supplied Period.
//
// Behavior highlights:
//   - Use atr.New directly to test invalid construction.
//
// Inputs:
//   - positive period and testing.TB.
//
// Returns:
//   - An independent initialized *atr.Stream with the supplied Period.
//
// Errors:
//   - Unexpected New failure stops the owning test.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(W), W=34.
//
// Notes:
//   - Use atr.New directly to test invalid construction.
//
// AI-Hints:
//   - Do not synthesize private stream fields.
func newStream(t testing.TB, period int) *atr.Stream {
	t.Helper()
	s, err := atr.New(atr.Options{Period: period})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// push accepts one known-valid fixture through the public streaming boundary.
// No snapshot or persistence work is included in this helper.
//
// Implementation:
//   - Stage 1: constructed stream and finite candle fixture.
//   - Stage 2: call Push with an active background context and check its error.
//   - Stage 3: The complete atr.Value returned by a successful public Push.
//
// Behavior highlights:
//   - No snapshot or persistence work is included in this helper.
//
// Inputs:
//   - constructed stream and finite candle fixture.
//
// Returns:
//   - The complete atr.Value returned by a successful public Push.
//
// Errors:
//   - Unexpected Push failure stops the owning test.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(W), W=34.
//
// Notes:
//   - No snapshot or persistence work is included in this helper.
//
// AI-Hints:
//   - Use direct Push when the test expects failure or cancellation.
func push(t testing.TB, s *atr.Stream, bar candle) atr.Value {
	t.Helper()
	v, err := s.Push(context.Background(), bar.high, bar.low, bar.close)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// compute calculates a known-valid fixture through the canonical batch facade.
// Inputs belong to the test and remain read-only during the call.
//
// Implementation:
//   - Stage 1: equal-length HLC columns and an explicit positive period.
//   - Stage 2: call atr.Compute and reject an unexpected error.
//   - Stage 3: An owning nonnil *atr.Result from the canonical public facade.
//
// Behavior highlights:
//   - Inputs belong to the test and remain read-only during the call.
//
// Inputs:
//   - equal-length HLC columns and an explicit positive period.
//
// Returns:
//   - An owning nonnil *atr.Result from the canonical public facade.
//
// Errors:
//   - Unexpected Compute failure stops the owning test.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(n*W), Space O(n+W).
//
// Notes:
//   - Inputs belong to the test and remain read-only during the call.
//
// AI-Hints:
//   - Expected failures must call the facade directly instead.
func compute(t testing.TB, high, low, closePrices []float64, period int) *atr.Result {
	t.Helper()
	r, err := atr.Compute(context.Background(), high, low, closePrices, atr.Options{Period: period})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// marketBars builds reproducible binary-exact price intervals with moving gaps.
// Fixture generation allocates and must precede benchmark ResetTimer.
//
// Implementation:
//   - Stage 1: n nonnegative observations.
//   - Stage 2: allocate owning columns and generate fractional prices in ascending index order.
//   - Stage 3: Three fresh equal-length HLC slices; every close lies in its interval.
//
// Behavior highlights:
//   - Fixture generation allocates and must precede benchmark ResetTimer.
//
// Inputs:
//   - n nonnegative observations.
//
// Returns:
//   - Three fresh equal-length HLC slices; every close lies in its interval.
//
// Errors:
//   - No returned error; tests supply nonnegative bounded n.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(n) time/space.
//
// Notes:
//   - Fixture generation allocates and must precede benchmark ResetTimer.
//
// AI-Hints:
//   - The modular arithmetic is a deterministic workload, not financial data.
func marketBars(n int) (high, low, closed []float64) {
	high, low, closed = make([]float64, n), make([]float64, n), make([]float64, n)
	for i := range closed {
		closed[i] = 100 + float64((i*17)%37-18)/4
		high[i] = closed[i] + float64(1+i%7)/8
		low[i] = closed[i] - float64(1+(i*3)%5)/8
	}
	return high, low, closed
}

// rationalTR independently evaluates all three exact true-range candidates.
// The oracle uses arbitrary precision only in tests, with three independently formed candidates.
//
// Implementation:
//   - Stage 1: finite valid candle and optional prior represented close.
//   - Stage 2: compute rational differences, choose the largest, then round once.
//   - Stage 3: The nearest-even float64 of the exact maximum; overflow may produce +Inf for the oracle caller to classify.
//
// Behavior highlights:
//   - The oracle uses arbitrary precision only in tests, with three independently formed candidates.
//
// Inputs:
//   - finite valid candle and optional prior represented close.
//
// Returns:
//   - The nearest-even float64 of the exact maximum; overflow may produce +Inf for the oracle caller to classify.
//
// Errors:
//   - No sentinel; callers admit finite valid candle geometry before rational construction.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - bounded binary64 rational work; test allocations allowed.
//
// Notes:
//   - The oracle uses arbitrary precision only in tests, with three independently formed candidates.
//
// AI-Hints:
//   - Do not replace the three-candidate oracle by the production envelope reduction.
func rationalTR(bar candle, previous float64, hasPrevious bool) float64 {
	high, low := new(big.Rat).SetFloat64(bar.high), new(big.Rat).SetFloat64(bar.low)
	maximum := new(big.Rat).Sub(high, low)
	if hasPrevious {
		prev := new(big.Rat).SetFloat64(previous)
		for _, endpoint := range []*big.Rat{high, low} {
			difference := new(big.Rat).Sub(endpoint, prev)
			difference.Abs(difference)
			if difference.Cmp(maximum) > 0 {
				maximum.Set(difference)
			}
		}
	}
	rounded, _ := maximum.Float64()
	return rounded
}

// rationalStep rounds one exact rational Wilder affine update.
// The represented previous ATR, not an unrounded entire-history value, is the recurrence operand.
//
// Implementation:
//   - Stage 1: finite previous ATR and TR; positive integer period.
//   - Stage 2: form exact integer-weighted rational numerator, divide, then round once.
//   - Stage 3: One nearest-even rounded finite nonnegative affine result.
//
// Behavior highlights:
//   - The represented previous ATR, not an unrounded entire-history value, is the recurrence operand.
//
// Inputs:
//   - finite previous ATR and TR; positive integer period.
//
// Returns:
//   - One nearest-even rounded finite nonnegative affine result.
//
// Errors:
//   - No returned error under the finite-nonnegative and positive-period preconditions.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - bounded rational work in the input bit width.
//
// Notes:
//   - The represented previous ATR, not an unrounded entire-history value, is the recurrence operand.
//
// AI-Hints:
//   - Never pre-round alpha or keep unrounded previous ATR across steps.
func rationalStep(previous, tr float64, period int) float64 {
	numerator := new(big.Rat).Mul(new(big.Rat).SetFloat64(previous), big.NewRat(int64(period-1), 1))
	numerator.Add(numerator, new(big.Rat).SetFloat64(tr))
	numerator.Quo(numerator, big.NewRat(int64(period), 1))
	rounded, _ := numerator.Float64()
	return rounded
}

// rationalHistory reconstructs every expected output with independent rational arithmetic.
// Rational seed storage is discarded only after its exact mean is rounded.
//
// Implementation:
//   - Stage 1: valid finite ordered HLC columns and a positive period.
//   - Stage 2: round TR, accumulate exact seed, and round each later affine step.
//   - Stage 3: A fresh aligned *atr.Result with independent oracle values and readiness.
//
// Behavior highlights:
//   - Rational seed storage is discarded only after its exact mean is rounded.
//
// Inputs:
//   - valid finite ordered HLC columns and a positive period.
//
// Returns:
//   - A fresh aligned *atr.Result with independent oracle values and readiness.
//
// Errors:
//   - No returned error; the caller supplies admitted finite candles and a positive period.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(n) bounded-rational operations; O(n) output.
//
// Notes:
//   - Rational seed storage is discarded only after its exact mean is rounded.
//
// AI-Hints:
//   - Seed averages already-rounded TR, not unrounded price differences.
func rationalHistory(high, low, closePrices []float64, period int) *atr.Result {
	n := len(high)
	r := &atr.Result{TR: make([]float64, n), ATR: make([]float64, n), Valid: make([]bool, n)}
	seed := new(big.Rat)
	var previous, current float64
	for i := range high {
		tr := rationalTR(candle{high[i], low[i], closePrices[i]}, previous, i > 0)
		r.TR[i] = tr
		switch {
		case i < period:
			seed.Add(seed, new(big.Rat).SetFloat64(tr))
			if i == period-1 {
				exact := new(big.Rat).Quo(seed, big.NewRat(int64(period), 1))
				current, _ = exact.Float64()
				r.Valid[i] = true
			}
		default:
			current = rationalStep(current, tr, period)
			r.Valid[i] = true
		}
		r.ATR[i] = current
		previous = closePrices[i]
	}
	return r
}

// binary64Units converts a represented finite value to exact subnormal units.
// Neither the input nor any production accumulator is accessed or modified.
//
// Implementation:
//   - Stage 1: finite nonnegative binary64 value.
//   - Stage 2: multiply by exact 2^1074 and return the integer numerator.
//   - Stage 3: A fresh *big.Int containing the exact integer number of 2^-1074 units.
//
// Behavior highlights:
//   - Neither the input nor any production accumulator is accessed or modified.
//
// Inputs:
//   - finite nonnegative binary64 value.
//
// Returns:
//   - A fresh *big.Int containing the exact integer number of 2^-1074 units.
//
// Errors:
//   - No returned error; the supplied float must be finite.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - bounded big.Int storage for binary64.
//
// Notes:
//   - Neither the input nor any production accumulator is accessed or modified.
//
// AI-Hints:
//   - This oracle conversion must not call the production limb codec.
func binary64Units(x float64) *big.Int {
	scaled := new(big.Rat).Mul(new(big.Rat).SetFloat64(x), new(big.Rat).SetInt(new(big.Int).Lsh(big.NewInt(1), 1074)))
	return new(big.Int).Set(scaled.Num())
}

// cancelAfterContext is a valid context whose cancellation occurs at a logical
// Err observation, eliminating timing races from publication-boundary tests.
type cancelAfterContext struct {
	context.Context
	cancel    context.CancelFunc
	calls, at int
}

// Err cancels at a specified logical context observation.
// This test context is single-owner; it is not used to claim concurrent context safety.
//
// Implementation:
//   - Stage 1: a single-owner cancelAfterContext with its real parent context.
//   - Stage 2: increment the call counter, cancel at the chosen boundary, return parent Err.
//   - Stage 3: The real parent context error, nil until the configured logical cancellation.
//
// Behavior highlights:
//   - This test context is single-owner; it is not used to claim concurrent context safety.
//
// Inputs:
//   - a single-owner cancelAfterContext with its real parent context.
//
// Returns:
//   - The real parent context error, nil until the configured logical cancellation.
//
// Errors:
//   - Returns the parent context cancellation category unchanged.
//
// Determinism:
//   - Cancellation occurs at exactly the configured Err observation and remains permanent.
//
// Complexity:
//   - O(1).
//
// Notes:
//   - This test context is single-owner; it is not used to claim concurrent context safety.
//
// AI-Hints:
//   - This context is single-owner test instrumentation, not a concurrent service context.
func (c *cancelAfterContext) Err() error {
	c.calls++
	if c.calls == c.at {
		c.cancel()
	}
	return c.Context.Err()
}

// logicalCancellation registers deterministic context cancellation and cleanup.
// The parent context is genuinely canceled; no fake error replaces Done/Err consistency.
//
// Implementation:
//   - Stage 1: positive observation number and the owning testing.TB.
//   - Stage 2: construct a real cancelable context and register its cancel function.
//   - Stage 3: A fresh single-owner *cancelAfterContext with cleanup registered on t.
//
// Behavior highlights:
//   - The parent context is genuinely canceled; no fake error replaces Done/Err consistency.
//
// Inputs:
//   - positive observation number and the owning testing.TB.
//
// Returns:
//   - A fresh single-owner *cancelAfterContext with cleanup registered on t.
//
// Errors:
//   - No returned error; at is a positive logical observation index.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(1).
//
// Notes:
//   - The parent context is genuinely canceled; no fake error replaces Done/Err consistency.
//
// AI-Hints:
//   - Use no clock delays to target publication boundaries.
func logicalCancellation(t testing.TB, at int) *cancelAfterContext {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	return &cancelAfterContext{Context: ctx, cancel: cancel, at: at}
}

// matureState constructs an admitted mature checkpoint through public Restore.
// The synthesized checkpoint is not claimed to have a uniquely realizable source history.
//
// Implementation:
//   - Stage 1: positive period and a finite nonnegative Current.
//   - Stage 2: obtain schema from New, set mature fields, then validate with Restore.
//   - Stage 3: A constructed *atr.Stream with a structurally admitted mature Current.
//
// Behavior highlights:
//   - The synthesized checkpoint is not claimed to have a uniquely realizable source history.
//
// Inputs:
//   - positive period and a finite nonnegative Current.
//
// Returns:
//   - A constructed *atr.Stream with a structurally admitted mature Current.
//
// Errors:
//   - Construction or Restore failure stops the owning test.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(W), W=34.
//
// Notes:
//   - The synthesized checkpoint is not claimed to have a uniquely realizable source history.
//
// AI-Hints:
//   - The checkpoint is structural test input, not claimed historical provenance.
func matureState(t testing.TB, period int, current float64) *atr.Stream {
	t.Helper()
	s := newStream(t, period)
	snapshot := s.Snapshot()
	snapshot.Count, snapshot.Current = uint64(period), current
	if err := s.Restore(snapshot); err != nil {
		t.Fatal(err)
	}
	return s
}

// rationalSnapshotStep predicts continuation from an admissible checkpoint independently.
// State is a copied snapshot and remains unchanged.
//
// Implementation:
//   - Stage 1: valid snapshot and a finite geometrically valid candle.
//   - Stage 2: check counter/TR overflow, then use rational seed or mature step.
//   - Stage 3: Expected atr.Value and nil, or Value{} with the required overflow category.
//
// Behavior highlights:
//   - State is a copied snapshot and remains unchanged.
//
// Inputs:
//   - valid snapshot and a finite geometrically valid candle.
//
// Returns:
//   - Expected atr.Value and nil, or Value{} with the required overflow category.
//
// Errors:
//   - ErrOverflow for count/range overflow; malformed admitted seed fails the owning test.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - bounded rational work and test-only allocations.
//
// Notes:
//   - State is a copied snapshot and remains unchanged.
//
// AI-Hints:
//   - Compare continuation after Restore, not just serialization equality.
func rationalSnapshotStep(t testing.TB, state atr.Snapshot, bar candle) (atr.Value, error) {
	t.Helper()
	if state.Count == ^uint64(0) {
		return atr.Value{}, atr.ErrOverflow
	}
	tr := rationalTR(bar, state.PreviousClose, state.Count != 0)
	if math.IsInf(tr, 0) {
		return atr.Value{}, atr.ErrOverflow
	}
	value := atr.Value{TR: tr}
	period := uint64(state.Options.Period)
	if state.Count >= period {
		value.ATR = rationalStep(state.Current, tr, state.Options.Period)
		value.Valid = true
	} else if state.Count+1 == period {
		sum, ok := new(big.Int).SetString(state.SeedSum, 10)
		if !ok {
			t.Fatal("accepted snapshot has invalid seed")
		}
		sum.Add(sum, binary64Units(tr))
		divisor := new(big.Int).Lsh(new(big.Int).SetUint64(period), 1074)
		exact := new(big.Rat).SetFrac(sum, divisor)
		value.ATR, _ = exact.Float64()
		value.Valid = true
	}
	return value, nil
}

// snapshotValidOracle checks structural snapshot admissibility with independent big.Int arithmetic.
// Input length is bounded before big.Int parsing; no history authentication is attempted.
//
// Implementation:
//   - Stage 1: candidate snapshot and the receiving Options.
//   - Stage 2: check metadata, scalar phase, canonical decimal, and exact magnitude bound.
//   - Stage 3: true exactly for the independently checked structural domain; false otherwise.
//
// Behavior highlights:
//   - Input length is bounded before big.Int parsing; no history authentication is attempted.
//
// Inputs:
//   - candidate snapshot and the receiving Options.
//
// Returns:
//   - true exactly for the independently checked structural domain; false otherwise.
//
// Errors:
//   - No error return; rejection is expressed as false.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(D*W) bounded work after the length guard.
//
// Notes:
//   - Input length is bounded before big.Int parsing; no history authentication is attempted.
//
// AI-Hints:
//   - Do not treat a syntactically valid altered snapshot as authenticated history.
func snapshotValidOracle(state atr.Snapshot, options atr.Options) bool {
	if state.Version != 1 || state.Algorithm != "atr/wilder-rn/v1" || state.Options != options {
		return false
	}
	if math.IsNaN(state.PreviousClose) || math.IsInf(state.PreviousClose, 0) || math.IsNaN(state.Current) || math.IsInf(state.Current, 0) || state.Current < 0 {
		return false
	}
	if state.Count == 0 && state.PreviousClose != 0 {
		return false
	}
	if state.Count >= uint64(options.Period) {
		return state.SeedSum == "0"
	}
	if math.Float64bits(state.Current) != 0 || len(state.SeedSum) == 0 || len(state.SeedSum) > 660 {
		return false
	}
	sum, ok := new(big.Int).SetString(state.SeedSum, 10)
	if !ok || sum.Sign() < 0 || sum.String() != state.SeedSum {
		return false
	}
	bound := new(big.Int).Mul(binary64Units(math.MaxFloat64), new(big.Int).SetUint64(state.Count))
	return sum.Cmp(bound) <= 0
}

// datedCandle keeps source identity separate from the three numeric algorithm inputs.
// The date is an ordered session label, not a claim that the library validates calendars.
type datedCandle struct {
	date   string // Fixed ISO session label; it is not a validated market calendar.
	candle        // Literal HLC prices in one consistent scenario unit.
}

// replayEnvelope couples an indicator checkpoint to the next source record.
// Persistence must atomically save the whole envelope; this carrier adds no authentication.
type replayEnvelope struct {
	Instrument string       // Instrument identity, including the feed's price convention.
	Revision   string       // Fixed dataset revision; a different revision needs replay.
	NextIndex  int          // First source record not yet accepted by the stream.
	State      atr.Snapshot // Exact indicator continuation for the accepted prefix.
}

// dailyBars builds the fixed twenty-session range and restart scenario entirely in Go.
// Each call owns its records; no encoded asset, parser, file, clock, or network is used.
//
// Implementation:
//   - Stage 1: materialize ordered session labels and typed HLC literal values.
//   - Stage 2: keep the original binary64 prices, gaps, and warmup boundary intact.
//   - Stage 3: return fresh storage so correction tests cannot alter another caller.
//
// Behavior highlights:
//   - The first fourteen records seed ATR14; later gaps exercise mature filtering.
//   - Inputs are regression observations, not expected ATR outputs or live quotations.
//
// Inputs:
//   - None; every input value is visible in the literal below.
//
// Returns:
//   - []datedCandle: twenty caller-owned records in increasing session-label order.
//
// Errors:
//   - None; there is no fallible decoding or external input in this constructor.
//   - Compute and Push still validate the supplied prices through their public APIs.
//
// Determinism:
//   - Literal order and float64 representations are fixed; no randomness is involved.
//
// Complexity:
//   - Time O(n), Space O(n) for n=20 copied records; no parsing workspace is needed.
//
// Notes:
//   - Dates describe the scenario only; no verified exchange provenance is claimed.
//   - Returning a package-level slice would permit cross-test mutation and is forbidden.
//
// AI-Hints:
//   - Use rationalHistory for expected values; never derive inputs from an ATR result.
//   - Keep test-only data here rather than adding public fixtures or runtime I/O.
func dailyBars() []datedCandle {
	// Construct independent input storage on every call, including all twenty closes.
	return []datedCandle{
		// The first bar has no previous close; only its own interval defines TR.
		{date: "2026-10-26", candle: candle{high: 256.0000, low: 250.8000, close: 253.4400}},
		{date: "2026-10-27", candle: candle{high: 253.1800, low: 249.1250, close: 250.2500}},
		{date: "2026-10-28", candle: candle{high: 253.8500, low: 247.6700, close: 253.2200}},
		{date: "2026-10-29", candle: candle{high: 258.3300, low: 247.8001, close: 258.2800}},
		{date: "2026-10-30", candle: candle{high: 259.5900, low: 254.7105, close: 257.8100}},
		// Failover resumes here after preserving five accepted warmup bars.
		{date: "2026-11-02", candle: candle{high: 259.8700, low: 252.3600, close: 252.5100}},
		{date: "2026-11-03", candle: candle{high: 250.9500, low: 244.6100, close: 250.0400}},
		{date: "2026-11-04", candle: candle{high: 258.5899, low: 253.5100, close: 254.3200}},
		{date: "2026-11-05", candle: candle{high: 256.1750, low: 245.7200, close: 247.1100}},
		{date: "2026-11-06", candle: candle{high: 246.3400, low: 240.3500, close: 245.0000}},
		{date: "2026-11-09", candle: candle{high: 248.8100, low: 245.1957, close: 246.1100}},
		{date: "2026-11-10", candle: candle{high: 246.0000, low: 242.7000, close: 244.5300}},
		{date: "2026-11-11", candle: candle{high: 249.3100, low: 242.8800, close: 248.2200}},
		// ATR14 first becomes ready after this fourteenth accepted observation.
		{date: "2026-11-12", candle: candle{high: 248.7700, low: 244.3700, close: 244.6700}},
		{date: "2026-11-13", candle: candle{high: 247.8500, low: 244.8300, close: 247.1300}},
		{date: "2026-11-16", candle: candle{high: 247.0700, low: 241.3400, close: 242.5100}},
		// The prior close is above this interval, so TR must include the downward gap.
		{date: "2026-11-17", candle: candle{high: 240.9985, low: 230.6400, close: 231.0900}},
		{date: "2026-11-18", candle: candle{high: 232.8450, low: 228.4000, close: 228.5300}},
		{date: "2026-11-19", candle: candle{high: 232.7700, low: 222.7437, close: 223.8800}},
		{date: "2026-11-20", candle: candle{high: 233.0000, low: 223.3900, close: 224.4400}},
	}
}

// dailyChannels projects dated source records into independent numeric columns.
// Zero-length input returns three nonnil empty columns.
//
// Implementation:
//   - Stage 1: ordered datedCandle slice.
//   - Stage 2: allocate three columns and preserve original row order.
//   - Stage 3: Fresh high/low/closed slices detached from the dated input.
//
// Behavior highlights:
//   - Zero-length input returns three nonnil empty columns.
//
// Inputs:
//   - ordered datedCandle slice.
//
// Returns:
//   - Fresh high/low/closed slices detached from the dated input.
//
// Errors:
//   - None; projection copies values without accepting or repairing their domain.
//   - Compute and Push remain responsible for finite-input and candle validation.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(n) time/space.
//
// Notes:
//   - Zero-length input returns three nonnil empty columns.
//
// AI-Hints:
//   - Keep source dates outside ATR; do not silently fill missing sessions.
func dailyChannels(bars []datedCandle) (high, low, closed []float64) {
	// Allocate distinct columns, including nonnil empty columns for zero records.
	high, low, closed = make([]float64, len(bars)), make([]float64, len(bars)), make([]float64, len(bars))
	// Preserve positional correspondence without consulting the session labels.
	for i, bar := range bars {
		// Copy each numeric field; changes to a returned column cannot mutate the records.
		high[i], low[i], closed[i] = bar.high, bar.low, bar.close
	}
	// Transfer ownership of all three independent columns to the caller.
	return high, low, closed
}

// sameValueBits compares scalar payload bits and readiness for replay examples.
// The comparison does not validate the domain of either Value.
//
// Implementation:
//   - Stage 1: two detached atr.Value carriers.
//   - Stage 2: compare both binary64 fields and the boolean mask.
//   - Stage 3: true only when both payload bit patterns and readiness agree.
//
// Behavior highlights:
//   - The comparison does not validate the domain of either Value.
//
// Inputs:
//   - two detached atr.Value carriers.
//
// Returns:
//   - true only when both payload bit patterns and readiness agree.
//
// Errors:
//   - None; arbitrary Value scalars can be compared without dereferencing pointers.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - O(1).
//
// Notes:
//   - The comparison does not validate the domain of either Value.
//
// AI-Hints:
//   - Float equality alone loses signed-zero differences.
func sameValueBits(a, b atr.Value) bool {
	return math.Float64bits(a.TR) == math.Float64bits(b.TR) &&
		math.Float64bits(a.ATR) == math.Float64bits(b.ATR) && a.Valid == b.Valid
}
