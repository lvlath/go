// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File test_helper_test.go centralizes bit-exact assertions and deterministic fixtures.
package drawdown_test

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/lvlath/go/drawdown"
)

// mustErrorIs checks protocol identity through errors.Is, not message text.
func mustErrorIs(t *testing.T, got, want error) {
	t.Helper()
	if !errors.Is(got, want) {
		t.Fatalf("error=%v, want errors.Is(..., %v)", got, want)
	}
}

// mustEqualBool compares a predicate and reports the named contract on failure.
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t, want %t", op, got, want)
	}
}

// mustEqualInt compares exact index or count metadata without tolerances.
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d, want %d", op, got, want)
	}
}

// isNilLike handles ordinary nil and the structural IsNil convention without reflection.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(interface{ IsNil() bool }); ok {
		return nilable.IsNil()
	}
	return false
}

// mustNilState asserts ordinary or typed-nil result presence without dereferencing.
func mustNilState(t *testing.T, value any, want bool, op string) {
	t.Helper()
	mustEqualBool(t, isNilLike(value), want, op)
}

// mustEqualBits compares binary64 representations, distinguishing signed zeros and ULPs.
func mustEqualBits(t *testing.T, got, want float64, op string) {
	t.Helper()
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("%s: got %g (%016x), want %g (%016x)", op, got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

// mustEqualSlice checks shape and every float64 bit pattern, including signed zeros and NaN payloads.
func mustEqualSlice(t *testing.T, got, want []float64) {
	t.Helper()
	mustEqualInt(t, len(got), len(want), "slice length")
	for i, value := range want {
		if math.Float64bits(got[i]) != math.Float64bits(value) {
			t.Fatalf("slice[%d]: got %g (%016x), want %g (%016x)", i, got[i], math.Float64bits(got[i]), value, math.Float64bits(value))
		}
	}
}

// mustCompute calls the public facade with a stable context and rejects absent results.
func mustCompute(t *testing.T, equity []float64) *drawdown.Result {
	t.Helper()
	result, err := drawdown.Compute(context.Background(), equity)
	if err != nil || result == nil {
		t.Fatalf("Compute: result=%v, err=%v", result, err)
	}
	return result
}

// mustResult compares every published field and the exact Maximum/Depth identity.
func mustResult(t *testing.T, got, want *drawdown.Result) {
	t.Helper()
	mustNilState(t, got, false, "result")
	if got.Depth == nil {
		t.Fatal("nil Depth on success")
	}
	mustEqualSlice(t, got.Depth, want.Depth)
	mustEqualBits(t, got.Maximum, want.Maximum, "maximum")
	mustEqualInt(t, got.PeakIndex, want.PeakIndex, "peak")
	mustEqualInt(t, got.TroughIndex, want.TroughIndex, "trough")
	mustEqualInt(t, got.RecoveryIndex, want.RecoveryIndex, "recovery")
	// Validate the index before dereferencing; a broken result must fail diagnostically.
	if got.TroughIndex < 0 || got.TroughIndex >= len(got.Depth) {
		t.Fatalf("trough %d outside profile of length %d", got.TroughIndex, len(got.Depth))
	}
	mustEqualBits(t, got.Maximum, got.Depth[got.TroughIndex], "maximum/trough bit parity")
}

// mustGolden constructs the expected canonical artifact from independently known values.
func mustGolden(t *testing.T, equity, depth []float64, peak, trough, recovery int) {
	t.Helper()
	mustResult(t, mustCompute(t, equity), &drawdown.Result{
		Depth: depth, Maximum: depth[trough], PeakIndex: peak,
		TroughIndex: trough, RecoveryIndex: recovery,
	})
}

// finitePositive draws directly from the finite positive binary64 bit domain.
// Rejection is confined to fixture generation, outside benchmark timers.
func finitePositive(rng *rand.Rand) float64 {
	for {
		value := math.Float64frombits(rng.Uint64() & 0x7fffffffffffffff)
		if value > 0 && !math.IsInf(value, 0) {
			return value
		}
	}
}

// sawEquity produces growing peaks separated by shallow/deep observed losses.
func sawEquity(n int, deep bool) []float64 {
	equity := make([]float64, n)
	for i := range equity {
		peak := float64(1000 + i/16)
		if deep {
			equity[i] = peak * float64(16-i%16) / 16
		} else {
			equity[i] = peak - float64(i%16)
		}
	}
	return equity
}

// backtestFixture generates a 500-mark synthetic institutional equity ledger.
// Integer-dollar anchor values represent recurring risk episodes, a major stress
// period, a recovery and a later correction. Deterministic changes between adjacent anchors
// cannot create an unintended major peak.
// Dates and provenance are supplied separately by the example, not the kernel.
func backtestFixture() []float64 {
	const observations = 500
	// Anchor indices describe the scenario's event chronology, not a trading calendar.
	anchors := []struct{ index, usd int }{
		{0, 100000}, {40, 108000}, {52, 103000}, {76, 115000},
		{100, 120000}, {115, 112000}, {130, 115000}, {160, 90000},
		{180, 104000}, {195, 101000}, {220, 120000}, {270, 134000},
		{300, 124000}, {355, 142000}, {400, 132000}, {499, 152000},
	}
	equity := make([]float64, observations)
	// Interpolate the engineered scenario in whole dollars, never market data.
	for a := 1; a < len(anchors); a++ {
		left, right := anchors[a-1], anchors[a]
		for i := left.index; i <= right.index; i++ {
			equity[i] = float64(left.usd + (right.usd-left.usd)*(i-left.index)/(right.index-left.index))
		}
	}
	return equity
}

// pollContext closes Done on a specified Err call. This deterministic context
// simulates cancellation at exact observation checkpoints without wall-clock races.
// It is single-owner, like the test that invokes it.
type pollContext struct {
	context.Context
	done     chan struct{}
	cancelAt int
	polls    int
}

// newPollContext builds a single-owner deterministic cancellation fixture.
func newPollContext(cancelAt int) *pollContext {
	return &pollContext{Context: context.Background(), done: make(chan struct{}), cancelAt: cancelAt}
}

// Done exposes the stable channel closed at the specified logical poll.
func (c *pollContext) Done() <-chan struct{} { return c.done }

// Err advances the logical polling clock; cancellation is permanent once observed.
// Only the invoking test goroutine owns this fixture.
func (c *pollContext) Err() error {
	c.polls++
	if c.polls == c.cancelAt {
		close(c.done)
	}
	if c.polls >= c.cancelAt {
		return context.Canceled
	}
	return nil
}

// encodeEquity encodes raw float64 bits for the public-boundary fuzzer.
func encodeEquity(values ...float64) []byte {
	data := make([]byte, 8*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint64(data[8*i:], math.Float64bits(v))
	}
	return data
}

// rationalOracle intentionally rescans each small prefix, then separately finds
// the exact argmax and recovery. All values and differences are exact rationals.
func rationalOracle(equity []float64) *drawdown.Result {
	values := make([]*big.Rat, len(equity))
	for i, v := range equity {
		values[i] = new(big.Rat).SetFloat64(v)
	}
	depths := make([]*big.Rat, len(equity))
	peaks := make([]int, len(equity))
	result := &drawdown.Result{Depth: make([]float64, len(equity))}
	for i := range values {
		peak := 0
		for j := 1; j <= i; j++ {
			if values[j].Cmp(values[peak]) > 0 {
				peak = j
			}
		}
		peaks[i] = peak
		delta := new(big.Rat).Sub(values[peak], values[i])
		depths[i] = new(big.Rat).Quo(delta, values[peak])
		result.Depth[i], _ = depths[i].Float64()
	}
	for i := range depths {
		if depths[i].Cmp(depths[result.TroughIndex]) > 0 {
			result.TroughIndex = i
		}
	}
	result.PeakIndex = peaks[result.TroughIndex]
	result.Maximum = result.Depth[result.TroughIndex]
	if depths[result.TroughIndex].Sign() != 0 {
		result.RecoveryIndex = -1
		for j := result.TroughIndex + 1; j < len(values); j++ {
			if values[j].Cmp(values[result.PeakIndex]) >= 0 {
				result.RecoveryIndex = j
				break
			}
		}
	}
	return result
}

// oracleError classifies invalid input independently of production validators.
// It scans the original row order and never computes from an invalid history.
func oracleError(equity []float64) error {
	if len(equity) == 0 {
		return drawdown.ErrInsufficient
	}
	for i, v := range equity {
		if math.IsInf(v, 0) || math.IsNaN(v) {
			return drawdown.ErrNonFinite
		}
		if v < 0 || (i == 0 && v == 0) {
			return drawdown.ErrEquity
		}
	}
	return nil
}
