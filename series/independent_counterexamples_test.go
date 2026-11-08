package series_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/series"
)

// TestIndependentCounterexample_GridDeltaRejectsMinToZero verifies that
// subtraction is checked before any cadence-free early exit.
func TestIndependentCounterexample_GridDeltaRejectsMinToZero(t *testing.T) {
	got, err := series.Validate([]int64{math.MinInt64, 0}, []float64{1, 2}, series.GridOptions{})
	if got != nil || !errors.Is(err, series.ErrOverflow) {
		t.Fatalf("expected ErrOverflow, got result=%v error=%v", got, err)
	}
}

// TestIndependentCounterexample_LogReturnsResolveNextafter checks a positive
// return at large magnitude where rounded logarithms can become equal.
func TestIndependentCounterexample_LogReturnsResolveNextafter(t *testing.T) {
	prev := 1e300
	curr := math.Nextafter(prev, math.Inf(1))
	got, err := series.LogReturns([]float64{prev, curr})
	if err != nil || got == nil || !(got.Values[1] > 0) {
		t.Fatalf("expected positive log return, got result=%v error=%v", got, err)
	}
}

// TestIndependentCounterexample_WindowCountAllocationOverflow confirms
// unrepresentable slice storage is rejected as an error, not a panic.
func TestIndependentCounterexample_WindowCountAllocationOverflow(t *testing.T) {
	defer func() {
		if v := recover(); v != nil {
			t.Errorf("unexpected panic: %v", v)
		}
	}()
	got, err := series.WindowRanges(math.MaxInt, 1, 1)
	if got != nil || !errors.Is(err, series.ErrOverflow) {
		t.Errorf("expected ErrOverflow, got result=%v error=%v", got, err)
	}
}
