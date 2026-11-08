package series_test

import (
	"sync"
	"testing"

	"github.com/lvlath/go/series"
)

// TestSeries_ConcurrentReadOnlyCalls checks independently allocated results
// when callers share an immutable source array. No input writes are performed.
func TestSeries_ConcurrentReadOnlyCalls(t *testing.T) {
	values := []float64{1, 2, 4, 8, 16, 32}
	times := []int64{0, 10, 20, 30, 40, 50}
	var wg sync.WaitGroup
	for worker := 0; worker < 12; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for iteration := 0; iteration < 20; iteration++ {
				grid, err := series.Validate(times, values, series.GridOptions{CadenceNS: 10})
				if err != nil || grid.Samples != len(values) {
					t.Errorf("Validate: result=%v err=%v", grid, err)
					return
				}
				result, err := series.LogReturns(values)
				if err != nil || !result.Valid[5] {
					t.Errorf("LogReturns: result=%v err=%v", result, err)
					return
				}
				result.Values[5] = 100 // only the caller's output is mutated.
			}
		}()
	}
	wg.Wait()
}
