package series_test

import (
	"fmt"

	"github.com/lvlath/go/series"
)

// ExampleValidate demonstrates a missing slot on an integer-nanosecond grid.
func ExampleValidate() {
	r, err := series.Validate([]int64{0, 10, 30}, []float64{2, 4, 8}, series.GridOptions{CadenceNS: 10, AllowGaps: true})
	fmt.Println(r.Samples, r.Gaps, err)
	// Output: 3 [{1 1}] <nil>
}

// ExampleLag demonstrates strict causality and a detached warmup cell.
func ExampleLag() {
	r, _ := series.Lag([]float64{2, 4, 8}, 1)
	fmt.Println(r.Values, r.Valid)
	// Output: [0 2 4] [false true true]
}

// ExampleDifference demonstrates a two-step finite difference.
func ExampleDifference() {
	r, _ := series.Difference([]float64{2, 4, 8, 16}, 2)
	fmt.Println(r.Values, r.Valid)
	// Output: [0 0 6 12] [false false true true]
}

// ExampleSimpleReturns demonstrates negative returns and a valid final zero.
func ExampleSimpleReturns() {
	r, _ := series.SimpleReturns([]float64{2, 4, 0})
	fmt.Println(r.Values, r.Valid)
	// Output: [0 1 -1] [false true true]
}

// ExampleWindowRanges demonstrates complete-only half-open windows.
func ExampleWindowRanges() {
	r, _ := series.WindowRanges(8, 3, 2)
	fmt.Println(r)
	// Output: [{0 3} {2 5} {4 7}]
}
