package series_test

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/lvlath/go/series"
)

func TestValidate_DuplicateAndGap(t *testing.T) {
	cases := []struct {
		name  string
		times []int64
		vals  []float64
		opts  series.GridOptions
		want  error
		gaps  []series.Gap
	}{
		{"duplicate", []int64{0, 10, 10}, []float64{1, 2, 3}, series.GridOptions{}, series.ErrOrder, nil},
		{"decreasing", []int64{3, 2}, []float64{1, 2}, series.GridOptions{}, series.ErrOrder, nil},
		{"disallowed gap", []int64{0, 10, 30}, []float64{1, 2, 3}, series.GridOptions{CadenceNS: 10}, series.ErrGap, nil},
		{"off-grid allowed", []int64{0, 10, 31}, []float64{1, 2, 3}, series.GridOptions{CadenceNS: 10, AllowGaps: true}, series.ErrGap, nil},
		{"allowed gap", []int64{0, 10, 30}, []float64{1, 2, 3}, series.GridOptions{CadenceNS: 10, AllowGaps: true}, nil, []series.Gap{{AfterIndex: 1, MissingCount: 1}}},
		{"order only", []int64{-5, 10, 31}, []float64{1, 2, 3}, series.GridOptions{}, nil, []series.Gap{}},
		{"negative timestamps", []int64{-30, -20, -10}, []float64{1, 2, 3}, series.GridOptions{CadenceNS: 10}, nil, []series.Gap{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := series.Validate(tc.times, tc.vals, tc.opts)
			if !errors.Is(err, tc.want) {
				if tc.want != nil || err != nil {
					t.Fatalf("error = %v; want %v", err, tc.want)
				}
			}
			if tc.want != nil {
				if got != nil {
					t.Fatalf("error returned non-nil result: %+v", got)
				}
				return
			}
			if got.Samples != len(tc.vals) || !reflect.DeepEqual(got.Gaps, tc.gaps) {
				t.Fatalf("result = %+v; expected samples=%d gaps=%v", got, len(tc.vals), tc.gaps)
			}
		})
	}
}

func TestValidate_Int64DeltaOverflow(t *testing.T) {
	for _, opts := range []series.GridOptions{{}, {CadenceNS: 1, AllowGaps: true}} {
		got, err := series.Validate([]int64{math.MinInt64, math.MaxInt64}, []float64{1, 2}, opts)
		if got != nil || !errors.Is(err, series.ErrOverflow) {
			t.Fatalf("got=%v err=%v", got, err)
		}
	}
	// Exact largest representable delta is admissible without a cadence.
	got, err := series.Validate([]int64{-1, math.MaxInt64 - 1}, []float64{1, 2}, series.GridOptions{})
	if err != nil || got.Samples != 2 {
		t.Fatalf("got=%v err=%v", got, err)
	}
}

func TestValidate_InputPrecedenceAndFiniteness(t *testing.T) {
	cases := []struct {
		times  []int64
		values []float64
		opts   series.GridOptions
		want   error
	}{
		{nil, nil, series.GridOptions{CadenceNS: -1}, series.ErrInvalidOption},
		{nil, []float64{}, series.GridOptions{}, series.ErrNilInput},
		{[]int64{}, nil, series.GridOptions{}, series.ErrNilInput},
		{[]int64{}, []float64{1}, series.GridOptions{}, series.ErrShape},
		{[]int64{}, []float64{}, series.GridOptions{}, series.ErrEmptyInput},
		{[]int64{5, 5}, []float64{math.NaN(), 2}, series.GridOptions{}, series.ErrNonFinite},
		{[]int64{1}, []float64{math.Inf(1)}, series.GridOptions{}, series.ErrNonFinite},
	}
	for _, tc := range cases {
		got, err := series.Validate(tc.times, tc.values, tc.opts)
		if got != nil || !errors.Is(err, tc.want) {
			t.Errorf("got %v, %v; want %v", got, err, tc.want)
		}
	}
}

func TestValidate_GapsAndInputsDetached(t *testing.T) {
	times, values := []int64{0, 10, 40}, []float64{1, 2, 3}
	got, err := series.Validate(times, values, series.GridOptions{CadenceNS: 10, AllowGaps: true})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Gaps, []series.Gap{{AfterIndex: 1, MissingCount: 2}}) {
		t.Fatal(got.Gaps)
	}
	got.Gaps[0].MissingCount = 44
	if !reflect.DeepEqual(times, []int64{0, 10, 40}) || !reflect.DeepEqual(values, []float64{1, 2, 3}) {
		t.Fatal("inputs mutated")
	}
}
