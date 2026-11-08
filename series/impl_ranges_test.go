package series_test

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/lvlath/go/series"
)

func TestWindowRanges_HalfOpenAndNoPartial(t *testing.T) {
	for _, tc := range []struct {
		length, width, stride int
		want                  []series.Range
	}{
		{8, 3, 2, []series.Range{{Start: 0, End: 3}, {Start: 2, End: 5}, {Start: 4, End: 7}}},
		{8, 3, 3, []series.Range{{Start: 0, End: 3}, {Start: 3, End: 6}}},
		{8, 3, 4, []series.Range{{Start: 0, End: 3}, {Start: 4, End: 7}}},
		{3, 3, 1, []series.Range{{Start: 0, End: 3}}},
		{0, 3, 1, []series.Range{}},
		{2, 3, 1, []series.Range{}},
	} {
		got, err := series.WindowRanges(tc.length, tc.width, tc.stride)
		if err != nil || !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("%+v got=%v err=%v", tc, got, err)
		}
	}
}

func TestWindowRanges_InvalidAndBoundary(t *testing.T) {
	for _, tc := range [][3]int{{-1, 1, 1}, {1, 0, 1}, {1, 1, 0}, {0, -1, 1}, {0, 1, -1}} {
		got, err := series.WindowRanges(tc[0], tc[1], tc[2])
		if got != nil || !errors.Is(err, series.ErrInvalidOption) {
			t.Fatalf("%v got=%v err=%v", tc, got, err)
		}
	}
	got, err := series.WindowRanges(math.MaxInt, math.MaxInt, math.MaxInt)
	if err != nil || !reflect.DeepEqual(got, []series.Range{{Start: 0, End: math.MaxInt}}) {
		t.Fatalf("boundary got=%v err=%v", got, err)
	}
	got, err = series.WindowRanges(math.MaxInt, math.MaxInt-1, math.MaxInt)
	if err != nil || len(got) != 1 {
		t.Fatalf("boundary got=%v err=%v", got, err)
	}
}
