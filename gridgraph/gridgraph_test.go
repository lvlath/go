package gridgraph_test

import (
	"errors"
	"testing"

	"github.com/lvlath/go/gridgraph"
)

//----------------------------------------------------------------------------//
// NewGridGraph and InBounds Tests
//----------------------------------------------------------------------------//

// TestNewGridGraph_Errors verifies that NewGridGraph rejects empty or ragged inputs.
func TestNewGridGraph_Errors(t *testing.T) {
	cases := []struct {
		name string
		grid [][]int
		opts gridgraph.GridOptions
		err  error
	}{
		{"EmptyRows", [][]int{}, gridgraph.DefaultGridOptions(), gridgraph.ErrEmptyGrid},
		{"EmptyCols", [][]int{{}}, gridgraph.DefaultGridOptions(), gridgraph.ErrEmptyGrid},
		{"NonRectangular", [][]int{{1, 2}, {3}}, gridgraph.DefaultGridOptions(), gridgraph.ErrNonRectangular},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := gridgraph.NewGridGraph(tc.grid, tc.opts)
			if !errors.Is(err, tc.err) {
				t.Errorf("NewGridGraph(%v) error = %v; want %v", tc.grid, err, tc.err)
			}
		})
	}
}

// TestInBounds checks InBounds on a 3×2 grid under Conn4.
func TestInBounds(t *testing.T) {
	grid := [][]int{
		{0, 1, 0},
		{1, 0, 1},
	}
	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn4
	gg, err := gridgraph.NewGridGraph(grid, opts)
	if err != nil {
		t.Fatalf("NewGridGraph error: %v", err)
	}

	valid := [][2]int{{0, 0}, {2, 1}, {1, 1}}
	for _, xy := range valid {
		if !gg.InBounds(xy[0], xy[1]) {
			t.Errorf("InBounds(%d,%d)=false; want true", xy[0], xy[1])
		}
	}
	invalid := [][2]int{{-1, 0}, {3, 0}, {1, 2}, {2, -1}}
	for _, xy := range invalid {
		if gg.InBounds(xy[0], xy[1]) {
			t.Errorf("InBounds(%d,%d)=true; want false", xy[0], xy[1])
		}
	}
}

//----------------------------------------------------------------------------//
// ToCoreGraph Tests
//----------------------------------------------------------------------------//

// TestToCoreGraph_Conn4 verifies that only orthogonal edges exist under Conn4.
func TestToCoreGraph_Conn4(t *testing.T) {
	grid := [][]int{{1, 0}, {1, 1}}
	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn4
	gg, err := gridgraph.NewGridGraph(grid, opts)
	if err != nil {
		t.Fatalf("NewGridGraph error: %v", err)
	}
	cg := gg.ToCoreGraph()

	if len(cg.Vertices()) != 4 {
		t.Errorf("Vertices count = %d; want 4", len(cg.Vertices()))
	}

	have := []struct{ u, v string }{{"0,0", "0,1"}, {"0,1", "1,1"}}
	for _, e := range have {
		if !cg.HasEdge(e.u, e.v) {
			t.Errorf("Edge %s↔%s missing under Conn4", e.u, e.v)
		}
	}

	if cg.HasEdge("0,0", "1,1") {
		t.Error("Unexpected diagonal edge 0,0↔1,1 under Conn4")
	}
}

// TestToCoreGraph_Conn8 verifies diagonal connectivity under Conn8.
func TestToCoreGraph_Conn8(t *testing.T) {
	grid := [][]int{{1, 0}, {0, 1}}
	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn8
	gg, err := gridgraph.NewGridGraph(grid, opts)
	if err != nil {
		t.Fatalf("NewGridGraph error: %v", err)
	}
	cg := gg.ToCoreGraph()

	if !cg.HasEdge("0,0", "1,1") {
		t.Error("Expected diagonal edge 0,0↔1,1 under Conn8")
	}
	if !cg.HasEdge("0,0", "0,1") {
		t.Error("Expected vertical edge 0,0↔0,1 under Conn8")
	}
	if !cg.HasEdge("0,0", "1,0") {
		t.Error("Expected horizontal edge 0,0↔1,0 under Conn8")
	}
}
