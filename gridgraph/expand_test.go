package gridgraph_test

import (
	"errors"
	"reflect"
	"testing"

	"github.com/lvlath/go/gridgraph"
)

// Basic test on a small grid (1×3) verifying minimal water conversion.
func TestExpandIsland_BasicLine(t *testing.T) {
	grid := [][]int{{1, 0, 1}}
	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn4
	gg, err := gridgraph.NewGridGraph(grid, opts)
	if err != nil {
		t.Fatalf("NewGridGraph failed: %v", err)
	}

	// Two land components at ends
	comps := gg.ConnectedComponents()[1]
	if len(comps) != 2 {
		t.Fatalf("expected 2 components of value=1; got %d", len(comps))
	}
	src, dst := comps[0], comps[1]

	path, cost, err := gg.ExpandIsland(src, dst)
	if err != nil {
		t.Fatalf("ExpandIsland error: %v", err)
	}

	// Expect cost=1 and path through the single water cell
	expected := []gridgraph.Cell{{0, 0, 1}, {1, 0, 0}, {2, 0, 1}}
	if cost != 1 || !reflect.DeepEqual(path, expected) {
		t.Errorf("BasicLine: got cost=%d, path=%v; want cost=1, path=%v", cost, path, expected)
	}
}

// Medium test on a larger random-like 10×10 grid with two diagonal clusters.
func TestExpandIsland_10x10Grid(t *testing.T) {
	// Build a 10×10 grid: two 3×3 land blocks at corners, water elsewhere
	n := 10
	grid := make([][]int, n)
	for i := range grid {
		grid[i] = make([]int, n)
	}
	// Top-left 3x3 block = 1, bottom-right 3x3 block = 2
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			grid[y][x] = 1
			grid[n-1-y][n-1-x] = 2
		}
	}

	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn4
	gg, err := gridgraph.NewGridGraph(grid, opts)
	if err != nil {
		t.Fatalf("NewGridGraph failed: %v", err)
	}

	comps := gg.ConnectedComponents()
	srcComps, ok1 := comps[1]
	dstComps, ok2 := comps[2]
	if !ok1 || !ok2 {
		t.Fatalf("Missing components: keys1=%v, keys2=%v", ok1, ok2)
	}
	src, dst := srcComps[0], dstComps[0]

	path, cost, err := gg.ExpandIsland(src, dst)
	if err != nil {
		t.Fatalf("ExpandIsland on 10x10 failed: %v", err)
	}

	// Path length should be >0 and cost ~ (distance between centers) minus contiguous land steps
	dx := (n - 3) / 2 * (2 + 1) // approximate manhattan distance
	expectedMinCost := dx       // + dx
	if cost != expectedMinCost {
		t.Errorf("10x10: got cost=%d; want %d", cost, expectedMinCost)
	}
	if len(path) == 0 {
		t.Error("10x10: unexpected empty path")
	}
}

// Diagonal connectivity allows zero-cost path on corner-touching lands under Conn8.
func TestExpandIsland_Diagonal8(t *testing.T) {
	grid := [][]int{{1, 0}, {0, 1}}
	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn8
	gg, err := gridgraph.NewGridGraph(grid, opts)
	if err != nil {
		t.Fatalf("NewGridGraph failed: %v", err)
	}

	// Single component of value=1 under Conn8
	comps := gg.ConnectedComponents()[1]
	src, dst := comps[0], comps[0]

	path, cost, err := gg.ExpandIsland(src, dst)
	if err != nil {
		t.Fatalf("ExpandIsland error: %v", err)
	}
	if cost != 0 {
		t.Errorf("Diagonal8: expected cost=0; got %d", cost)
	}
	if len(path) != 1 {
		t.Errorf("Diagonal8: expected single-cell path; got %v", path)
	}
}

// TestExpandIsland_ErrorCases verifies error conditions: empty inputs.
func TestExpandIsland_ErrorCases(t *testing.T) {
	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn4
	gg, _ := gridgraph.NewGridGraph([][]int{{1, 0, 1}}, opts)

	// Empty src
	_, _, err := gg.ExpandIsland(nil, []gridgraph.Cell{{0, 0, 1}})
	if !errors.Is(err, gridgraph.ErrComponentIndex) {
		t.Errorf("empty src: got %v; want ErrComponentIndex", err)
	}

	// Empty dst
	_, _, err = gg.ExpandIsland([]gridgraph.Cell{{0, 0, 1}}, nil)
	if !errors.Is(err, gridgraph.ErrComponentIndex) {
		t.Errorf("empty dst: got %v; want ErrComponentIndex", err)
	}
}
