package gridgraph_test

import (
	"errors"
	"testing"

	"github.com/lvlath/go/gridgraph"
)

// TestConnectedComponents_Simple4 verifies two islands of values 1 and 2 on a 3×4 grid under Conn4.
func TestConnectedComponents_Simple4(t *testing.T) {
	grid := [][]int{
		{0, 1, 1, 0},
		{1, 1, 0, 0},
		{0, 0, 2, 2},
	}
	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn4
	gg, err := gridgraph.NewGridGraph(grid, opts)
	if err != nil {
		t.Fatalf("NewGridGraph error: %v", err)
	}

	comps := gg.ConnectedComponents()

	// Expect two keys: 1 and 2
	if len(comps) != 2 {
		t.Fatalf("Components map size = %d; want 2 keys", len(comps))
	}

	// For value 1: four cells
	c1 := comps[1]
	if len(c1) != 1 {
		t.Errorf("Value=1: number of components = %d; want 1", len(c1))
	} else if len(c1[0]) != 4 {
		t.Errorf("Value=1: component size = %d; want 4", len(c1[0]))
	}

	// For value 2: two cells
	c2 := comps[2]
	if len(c2) != 1 {
		t.Errorf("Value=2: number of components = %d; want 1", len(c2))
	} else if len(c2[0]) != 2 {
		t.Errorf("Value=2: component size = %d; want 2", len(c2[0]))
	}
}

// TestConnectedComponents_Diagonal8 verifies diagonal connectivity merges corner-touching cells.
func TestConnectedComponents_Diagonal8(t *testing.T) {
	grid := [][]int{
		{1, 0, 0, 0, 1},
		{0, 1, 0, 1, 0},
		{0, 0, 1, 0, 0},
		{0, 1, 0, 1, 0},
		{1, 0, 0, 0, 1},
	}
	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn8
	gg, err := gridgraph.NewGridGraph(grid, opts)
	if err != nil {
		t.Fatalf("NewGridGraph error: %v", err)
	}

	comps := gg.ConnectedComponents()

	// Expect one component for value 1
	if comps1, ok := comps[1]; !ok {
		t.Fatal("Missing key 1 in components map")
	} else if len(comps1) != 1 {
		t.Errorf("Value=1: number of components = %d; want 1", len(comps1))
	} else if len(comps1[0]) != 9 {
		t.Errorf("Value=1: component size = %d; want 9", len(comps1[0]))
	}
}

// TestConnectedComponents_EmptyAndSingle tests empty-water and single-land cases.
func TestConnectedComponents_EmptyAndSingle(t *testing.T) {
	// All water grid
	gridAllWater := [][]int{{0, 0}, {0, 0}}
	opts := gridgraph.DefaultGridOptions()
	ggw, _ := gridgraph.NewGridGraph(gridAllWater, opts)
	cw := ggw.ConnectedComponents()
	if len(cw) != 0 {
		t.Errorf("All-water: got %d keys; want 0", len(cw))
	}

	// Single land cell
	gridSingle := [][]int{{0, 1}}
	ggs, _ := gridgraph.NewGridGraph(gridSingle, opts)
	cs := ggs.ConnectedComponents()
	if comps1, ok := cs[1]; !ok {
		t.Fatal("Missing key 1 for single cell")
	} else if len(comps1) != 1 {
		t.Errorf("Single land: number of components = %d; want 1", len(comps1))
	} else if len(comps1[0]) != 1 {
		t.Errorf("Single land: component size = %d; want 1", len(comps1[0]))
	}
}

// TestConnectedComponents_InvalidGrid verifies NewGridGraph rejects invalid shapes for CC.
func TestConnectedComponents_InvalidGrid(t *testing.T) {
	// Nil grid
	if _, err := gridgraph.NewGridGraph(nil, gridgraph.DefaultGridOptions()); !errors.Is(err, gridgraph.ErrEmptyGrid) {
		t.Errorf("nil grid error = %v; want ErrEmptyGrid", err)
	}
	// Jagged grid
	jagged := [][]int{{1}, {}}
	if _, err := gridgraph.NewGridGraph(jagged, gridgraph.DefaultGridOptions()); !errors.Is(err, gridgraph.ErrNonRectangular) {
		t.Errorf("jagged grid error = %v; want ErrNonRectangular", err)
	}
}
