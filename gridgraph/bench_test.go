package gridgraph_test

import (
	"math/rand"
	"testing"

	"github.com/lvlath/go/gridgraph"
)

// BenchmarkConnectedComponents measures performance of ConnectedComponents
// on a randomly generated 1000×1000 grid with values in [0,4].
// Complexity: O(W×H×d)
func BenchmarkConnectedComponents(b *testing.B) {
	const n = 1000
	// Setup: deterministic random grid
	//rand.Seed(42)
	rand.New(rand.NewSource(42))
	grid := make([][]int, n)
	for y := 0; y < n; y++ {
		row := make([]int, n)
		for x := 0; x < n; x++ {
			row[x] = rand.Intn(5) // values 0..4
		}
		grid[y] = row
	}
	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn4
	gg, err := gridgraph.NewGridGraph(grid, opts)
	if err != nil {
		b.Fatalf("setup NewGridGraph failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = gg.ConnectedComponents()
	}
}

// BenchmarkExpandIsland measures performance of ExpandIsland
// on a 1000×1000 grid with two 1-cell islands at opposite corners.
// Complexity: O(W×H×d)
func BenchmarkExpandIsland(b *testing.B) {
	const n = 1000
	// Setup: grid with land at top-left (value=1) and bottom-right (value=2)
	grid := make([][]int, n)
	for y := 0; y < n; y++ {
		row := make([]int, n)
		grid[y] = row
	}
	// Place two islands
	grid[0][0] = 1
	grid[n-1][n-1] = 2

	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn8 // use diagonal connectivity for faster path
	gg, err := gridgraph.NewGridGraph(grid, opts)
	if err != nil {
		b.Fatalf("setup NewGridGraph failed: %v", err)
	}
	comps := gg.ConnectedComponents()
	srcList := comps[1]
	dstList := comps[2]
	if len(srcList) == 0 || len(dstList) == 0 {
		b.Fatal("expected two islands in setup grid")
	}
	src := srcList[0]
	dst := dstList[0]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = gg.ExpandIsland(src, dst)
	}
}
