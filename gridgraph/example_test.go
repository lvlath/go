package gridgraph_test

import (
	"fmt"
	"sort"

	"github.com/lvlath/go/gridgraph"
)

// //////////////////////////////////////////////////////////////////////////////
// Example: ConnectedComponents on an 8×6 grid
// //////////////////////////////////////////////////////////////////////////////
// Diagram (8×6):
//
//	0 1 1 0 2 2 0 3
//	1 1 0 2 2 2 0 3
//	1 0 0 0 0 0 0 3
//	4 4 4 0 5 5 5 3
//	4 0 4 0 5 0 5 3
//	4 4 4 0 5 5 5 3
//
// Values: 0=water, 1-5=land IDs
//
// //////////////////////////////////////////////////////////////////////////////
// Playground: [![Playground - gridgraph](https://img.shields.io/badge/Go_Playground-gridgraph-blue?logo=go)](https://go.dev/play/p/Bv2kVqdRtI6)
// //////////////////////////////////////////////////////////////////////////////
//
// ExampleGridGraph_ConnectedComponents demonstrates how to identify
// contiguous “islands” of cells with value ≥ LandThreshold in a 2D grid.
// It groups components by their value and lists cell coordinates.
func ExampleGridGraph_ConnectedComponents() {
	grid := [][]int{
		{0, 1, 1, 0, 2, 2, 0, 3},
		{1, 1, 0, 2, 2, 2, 0, 3},
		{1, 0, 0, 0, 0, 0, 0, 3},
		{4, 4, 4, 0, 5, 5, 5, 3},
		{4, 0, 4, 0, 5, 0, 5, 3},
		{4, 4, 4, 0, 5, 5, 5, 3},
	}
	opts := gridgraph.DefaultGridOptions()
	opts.LandThreshold = 1
	opts.Conn = gridgraph.Conn4
	gg, _ := gridgraph.NewGridGraph(grid, opts)

	comps := gg.ConnectedComponents()
	// Sort keys for deterministic output
	keys := make([]int, 0, len(comps))
	for k := range comps {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	for _, val := range keys {
		regions := comps[val]
		fmt.Printf("Value %d: %d region(s)\n", val, len(regions))
		// Sort each region's cells by (Y, X)
		for _, region := range regions {
			sort.Slice(region, func(i, j int) bool {
				if region[i].Y != region[j].Y {
					return region[i].Y < region[j].Y
				}

				return region[i].X < region[j].X
			})

			for i, cell := range region {
				if i == 0 {
					fmt.Printf(" Region 0:")
				}
				fmt.Printf(" (%d,%d)", cell.X, cell.Y)
			}
			fmt.Println()
		}
	}
	// Output:
	// Value 1: 1 region(s)
	//  Region 0: (1,0) (2,0) (0,1) (1,1) (0,2)
	// Value 2: 1 region(s)
	//  Region 0: (4,0) (5,0) (3,1) (4,1) (5,1)
	// Value 3: 1 region(s)
	//  Region 0: (7,0) (7,1) (7,2) (7,3) (7,4) (7,5)
	// Value 4: 1 region(s)
	//  Region 0: (0,3) (1,3) (2,3) (0,4) (2,4) (0,5) (1,5) (2,5)
	// Value 5: 1 region(s)
	//  Region 0: (4,3) (5,3) (6,3) (4,4) (6,4) (4,5) (5,5) (6,5)
}

// //////////////////////////////////////////////////////////////////////////////
// Example: ExpandIsland on a 10×10 grid
// //////////////////////////////////////////////////////////////////////////////
// Diagram (10×10):
//
//	1 1 1 0 0 0 0 0 0 0
//	1 1 1 0 0 0 0 0 0 0
//	1 1 1 0 0 0 0 0 0 0
//	0 0 0 0 0 0 0 0 0 0
//	0 0 0 0 0 0 0 0 0 0
//	0 0 0 0 0 0 0 0 0 0
//	0 0 0 0 0 0 0 0 0 0
//	0 0 0 0 0 0 0 2 2 2
//	0 0 0 0 0 0 0 2 2 2
//	0 0 0 0 0 0 0 2 2 2
//
// Values: 0=water, 1-2=land IDs
//
// //////////////////////////////////////////////////////////////////////////////
// Playground: [![Playground - gridgraph](https://img.shields.io/badge/Go_Playground-gridgraph-10x10-blue?logo=go)](https://go.dev/play/p/vt9Q2VDwvJO)
// //////////////////////////////////////////////////////////////////////////////
//
// Create a 10×10 grid with two 3×3 clusters at opposite corners and connected it.
func ExampleGridGraph_ExpandIsland() {
	n := 10
	grid := make([][]int, n)
	for i := range grid {
		grid[i] = make([]int, n)
	}
	// Top-left cluster: value=1; bottom-right: value=2
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			grid[y][x] = 1
			grid[n-1-y][n-1-x] = 2
		}
	}
	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn4
	gg, _ := gridgraph.NewGridGraph(grid, opts)

	comps := gg.ConnectedComponents()
	src := comps[1][0]
	dst := comps[2][0]

	path, cost, _ := gg.ExpandIsland(src, dst)
	fmt.Printf("Converted %d water cells. Path length: %d\n", cost, len(path))

	// Output:
	// Converted 9 water cells. Path length: 11
}
