// Package main demonstrates connecting two “islands” on a game‐style grid
// using lvlath/gridgraph’s ConnectedComponents and ExpandIsland.
//
// Playground: https://go.dev/play/p/y-aCwuVj4X1
//
// Scenario:
//
//	Imagine a 2D game map where cells ≥1 are “land” (different resource IDs),
//	and 0 is water.  We need to construct a minimal‐cost bridge (convert water
//	cells) between two land clusters (islands).
//
// Grid (5×5):
//
//	0 1 1 0 2
//	1 1 0 0 2
//	0 0 0 2 2
//	3 0 0 0 0
//	3 3 0 4 4
//
// Connectivity: 4‐direction (N/E/S/W).
// We will connect the island of “1”s (component 0) to the island of “2”s
// (component 1), minimizing water‐cell conversions.
//
// Use case:
//
//	Terrain generation: build shortest “bridge” between resource areas.
//
// Complexity: O(W·H) time & memory for component analysis + 0-1 BFS expand.
package main

//
//import (
//	"fmt"
//	"log"
//
//	"github.com/lvlath/go/gridgraph"
//)
//
//func main8() {
//	// 1) Define the game map: 0=water, 1-4=different lands
//	grid := [][]int{
//		{0, 1, 1, 0, 2},
//		{1, 1, 0, 0, 2},
//		{0, 0, 0, 2, 2},
//		{3, 0, 0, 0, 0},
//		{3, 3, 0, 4, 4},
//	}
//
//	// 2) Build GridGraph with 4‐way connectivity
//	gg, err := gridgraph.From2D(grid, gridgraph.Conn4)
//	if err != nil {
//		log.Fatalf("Failed to build grid: %v", err)
//	}
//
//	// 3) Identify connected components (“islands”)
//	comps := gg.ConnectedComponents()
//	fmt.Printf("Found %d islands:\n", len(comps))
//	for i, comp := range comps {
//		fmt.Printf("  Component %d: size=%d\n", i, len(comp))
//	}
//
//	// 4) Choose to connect island 0 (value=1) → island 1 (value=2)
//	src, dst := 0, 1
//	path, cost, err := gg.ExpandIsland(src, dst)
//	if err != nil {
//		log.Fatalf("ExpandIsland error: %v", err)
//	}
//
//	// 5) Print bridge plan
//	fmt.Printf("\n🔗 Bridge from island %d to %d:\n", src, dst)
//	fmt.Printf("  Convert %d water cells (cost) along path:\n", cost)
//	for _, idx := range path {
//		x, y := gg.Coordinate(idx)
//		fmt.Printf("    → (%d,%d)\n", x, y)
//	}
//
//	// 6) ASCII view of path overlay
//	fmt.Println("\nMap with bridge path (X):")
//	// create a copy for display
//	display := make([][]rune, gg.Height)
//	for y := range display {
//		display[y] = make([]rune, gg.Width)
//		for x := range display[y] {
//			val := gg.CellValues[y][x]
//			if val == 0 {
//				display[y][x] = '~' // water
//			} else {
//				display[y][x] = rune('0' + val) // land ID
//			}
//		}
//	}
//	// mark path
//	for _, idx := range path {
//		x, y := gg.Coordinate(idx)
//		display[y][x] = 'X'
//	}
//	// print
//	for y := 0; y < gg.Height; y++ {
//		for x := 0; x < gg.Width; x++ {
//			fmt.Printf("%c ", display[y][x])
//		}
//		fmt.Println()
//	}
//}
