# 9. GridGraph

`lvlath/gridgraph` models a 2D grid of integer cell values as a graph. Each cell becomes a vertex; edges connect neighboring cells based on **Connectivity** (4- or 8-directional). Cells with value `0` are “water” (impassable); cells with value ≥1 are “land” (passable). This enables common grid tasks like island counting, terrain bridging, and conversion to general graph algorithms.

---

## 9.1 Definitions

* **Conn4 (4-connectivity)**: each cell connects to its four orthogonal neighbors (N, E, S, W).
* **Conn8 (8-connectivity)**: also includes the four diagonal neighbors (NE, SE, SW, NW).

```ascii
 Conn4:           Conn8:
   . X .            X X X
   X X X            X X X
   . X .            X X X
```

*Use Conn4 for strict orthogonal movement (e.g., grid mazes); use Conn8 for natural terrain connectivity.*

---

## 9.2 Use Cases

1. **Island Counting**: identify clusters of adjacent “land” cells via `ConnectedComponents()`.
2. **Terrain Bridging**: find minimal-cost paths over water (cost=1) vs land (cost=0) using `ExpandIsland()`.
3. **Graph Conversion**: transform any grid into a `*core.Graph` for shortest-path or flow algorithms via `ToCoreGraph()`.

---

## 9.3 Core Functions

These methods are defined on `*gridgraph.GridGraph` in `gridgraph.go` and related files:

| Function                                | Signature                                                        | Ref               |
| --------------------------------------- | ---------------------------------------------------------------- | ----------------- |
| Construct from 2D slice                 | `From2D(values [][]int, conn Connectivity) (*GridGraph, error)`  | citeturn4file3 |
| Connected components (clusters of land) | `ConnectedComponents() [][]int`                                  | citeturn4file5 |
| Minimal-cost bridge between two islands | `ExpandIsland(srcComp, dstComp int) (path []int, cost int, err)` | citeturn4file6 |
| Convert to general graph                | `ToCoreGraph() *core.Graph`                                      | citeturn4file0 |

### 9.3.1 ConnectedComponents

Performs a BFS flood-fill over all `CellValues[y][x] >= 1`, grouping each contiguous region of land into a slice of row-major indices. Returns all components in arbitrary order.

### 9.3.2 ExpandIsland (0-1 BFS)

Implements a 0-1 BFS from every cell in source component:

* **Cost 0** for stepping on land (`CellValues>=1`).
* **Cost 1** for stepping on water (`CellValues<1`).

Stops when any target-component cell is reached; reconstructs the shortest conversion path and its total cost.

### 9.3.3 ToCoreGraph

Transforms the grid into a `*core.Graph`:

* Each cell at `(x,y)` becomes a vertex with ID `"x,y"` and metadata `{x, y, value}`.
* Edges of weight 1 connect every neighboring pair per the chosen connectivity.

Time: O(W·H·d), Memory: O(W·H + E), where d = 4 or 8, E≈W·H·d.

---

## 9.4 ASCII Visualization

Consider the following 5×5 height map (`0`=water, `1`=land):

```ascii
 0 1 1 0 0
 1 1 0 1 1
 0 0 1 1 0
 1 1 1 0 0
 0 0 0 1 1
```

* **Conn4** components (two islands A and B):

```ascii
   . # # . .    (# marks one island)
   # # . # #    (second island appears separately)
   . . . . .    
   # # # . .    
   . . . . .    
```

* **Conn8** merges diagonals into a single component:

```ascii
   . # # . .
   # # . # #
   . . # # .
   # # # . .
   . . . . .
```

*(Here all `#` form one contiguous island under Conn8.)*

---

## 9.5 Go Playground Example

```go
package main

import (
	"fmt"
	"log"
	"sort"

	"github.com/lvlath/go/gridgraph"
)

func main() {
	// Emergency island-routing map, 8×8.
	//
	// Values:
	//   0 = water / blocked cell
	//   1 = logistics road islands that must be connected
	//   2 = hospital district
	//   3 = port district
	//   4 = power station
	//   5 = warehouse zone
	//   6 = rail yard
	//   7 = repair depot
	//   8 = command center
	//
	// Conn4 is intentionally used:
	// city movement is orthogonal, so diagonal touching is NOT a road connection.
	grid := [][]int{
		{1, 1, 0, 0, 0, 0, 1, 1},
		{1, 1, 0, 2, 2, 0, 1, 1},
		{0, 0, 0, 2, 2, 0, 0, 0},
		{3, 3, 0, 0, 0, 0, 4, 4},
		{3, 0, 0, 5, 5, 0, 0, 4},
		{3, 3, 0, 5, 5, 0, 4, 4},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{6, 6, 0, 7, 7, 0, 8, 8},
	}

	opts := gridgraph.DefaultGridOptions()
	opts.Conn = gridgraph.Conn4
	opts.LandThreshold = 1

	gg, err := gridgraph.NewGridGraph(grid, opts)
	if err != nil {
		log.Fatalf("build grid graph: %v", err)
	}

	// 1) Detect all dry connected regions, grouped by their terrain value.
	//
	// ConnectedComponents returns:
	//   map[value][][]Cell
	//
	// Meaning:
	//   components[1] is a list of all disconnected logistics-road regions.
	//   components[2] is a list of all disconnected hospital regions.
	//   etc.
	components := gg.ConnectedComponents()

	values := make([]int, 0, len(components))
	totalRegions := 0
	for value, regions := range components {
		values = append(values, value)
		totalRegions += len(regions)
	}
	sort.Ints(values)

	fmt.Printf("Detected %d terrain value(s), %d connected region(s):\n", len(values), totalRegions)

	for _, value := range values {
		regions := components[value]

		// Deterministic example output:
		// sort regions by their first top-left cell, and sort cells inside each region.
		for _, region := range regions {
			sort.Slice(region, func(i, j int) bool {
				if region[i].Y != region[j].Y {
					return region[i].Y < region[j].Y
				}
				return region[i].X < region[j].X
			})
		}
		sort.Slice(regions, func(i, j int) bool {
			a, b := regions[i][0], regions[j][0]
			if a.Y != b.Y {
				return a.Y < b.Y
			}
			return a.X < b.X
		})

		fmt.Printf("  value=%d: %d region(s)\n", value, len(regions))
		for i, region := range regions {
			fmt.Printf("    region %d: %d cells:", i, len(region))
			for _, cell := range region {
				fmt.Printf(" (%d,%d)", cell.X, cell.Y)
			}
			fmt.Println()
		}
	}

	// 2) Connect two disconnected logistics-road islands.
	//
	// We deliberately connect components[1][0] and components[1][1]:
	// both are value=1 logistics regions, but they are separated by water.
	logisticsRegions := components[1]
	if len(logisticsRegions) < 2 {
		log.Fatalf("need at least two disconnected logistics regions, got %d", len(logisticsRegions))
	}

	src := logisticsRegions[0]
	dst := logisticsRegions[1]

	path, cost, err := gg.ExpandIsland(src, dst)
	if err != nil {
		log.Fatalf("expand logistics island: %v", err)
	}

	fmt.Println()
	fmt.Println("Minimum-cost emergency bridge:")
	fmt.Printf("  converted water cells: %d\n", cost)
	fmt.Printf("  path length: %d cells\n", len(path))
	fmt.Print("  path:")

	for _, cell := range path {
		mark := "dry"
		if cell.Value < opts.LandThreshold {
			mark = "water→bridge"
		}
		fmt.Printf(" (%d,%d:%s)", cell.X, cell.Y, mark)
	}

	fmt.Println()

	// Example interpretation:
	//
	// If cost=2, then the planner found a route where only two flooded cells
	// need conversion. Dry infrastructure already present on the path costs 0.
	//
	// This is exactly what 0–1 BFS is good at:
	//   - step into dry land:   cost 0
	//   - step into water:      cost 1
	//   - total bridge cost:    number of water cells converted
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-GridGraph-blue?logo=go)](https://go.dev/play/p/NH0UW0MPJUu)

---

## 9.6 Pitfalls & Best Practices

* **Connectivity choice**: use **Conn4** for strict orthogonal grids (e.g., mazes), **Conn8** for realistic movements over terrain.
* **0-1 BFS** offers **O(V+E)** performance: ideal for binary-cost grids.
* **Consistent indexing**: row-major indexing (`index(x,y)=y*W+x`) ensures reproducible component ordering.
* **Memory**: avoid deep recursion; both `ConnectedComponents` and `ExpandIsland` are iterative.
* **Metadata**: leverage `ToCoreGraph()` to annotate vertices for use with other algorithms.

---

Next: [10. Matrices: Adjacency & Incidence →](MATRICES.md)
