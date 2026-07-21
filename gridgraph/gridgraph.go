// Package gridgraph provides utilities to treat a 2D grid of integer cell values
// as a graph. It supports:
//
//   - Four- or eight-connectivity (Conn4 or Conn8)
//   - Conversion to a *core.Graph
//   - Identification of connected components of “land” cells
//   - Shortest-path expansions between components
//
// Cells with value < LandThreshold are considered “water”; cells with value ≥ LandThreshold are “land”.
package gridgraph

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// NewGridGraph constructs a GridGraph from a non-empty, rectangular 2D slice.
// It deep-copies the input to ensure immutability.
// Returns ErrEmptyGrid if grid has no rows or no columns,
// ErrNonRectangular if any row length differs.
// Algorithmic complexity: O(W×H) time and memory.
func NewGridGraph(values [][]int, opts GridOptions) (*GridGraph, error) {
	if len(values) == 0 || len(values[0]) == 0 {
		return nil, ErrEmptyGrid
	}
	h, w := len(values), len(values[0])
	for _, row := range values {
		if len(row) != w {
			return nil, ErrNonRectangular
		}
	}
	// Deep copy to prevent external mutation
	cells := make([][]int, h)
	for y := 0; y < h; y++ {
		cells[y] = make([]int, w)
		copy(cells[y], values[y])
	}
	// Precompute neighbor offsets based on connectivity
	var offsets [][2]int
	//offsets := make([][2]int, 0, 8)
	if opts.Conn == Conn8 {
		offsets = [][2]int{{0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}}
	} else {
		offsets = [][2]int{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}
	}
	gg := &GridGraph{
		Width:           w,
		Height:          h,
		CellValues:      cells,
		Conn:            opts.Conn,
		LandThreshold:   opts.LandThreshold,
		neighborOffsets: offsets,
	}

	return gg, nil
}

// InBounds reports whether (x,y) lies within the grid boundaries.
// Complexity: O(1).
func (gg *GridGraph) InBounds(x, y int) bool {
	return x >= 0 && x < gg.Width && y >= 0 && y < gg.Height
}

// NeighborOffsets returns the precomputed neighbor offsets slice.
// Should be used in all adjacency traversals to avoid branching.
// Complexity: O(1).
func (gg *GridGraph) NeighborOffsets() [][2]int {
	return gg.neighborOffsets
}

// vertexID formats the unique vertex identifier for cell (x,y).
// Used when converting to a core.Graph.
func (gg *GridGraph) vertexID(x, y int) string {
	return fmt.Sprintf("%d,%d", x, y)
}

// ToCoreGraph converts the GridGraph into a weighted, undirected *core.Graph.
// Each cell at (x,y) becomes a vertex with ID "x,y" and metadata {x,y,value}.
// Edges of unit weight (1) connect neighboring cells according to gg.Conn.
// Complexity: O(W×H×d + E) time, Memory: O(W×H + E).
func (gg *GridGraph) ToCoreGraph() *core.Graph {
	g, _ := core.NewGraph(core.WithWeighted())
	// Add all vertices
	for y := 0; y < gg.Height; y++ {
		for x := 0; x < gg.Width; x++ {
			id := gg.vertexID(x, y)
			_ = g.AddVertex(id)
		}
	}
	// Populate metadata maps
	verts := g.InternalVertices()
	for y := 0; y < gg.Height; y++ {
		for x := 0; x < gg.Width; x++ {
			id := gg.vertexID(x, y)
			v := verts[id]
			v.Metadata["x"] = x
			v.Metadata["y"] = y
			v.Metadata["value"] = gg.CellValues[y][x]
		}
	}
	// Add edges for each neighbor pair
	for y := 0; y < gg.Height; y++ {
		for x := 0; x < gg.Width; x++ {
			uID := gg.vertexID(x, y)
			for _, d := range gg.NeighborOffsets() {
				nx, ny := x+d[0], y+d[1]
				if !gg.InBounds(nx, ny) {
					continue
				}
				vID := gg.vertexID(nx, ny)
				_, _ = g.AddEdge(uID, vID, 1)
			}
		}
	}

	return g
}

// index maps (x,y) to a row‑major index: y*Width + x.
// Complexity: O(1).
func (gg *GridGraph) index(x, y int) int {
	return y*gg.Width + x
}

// Coordinate converts a row‑major index back to (x,y).
// Complexity: O(1).
func (gg *GridGraph) Coordinate(idx int) (x, y int) {
	return idx % gg.Width, idx / gg.Width
}
