// Package gridgraph defines core types, options, and sentinel errors
// for the gridgraph subpackage of github.com/lvlath/go.
package gridgraph

import (
	"errors"
)

// Sentinel errors for gridgraph operations.
var (
	// ErrEmptyGrid indicates input grid has no rows or no columns.
	ErrEmptyGrid = errors.New("gridgraph: input grid must have at least one row and one column")
	// ErrNonRectangular indicates rows of differing lengths.
	ErrNonRectangular = errors.New("gridgraph: all rows must have the same length")
	// ErrComponentIndex indicates a requested component index is out of range.
	ErrComponentIndex = errors.New("gridgraph: component index out of range")
	// ErrNoPath indicates no conversion path exists between two components.
	ErrNoPath = errors.New("gridgraph: no path between specified components")
)

// Connectivity selects neighbor connectivity: orthogonal (Conn4) or including diagonals (Conn8).
type Connectivity int

const (
	// Conn4 uses 4-directional connectivity: N, E, S, W.
	Conn4 Connectivity = iota
	// Conn8 uses 8-directional connectivity: N, NE, E, SE, S, SW, W, NW.
	Conn8
)

// Cell represents a single grid cell with its coordinates and stored value.
type Cell struct {
	X, Y  int // Coordinates within the grid
	Value int // Original grid value at (X, Y)
}

// GridOptions contains tunable parameters for grid analysis.
type GridOptions struct {
	// LandThreshold specifies the minimum cell value considered "land".
	LandThreshold int
	// Conn chooses 4- or 8-directional connectivity.
	Conn Connectivity
}

// DefaultGridOptions returns a GridOptions with default settings:
// LandThreshold=1 (values ≥1 are land), Conn=Conn4.
func DefaultGridOptions() GridOptions {
	return GridOptions{
		LandThreshold: 1,
		Conn:          Conn4,
	}
}

// GridGraph treats a 2D integer grid as a graph. It is immutable once built.
// Width and Height define dimensions; CellValues[y][x] holds the original input value.
// Conn and LandThreshold are set from GridOptions during construction.
// neighborOffsets is precomputed for efficient adjacency lookups.
type GridGraph struct {
	Width, Height   int
	CellValues      [][]int
	Conn            Connectivity
	LandThreshold   int
	neighborOffsets [][2]int
}
