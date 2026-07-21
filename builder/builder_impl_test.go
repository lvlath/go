// File: builders_impl_test.go
// Package builder_test contains functional tests for all GraphConstructor
// implementations in the builder package, verifying correct topology, counts,
// idempotence, and default weights.
package builder_test

import (
	"fmt"
	"sort"
	"testing"

	"github.com/lvlath/go/builder"
	"github.com/lvlath/go/core"
)

// edgeKey identifies an edge by its endpoints.
type edgeKey struct{ U, V string }

// sortedVertices returns the sorted slice of vertex IDs in g.
func sortedVertices(g *core.Graph) []string {
	vs := g.Vertices() // get all vertex IDs
	sort.Strings(vs)   // sort for deterministic comparison
	return vs
}

// sortedEdgeWeights returns a map from edgeKey to weight for all edges in g.
func sortedEdgeWeights(g *core.Graph) map[edgeKey]float64 {
	m := make(map[edgeKey]float64)
	for _, e := range g.Edges() {
		m[edgeKey{U: e.From, V: e.To}] = e.Weight
	}
	return m
}

// TestBuilders_Functional runs table-driven functional tests for each builder.
func TestBuilders_Functional(t *testing.T) {
	t.Parallel() // allow this test to run in parallel with others

	// helper to count undirected edges: since builder uses undirected graphs by default,
	// each AddEdge call creates exactly one entry in Edges().
	// For symmetric constructions, counts must match expected.
	tests := []struct {
		name        string
		ctor        builder.Constructor
		wantV       int                               // expected number of vertices
		wantE       int                               // expected number of edges
		sampleCheck func(t *testing.T, g *core.Graph) // additional topology-specific checks
		bopts       []builder.Option
	}{
		{
			name:  "Cycle(5)",
			ctor:  builder.Cycle(5),
			wantV: 5, wantE: 5,
			sampleCheck: func(t *testing.T, g *core.Graph) {
				edges := sortedEdgeWeights(g)
				// verify each i->(i+1)%5 exists with default weight
				for i := 0; i < 5; i++ {
					from := fmt.Sprint(i)
					to := fmt.Sprint((i + 1) % 5)
					if w, ok := edges[edgeKey{from, to}]; !ok || w != builder.DefaultEdgeWeight {
						t.Errorf("Cycle: missing or wrong weight for edge %s→%s: got %g, ok=%v", from, to, w, ok)
					}
				}
			},
		},
		{
			name:  "Path(4)",
			ctor:  builder.Path(4),
			wantV: 4, wantE: 3,
			sampleCheck: func(t *testing.T, g *core.Graph) {
				edges := sortedEdgeWeights(g)
				// verify edges 0→1,1→2,2→3
				for i := 0; i < 3; i++ {
					from, to := fmt.Sprint(i), fmt.Sprint(i+1)
					if w, ok := edges[edgeKey{from, to}]; !ok || w != builder.DefaultEdgeWeight {
						t.Errorf("Path: missing or wrong weight for edge %s→%s", from, to)
					}
				}
			},
		},
		{
			name:  "Star(4)",
			ctor:  builder.Star(4),
			wantV: 4, wantE: 3,
			sampleCheck: func(t *testing.T, g *core.Graph) {
				edges := sortedEdgeWeights(g)
				// leaves are IDs "1","2","3" all from "Center"
				for i := 1; i < 4; i++ {
					leaf := fmt.Sprint(i)
					if w, ok := edges[edgeKey{"Center", leaf}]; !ok || w != builder.DefaultEdgeWeight {
						t.Errorf("Star: missing or wrong weight for edge Center→%s", leaf)
					}
				}
			},
		},
		{
			name:  "Wheel(4)",
			ctor:  builder.Wheel(4),
			wantV: 5, wantE: 8, // 4 cycle + 4 spokes
			sampleCheck: func(t *testing.T, g *core.Graph) {
				edges := sortedEdgeWeights(g)
				// check one cycle edge and one spoke
				if _, ok := edges[edgeKey{"0", "1"}]; !ok {
					t.Error("Wheel: missing cycle edge 0→1")
				}
				if _, ok := edges[edgeKey{"Center", "2"}]; !ok {
					t.Error("Wheel: missing spoke Center→2")
				}
			},
		},
		{
			name:  "Complete(4)",
			ctor:  builder.Complete(4),
			wantV: 4, wantE: 6, // undirected K4 has 4*3/2 = 6 edges
			sampleCheck: func(t *testing.T, g *core.Graph) {
				edges := sortedEdgeWeights(g)
				// verify a few unordered pairs exist
				pairs := [][2]string{{"0", "1"}, {"1", "2"}, {"2", "3"}}
				for _, p := range pairs {
					if _, ok := edges[edgeKey{p[0], p[1]}]; !ok {
						t.Errorf("Complete: missing edge %s→%s", p[0], p[1])
					}
				}
			},
		},
		{
			name:  "CompleteBipartite(2,3)",
			ctor:  builder.CompleteBipartite(2, 3),
			wantV: 5, wantE: 6, // 2*3 = 6 edges
			sampleCheck: func(t *testing.T, g *core.Graph) {
				edges := sortedEdgeWeights(g)
				// check edge L0→R0 and L1→R2
				if _, ok := edges[edgeKey{"L0", "R0"}]; !ok {
					t.Error("CompleteBipartite: missing L0→R0")
				}
				if _, ok := edges[edgeKey{"L1", "R2"}]; !ok {
					t.Error("CompleteBipartite: missing L1→R2")
				}
			},
		},
		{
			name:  "RandomSparse_p0(5)",
			ctor:  builder.RandomSparse(5, 0.0),
			wantV: 5, wantE: 0, // p=0 yields no edges
			bopts: []builder.Option{builder.WithSeed(1)}, // satisfy rng-required contract
			sampleCheck: func(t *testing.T, g *core.Graph) {
				if len(g.Edges()) != 0 {
					t.Errorf("RandomSparse(p=0): expected 0 edges, got %d", len(g.Edges()))
				}
			},
		},
		{
			name:  "RandomSparse_p1(5)",
			ctor:  builder.RandomSparse(5, 1.0),
			wantV: 5, wantE: 10, // 5*4/2 = 10
			bopts: []builder.Option{builder.WithSeed(1)}, // satisfy rng-required contract
			sampleCheck: func(t *testing.T, g *core.Graph) {
				if len(g.Edges()) != 10 {
					t.Errorf("RandomSparse(p=1): expected 10 edges, got %d", len(g.Edges()))
				}
			},
		},
		{
			name:  "RandomRegular(6,2)",
			ctor:  builder.RandomRegular(6, 2),
			wantV: 6, wantE: 6, // n*d/2 = 6*2/2 = 6 edges
			bopts: []builder.Option{builder.WithSeed(42)}, // rng required by contract
			sampleCheck: func(t *testing.T, g *core.Graph) {
				if len(g.Edges()) != 6 {
					t.Errorf("RandomRegular: expected 6 edges, got %d", len(g.Edges()))
				}
			},
		},
		{
			name:  "Grid(2x3)",
			ctor:  builder.Grid(2, 3),
			wantV: 6, wantE: 7, // (2*(3-1)) + ((2-1)*3) = 4+3 = 7
			sampleCheck: func(t *testing.T, g *core.Graph) {
				// check edge "0,0"→"0,1" and "0,0"→"1,0"
				edges := sortedEdgeWeights(g)
				if _, ok := edges[edgeKey{"0,0", "0,1"}]; !ok {
					t.Error("Grid: missing horizontal edge 0,0→0,1")
				}
				if _, ok := edges[edgeKey{"0,0", "1,0"}]; !ok {
					t.Error("Grid: missing vertical edge 0,0→1,0")
				}
			},
		},
		{
			name:  "Hexagram(Default)",
			ctor:  builder.Hexagram(builder.HexDefault),
			wantV: 6, wantE: 12, // 6-cycle + 6 chords
			sampleCheck: func(t *testing.T, g *core.Graph) {
				// check chord edges exist, e.g. 0→2
				edges := sortedEdgeWeights(g)
				if _, ok := edges[edgeKey{"0", "2"}]; !ok {
					t.Error("Hexagram: missing chord edge 0→2")
				}
			},
		},
		{
			name:  "PlatonicSolid(Tetrahedron,noCenter)",
			ctor:  builder.PlatonicSolid(builder.Tetrahedron, false),
			wantV: 4, wantE: 6, // K4 has 6 edges
			sampleCheck: func(t *testing.T, g *core.Graph) {
				edges := sortedEdgeWeights(g)
				if _, ok := edges[edgeKey{"0", "1"}]; !ok {
					t.Error("PlatonicSolid: missing edge 0→1")
				}
			},
		},
		{
			name:  "PlatonicSolid(Tetrahedron,withCenter)",
			ctor:  builder.PlatonicSolid(builder.Tetrahedron, true),
			wantV: 5, wantE: 10, // 6 shell edges + 4 spokes = 10
			sampleCheck: func(t *testing.T, g *core.Graph) {
				if _, ok := sortedEdgeWeights(g)[edgeKey{"Center", "0"}]; !ok {
					t.Error("PlatonicSolid: missing spoke Center→0")
				}
			},
		},
	}

	// Execute each subtest in parallel
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			// build into a weighted graph so AddEdge never returns ErrBadWeight
			graphOpts := []core.GraphOption{core.WithWeighted()}
			g, err := builder.BuildGraph(graphOpts, tc.bopts, tc.ctor)
			if err != nil {
				t.Fatalf("BuildGraph(%s) returned error: %v", tc.name, err)
			}

			// verify vertex count
			if got := len(sortedVertices(g)); got != tc.wantV {
				t.Errorf("vertices: got %d, want %d", got, tc.wantV)
			}

			// verify edge count
			if got := len(g.Edges()); got != tc.wantE {
				t.Errorf("edges: got %d, want %d", got, tc.wantE)
			}

			// topology‐specific checks
			tc.sampleCheck(t, g)

			// idempotence: rerun builder on a fresh weighted graph
			g2, err2 := builder.BuildGraph(graphOpts, tc.bopts, tc.ctor)
			if err2 != nil {
				t.Fatalf("second BuildGraph(%s) returned error: %v", tc.name, err2)
			}
			if len(g2.Vertices()) != tc.wantV || len(g2.Edges()) != tc.wantE {
				t.Errorf("idempotence: counts changed after re-run of %s", tc.name)
			}
		})
	}
}
