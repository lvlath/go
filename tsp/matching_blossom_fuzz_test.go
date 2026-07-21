package tsp

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/lvlath/go/matrix"
)

/*func FuzzBlossomMatchesOracleSmall(f *testing.F) {
	for _, seed := range []int64{1, 2, 3, 5, 8, 13, 21, 34, 55, 89} {
		f.Add(uint64(seed), 2)
		f.Add(uint64(seed), 4)
		f.Add(uint64(seed), 6)
		f.Add(uint64(seed), 8)
		f.Add(uint64(seed), 10)
	}

	f.Fuzz(func(t *testing.T, rawSeed uint64, rawK int) {
		k := rawK
		if k < 2 {
			k = 2
		}
		if k > 10 {
			k = 10
		}
		if (k & 1) == 1 {
			k++
		}

		seed := int64(rawSeed)
		problem := internalSeededMatchingProblem(k, seed)

		_, wantCost, err := exactMatchingOracleForTest(problem)
		if err != nil {
			t.Fatalf("oracle: %v", err)
		}

		match, gotCost, stats, err := solveMinimumWeightPerfectMatching(problem, blossomOptions{Eps: DefaultEps})
		if err != nil {
			t.Fatalf("blossom: %v k=%d seed=%d stats=%+v", err, k, seed, stats)
		}
		if err = verifyPerfectMatching(match); err != nil {
			t.Fatalf("verify: %v k=%d seed=%d match=%v stats=%+v", err, k, seed, match, stats)
		}
		if math.Abs(gotCost-wantCost) > DefaultEps {
			t.Fatalf("cost got %.12f want %.12f k=%d seed=%d match=%v stats=%+v",
				gotCost, wantCost, k, seed, match, stats)
		}
	})
}*/

func FuzzChristofidesBlossomMetricComplete(f *testing.F) {
	seeds := []uint64{1, 2, 3, 5, 8, 13, 21, 34, 55, 80, 86, 89, 1001, 9001}
	for _, seed := range seeds {
		for _, n := range []int{4, 8, 16, 32, 48} {
			f.Add(seed, n)
		}
	}

	f.Fuzz(func(t *testing.T, rawSeed uint64, rawN int) {
		n := rawN
		if n < 4 {
			n = 4
		}
		if n > 64 {
			n = 64
		}

		dist := seededMetricCompleteForTest(n, int64(rawSeed))

		opts := DefaultOptions()
		opts.Algo = Christofides
		opts.Symmetric = true
		opts.MatchingAlgo = BlossomMatch
		opts.EnableLocalSearch = false
		opts.StartVertex = int(rawSeed % uint64(n))

		first, err := ChristofidesSolve(dist, opts)
		if err != nil {
			t.Fatalf("first Christofides: %v n=%d seed=%d", err, n, rawSeed)
		}
		if err = validateResultForTest(first, n, opts.StartVertex, Christofides); err != nil {
			t.Fatalf("first result: %v result=%+v", err, first)
		}

		second, err := ChristofidesSolve(dist, opts)
		if err != nil {
			t.Fatalf("second Christofides: %v n=%d seed=%d", err, n, rawSeed)
		}
		if !slices.Equal(first.Tour, second.Tour) {
			t.Fatalf("nondeterministic tour first=%v second=%v n=%d seed=%d", first.Tour, second.Tour, n, rawSeed)
		}
		if first.Cost != second.Cost {
			t.Fatalf("nondeterministic cost got %.12f want %.12f n=%d seed=%d", second.Cost, first.Cost, n, rawSeed)
		}
	})
}

func seededMetricCompleteForTest(n int, seed int64) *matrix.Dense {
	rng := rand.New(rand.NewSource(seed))

	type point struct {
		x float64
		y float64
	}

	points := make([]point, n)
	for vertex := 0; vertex < n; vertex++ {
		points[vertex] = point{
			x: rng.Float64() * 10_000,
			y: rng.Float64() * 10_000,
		}
	}

	data := make([]float64, n*n)
	for row := 0; row < n; row++ {
		for col := row + 1; col < n; col++ {
			dx := points[row].x - points[col].x
			dy := points[row].y - points[col].y
			value := math.Hypot(dx, dy)

			data[row*n+col] = value
			data[col*n+row] = value
		}
	}

	dence, _ := matrix.NewDense(n, n)
	dence.Fill(data)

	return dence
}

func FuzzBlossomMatchesOracleMixedSmall(f *testing.F) {
	seeds := []uint64{1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 1001, 9001}
	for _, seed := range seeds {
		for _, k := range []int{2, 4, 6, 8, 10, 12} {
			for mode := 0; mode < 4; mode++ {
				f.Add(seed, k, mode)
			}
		}
	}

	f.Fuzz(func(t *testing.T, rawSeed uint64, rawK int, rawMode int) {
		k := rawK
		if k < 2 {
			k = 2
		}
		if k > 12 {
			k = 12
		}
		if (k & 1) == 1 {
			k++
		}

		mode := rawMode % 4
		if mode < 0 {
			mode += 4
		}

		var problem matchingProblem
		switch mode {
		case 0:
			problem = internalSeededMatchingProblem(k, int64(rawSeed))
		case 1:
			problem = internalConstantMatchingProblem(k, 1)
		case 2:
			problem = internalWideRangeMatchingProblem(k, int64(rawSeed))
		case 3:
			problem = internalNearTieMatchingProblem(k, int64(rawSeed))
		}

		_, wantCost, err := exactMatchingOracleForTest(problem)
		if err != nil {
			t.Fatalf("oracle: %v", err)
		}

		first, firstCost, firstStats, err := solveMinimumWeightPerfectMatching(problem, blossomOptions{Eps: DefaultEps})
		if err != nil {
			t.Fatalf("first blossom: %v k=%d seed=%d mode=%d stats=%+v", err, k, rawSeed, mode, firstStats)
		}
		if err = verifyPerfectMatching(first); err != nil {
			t.Fatalf("first verify: %v match=%v stats=%+v", err, first, firstStats)
		}
		if math.Abs(firstCost-wantCost) > 1e-7*math.Max(1, math.Abs(wantCost)) {
			t.Fatalf("cost got %.12f want %.12f k=%d seed=%d mode=%d match=%v stats=%+v",
				firstCost, wantCost, k, rawSeed, mode, first, firstStats)
		}

		second, secondCost, secondStats, err := solveMinimumWeightPerfectMatching(problem, blossomOptions{Eps: DefaultEps})
		if err != nil {
			t.Fatalf("second blossom: %v k=%d seed=%d mode=%d stats=%+v", err, k, rawSeed, mode, secondStats)
		}
		if secondCost != firstCost {
			t.Fatalf("nondeterministic cost got %.12f want %.12f k=%d seed=%d mode=%d first=%v second=%v",
				secondCost, firstCost, k, rawSeed, mode, first, second)
		}
		for vertex := range first {
			if second[vertex] != first[vertex] {
				t.Fatalf("nondeterministic match[%d] got %d want %d k=%d seed=%d mode=%d",
					vertex, second[vertex], first[vertex], k, rawSeed, mode)
			}
		}
	})
}

func internalNearTieMatchingProblem(k int, seed int64) matchingProblem {
	rng := rand.New(rand.NewSource(seed))

	odd := make([]int, k)
	w := make([]float64, k*k)

	for vertex := 0; vertex < k; vertex++ {
		odd[vertex] = vertex
	}

	for row := 0; row < k; row++ {
		for col := row + 1; col < k; col++ {
			value := 1000 + float64(rng.Intn(5)) + rng.Float64()*1e-6
			w[row*k+col] = value
			w[col*k+row] = value
		}
	}

	return matchingProblem{
		odd: odd,
		w:   w,
		n:   k,
	}
}

func validateResultForTest(result *Result, n int, start int, algo Algorithm) error {
	if result.Algorithm != algo {
		return fmt.Errorf("algorithm got %v want %v", result.Algorithm, algo)
	}
	if len(result.Tour) != n+1 {
		return fmt.Errorf("tour length got %d want %d", len(result.Tour), n+1)
	}
	if result.Tour[0] != start {
		return fmt.Errorf("tour start got %d want %d", result.Tour[0], start)
	}
	if result.Tour[n] != start {
		return fmt.Errorf("tour closure got %d want %d", result.Tour[n], start)
	}

	if err := ValidatePermutation(result.Tour[:n], n); err != nil {
		return err
	}

	if math.IsNaN(result.Cost) || math.IsInf(result.Cost, 0) || result.Cost < 0 {
		return fmt.Errorf("invalid result cost %.17g", result.Cost)
	}

	return nil
}

func FuzzEulerianCircuitParallelMultigraph(f *testing.F) {
	for _, seed := range []uint64{1, 2, 3, 7, 42, 128, 1001} {
		for _, n := range []int{2, 3, 4, 8, 16} {
			f.Add(seed, n)
		}
	}

	f.Fuzz(func(t *testing.T, rawSeed uint64, rawN int) {
		n := rawN
		if n < 2 {
			n = 2
		}
		if n > 24 {
			n = 24
		}

		adj := seededEulerianMultigraphForTest(n, int64(rawSeed))

		start := int(rawSeed % uint64(n))
		walk, err := EulerianCircuit(adj, start)
		if err != nil {
			t.Fatalf("EulerianCircuit: %v n=%d seed=%d adj=%v", err, n, rawSeed, adj)
		}

		if err = verifyEulerianWalkUsesEveryEdgeForTest(adj, walk, start); err != nil {
			t.Fatalf("verify walk: %v n=%d seed=%d walk=%v adj=%v", err, n, rawSeed, walk, adj)
		}
	})
}

func seededEulerianMultigraphForTest(n int, seed int64) [][]int {
	rng := rand.New(rand.NewSource(seed))

	adj := make([][]int, n)

	// Start with one simple cycle so the non-zero graph is connected.
	for vertex := 0; vertex < n; vertex++ {
		next := (vertex + 1) % n
		adj[vertex] = append(adj[vertex], next)
		adj[next] = append(adj[next], vertex)
	}

	// Add random pairs of parallel undirected edges.
	// Adding two copies preserves even degree at both endpoints.
	extraPairs := n + rng.Intn(3*n)
	for i := 0; i < extraPairs; i++ {
		u := rng.Intn(n)
		v := rng.Intn(n)
		if u == v {
			v = (v + 1) % n
		}

		for copyIndex := 0; copyIndex < 2; copyIndex++ {
			adj[u] = append(adj[u], v)
			adj[v] = append(adj[v], u)
		}
	}

	return adj
}
func verifyEulerianWalkUsesEveryEdgeForTest(adj [][]int, walk []int, start int) error {
	if len(walk) == 0 {
		return fmt.Errorf("empty walk")
	}
	if walk[0] != start || walk[len(walk)-1] != start {
		return fmt.Errorf("walk is not closed at start=%d: %v", start, walk)
	}

	want := make(map[uint64]int)
	edgeCount := 0

	for from := range adj {
		if len(adj[from])&1 == 1 {
			return fmt.Errorf("odd degree vertex=%d degree=%d", from, len(adj[from]))
		}
		for _, to := range adj[from] {
			key := packUndirectedKey(from, to)
			want[key]++
			edgeCount++
		}
	}

	for key, count := range want {
		if count&1 == 1 {
			return fmt.Errorf("non-reciprocal edge key=%d count=%d", key, count)
		}
		want[key] = count / 2
	}

	if len(walk) != edgeCount/2+1 {
		return fmt.Errorf("walk length got %d want %d", len(walk), edgeCount/2+1)
	}

	for i := 0; i+1 < len(walk); i++ {
		u := walk[i]
		v := walk[i+1]
		if u < 0 || u >= len(adj) || v < 0 || v >= len(adj) {
			return fmt.Errorf("walk edge %d has OOB endpoint %d->%d", i, u, v)
		}

		key := packUndirectedKey(u, v)
		if want[key] == 0 {
			return fmt.Errorf("walk uses missing/excess edge %d->%d at index %d", u, v, i)
		}
		want[key]--
	}

	for key, remaining := range want {
		if remaining != 0 {
			return fmt.Errorf("unused edge key=%d remaining=%d", key, remaining)
		}
	}

	return nil
}
func packUndirectedKey(u, v int) uint64 {
	var (
		a = uint64(u)
		b = uint64(v)
	)
	// Order the endpoints to make the key direction-agnostic.
	if a < b {
		return (a << 32) | b
	}

	return (b << 32) | a
}
