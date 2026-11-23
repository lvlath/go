// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test demonstrates independent matching and its composition with TSP.
// Scenarios use deterministic operational inputs and print verifiable invariants.
package match_test

import (
	"errors"
	"fmt"

	"github.com/lvlath/go/match"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// exampleSite identifies one operational asset at integer grid coordinates.
// Coordinates are illustrative kilometer offsets rather than live geocoded positions.
// Implementation:
//   - Store an operational identifier and two fixed spatial coordinates.
//
// Behavior highlights:
//   - The identifiers remain detached from the matching matrix index domain.
//
// Inputs:
//   - id: stable asset label; east, north: kilometer-scale offsets.
//
// Returns:
//   - exampleSite: an immutable-by-convention test fixture value.
//
// Errors:
//   - None; data is constructed from deterministic literals.
//
// Determinism:
//   - Callers enumerate sites in a fixed slice order.
//
// Complexity:
//   - Storage O(1) per site.
//
// Notes:
//   - The scenario is a reproducible dispatch model, not a live GIS data feed.
//
// AI-Hints:
//   - Do not confuse a site label with the original numeric matrix index.
type exampleSite struct {
	id    string
	east  int
	north int
}

// exampleManhattan returns the Manhattan cost between two operational sites.
// The nonnegative symmetric metric is suitable for pairing and route examples.
// Implementation:
//   - Stage 1: Compute both signed coordinate differences.
//   - Stage 2: Sum their absolute distances in kilometer-equivalent units.
//
// Behavior highlights:
//   - The fixed fixture coordinates fit within machine-size integers.
//
// Inputs:
//   - a, b: independent site coordinates in illustrative kilometers.
//
// Returns:
//   - float64: nonnegative symmetric distance; zero only for co-located sites.
//
// Errors:
//   - None for the bounded example inputs.
//
// Determinism:
//   - Integer arithmetic, no randomness or external traffic service.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Road closures and travel-time constraints need a richer domain cost model.
//
// AI-Hints:
//   - Do not interpret this metric as observed fleet travel time.
func exampleManhattan(a, b exampleSite) float64 {
	dx, dy := a.east-b.east, a.north-b.north
	if dx < 0 {
		dx = -dx
	}
	if dy < 0 {
		dy = -dy
	}
	return float64(dx + dy)
}

// exampleDistanceMatrix builds a complete symmetric matrix of site-to-site costs.
// It converts each operational asset's coordinates into one reusable matrix index.
// Implementation:
//   - Stage 1: Allocate a k-by-k Dense matrix.
//   - Stage 2: Evaluate the same deterministic Manhattan metric for every cell.
//   - Stage 3: Return the complete matrix or the first allocation/write error.
//
// Behavior highlights:
//   - Set failures are never dropped; the caller owns the resulting Dense matrix.
//
// Inputs:
//   - sites: ordered, caller-owned assets, read but not modified or retained.
//
// Returns:
//   - *matrix.Dense: complete nonnegative symmetric costs indexed by site positions.
//
// Errors:
//   - Propagated matrix constructor and Set errors.
//
// Determinism:
//   - Row-major traversal in the original site slice order.
//
// Complexity:
//   - Time O(k^2), Space O(k^2).
//
// Notes:
//   - A production service can substitute measured symmetric costs with the same API.
//
// AI-Hints:
//   - Do not replace the matching input subset with physical site IDs.
func exampleDistanceMatrix(sites []exampleSite) (*matrix.Dense, error) {
	m, err := matrix.NewDense(len(sites), len(sites))
	if err != nil {
		return nil, err
	}
	for i := range sites {
		for j := range sites {
			if err := m.Set(i, j, exampleManhattan(sites[i], sites[j])); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

// exampleFleetSites builds a reproducible twenty-node vehicle-handoff schedule.
// Ten named operational zones have a pickup and a nearby relay depot each.
// Implementation:
//   - Stage 1: Enumerate the ordered zone labels.
//   - Stage 2: Place each zone's two sites one kilometer apart.
//
// Behavior highlights:
//   - Cross-zone repositioning is intentionally more expensive than local handoff.
//
// Inputs:
//   - None; the fixture is fixed.
//
// Returns:
//   - []exampleSite: a newly allocated sequence of twenty named assets.
//
// Errors:
//   - None for these fixed literals.
//
// Determinism:
//   - Zone and endpoint order are stable across executions.
//
// Complexity:
//   - Time O(k), Space O(k), for the number of sites.
//
// Notes:
//   - Coordinates are a simplified operations scenario, not a live city map.
//
// AI-Hints:
//   - The unique low-cost pairings are intentional, making Output stable.
func exampleFleetSites() []exampleSite {
	zones := [...]string{
		"airport", "hospital", "old-town", "north-terminal", "seaport",
		"research", "university", "logistics", "central-station", "night-depot",
	}
	sites := make([]exampleSite, 0, len(zones)*2)
	for zone, name := range zones {
		center := zone * 20
		sites = append(sites,
			exampleSite{id: name + "-pickup", east: center, north: 0},
			exampleSite{id: name + "-relay", east: center + 1, north: 0},
		)
	}
	return sites
}

// exampleFleetHandoffScores constructs a complete symmetric dispatch-planning objective.
// Four urgent jobs illustrate a nonmetric compatibility trap: the cheapest
// first handoff can force a very expensive remaining assignment. Eight additional
// independent vehicle groups provide a full twenty-job operations batch.
//
// Implementation:
//   - Stage 1: Allocate a twenty-by-twenty nonnegative score matrix.
//   - Stage 2: Assign the fallback cross-group mismatch penalty to every edge.
//   - Stage 3: Configure eight local work-order pairs and six urgent cross-pairs.
//
// Behavior highlights:
//   - Scores are illustrative dispatch-effort points, not physical kilometers.
//   - The high penalty represents a costly but permitted staffing mismatch.
//   - The resulting matrix is symmetric and complete but need not be metric.
//
// Inputs:
//   - sites: twenty ordered job labels returned by exampleFleetSites.
//
// Returns:
//   - *matrix.Dense: detached full symmetric pairing costs in planning points.
//
// Errors:
//   - Propagates matrix construction and write errors; no partial matrix escapes.
//
// Determinism:
//   - Fixed row-major fill and fixed index-based compatibility overrides.
//
// Complexity:
//   - Time O(n^2), Space O(n^2) for n=20 scheduled jobs.
//
// Notes:
//   - In production, feed audited travel-time and assignment-constraint scores.
//   - This scores disjoint pairs, not a vehicle-routing or shift-coverage solution.
//
// AI-Hints:
//   - Matching does not require triangle inequality; do not metric-close these scores.
//   - A large mismatch penalty is soft, not a forbidden sparse edge.
func exampleFleetHandoffScores(sites []exampleSite) (*matrix.Dense, error) {
	const (
		fleetJobCount    = 20
		urgentJobs       = 4
		localPairScore   = 1.0
		mismatchScore    = 250.0
		unfavorableScore = 100.0
	)
	if len(sites) != fleetJobCount {
		return nil, fmt.Errorf("expected twenty fleet jobs, got %d", len(sites))
	}
	matrixValue, err := matrix.NewDense(len(sites), len(sites))
	if err != nil {
		return nil, err
	}
	// Every pair remains allowed, but unrelated jobs incur a costly reassignment.
	for from := range sites {
		for to := range sites {
			score := mismatchScore
			if from == to {
				score = 0
			}
			if err := matrixValue.Set(from, to, score); err != nil {
				return nil, err
			}
		}
	}
	// These ordinary low-cost job pairs do not obscure the four urgent cases.
	for from := urgentJobs; from < len(sites); from += 2 {
		if err := matrixValue.Set(from, from+1, localPairScore); err != nil {
			return nil, err
		}
		if err := matrixValue.Set(from+1, from, localPairScore); err != nil {
			return nil, err
		}
	}
	// Local urgent identifiers retain the caller's order; only their costs vary.
	// The three urgent perfect matchings score 4, 5, and 101 respectively.
	urgent := []struct {
		from  int
		to    int
		score float64
	}{
		{0, 1, 2}, {0, 2, 2}, {0, 3, 1},
		{1, 2, unfavorableScore}, {1, 3, 2}, {2, 3, 3},
	}
	for _, assignment := range urgent {
		if err := matrixValue.Set(assignment.from, assignment.to, assignment.score); err != nil {
			return nil, err
		}
		if err := matrixValue.Set(assignment.to, assignment.from, assignment.score); err != nil {
			return nil, err
		}
	}
	return matrixValue, nil
}

// ExamplePerfect_autonomousFleetHandoffs compares exact and heuristic scheduling.
// A dispatch operator pairs twenty vehicle-repositioning jobs into ten joint
// assignments while penalizing incompatible driver availability and handoff windows.
//
// Implementation:
//   - Stage 1: Construct named shift jobs and a complete symmetric effort matrix.
//   - Stage 2: Compute minimum-effort Blossom pairs and explicit greedy pairs.
//   - Stage 3: Verify complete coverage and report the cost of greedy myopia.
//
// Behavior highlights:
//   - A local cheap handoff can strand two expensive-to-combine urgent jobs.
//   - Compatibility scores are planning points, not kilometers or travel time.
//   - Exact optimization applies to matching, not an entire dispatch schedule.
//
// Inputs:
//   - None; site names and compatibility scores are reproducible sample data.
//
// Returns:
//   - Printed pair count, exact cost, greedy cost and two actionable handoffs.
//
// Errors:
//   - Matrix and algorithm failures are reported without partial assignments.
//
// Determinism:
//   - Fixed source order, strict deterministic tie rules and canonical pairs.
//
// Complexity:
//   - O(n^2) score preparation plus weighted Blossom and O(n^2) greedy selection.
//
// Notes:
//   - A real scheduler also models temporal capacity and assignment eligibility;
//     those constraints must be reflected in a valid score model or a richer solver.
//
// AI-Hints:
//   - Do not interpret Greedy's completion as minimal total assignment effort.
//   - The large mismatch score is a finite penalty, not an absent graph edge.
func ExamplePerfect_autonomousFleetHandoffs() {
	sites := exampleFleetSites()
	dist, err := exampleFleetHandoffScores(sites)
	if err != nil {
		fmt.Println("fleet scores failed:", err)
		return
	}
	selected := make([]int, len(sites))
	for i := range selected {
		selected[i] = i
	}

	exact, err := match.Perfect(dist, selected)
	if err != nil {
		fmt.Println("fleet optimization failed:", err)
		return
	}
	if exact == nil {
		fmt.Println("unexpected nil result from solver")
		return
	}
	greedy, err := match.GreedyPerfect(dist, selected)
	if err != nil {
		fmt.Println("fleet heuristic failed:", err)
		return
	}
	if greedy == nil {
		fmt.Println("unexpected nil result from greedy solver")
		return
	}
	fmt.Printf("handoffs=%d exact-score=%.0f greedy-score=%.0f avoidable-penalty=%.0f\n",
		len(exact.Pairs), exact.Cost, greedy.Cost, greedy.Cost-exact.Cost)
	for _, pair := range exact.Pairs {
		if pair.U < 4 {
			fmt.Printf("urgent assignment: %s <-> %s\n", sites[pair.U].id, sites[pair.V].id)
		}
	}
	// Output:
	// handoffs=10 exact-score=12 greedy-score=109 avoidable-penalty=97
	// urgent assignment: airport-pickup <-> hospital-pickup
	// urgent assignment: airport-relay <-> hospital-relay
}

// ExamplePerfect_sensorRelayMaintenance assigns twelve radio nodes to pairs.
// The explicit greedy mode creates a feasible maintenance handoff schedule.
// Implementation:
//   - Stage 1: Create twelve named transmit/receive stations and their link costs.
//   - Stage 2: Supply the technician's unsorted priority subset to GreedyPerfect.
//   - Stage 3: Audit each selected link against a technician radio-range policy.
//
// Behavior highlights:
//   - The greedy pivot order is not replaced by global minimum-edge matching.
//
// Inputs:
//   - None; site positions and technician priority are fixed.
//
// Returns:
//   - Printed relay count, total link distance, range compliance and equipment links.
//
// Errors:
//   - Matrix/solver failures are printed; partial matches are never reported.
//
// Determinism:
//   - Original subset ordering drives selection; output endpoints are canonical.
//
// Complexity:
//   - Time O(k^2), Space O(k^2) for matrix input plus greedy admission.
//
// Notes:
//   - Costs are illustrative path lengths; interference and redundancy are not modeled.
//   - Distance is a pairing objective, not a guarantee of radio propagation.
//
// AI-Hints:
//   - Never present Greedy's feasible result as a certified minimum.
//   - A separate radio-range policy must check the returned links before deployment.
func ExamplePerfect_sensorRelayMaintenance() {
	sites := []exampleSite{
		{"plant-a-tx", 0, 0}, {"plant-a-rx", 0, 2},
		{"plant-b-tx", 30, 0}, {"plant-b-rx", 30, 2},
		{"solar-field-tx", 60, 0}, {"solar-field-rx", 60, 2},
		{"cold-store-tx", 90, 0}, {"cold-store-rx", 90, 2},
		{"rail-hub-tx", 120, 0}, {"rail-hub-rx", 120, 2},
		{"water-plant-tx", 150, 0}, {"water-plant-rx", 150, 2},
	}
	dist, err := exampleDistanceMatrix(sites)
	if err != nil {
		fmt.Println("network construction failed:", err)
		return
	}
	// Permutation represents the incident team's priority order, not sorted IDs.
	chosen := []int{11, 4, 9, 0, 7, 2, 10, 5, 8, 1, 6, 3}
	result, err := match.GreedyPerfect(dist, chosen)
	if err != nil {
		fmt.Println("sensor matching failed:", err)
		return
	}
	if result == nil || len(result.Pairs) != len(sites)/2 {
		fmt.Println("sensor matching returned incomplete pairing")
		return
	}
	const maxCertifiedHop = 3.0 // technician-verified grid-distance limit
	withinRange := 0
	for _, pair := range result.Pairs {
		length, readErr := dist.At(pair.U, pair.V)
		if readErr != nil {
			fmt.Println("relay link verification failed:", readErr)
			return
		}
		if length <= maxCertifiedHop {
			withinRange++
		}
	}
	fmt.Printf("relay-pairs=%d total-link-distance=%.0f verified-links=%d/%d heuristic=%v\n",
		len(result.Pairs), result.Cost, withinRange, len(result.Pairs), !result.Optimal)
	// A complete pairing is not a radio-link deployment authorization. Every
	// selected edge must pass the independently measured range policy.
	deploymentAuthorized := withinRange == len(result.Pairs)
	fmt.Printf("deployment-authorized=%v\n", deploymentAuthorized)
	fmt.Printf("first-link: %s <-> %s\n", sites[result.Pairs[0].U].id, sites[result.Pairs[0].V].id)
	fmt.Printf("last-link: %s <-> %s\n", sites[result.Pairs[len(result.Pairs)-1].U].id, sites[result.Pairs[len(result.Pairs)-1].V].id)
	// Output:
	// relay-pairs=6 total-link-distance=12 verified-links=6/6 heuristic=true
	// deployment-authorized=true
	// first-link: plant-a-tx <-> plant-a-rx
	// last-link: water-plant-tx <-> water-plant-rx
}

// ExamplePerfect_maintenanceRouteBridge composes MST, matching and route inspection.
// Twelve facility stations are visited by a depot-anchored closed audit tour.
// Implementation:
//   - Stage 1: Build symmetric station distances and compute a dense MST.
//   - Stage 2: Select its odd-degree endpoints and solve their exact matching.
//   - Stage 3: Add disjoint reciprocal edges, traverse Eulerian topology and shortcut.
//
// Behavior highlights:
//   - Matching solves edge pairing, while TSP owns multigraph mutation and routing.
//
// Inputs:
//   - None; station positions and the depot are fixed operational fixtures.
//
// Returns:
//   - Printed parity counts, MST/matching costs, closed route cost and validity.
//
// Errors:
//   - Every matrix, MST, matching, Eulerian and tour-cost error is checked.
//
// Determinism:
//   - Input station order, MST tie rules and matching ties are unchanged.
//
// Complexity:
//   - O(n^2) matrix/MST work, plus Blossom search, Euler traversal and shortcut.
//
// Notes:
//   - This demonstrates composition; reusable match does not implement Christofides.
//
// AI-Hints:
//   - The formal 1.5 approximation requires metric input and exact MWPM.
func ExamplePerfect_maintenanceRouteBridge() {
	sites := []exampleSite{
		{"depot", 0, 0}, {"junction-a", 2, 0}, {"substation-a", 4, 1},
		{"substation-b", 8, 0}, {"rail-crossing", 11, 3}, {"bridge-south", 13, 0},
		{"pump-north", 1, 9}, {"pump-east", 5, 10}, {"control-room", 8, 11},
		{"reservoir", 13, 9}, {"hill-relay", 5, 16}, {"mountain-gate", 15, 17},
	}
	dist, err := exampleDistanceMatrix(sites)
	if err != nil {
		fmt.Println("sites construction failed:", err)
		return
	}
	mstCost, adjacency, err := tsp.MinimumSpanningTree(dist)
	if err != nil {
		fmt.Println("minimum tree failed:", err)
		return
	}
	odd := make([]int, 0, len(adjacency))
	for i, neighbors := range adjacency {
		if len(neighbors)&1 != 0 {
			odd = append(odd, i)
		}
	}
	result, err := match.Perfect(dist, odd)
	if err != nil {
		fmt.Println("odd-degree matching failed:", err)
		return
	}
	if result == nil {
		fmt.Println("unexpected nil result from solver")
		return
	}
	// The caller owns the TSP multigraph and adds every complete pair exactly once.
	for _, pair := range result.Pairs {
		adjacency[pair.U] = append(adjacency[pair.U], pair.V)
		adjacency[pair.V] = append(adjacency[pair.V], pair.U)
	}
	walk, err := tsp.EulerianCircuit(adjacency, 0)
	if err != nil {
		fmt.Println("Euler circuit failed:", err)
		return
	}
	tour, err := tsp.ShortcutEulerianToHamiltonian(walk, len(sites), 0)
	if err != nil {
		fmt.Println("shortcut failed:", err)
		return
	}
	tourCost, err := tsp.TourCost(dist, tour)
	if err != nil {
		fmt.Println("tour cost failed:", err)
		return
	}
	fmt.Printf("stations=%d odd-vertices=%d matched-edges=%d\n", len(sites), len(odd), len(result.Pairs))
	fmt.Printf("mst-km=%.0f matching-km=%.0f closed-tour-km=%.0f\n", mstCost, result.Cost, tourCost)
	fmt.Printf("valid-cycle=%v start=%s\n", tour[0] == tour[len(tour)-1] && len(tour) == len(sites)+1, sites[tour[0]].id)
	// Output:
	// stations=12 odd-vertices=6 matched-edges=3
	// mst-km=60 matching-km=26 closed-tour-km=84
	// valid-cycle=true start=depot
}

// ExamplePerfect_emptySubset demonstrates valid empty perfect matching.
// An empty subset denotes no selected assets even when the matrix is absent.
// Implementation:
//   - Stage 1: Request the default exact policy with nil source and subset.
//   - Stage 2: Print the empty detached result's count and zero cost.
//
// Behavior highlights:
//   - Successful cardinality-zero pairing does not access matrix.Matrix.
//
// Inputs:
//   - None; nil expresses a zero-length subset.
//
// Returns:
//   - Printed zero pairs and zero matching cost.
//
// Errors:
//   - Unexpected errors are printed rather than ignored.
//
// Determinism:
//   - Constant-time empty fast path.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Nil vertices do not mean a request for every vertex in the matrix.
//
// AI-Hints:
//   - Always pass explicit indices for a nonempty matching request.
func ExamplePerfect_emptySubset() {
	result, err := match.Perfect(nil, nil)
	if err != nil {
		fmt.Println("unexpected error:", err)
		return
	}
	if result == nil {
		fmt.Println("unexpected nil result from solver")
		return
	}
	fmt.Printf("pairs=%d cost=%.0f\n", len(result.Pairs), result.Cost)
	// Output: pairs=0 cost=0
}

// ExamplePerfect_oddSubset demonstrates the odd-subset validation boundary.
// No perfect pairing exists for an odd number of distinct selected vertices.
// Implementation:
//   - Stage 1: Request matching for three original matrix indices.
//   - Stage 2: Check the stable ErrOddVertices sentinel and nil result.
//
// Behavior highlights:
//   - Odd cardinality is rejected before attempting any matrix access.
//
// Inputs:
//   - None; the malformed request is a deterministic law demonstration.
//
// Returns:
//   - Printed nil-result state and errors.Is classification.
//
// Errors:
//   - ErrOddVertices from the pre-admission fast path.
//
// Determinism:
//   - Fixed precondition checks, no solver execution.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The facade never returns a half-matching with an error.
//
// AI-Hints:
//   - Classify with errors.Is rather than comparing text.
func ExamplePerfect_oddSubset() {
	result, err := match.Perfect(nil, []int{7, 2, 1})
	fmt.Printf("nil-result=%v odd-error=%v\n", result == nil, errors.Is(err, match.ErrOddVertices))
	// Output: nil-result=true odd-error=true
}
