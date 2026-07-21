// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
// Package core_test verifies thread-safety of core.Graph under concurrent operations.
//
// Purpose:
//   - Demonstrate race-free behavior under concurrent Add/Remove/Read/Clone.
//   - Enforce the test rule: goroutines never call *testing.T (errors flow via channels).

package core_test

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/lvlath/go/core"
)

// TestConcurrentAddEdge ENSURES concurrent AddEdge is safe and observable via Neighbors.
//
// Implementation:
//   - Stage 1: Create a multi-edge graph to avoid endpoint-collision rejections.
//   - Stage 2: Spawn N goroutines calling AddEdge(from, V{i}, 0).
//   - Stage 3: Collect errors via channel (no *testing.T inside goroutines).
//   - Stage 4: Assert Neighbors(from) length equals N.
//
// Behavior highlights:
//   - Validates internal locking/atomic edge-ID generation under contention.
//   - Validates read API (Neighbors) after concurrent writes.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal if any AddEdge returns error.
//   - Fatal if Neighbors returns error or wrong length.
//
// Determinism:
//   - Deterministic assertions; scheduling nondeterminism is tolerated.
//
// Complexity:
//   - Time O(N) adds + O(N) neighbor enumeration, Space O(N) for channel buffering.
//
// Notes:
//   - The test asserts COUNT, not ordering; ordering determinism is covered elsewhere.
//
// AI-Hints:
//   - Keep N moderate; pair with `go test -race` to validate locking rigorously.
func TestConcurrentAddEdge(t *testing.T) {
	g, _ := core.NewGraph(core.WithMultiEdges())

	var (
		wg    sync.WaitGroup
		errCh chan error
	)

	wg.Add(NConcurrentAdds)
	errCh = make(chan error, NConcurrentAdds)

	var i int
	for i = 0; i < NConcurrentAdds; i++ {
		go func(id int) {
			defer wg.Done()

			_, err := g.AddEdge(VertexX, fmt.Sprintf("V%d", id), Weight0)
			if err != nil {
				errCh <- err
			}
		}(i)
	}

	wg.Wait()
	close(errCh)

	MustAllErrorsNil(t, errCh, "Concurrent AddEdge")

	var (
		nbs []*core.Edge
		err error
	)

	nbs, err = g.Neighbors(VertexX)
	MustErrorNil(t, err, "Neighbors(X)")
	MustEqualInt(t, len(nbs), NConcurrentAdds, "Neighbors(X) length after concurrent AddEdge")
}

// TestConcurrentAddRemoveEdge MIXES AddEdge and RemoveEdge to validate no panics/races.
//
// Implementation:
//   - Stage 1: Create a weighted multi-edge graph.
//   - Stage 2: Ensure a base vertex exists.
//   - Stage 3: For each round:
//   - Spawn one goroutine adding an edge Base->V{i}.
//   - Spawn one goroutine iterating current Edges() and attempting RemoveEdge.
//   - Stage 4: Collect only UNEXPECTED errors via channel.
//
// Behavior highlights:
//   - Accepts ErrEdgeNotFound during removals as a valid interleaving outcome.
//   - Detects unexpected internal-state violations under contention.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal if AddEdge returns error.
//   - Fatal if RemoveEdge returns an unexpected error.
//
// Determinism:
//   - Nondeterministic schedule; deterministic acceptance policy.
//
// Complexity:
//   - Time O(R * (workload)), Space O(R) channel buffering.
//
// Notes:
//   - This is primarily a race/panic detector; run with `-race`.
//
// AI-Hints:
//   - Keep the “acceptable error set” explicit; never silently ignore unknown errors.
func TestConcurrentAddRemoveEdge(t *testing.T) {
	g, _ := core.NewGraph(core.WithWeighted(), core.WithMultiEdges())

	MustErrorNil(t, g.AddVertex(VertexBase), "AddVertex(Base)")

	var (
		wg    sync.WaitGroup
		errCh chan error
	)

	wg.Add(2 * NConcurrentRounds)
	errCh = make(chan error, 2*NConcurrentRounds)

	var i int
	for i = 0; i < NConcurrentRounds; i++ {
		go func(id int) {
			defer wg.Done()

			_, err := g.AddEdge(VertexBase, fmt.Sprintf("V%d", id), float64(id))
			if err != nil {
				errCh <- err
			}
		}(i)

		go func() {
			defer wg.Done()

			var e *core.Edge
			for _, e = range g.Edges() {
				err := g.RemoveEdge(e.ID)
				if err == nil {
					continue
				}
				if errors.Is(err, core.ErrEdgeNotFound) {
					continue
				}
				errCh <- err
			}
		}()
	}

	wg.Wait()
	close(errCh)

	MustAllErrorsNil(t, errCh, "Concurrent Add/Remove")
}

// TestConcurrentNeighborsAndClone VALIDATES concurrent Neighbors and Clone do not race.
//
// Implementation:
//   - Stage 1: Create loop-enabled multi-edge weighted graph.
//   - Stage 2: Add NLoops self-loops on A.
//   - Stage 3: Spawn NReaders goroutines calling Neighbors(A) and verifying length.
//   - Stage 4: Spawn NCloners goroutines calling Clone().
//   - Stage 5: Report all errors via channel.
//
// Behavior highlights:
//   - Ensures read-heavy workloads remain safe under clone pressure.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal if Neighbors/Clone triggers errors or length mismatch.
//
// Determinism:
//   - Schedule nondeterministic; contract checks deterministic.
//
// Complexity:
//   - Time O(NReaders*NLoops) neighbor enumeration, Space O(NReaders+NCloners).
//
// Notes:
//   - Neighbors length equality assumes loops are represented once in adjacency for self-loop vertex.
//
// AI-Hints:
//   - If internal representation changes, update ONLY the contract statement, not the test harness style.
func TestConcurrentNeighborsAndClone(t *testing.T) {
	g, _ := core.NewGraph(core.WithWeighted(), core.WithMultiEdges(), core.WithLoops())

	var (
		i   int
		err error
	)

	for i = 0; i < NLoops; i++ {
		_, err = g.AddEdge(VertexA, VertexA, float64(i))
		MustErrorNil(t, err, "AddEdge(A,A,w) setup")
	}

	var (
		wg    sync.WaitGroup
		errCh chan error
	)

	wg.Add(NReaders + NCloners)
	errCh = make(chan error, NReaders+NCloners)

	for i = 0; i < NReaders; i++ {
		go func() {
			defer wg.Done()

			nbs, err := g.Neighbors(VertexA)
			if err != nil {
				errCh <- err
				return
			}
			if len(nbs) != NLoops {
				errCh <- fmt.Errorf("Neighbors(A) length mismatch: got=%d want=%d", len(nbs), NLoops)
				return
			}
		}()
	}

	for i = 0; i < NCloners; i++ {
		go func() {
			defer wg.Done()

			_ = g.Clone()
		}()
	}

	wg.Wait()
	close(errCh)

	MustAllErrorsNil(t, errCh, "Concurrent Neighbors/Clone")
}

// TestGraph_AtomicEdgeIDs ASSERTS concurrent AddEdge yields unique IDs.
//
// Implementation:
//   - Stage 1: Create feature-rich graph (multi-edge enabled).
//   - Stage 2: Spawn NAtomicEdgeIDs goroutines adding edges A->B with varying weights.
//   - Stage 3: Goroutines send errors/IDs to channels (no *testing.T inside goroutines).
//   - Stage 4: Assert no errors, and set size equals NAtomicEdgeIDs.
//
// Behavior highlights:
//   - Locks in uniqueness property of edge IDs under contention.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal if any AddEdge fails or uniqueness is violated.
//
// Determinism:
//   - Schedule nondeterministic; uniqueness assertion deterministic.
//
// Complexity:
//   - Time O(N), Space O(N).
//
// Notes:
//   - This test does not assert the *format* of IDs (only uniqueness/non-emptiness).
//
// AI-Hints:
//   - If you later formalize ID format, extend this test with a parser and pattern assertions.
func TestGraph_AtomicEdgeIDs(t *testing.T) {
	g := NewGraphFull(t)

	idCh := make(chan string, NAtomicEdgeIDs)
	errCh := make(chan error, NAtomicEdgeIDs)

	var wg sync.WaitGroup
	wg.Add(NAtomicEdgeIDs)

	var i int
	for i = 0; i < NAtomicEdgeIDs; i++ {
		go func(i int) {
			defer wg.Done()

			eid, err := g.AddEdge(VertexA, VertexB, float64(i))
			if err != nil {
				errCh <- err
				return
			}
			if eid == "" {
				errCh <- fmt.Errorf("empty edge ID returned")
				return
			}
			idCh <- eid
		}(i)
	}

	wg.Wait()
	close(idCh)
	close(errCh)

	MustAllErrorsNil(t, errCh, "Atomic edge IDs")

	ids := make(map[string]struct{}, NAtomicEdgeIDs)

	for eid := range idCh {
		ids[eid] = struct{}{}
	}

	MustEqualInt(t, len(ids), NAtomicEdgeIDs, "unique edge IDs count")
}

// TestGraph_HasVertexConcurrency ASSERTS concurrent HasVertex/AddVertex does not panic.
//
// Implementation:
//   - Stage 1: Create graph.
//   - Stage 2: Spawn M goroutines adding vertices and M goroutines reading HasVertex.
//   - Stage 3: Wait; test passes if no panic.
//
// Behavior highlights:
//   - This is a race/panic detector; validate with `go test -race`.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal only if a panic occurs (implicit).
//
// Determinism:
//   - Nondeterministic schedule; expected stability.
//
// Complexity:
//   - Time O(M), Space O(1) extra.
//
// Notes:
//   - This test intentionally does not assert final counts: it targets safety, not outcome.
//
// AI-Hints:
//   - Keep this test lightweight; rely on -race to detect unsynchronized access.
func TestGraph_HasVertexConcurrency(t *testing.T) {
	g := NewGraphFull(t)

	const M = 50

	var wg sync.WaitGroup
	wg.Add(2 * M)

	var i int
	for i = 0; i < M; i++ {
		go func(i int) {
			defer wg.Done()
			_ = g.AddVertex(fmt.Sprintf("V%d", i))
		}(i)

		go func(i int) {
			defer wg.Done()
			_ = g.HasVertex(fmt.Sprintf("V%d", i))
		}(i)
	}

	wg.Wait()
}
