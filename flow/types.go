// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package flow

import (
	"context"

	"github.com/lvlath/go/core"
)

// Algorithm identifies the maximum-flow kernel used by the canonical facade.
//
// Implementation:
//   - Algorithm values are stable strings so logs, tests, and examples can print
//     them without depending on integer enum ordering.
//   - MaxFlow validates Algorithm through WithAlgorithm before selecting a kernel.
//
// Behavior highlights:
//   - AlgorithmDinic is the default canonical algorithm.
//   - Legacy wrappers force the corresponding Algorithm explicitly.
//
// Determinism:
//   - Algorithm selection is explicit; no runtime heuristic changes the kernel.
//
// AI-Hints:
//   - Do not reorder or rename constants casually; persisted diagnostics may use them.
type Algorithm string

const (
	AlgorithmDinic         Algorithm = "dinic"
	AlgorithmEdmondsKarp   Algorithm = "edmonds_karp"
	AlgorithmFordFulkerson Algorithm = "ford_fulkerson"
)

// MaxFlowResult is the canonical result artifact for maximum-flow computations.
//
// Implementation:
//   - Stage 1: The selected kernel computes Value over an internal residualNetwork.
//   - Stage 2: Finalization publishes a detached directed weighted Residual graph.
//   - Stage 3: Min-cut sides are derived from the final residual network.
//
// Behavior highlights:
//   - Residual is never the input graph and is safe to mutate independently.
//   - CutSourceSide and CutSinkSide are sorted, caller-owned snapshots.
//   - Partial marks cancellation or observer interruption after some flow was pushed.
//
// Inputs:
//   - Source: source vertex ID used by the run.
//   - Sink: sink vertex ID used by the run.
//
// Returns:
//   - Value: maximum flow value accumulated before termination.
//   - Residual: final residual graph on success; may be nil on partial interruption.
//
// Errors:
//   - Helper methods return ErrNilResult or ErrNoResidual.
//
// Determinism:
//   - Cut sides follow core.Vertices() lexical order preserved by residualNetwork.
//
// Complexity:
//   - Result helper methods are O(1) except ResidualClone, which is O(V+E_res).
//
// Notes:
//   - Partial=true does not mean Value is wrong; it means the run did not publish
//     the full success certificate.
//
// AI-Hints:
//   - Do not replace this artifact with tuple returns in canonical APIs.
//   - Do not expose internal residual maps as the public result contract.
type MaxFlowResult struct {
	Value float64

	Source string
	Sink   string

	Algorithm Algorithm

	Residual *core.Graph

	CutSourceSide []string
	CutSinkSide   []string

	Augmentations int
	Partial       bool
}

// IsNil reports whether the receiver is nil.
//
// Implementation:
//   - Stage 1: Compare the receiver pointer to nil.
//
// Behavior highlights:
//   - This mirrors the nil-state convention used across lvlath result carriers.
//
// Returns:
//   - bool: true for nil receiver, false otherwise.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure O(1) predicate.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Keep this method panic-free; nil-result helpers rely on it.
func (r *MaxFlowResult) IsNil() bool {
	return r == nil
}

// ResidualClone returns a detached clone of the residual graph.
//
// Implementation:
//   - Stage 1: Validate the result receiver.
//   - Stage 2: Validate that a residual certificate was published.
//   - Stage 3: Delegate to core.Graph.Clone for detached topology copying.
//
// Behavior highlights:
//   - The returned graph can be mutated without affecting the result.
//
// Inputs:
//   - receiver: MaxFlowResult produced by MaxFlow or a legacy wrapper.
//
// Returns:
//   - *core.Graph: detached residual graph clone.
//
// Errors:
//   - ErrNilResult when the receiver is nil.
//   - ErrNoResidual when the result has no residual graph.
//
// Determinism:
//   - Clone preserves core deterministic graph inventory semantics.
//
// Complexity:
//   - Time O(V+E_res), Space O(V+E_res).
//
// Notes:
//   - Vertex metadata follows core.Clone semantics.
//
// AI-Hints:
//   - Use this helper when caller code needs to mutate or annotate residual output.
func (r *MaxFlowResult) ResidualClone() (*core.Graph, error) {
	if r == nil {
		return nil, ErrNilResult
	}
	if r.Residual == nil {
		return nil, ErrNoResidual
	}

	return r.Residual.Clone(), nil
}

// newPartialResult creates a partial result after cancellation or observer failure.
// It intentionally does not publish a residual graph or min-cut certificate.
//
// Implementation:
//   - Stage 1: Copy stable run metadata.
//   - Stage 2: Preserve the flow value accumulated before interruption.
//   - Stage 3: Mark Partial=true and leave Residual/Cut fields empty.
//
// Behavior highlights:
//   - Partial result value is meaningful as "flow pushed so far".
//   - Residual is nil because final certificate publication did not complete.
//
// Inputs:
//   - source: source vertex ID.
//   - sink: sink vertex ID.
//   - algorithm: active kernel.
//   - value: flow accumulated before interruption.
//   - augmentations: successful augmentation count before interruption.
//
// Returns:
//   - *MaxFlowResult: partial result artifact.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure construction; no traversal.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Callers must check Partial before treating result as an optimality certificate.
//
// AI-Hints:
//   - Do not publish min-cut sides on partial results; they may not be optimal.
func newPartialResult(
	source, sink string,
	algorithm Algorithm,
	value float64,
	augmentations int,
) *MaxFlowResult {
	return &MaxFlowResult{
		Value:         value,
		Source:        source,
		Sink:          sink,
		Algorithm:     algorithm,
		Augmentations: augmentations,
		Partial:       true,
	}
}

// Options configures legacy maximum-flow wrappers.
//
// Implementation:
//   - Stage 1: Legacy wrappers convert Options into canonical Option values.
//   - Stage 2: Canonical option assembly validates numeric and domain policy.
//
// Behavior highlights:
//   - Prefer MaxFlow with Option values for new code.
//   - Verbose is retained for compatibility and maps to an observer-style event path.
//
// Inputs:
//   - Ctx: optional cancellation context.
//   - Epsilon: capacity threshold; values <= epsilon are treated as absent.
//   - Verbose: compatibility logging switch.
//   - LevelRebuildInterval: Dinic-specific rebuild interval.
//
// Errors:
//   - Invalid Epsilon or interval are reported through canonical option validation.
//
// Determinism:
//   - Options do not change traversal ordering; residualNetwork owns ordering.
//
// Complexity:
//   - Option conversion is O(1).
//
// Notes:
//   - This type is maintained for source compatibility with older callers.
//
// AI-Hints:
//   - Do not add new contract-changing fields here; add canonical WithXxx options.
type Options struct {
	// Ctx is the context used to cancel or timeout long-running computations.
	// It is normalized to context.Background() if left nil.
	Ctx context.Context

	// Epsilon defines the minimum significant capacity when aggregating parallel edges.
	// Any edge whose total capacity ≤ Epsilon is ignored in the initial residual map.
	// Algorithms then manipulate int64 capacities exactly without further rounding.
	Epsilon float64

	// Verbose enables step-by-step logging of each augmenting path and flow push.
	Verbose bool

	// LevelRebuildInterval controls how often (in augmentation count) Dinic rebuilds
	// its level graph. A value of 0 means "never rebuild" until the next BFS naturally.
	LevelRebuildInterval int

	// MaxAugmentations limits successful augmenting pushes for safety-sensitive runs.
	// Zero means unlimited. This is primarily useful for Ford-Fulkerson on real-valued
	// capacity networks where DFS path choice can be a poor practical strategy.
	MaxAugmentations int
}

// DefaultOptions returns legacy options populated with safe defaults.
//
// Implementation:
//   - Stage 1: Set context.Background as cancellation root.
//   - Stage 2: Set defaultEpsilon for capacity filtering.
//
// Behavior highlights:
//   - New code should prefer MaxFlow with no options or explicit WithXxx options.
//
// Returns:
//   - Options: compatibility configuration.
//
// Errors:
//   - None.
//
// Determinism:
//   - Returns the same value every call.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Keep this as a legacy bridge; canonical defaults live in options.go.
func DefaultOptions() Options {
	return Options{
		Ctx:                  context.Background(),
		Epsilon:              defaultEpsilon,
		Verbose:              false,
		LevelRebuildInterval: 0,
		MaxAugmentations:     0,
	}
}
