// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap declares the resampling policies, immutable plan values,
// and detached result artifacts shared by its block-bootstrap algorithms.
package bootstrap

// Kind selects the mathematical transition law for an index-resampling plan.
// It is a discrete policy: changing the value changes the bootstrap distribution.
//
// Implementation:
//   - Stage 1: The plan validator accepts exactly the three defined values.
//   - Stage 2: The sampler dispatches to the corresponding block kernel.
//   - Stage 3: Only fully populated index rows enter the public result.
//
// Behavior highlights:
//   - The zero value is MovingBlocks, not a request to auto-detect a method.
//   - Any other value is rejected, rather than falling back to independent draws.
//
// Inputs:
//   - Plan.Kind: one of MovingBlocks, CircularBlocks or StationaryBlocks.
//
// Returns:
//   - An exact policy selector; it owns no input or output memory.
//
// Errors:
//   - ErrInvalidOption: an unrecognized Kind is rejected before allocation.
//
// Determinism:
//   - Each Kind fixes its own order of block starts, continuations and RNG draws.
//
// Complexity:
//   - Time O(1), Space O(1) for selecting and validating a value.
//
// Notes:
//   - All methods require 1 <= BlockLength <= rows, including CircularBlocks.
//
// AI-Hints:
//   - Do not reinterpret a missing Kind as StationaryBlocks or an IID shuffle.
//   - Set Kind explicitly when the statistical model depends on block topology.
//
// Kind is stored as uint8; the API is not a bit-mask or flags collection.
type Kind uint8

const (
	// MovingBlocks draws nonwrapping contiguous index blocks of fixed size B.
	// Each independent start is uniform on [0,rows-B]; the final block is
	// truncated to the remaining destination size, never wrapped.
	// AI-Hints: The source-boundary restriction changes start probabilities.
	MovingBlocks Kind = iota

	// CircularBlocks draws fixed-length index blocks on a cyclic source.
	// Every start is uniform on [0,rows); within a block, rows-1 precedes 0.
	// AI-Hints: One start is drawn per block, not per output index.
	CircularBlocks

	// StationaryBlocks draws a stationary uniformly started index chain.
	// Each later step restarts with exact probability 1/B; otherwise it
	// advances cyclically. Its geometric run length has mean B, not maximum B.
	// AI-Hints: B=1 samples IID indices without consuming Bernoulli words.
	StationaryBlocks
)

// Plan describes the complete, read-only specification of index resampling.
// Its integer dimensions count observations or replicates, never bytes or time.
//
// Implementation:
//   - Stage 1: Validate Kind and strictly positive dimensions against rows.
//   - Stage 2: Reserve a checked budget for Replicates independent output rows.
//   - Stage 3: Derive each row from Seed and its zero-based replicate number.
//
// Behavior highlights:
//   - Length is the length of each generated row and may differ from rows.
//   - BlockLength is fixed for Moving/Circular and is the mean run length for Stationary.
//   - Zero Seed is valid and does not trigger a time-based or random default.
//
// Inputs:
//   - Kind: exact sampling policy; defaults to MovingBlocks by enum definition.
//   - Length: required generated observations per row; must be at least 1.
//   - BlockLength: fixed or mean block length; in [1, rows].
//   - Replicates: number of independent logical streams; must be at least 1.
//   - Seed: explicit unsigned stream root, including the valid value 0.
//
// Returns:
//   - A value-only specification; callers retain ownership of their Plan variable.
//
// Errors:
//   - ErrInvalidOption for invalid Kind/dimensions, before size arithmetic.
//   - ErrOverflow for unrepresentable index-buffer sizes.
//   - ErrBudget for a representable request above MaxResultBytes.
//
// Determinism:
//   - Replicate r is keyed by (Seed, uint64(r)); increasing Replicates preserves a prefix.
//
// Complexity:
//   - Time O(1), Space O(1) to copy a Plan; sampling costs O(Replicates*Length)
//     expected time and O(Replicates*Length) output space.
//
// Notes:
//   - There are no implicit dimension defaults or automatic block-length estimators.
//   - Input observations are identified by integer positions; their values are never read.
//
// AI-Hints:
//   - Choose BlockLength from training history, not from held-out outcomes.
//   - Do not interpret StationaryBlocks.BlockLength as a maximum run length.
type Plan struct {
	// Kind fixes the exact block transition law; its zero value means MovingBlocks.
	Kind Kind

	// Length is the strictly positive number of output indices in each replicate.
	Length int

	// BlockLength is the fixed block size or stationary mean, in [1, rows].
	BlockLength int

	// Replicates is the strictly positive number of independent logical index rows.
	Replicates int

	// Seed identifies the root RNG state; zero is a legitimate reproducible value.
	Seed uint64
}

// AlgorithmID records the exact pairing of a versioned RNG stream derivation
// and the bootstrap integer-mapping/consumption rules used for a result.
//
// Behavior highlights:
//   - A changed stream algorithm, mapping or draw order requires a changed ID.
//   - The ID is metadata, not a secret, random seed or user-chosen run identifier.
//
// Determinism:
//   - Fixed by the implementation version, not by the number of replicates.
//
// Complexity:
//   - Time O(1), Space O(1) to read the string value.
//
// AI-Hints:
//   - Persist RNG along with Seed and Plan when results must be replayed later.
type AlgorithmID string

// Result carries only completed, caller-owned bootstrap-index rows and provenance.
// It never references a caller's observations or any mutable RNG workspace.
//
// Implementation:
//   - Stage 1: Allocate a bounded outer result slice before generating rows.
//   - Stage 2: Build each row separately, without publishing an unfinished row.
//   - Stage 3: Append only after success and synchronize Completed with len(Indices).
//
// Behavior highlights:
//   - Success: Completed == requested Replicates, with Length entries in every row.
//   - Mid-computation cancellation: Completed may be zero, and all retained rows are complete.
//   - Zero-completion partial results have a non-nil, zero-length Indices slice.
//
// Inputs:
//   - Produced exclusively by the canonical sampler; no input data are retained.
//
// Returns:
//   - Indices: detached, mutable [][]int; every index is in [0, rows).
//   - Completed: exact count of published rows, equal to len(Indices).
//   - RNG: algorithm identity for replay and audit.
//
// Errors:
//   - A non-nil Result may be accompanied by context.Canceled or
//     context.DeadlineExceeded; such results must never be read as full success.
//
// Determinism:
//   - Rows are published in ascending logical replicate index order.
//
// Complexity:
//   - Storage O(Completed*Length + Completed), including row headers.
//
// Notes:
//   - Caller may modify Indices after return, but this destroys replay fidelity.
//   - Concurrent mutation of a returned Result requires caller synchronization.
//
// AI-Hints:
//   - Always check error before relying on the requested number of rows.
//   - Do not treat a non-nil Result as evidence that sampling completed successfully.
type Result struct {
	// Indices contains exactly Completed fully populated detached rows.
	// On cancellation before the first publication it is allocated but empty.
	Indices [][]int

	// Completed counts published rows and is always equal to len(Indices).
	Completed int

	// RNG identifies the generator version and bootstrap integer-mapping revision.
	RNG AlgorithmID
}

// IsNil reports whether the Result pointer receiver is nil, including a typed nil
// stored behind core.Nilable or another interface requiring IsNil() bool.
//
// Implementation:
//   - Stage 1: Compare the receiver directly with nil without dereferencing it.
//   - Stage 2: Return the exact boolean outcome.
//
// Behavior highlights:
//   - A non-nil zero-value Result is not nil, even if it has no sample rows.
//
// Inputs:
//   - r: pointer receiver, which may be nil.
//
// Returns:
//   - bool: true exactly when r == nil.
//
// Errors:
//   - None; no receiver fields are accessed.
//
// Determinism:
//   - Constant-time pointer identity comparison, with no global state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Nilability is distinct from structural validation or completion status.
//
// AI-Hints:
//   - Check Completed and the returned error; IsNil is not a success predicate.
func (r *Result) IsNil() bool {
	return r == nil
}

// MaxResultBytes is the upper bound for logical index storage and row headers.
// The budget is 256 MiB, checked against the native int and slice-header sizes.
//
// Behavior highlights:
//   - Includes Replicates*Length*sizeof(int) plus Replicates*sizeof([]int).
//   - Does not promise a process-RSS cap or survival of unrelated system OOM.
//
// Complexity:
//   - Time O(1), Space O(1) to read; validation uses checked O(1) arithmetic.
//
// AI-Hints:
//   - Do not silently increase this ceiling or exclude outer row headers.
const MaxResultBytes = 256 * 1024 * 1024
