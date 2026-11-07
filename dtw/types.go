// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File types.go defines DTW domain types and public result carriers.
// Algorithm execution lives in api.go and impl_dtw.go; this file owns the type contract.
package dtw

import (
	"math"

	"github.com/lvlath/go/matrix"
)

// MemoryMode controls how much accumulated-cost state the DTW kernel stores.
//
// What:
//   - MemoryMode selects the storage policy for the accumulated dynamic-programming state.
//   - It does not change the mathematical recurrence.
//
// Implementation:
//   - Stage 1: options are validated and finalized before allocation.
//   - Stage 2: TwoRows and NoMemory use rolling rows for distance computation.
//   - Stage 3: FullMatrix stores accumulated costs and enables path reconstruction.
//
// Behavior highlights:
//   - NoMemory is retained as a legacy compatibility alias for rolling-row distance mode.
//   - FullMatrix is required when the caller asks for path reconstruction or accumulated output.
//   - Exact DTW distance still needs a rolling frontier; NoMemory is not an O(1) exact kernel.
//
// Inputs:
//   - Used through Options.MemoryMode or WithMemoryMode.
//
// Returns:
//   - Stored in Result.MemoryMode as the finalized execution mode.
//
// Errors:
//   - Invalid modes passed through WithMemoryMode return ErrBadInput.
//   - Legacy ReturnPath with non-FullMatrix returns ErrPathNeedsMatrix.
//
// Determinism:
//   - The selected mode does not change row-major traversal or distance values.
//
// Complexity:
//   - TwoRows/NoMemory: Time O(n*m), Space O(m) for scalar orientation-preserving DP.
//   - FullMatrix: Time O(n*m), Space O(n*m).
//
// Notes:
//   - A future symmetric-cost optimization may reduce rolling-row storage to O(min(n,m)).
//   - That optimization must not be enabled for arbitrary asymmetric CostFunc without an explicit contract.
//
// AI-Hints:
//   - Do not document NoMemory as O(1) exact DTW unless a real O(1) exact kernel exists.
//   - Do not allow ReturnPath without FullMatrix in the legacy Options path.
type MemoryMode int

const (
	// NoMemory is a deprecated compatibility alias for rolling-row distance mode.
	//
	// Deprecated: use TwoRows for distance-only DTW.
	NoMemory MemoryMode = iota

	// TwoRows stores only the previous and current DP rows and returns distance-only results.
	TwoRows

	// FullMatrix stores the accumulated DP matrix and enables path reconstruction.
	FullMatrix
)

// Coord is one zero-based point in an optimal DTW warping path.
//
// What:
//   - I indexes sequence A.
//   - J indexes sequence B.
//
// Behavior highlights:
//   - A valid returned path starts at {0,0} and ends at {len(a)-1,len(b)-1}.
//   - Consecutive path steps are monotone and use one of: diagonal, vertical, horizontal.
//
// Complexity:
//   - Coord is O(1) storage.
type Coord struct {
	I int
	J int
}

// Path is an ordered DTW warping path from the first aligned pair to the last.
//
// What:
//   - Path stores the selected representative optimal path.
//   - The package currently returns one deterministic path, not all optimal paths.
//
// Ownership:
//   - Returned Path values are detached; callers may mutate them.
//
// Determinism:
//   - The forward recurrence records diagonal, then up, then left on
//     strict binary64 ties; backtracking follows those stored codes.
//
// Complexity:
//   - Length is at most len(a)+len(b)-1 for the standard step set.
type Path []Coord

// CostFunc computes one finite non-negative local cost for a scalar pair.
// Its values define the additive objective of the existing DTW recurrence.
//
// Implementation:
//   - Stage 1: Align selects the configured callback after applying options.
//   - Stage 2: the kernel evaluates each required pair once and validates its cost.
//   - Stage 3: it reuses the validated value for dynamic programming and artifacts.
//
// Behavior highlights:
//   - Without LocalCost output only admitted window cells invoke the callback.
//   - With WithReturnLocalCost(true), every n-by-m cell is evaluated once,
//     including off-band cells, and all local values must be representable.
//
// Inputs:
//   - ai, bj: immutable scalar sample values; interpretation is caller-defined.
//
// Returns:
//   - float64: finite non-negative local cost in the chosen metric units.
//   - error: caller failure propagated unchanged or package numeric validation.
//
// Errors:
//   - ErrNaNInf when the returned cost is NaN or Inf.
//   - ErrNegativeCost when the returned cost is negative.
//   - User-defined callback errors are propagated without replacing their identity.
//
// Determinism:
//   - The callback must be pure and deterministic for equal input sample pairs.
//
// Complexity:
//   - O(n*m) invocations with a full local artifact, or one per admitted cell
//     otherwise; callback runtime is paid separately for each invocation.
//
// Notes:
//   - No callback is invoked during path reconstruction. Concurrent mutation
//     of caller state read by the callback is unsupported.
//
// AI-Hints:
//   - Do not use callback invocation order as a business-state protocol.
//   - Use AbsoluteCost or SquaredCost unless a different metric is required.
type CostFunc func(ai, bj float64) (float64, error)

// Result is the canonical DTW result artifact.
//
// What:
//   - Result carries the distance, reachability, optional path, and optional matrix artifacts.
//   - It is the primary public result surface for Align.
//
// Implementation:
//   - Stage 1: Align validates options and inputs.
//   - Stage 2: the kernel computes the classified row-major DP recurrence
//     and records incoming predecessors when requested.
//   - Stage 3: optional path and matrix artifacts are attached as detached state.
//
// Behavior highlights:
//   - Reachable=false means the endpoint is structurally blocked by Window;
//     Distance=+Inf in that case, not after a failed numeric calculation.
//   - An unrepresentable terminal optimum returns nil with ErrOverflow.
//   - Path is populated only when PathTracked=true and Reachable=true.
//   - Accumulated is present only with WithReturnAccumulated(true); its numeric
//     +Inf cells do not distinguish overflowed from unreachable intermediates.
//   - LocalCost is a complete n-by-m artifact, including off-band cells, only
//     when WithReturnLocalCost(true) was requested.
//
// Inputs:
//   - Produced by Align, AlignCostMatrix, and AlignMatrix.
//
// Returns:
//   - Caller owns Path and may mutate it.
//   - Matrix artifacts are detached *matrix.Dense instances when present.
//
// Errors:
//   - Result helper methods return ErrNilResult, ErrNoPath, or ErrPathNotTracked.
//
// Determinism:
//   - Path order is start-to-end.
//   - Strict forward predecessor ties prefer diagonal, then up, then left;
//     reconstructed paths follow those codes without cost subtraction.
//
// Complexity:
//   - Result storage is O(1) without path/matrices.
//   - Path storage is O(n+m).
//   - Accumulated or complete local-cost artifact storage is O(n*m).
//
// Nilability:
//   - Methods on nil *Result never panic.
//
// AI-Hints:
//   - Do not expose internal DP buffers directly.
//   - A +Inf accumulated cell alone is not a structural reachability witness.
//   - Use PathOrError instead of checking Path nilness when the distinction matters.
type Result struct {
	Distance  float64
	Reachable bool

	Path        Path
	PathTracked bool

	Window       int
	SlopePenalty float64
	MemoryMode   MemoryMode

	Accumulated *matrix.Dense
	LocalCost   *matrix.Dense
}

// IsNil reports whether r is nil.
//
// What:
//   - IsNil implements explicit nil-state behavior for Result.
//
// Returns:
//   - true when r == nil.
//   - false otherwise.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure pointer check.
//
// Complexity:
//   - Time O(1), Space O(1).
func (r *Result) IsNil() bool {
	return r == nil
}

// IsFinite reports whether the result exists and Distance is finite.
//
// What:
//   - IsFinite is a compact predicate for successful finite-distance alignments.
//
// Returns:
//   - true when r is non-nil and Distance is neither NaN nor ±Inf.
//   - false for nil results, no-path +Inf results, and invalid numeric states.
//
// Errors:
//   - None.
//
// Complexity:
//   - Time O(1), Space O(1).
func (r *Result) IsFinite() bool {
	return r != nil && !math.IsNaN(r.Distance) && !math.IsInf(r.Distance, 0)
}

// PathOrError returns a detached path or a sentinel-classified failure.
//
// What:
//   - PathOrError distinguishes nil result, unreachable result, and disabled path tracking.
//
// Returns:
//   - Path: detached caller-owned copy of the selected path.
//
// Errors:
//   - ErrNilResult when r is nil.
//   - ErrNoPath when the alignment is unreachable.
//   - ErrPathNotTracked when path tracking was disabled.
//
// Determinism:
//   - Preserves the deterministic path order computed by the kernel.
//
// Complexity:
//   - Time O(k), Space O(k), where k=len(r.Path).
//
// AI-Hints:
//   - Do not teach callers to treat nil Path as the only no-path signal.
//   - Reachability and tracking-disabled are different states.
func (r *Result) PathOrError() (Path, error) {
	if r == nil {
		return nil, ErrNilResult
	}
	if !r.Reachable {
		return nil, ErrNoPath
	}
	if !r.PathTracked {
		return nil, ErrPathNotTracked
	}

	out := make(Path, len(r.Path))
	copy(out, r.Path)

	return out, nil
}

// Clone returns a detached copy of the Result.
//
// What:
//   - Clone copies slice and matrix artifacts so callers can mutate the clone independently.
//
// Returns:
//   - *Result: detached result copy.
//   - nil when r is nil.
//
// Errors:
//   - None. Matrix Clone is expected to return a matrix.Dense-compatible snapshot.
//
// Complexity:
//   - Time O(k + A + C), where k is path length, A accumulated cells, C local-cost cells.
//   - Space O(k + A + C).
//
// Nilability:
//   - Calling Clone on nil returns nil.
//
// AI-Hints:
//   - Do not downgrade this to a shallow copy; Result ownership is part of the API contract.
func (r *Result) Clone() *Result {
	if r == nil {
		return nil
	}

	cp := *r

	if r.Path != nil {
		cp.Path = make(Path, len(r.Path))
		copy(cp.Path, r.Path)
	}

	if r.Accumulated != nil {
		if cloned, ok := r.Accumulated.Clone().(*matrix.Dense); ok {
			cp.Accumulated = cloned
		} else {
			cp.Accumulated = nil
		}
	}

	if r.LocalCost != nil {
		if cloned, ok := r.LocalCost.Clone().(*matrix.Dense); ok {
			cp.LocalCost = cloned
		} else {
			cp.LocalCost = nil
		}
	}

	return &cp
}

// Options configures the legacy DTW wrapper.
//
// What:
//   - Options preserves the old pointer-based configuration surface for DTW.
//   - New code should prefer Align with functional options.
//
// Behavior highlights:
//   - Window == -1 disables the Sakoe-Chiba band.
//   - Window >= 0 admits only cells satisfying |i-j| <= Window.
//   - Window == 0 is strict diagonal alignment.
//   - ReturnPath requires MemoryMode == FullMatrix in the legacy wrapper.
//
// AI-Hints:
//   - Do not describe Window <= 0 as unconstrained.
//   - Do not silently upgrade legacy ReturnPath+TwoRows; preserve ErrPathNeedsMatrix.
type Options struct {
	Window       int
	SlopePenalty float64
	ReturnPath   bool
	MemoryMode   MemoryMode
}

// DefaultOptions returns the legacy default Options value.
//
// What:
//   - DefaultOptions keeps backward-compatible defaults for callers still using DTW.
//
// Returns:
//   - Options with no window constraint, zero slope penalty, path tracking disabled,
//     and TwoRows distance-only storage.
//
// Complexity:
//   - Time O(1), Space O(1).
func DefaultOptions() Options {
	return Options{
		Window:       DefaultWindow,
		SlopePenalty: DefaultSlopePenalty,
		ReturnPath:   DefaultReturnPath,
		MemoryMode:   DefaultMemoryMode,
	}
}

// Validate checks the legacy Options value.
//
// What:
//   - Validate preserves the old validation entrypoint while using the new sentinel model.
//
// Returns:
//   - nil when the legacy options are valid.
//
// Errors:
//   - ErrNilOptions when the receiver is nil.
//   - ErrBadInput joined with ErrInvalidWindow for Window < -1.
//   - ErrBadInput joined with ErrInvalidPenalty for NaN, Inf, or negative penalty.
//   - ErrPathNeedsMatrix when ReturnPath is true without FullMatrix.
//
// Determinism:
//   - Pure validation; no allocation-heavy work.
//
// Complexity:
//   - Time O(1), Space O(1).
func (o *Options) Validate() error {
	if o == nil {
		return ErrNilOptions
	}

	_, err := optionsFromLegacy(o)
	return err
}
