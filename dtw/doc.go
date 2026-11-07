// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw implements Dynamic Time Warping for deterministic sequence alignment.
//
// DTW compares ordered signals that describe the same process at different speeds.
// It is designed for cases where direct index-by-index comparison is too rigid:
// price patterns may stretch across more candles, voice commands may contain held
// phonemes, vibration events may drift in phase, and sensor signatures may be
// sampled with slightly different timing.
//
// Public facades:
//
//   - Align compares scalar sequences.
//     Use it when each time step is one number: price impulse, temperature,
//     normalized amplitude, trend score, or any other scalar signal.
//
//   - AlignCostMatrix compares a precomputed local-cost surface.
//     Use it when another model already computed frame-to-frame costs: acoustic
//     model scores, embedding distances, classifier losses, graph-derived costs,
//     or a domain-specific distance matrix.
//
//   - AlignMatrix compares multivariate sequences.
//     Rows are time steps, columns are feature dimensions. The local cost is
//     squared L2 row distance:
//     C_{ij}=|X_i-Y_j|_2^2
//
//     Use it for OHLC candle vectors, sensor feature vectors, gesture features,
//     or any time-indexed feature matrix.
//
//   - AlignBounded classifies canonical scalar alignment against an inclusive
//     cost limit. Its BoundedResult distinguishes Exact, AboveLimit, and
//     Unreachable without publishing an approximate score.
//
//   - Search returns globally exact top-K equal-length squared-scalar DTW
//     windows, with optional machine-certified pruning and explicit partial results.
//     Radius, stride, normalization, and excluded start ranges are explicit.
//
//   - LBKeoghSquared computes a diagnostic rounded envelope deviation.
//     Search requires a stronger machine certificate before excluding candidates.
//
//   - DTW is the legacy tuple-returning wrapper.
//     It exists for compatibility. New code should use Align, AlignCostMatrix,
//     or AlignMatrix because Result carries reachability, path state, options,
//     and optional matrix artifacts explicitly.
//
// Mathematical law:
//   - Scalar local cost defaults to |a[i]-b[j]|.
//   - WithSquaredCost changes scalar local cost to (a[i]-b[j])².
//   - AlignMatrix uses squared L2 row distance:
//     C(i,j)=||X_i-Y_j||².
//   - AlignCostMatrix accepts caller-provided finite non-negative local costs.
//   - Accumulated cost:
//     D(0,0)=0;
//     D(i,0)=+Inf for i>0;
//     D(0,j)=+Inf for j>0;
//     D(i,j)=cost(i,j)+min(
//     D(i-1,j-1),
//     D(i-1,j)+SlopePenalty,
//     D(i,j-1)+SlopePenalty,
//     ).
//   - Structural absence and a reachable but overflowed accumulated state
//     are distinct classes inside the dynamic program.
//   - A discarded overflow candidate never invalidates a finite optimum.
//   - An unrepresentable optimal terminal sum returns nil with ErrOverflow
//     joined with ErrNaNInf, not Reachable=false.
//   - Finite candidates are compared strictly and ties preserve diagonal,
//     then up, then left predecessor priority during forward DP.
//
// Result model:
//
//   - Result.Distance is the accumulated cost of the selected optimal alignment.
//   - Result.Reachable is false when no admissible path reaches the final cell.
//   - Result.Distance is +Inf when Result.Reachable is false.
//   - Result.Path is filled only when WithReturnPath(true) is used and the result
//     is reachable.
//   - Result.PathTracked tells whether path tracking was requested.
//   - Result.Window, Result.SlopePenalty, and Result.MemoryMode record the executed
//     policy.
//   - Result.Accumulated is present only with WithReturnAccumulated(true).
//   - Result.LocalCost is present only with WithReturnLocalCost(true).
//   - PathOrError separates three states: nil result, no path, and path not tracked.
//   - BoundedResult publishes Exact with finite equal Distance/LowerBound,
//     AboveLimit with NaN Distance and certified LowerBound>limit, or Unreachable
//     with Distance=LowerBound=+Inf.
//   - SearchResult.Complete=true certifies globally exact sorted Hits;
//     Complete=false ranks only completed candidates and accompanies an error.
//
// Reading a path:
//
//   - A path element {I,J} means “time step I from the first sequence aligns with
//     time step J from the second sequence”.
//   - Diagonal steps consume both sequences together.
//   - Vertical or horizontal steps represent stretching: one side advances while
//     the other side is held.
//   - Repeated I values often mean the second sequence stretched a phase.
//   - Repeated J values often mean the first sequence stretched a phase.
//   - The returned path is one deterministic optimal path, not all possible
//     optimal paths.
//
// Window policy:
//
//   - WithWindow(-1) disables the Sakoe-Chiba band.
//   - WithWindow(0) enforces strict diagonal alignment and forbids warping.
//   - WithWindow(w) for w>0 admits cells satisfying |i-j| <= w.
//   - A small positive window is usually safer for production systems than an
//     unconstrained path because it prevents unrealistic temporal jumps.
//   - Window < -1 is invalid and returns ErrInvalidWindow joined with ErrBadInput.
//
// SlopePenalty policy:
//
//   - WithSlopePenalty(p) adds p to vertical and horizontal steps.
//   - p must be finite and non-negative.
//   - p=0 allows stretching without additional structural penalty.
//   - Larger p biases the solution toward diagonal movement.
//   - Use a small positive p when repeated frames are realistic but should still
//     affect the score: held speech sounds, delayed candles, slow sensor rise,
//     or jitter around a physical event.
//
// Cost policy:
//
//   - Align defaults to AbsoluteCost: |a-b|.
//   - WithSquaredCost switches scalar Align to (a-b)^2.
//   - WithCostFunc installs a custom scalar local-cost function.
//   - Custom local costs must be finite and non-negative.
//   - Scalar CostFunc is evaluated once per admitted cell without LocalCost
//     output, or once per full rectangular cell with WithReturnLocalCost(true).
//     Callbacks must be pure and deterministic; evaluation order is not a
//     mechanism for domain-state mutation.
//   - AlignMatrix always uses squared L2 row distance in the current contract.
//   - AlignCostMatrix ignores CostFunc because the supplied matrix already is
//     the local-cost source.
//
// Numeric policy:
//
//   - Scalar input samples must be finite by default.
//   - WithValidateFinite(false) only skips scalar input finite checks; it does
//     not allow invalid local costs.
//   - AlignMatrix validates matrix inputs with matrix.ValidateAllFinite.
//   - AlignCostMatrix rejects NaN, +Inf, -Inf, and negative local costs.
//   - Accumulated +Inf artifact cells can represent structural absence or
//     overflowed intermediate candidates; they do not expose the class by value.
//   - NaN is not an admitted local/accumulated cost; BoundAboveLimit deliberately
//     uses NaN only to mark the unavailable exact Distance field.
//
// Memory policy:
//
//   - TwoRows stores only rolling DP rows and is the default distance-only mode.
//   - FullMatrix stores the accumulated DP matrix and enables path reconstruction.
//   - WithReturnPath(true) and WithReturnAccumulated(true) require full accumulated
//     storage.
//   - NoMemory is a deprecated compatibility alias for rolling-row distance mode.
//   - Exact DTW distance in this package is not an O(1)-memory algorithm.
//
// Matrix artifacts:
//
//   - Result.Accumulated is the numeric DP surface D; it does not carry
//     separate structural/overflow state classification. The forward-selected
//     predecessor codes, not subtraction from D, determine the returned path.
//
//   - Result.LocalCost is the complete n-by-m local surface C, including
//     cells outside Window. Publication requires representable values for all
//     cells and is independent of which cells participate in the optimal path.
//
//   - Returned matrices are detached artifacts.
//     Mutating Result.LocalCost or Result.Accumulated does not mutate the caller’s
//     input matrix.
//
// Error classification:
//
//   - Use errors.Is; never match error strings.
//   - ErrNilInput means a scalar input slice is nil.
//   - ErrEmptyInput means a scalar sequence or local-cost matrix domain is empty.
//   - ErrNilOption means a nil functional option was passed.
//   - ErrInvalidWindow means the window is below -1.
//   - ErrInvalidPenalty means slope penalty is NaN, Inf, or negative.
//   - ErrNaNInf means a finite-only input or local cost contained NaN or Inf.
//   - ErrOverflow joined with ErrNaNInf means unrepresentable local squared
//     cost or selected final accumulated optimum.
//   - ErrInvalidOption, ErrShape, ErrNonFinite, ErrZeroVariance, and ErrBudget
//     classify validation, normalization, and search resource failures.
//   - ErrNegativeCost means a local cost is negative.
//   - ErrNoPath means path tracking was requested but the final cell is unreachable.
//   - ErrPathNotTracked means the caller requested a path from a result that did
//     not track one.
//   - Matrix-backed adapters preserve relevant matrix sentinels, such as
//     matrix.ErrDimensionMismatch, by joining them with DTW-level errors.
//
// Determinism:
//
//   - DP loops are row-major: i ascending, then j ascending.
//   - Matrix local costs are materialized in row-major order.
//   - Forward predecessor selection is strict: diagonal, then up, then left
//     on exact ties; backtracking reads recorded codes, not rounded differences.
//   - Completed Search hits sort by (squared distance, start, end); equal-cost
//     starts remain in canonical order across worker counts.
//   - Returned paths are ordered from start to end.
//   - No map iteration or randomized behavior participates in path selection.
//
// Operational recipes:
//
//   - Fast distance scan:
//     call Align or AlignMatrix without WithReturnPath and without matrix artifacts.
//     This keeps memory to rolling rows.
//
//   - Explain one match:
//     enable WithReturnPath(true), optionally WithReturnAccumulated(true), and inspect
//     the path anchors where one side repeats.
//
//   - Debug feature scaling:
//     use AlignMatrix with WithReturnLocalCost(true). If one feature dominates the
//     local-cost matrix, normalize outside DTW before alignment.
//
//   - Use external models:
//     pass their finite non-negative frame costs to AlignCostMatrix. Do not force
//     those costs through scalar slices.
//
//   - Search for matching windows:
//     call Search(ctx, query, data, DefaultSearchOptions()) and consume Hits
//     only as a global top-K when Complete is true. SearchRaw and population
//     z-normalization are distinct explicitly selected cost regimes.
//
//   - Classify an inclusive score threshold:
//     call AlignBounded and inspect BoundStatus; BoundAboveLimit.Distance is NaN.
//
// Common pitfalls:
//
//   - Window=0 is not “no constraint”; it is strict diagonal alignment.
//   - Unconstrained DTW can over-warp; production systems usually need a window.
//   - Very large slope penalties can turn DTW into near-diagonal comparison.
//   - DTW distance is not a metric in general; do not assume triangle inequality.
//   - AlignMatrix uses squared L2 distance; feature scale matters.
//   - Do not pass probabilities directly as costs unless the scale is intentional;
//     many models should convert probabilities to costs first, for example -log(p).
//
// Ownership:
//
//   - Input slices and input matrices are never mutated.
//   - Result.Path is detached and caller-owned.
//   - Result.Accumulated and Result.LocalCost are detached matrix.Dense artifacts.
//   - Result values are not internally synchronized; callers must synchronize their
//     own mutation of returned artifacts across goroutines.
//
// Partial-result law:
//
//   - When path tracking is requested and no admissible path exists, Align,
//     AlignCostMatrix, and AlignMatrix return Result{Distance:+Inf, Reachable:false}
//     together with ErrNoPath.
//   - AlignBounded never returns a partial BoundedResult on cancellation or
//     numeric failure. Its Unreachable status is not a legacy ErrNoPath.
//   - Search can return incomplete SearchResult with ErrBudget, context error,
//     or candidate runtime failure after computation begins. Partial Hits are
//     not a certified global top-K.
//
// Concurrency:
//
//   - Package functions are pure over caller-provided immutable inputs.
//   - Concurrent mutation of input slices or matrices during execution is unsupported.
//   - Result values are not internally synchronized; callers own synchronization if
//     they mutate returned Path or matrix artifacts across goroutines.
//
// Complexity summary:
//
//   - Scalar Align: O(n*m) time.
//   - AlignCostMatrix: O(n*m) materialization plus O(n*m) DP.
//   - AlignMatrix: O(n*m*d) local-cost construction plus O(n*m) DP.
//   - Distance-only rolling-row mode uses O(m) DP memory.
//   - Path, accumulated, or complete local-cost output uses O(n*m) memory.
//   - Align and AlignMatrix retain rectangular O(n*m) DP iteration even with
//     an active narrow window; only the specialized Search scorer is banded.
//   - For C equal-length candidates, query length L, W workers and band
//     width b=min(L,2r+1), exact Search is O(C*L*b + C log K) worst-case time
//     and O(K+W*b+L) raw-mode workspace plus exclusion storage and certification.
//     Z normalization additionally uses O(C*L) time and O(W*L) workspace.
//
// AI-Hints:
//   - Do not treat Window=0 as unconstrained.
//   - Do not classify errors by string matching; use errors.Is.
//   - Do not replace Result with raw parallel return values on canonical APIs.
//   - Do not silently normalize matrix inputs inside AlignMatrix.
//   - Do not allow +Inf local costs, and do not interpret every +Inf
//     accumulated artifact cell as structural unreachability.
//   - Do not reconstruct a path from D(i,j)-local(i,j) after rounding.
//   - Never treat a rounded public LBKeoghSquared value alone as a proof for
//     pruning at a near-equal cutoff; Search uses machine certificates.
//   - Do not attempt path reconstruction from rolling rows.
//   - Do not use AlignMatrix for heterogeneous feature scales without preprocessing.
//   - Do not document NoMemory as O(1) exact DTW.
package dtw
