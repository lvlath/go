// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian classifies public failures using stable sentinel identities.
// This file documents each reachable category and its originating validation boundary.
package donchian

import "errors"

var (
	// ErrNilContext classifies an absent context before any option or input inspection.
	// Its identity is stable under wrapping; its message is diagnostic rather than protocol.
	//
	// Implementation:
	//   - Stage 1: initialize one package-level sentinel with errors.New.
	//   - Stage 2: validateHeader rejects absence before calling any context method.
	//   - Stage 3: callers classify the returned error using errors.Is.
	//
	// Behavior highlights:
	//   - A canceled non-nil context instead returns ctx.Err() unchanged.
	//
	// Inputs:
	//   - ctx is a nil interface.
	//
	// Returns:
	//   - Compute and its wrappers return nil with an error matching this category.
	//
	// Errors:
	//   - This value is the classification target, not a separate fallible operation.
	//
	// Determinism:
	//   - Identity is stable within the process; validation order is defined by Compute.
	//
	// Complexity:
	//   - O(1) sentinel storage; classification follows the returned error's wrapping tree.
	//
	// Notes:
	//   - Do not reassign exported sentinels or depend on the exact diagnostic text.
	//
	// AI-Hints:
	//   - Pass context.Background() or a real derived context; do not pass typed-nil contexts.
	ErrNilContext = errors.New("donchian: nil context")

	// ErrNilInput classifies a nil channel paired with a nonempty peer.
	// Its identity is stable under wrapping; its message is diagnostic rather than protocol.
	//
	// Implementation:
	//   - Stage 1: initialize one package-level sentinel with errors.New.
	//   - Stage 2: validateHeader joins this category with ErrShape after option validation.
	//   - Stage 3: callers classify the returned error using errors.Is.
	//
	// Behavior highlights:
	//   - Both errors.Is(err, ErrNilInput) and errors.Is(err, ErrShape) are true.
	//
	// Inputs:
	//   - Exactly one price channel is nil and the other has positive length.
	//
	// Returns:
	//   - Compute and its wrappers return nil with an error matching this category.
	//
	// Errors:
	//   - This value is the classification target, not a separate fallible operation.
	//
	// Determinism:
	//   - Identity is stable within the process; validation order is defined by Compute.
	//
	// Complexity:
	//   - O(1) sentinel storage; classification follows the returned error's wrapping tree.
	//
	// Notes:
	//   - Do not reassign exported sentinels or depend on the exact diagnostic text.
	//
	// AI-Hints:
	//   - Two nil channels are an empty success, not a missing-input error.
	ErrNilInput = errors.New("donchian: nil channel paired with nonempty channel")

	// ErrInvalidOption classifies a nonpositive observation period.
	// Its identity is stable under wrapping; its message is diagnostic rather than protocol.
	//
	// Implementation:
	//   - Stage 1: initialize one package-level sentinel with errors.New.
	//   - Stage 2: validateHeader checks the period after context and before shape.
	//   - Stage 3: callers classify the returned error using errors.Is.
	//
	// Behavior highlights:
	//   - Every positive int is admissible; no wrapper silently supplies a default.
	//
	// Inputs:
	//   - Options.Period is zero or negative.
	//
	// Returns:
	//   - Compute and its wrappers return nil with an error matching this category.
	//
	// Errors:
	//   - This value is the classification target, not a separate fallible operation.
	//
	// Determinism:
	//   - Identity is stable within the process; validation order is defined by Compute.
	//
	// Complexity:
	//   - O(1) sentinel storage; classification follows the returned error's wrapping tree.
	//
	// Notes:
	//   - Do not reassign exported sentinels or depend on the exact diagnostic text.
	//
	// AI-Hints:
	//   - Do not reject Period > len(high); it is a valid all-warmup request.
	ErrInvalidOption = errors.New("donchian: period must be positive")

	// ErrShape classifies unequal high and low lengths.
	// Its identity is stable under wrapping; its message is diagnostic rather than protocol.
	//
	// Implementation:
	//   - Stage 1: initialize one package-level sentinel with errors.New.
	//   - Stage 2: validateHeader rejects mismatched dimensions before reading any elements.
	//   - Stage 3: callers classify the returned error using errors.Is.
	//
	// Behavior highlights:
	//   - Nil/nonempty mismatches also match ErrNilInput; non-nil empty mismatches do not.
	//
	// Inputs:
	//   - The two price-channel lengths differ.
	//
	// Returns:
	//   - Compute and its wrappers return nil with an error matching this category.
	//
	// Errors:
	//   - This value is the classification target, not a separate fallible operation.
	//
	// Determinism:
	//   - Identity is stable within the process; validation order is defined by Compute.
	//
	// Complexity:
	//   - O(1) sentinel storage; classification follows the returned error's wrapping tree.
	//
	// Notes:
	//   - Do not reassign exported sentinels or depend on the exact diagnostic text.
	//
	// AI-Hints:
	//   - Do not truncate the longer channel or align rows by dropping observations.
	ErrShape = errors.New("donchian: high and low lengths differ")

	// ErrNonFinite classifies NaN or either infinity in either price channel.
	// Its identity is stable under wrapping; its message is diagnostic rather than protocol.
	//
	// Implementation:
	//   - Stage 1: initialize one package-level sentinel with errors.New.
	//   - Stage 2: validateHistory scans indices ascending and checks high before low.
	//   - Stage 3: callers classify the returned error using errors.Is.
	//
	// Behavior highlights:
	//   - This category precedes finite ErrBar globally, including in masked rows.
	//
	// Inputs:
	//   - Any supplied high or low is not finite.
	//
	// Returns:
	//   - Compute and its wrappers return nil with an error matching this category.
	//
	// Errors:
	//   - This value is the classification target, not a separate fallible operation.
	//
	// Determinism:
	//   - Identity is stable within the process; validation order is defined by Compute.
	//
	// Complexity:
	//   - O(1) sentinel storage; classification follows the returned error's wrapping tree.
	//
	// Notes:
	//   - Do not reassign exported sentinels or depend on the exact diagnostic text.
	//
	// AI-Hints:
	//   - Do not replace invalid values by zero or skip the last excluded input row.
	ErrNonFinite = errors.New("donchian: non-finite price")

	// ErrBar classifies a finite high smaller than the corresponding finite low.
	// Its identity is stable under wrapping; its message is diagnostic rather than protocol.
	//
	// Implementation:
	//   - Stage 1: initialize one package-level sentinel with errors.New.
	//   - Stage 2: validateHistory remembers the first violating index and finishes finite admission.
	//   - Stage 3: callers classify the returned error using errors.Is.
	//
	// Behavior highlights:
	//   - Negative prices, signed zero, and equal boundaries are valid.
	//
	// Inputs:
	//   - At least one finite pair has high[i] < low[i], with all input values finite.
	//
	// Returns:
	//   - Compute and its wrappers return nil with an error matching this category.
	//
	// Errors:
	//   - This value is the classification target, not a separate fallible operation.
	//
	// Determinism:
	//   - Identity is stable within the process; validation order is defined by Compute.
	//
	// Complexity:
	//   - O(1) sentinel storage; classification follows the returned error's wrapping tree.
	//
	// Notes:
	//   - Do not reassign exported sentinels or depend on the exact diagnostic text.
	//
	// AI-Hints:
	//   - Do not swap high and low to conceal an invalid supplied candle.
	ErrBar = errors.New("donchian: high is below low")

	// ErrOverflow classifies an unrepresentable packed output allocation size.
	// Its identity is stable under wrapping; its message is diagnostic rather than protocol.
	//
	// Implementation:
	//   - Stage 1: initialize one package-level sentinel with errors.New.
	//   - Stage 2: Compute checks the length bound before multiplying or allocating output storage.
	//   - Stage 3: callers classify the returned error using errors.Is.
	//
	// Behavior highlights:
	//   - Finite midpoints cannot cause this error; runtime RAM exhaustion is not this sentinel.
	//
	// Inputs:
	//   - The required three-channel binary64 byte count would exceed math.MaxInt.
	//
	// Returns:
	//   - Compute and its wrappers return nil with an error matching this category.
	//
	// Errors:
	//   - This value is the classification target, not a separate fallible operation.
	//
	// Determinism:
	//   - Identity is stable within the process; validation order is defined by Compute.
	//
	// Complexity:
	//   - O(1) sentinel storage; classification follows the returned error's wrapping tree.
	//
	// Notes:
	//   - Do not reassign exported sentinels or depend on the exact diagnostic text.
	//
	// AI-Hints:
	//   - Do not fabricate a midpoint overflow branch or promise recovery from runtime OOM.
	ErrOverflow = errors.New("donchian: output allocation size overflows int")
)
