// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm centralizes tensor, matrix-presence and numeric input checks.
// This file enforces shape and precedence laws before allocating output buffers.
package lstm

import (
	"context"
	"math"

	"github.com/lvlath/go/matrix"
)

const maxInt = int(^uint(0) >> 1) // Maximum representable platform int.

const (
	inputGateBlock     = iota // PyTorch gate block order: input.
	forgetGateBlock           // Forget.
	candidateGateBlock        // Cell candidate.
	outputGateBlock           // Output.
	gateBlocks                // Number of fixed gate blocks in the import ABI.
)

const contextPollInterval = 1024 // Maximum unchecked input cells during a finite scan.

// checkedProduct safely calculates an allocation or flat-index element count.
// It rejects invalid geometry before any signed-int multiplication can wrap.
//
// Implementation:
//   - Stage 1: reject either negative factor.
//   - Stage 2: compare a with MaxInt/b only when b is positive.
//   - Stage 3: multiply checked factors and return the exact nonnegative count.
//
// Behavior highlights:
//   - Zero factors always produce zero without division by zero.
//
// Inputs:
//   - a, b: signed dimensions for tensor, matrix, gate, or state storage.
//
// Returns:
//   - int: exact product on success; zero and error on malformed geometry.
//
// Errors:
//   - ErrShape: negative dimension or product larger than platform MaxInt.
//
// Determinism:
//   - Fixed comparisons and arithmetic independent of runtime allocation policy.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Runtime OOM for a representable count cannot be intercepted without panic.
//
// AI-Hints:
//   - Always check before multiplying; wrapped products can corrupt bounds checks.
func checkedProduct(a, b int) (int, error) {
	if a < 0 || b < 0 || (b != 0 && a > maxInt/b) {
		return 0, ErrShape
	}
	return a * b, nil
}

// finite recognizes the permitted domain of caller-supplied binary64 values.
// Zero and subnormal numbers remain valid; NaN and both infinities are rejected.
//
// Implementation:
//   - Stage 1: classify NaN through math.IsNaN.
//   - Stage 2: classify positive and negative infinity through math.IsInf.
//   - Stage 3: permit only values belonging to neither class.
//
// Behavior highlights:
//   - Signed zero is allowed in input; Import normalizes model-storage zero.
//
// Inputs:
//   - value: one float64 from weights, input, or state.
//
// Returns:
//   - bool: true for finite values including ±0 and subnormals.
//
// Errors:
//   - None; callers return ErrNonFinite when the predicate is false.
//
// Determinism:
//   - IEEE-754 classification with no tolerance or branch on user-selected policy.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Computed arithmetic overflow uses ErrOverflow instead of ErrNonFinite.
//
// AI-Hints:
//   - Do not turn an infinity into a missing-data marker in this inference domain.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// matrixPresent checks whether an interface contains a usable matrix receiver.
// It follows the explicit IsNil convention instead of forbidden reflection.
//
// Implementation:
//   - Stage 1: reject a nil matrix interface.
//   - Stage 2: if the concrete type is Nilable, reject its typed-nil receiver.
//   - Stage 3: accept non-Nilable implementations under their own matrix contract.
//
// Behavior highlights:
//   - Handles typed-nil *matrix.Dense, whose IsNil method is defined by matrix.
//
// Inputs:
//   - value: an optional caller-owned matrix.Matrix interface value.
//
// Returns:
//   - bool: false for absent/Nilable-typed-nil matrices; true otherwise.
//
// Errors:
//   - None; callers convert an absent matrix into ErrNilInput.
//
// Determinism:
//   - One interface check and optional Nilable method call without global state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A custom typed-nil Matrix not implementing IsNil cannot be identified
//     without reflection; such implementations must enforce their own nil safety.
//
// AI-Hints:
//   - Do not add reflect/recover to probe arbitrary pointer implementations.
func matrixPresent(value matrix.Matrix) bool {
	if value == nil {
		return false
	}
	if nilable, ok := value.(interface{ IsNil() bool }); ok {
		return !nilable.IsNil()
	}
	return true
}

// validateInput enforces the batch-first tensor and local-length shape contract.
// It intentionally performs geometry validation before any numeric input scan.
//
// Implementation:
//   - Stage 1: validate B, T and D against the imported model input width.
//   - Stage 2: use checked products to compare required length with Data length.
//   - Stage 3: verify one valid timestep count per batch row if lengths is provided.
//
// Behavior highlights:
//   - A zero-time tensor may omit Data entirely; nonempty tensors may not.
//   - lengths=nil means every Time position is active for every batch row.
//
// Inputs:
//   - input: caller-owned row-major [B,T,D], B,D>=1, T>=0.
//   - inputWidth: model D>0; lengths: nil or B integers from 0 through T.
//
// Returns:
//   - error: nil only if complete geometry and mask structure are consistent.
//
// Errors:
//   - ErrShape: malformed shape, checked count overflow or wrong nonnil Data length.
//   - ErrNilInput: missing Data when the checked required count is nonzero.
//   - ErrInvalidOption: wrong lengths count or an invalid local length.
//
// Determinism:
//   - Fixed dimension order, then ascending mask entries; first failure wins.
//
// Complexity:
//   - Time O(B), Space O(1); it does not iterate over data values.
//
// Notes:
//   - Full finiteness validation is a later stage, including padded entries.
//
// AI-Hints:
//   - Never infer lengths from trailing zeros: zero is a valid sensor value.
func validateInput(input Tensor3, inputWidth int, lengths []int) error {
	if input.Batch < 1 || input.Features < 1 || input.Time < 0 || input.Features != inputWidth {
		return ErrShape
	}
	frameCount, err := checkedProduct(input.Batch, input.Time)
	if err != nil {
		return err
	}
	valueCount, err := checkedProduct(frameCount, input.Features)
	if err != nil {
		return err
	}
	if valueCount > 0 && input.Data == nil {
		return ErrNilInput
	}
	if len(input.Data) != valueCount {
		return ErrShape
	}
	if lengths != nil && len(lengths) != input.Batch {
		return ErrInvalidOption
	}
	for _, length := range lengths {
		if length < 0 || length > input.Time {
			return ErrInvalidOption
		}
	}
	return nil
}

// scanFinite rejects every caller-supplied NaN or infinity in a flat vector.
// It is the context-free variant used during immutable weight import.
//
// Implementation:
//   - Stage 1: read every element in ascending index order.
//   - Stage 2: return ErrNonFinite immediately on the first invalid element.
//   - Stage 3: return nil only when the complete slice is finite.
//
// Behavior highlights:
//   - Negative zero and subnormal values are accepted unchanged by validation.
//
// Inputs:
//   - values: read-only caller-owned or freshly copied float64 slice.
//
// Returns:
//   - error: nil when every element is finite.
//
// Errors:
//   - ErrNonFinite: any NaN, +Inf or -Inf.
//
// Determinism:
//   - Strictly ascending index traversal with no early accept for masked values.
//
// Complexity:
//   - Time O(n), Space O(1), n=len(values).
//
// Notes:
//   - Inference uses scanFiniteContext so long validation can be canceled.
//
// AI-Hints:
//   - Do not skip a candidate bias block or ignore Inf in a source matrix.
func scanFinite(values []float64) error {
	for _, value := range values {
		if !finite(value) {
			return ErrNonFinite
		}
	}
	return nil
}

// scanFiniteContext checks every input scalar while polling a valid context.
// Padded entries remain part of the validated payload even when gates skip them.
//
// Implementation:
//   - Stage 1: poll ctx at each fixed input-index interval.
//   - Stage 2: reject the first nonfinite scalar during ascending storage scan.
//   - Stage 3: recheck context after the final value, including zero-length input.
//
// Behavior highlights:
//   - Cancellation bounds long input scans independently of recurrent timestep count.
//
// Inputs:
//   - ctx: non-nil context already checked by Infer; values: full input.Data.
//
// Returns:
//   - error: nil if both context and all values remain valid.
//
// Errors:
//   - ctx.Err(): operation canceled during or immediately after the finite scan.
//   - ErrNonFinite: first observed NaN or infinity when the context is active.
//
// Determinism:
//   - Ascending contiguous index order with a fixed poll interval.
//
// Complexity:
//   - Time O(n), Space O(1), n=len(values).
//
// Notes:
//   - The context can cancel between polls; no partial Result is published.
//
// AI-Hints:
//   - Do not skip masked samples: serialized input must be wholly finite.
func scanFiniteContext(ctx context.Context, values []float64) error {
	for index, value := range values {
		if index%contextPollInterval == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if !finite(value) {
			return ErrNonFinite
		}
	}
	return ctx.Err()
}
