// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats defines the public statistical state, configuration, and results.
// This file fixes ownership, zero-value, and constant-column contracts.
package stats

import "github.com/lvlath/go/core"

// Moments accumulates finite observations using stable central-moment recurrences.
// Its zero value is a valid empty state; the exact binary mean sum is independent
// of the compensated extended-exponent center and second moment.
//
// Implementation:
//   - Stage 1: the first Update initializes count, extrema, and numeric state.
//   - Stage 2: subsequent Updates use Welford; independent partitions use Chan.
//   - Stage 3: Summary selects ddof and materializes a detached float64 snapshot.
//
// Behavior highlights:
//   - The exact sum survives cancellation between observations of distant exponents.
//   - The unnormalized M2 may exceed MaxFloat64 without making Update fail.
//
// Inputs:
//   - Update accepts one finite float64 observation in caller-specified order.
//   - Merge accepts a quiescent, independent accumulator with its own count.
//
// Returns:
//   - Summary returns a detached snapshot after checking Count>ddof.
//
// Errors:
//   - ErrNonFinite rejects invalid observations; ErrOverflow rejects count overflow.
//   - ErrNilInput/ErrInvalidOption reject nil/self-merge receiver configurations.
//
// Determinism: updates preserve input order; merge order is caller-controlled.
// Complexity: O(1) bounded state; O(1) arithmetic operations per Update/Merge.
// Notes: all private numeric fields are value types; mutation is not concurrent-safe.
// AI-Hints:
//   - Never replace central M2 with E[x*x]-E[x]*E[x].
//   - Aggregate partitions independently and Merge them in a fixed tree order.
type Moments struct {
	// count is the number of committed finite samples; zero means empty.
	count uint64

	// mean preserves sub-ULP center corrections for centered M2 updates.
	mean scaledNumber

	// exactMeanSum retains every original binary input bit for materialization.
	// This prevents loss across arbitrary exponent cancellation patterns.
	exactMeanSum exactSum

	// m2 is the nonnegative, unnormalized sum of squared central residuals.
	// Its exponent is deliberately NOT constrained to IEEE float64's range.
	m2 scaledNumber

	// min is the lowest committed original observation, not a normalized value.
	min float64

	// max is the highest committed original observation, not a normalized value.
	max float64
}

// Summary publishes a finite, detached description of committed observations.
// It separates exact Count/Min/Max and correctly rounded Mean from the
// compensated float64 central Variance using the caller-selected ddof.
//
// Implementation:
//   - Stage 1: validate sample-count sufficiency for ddof=0 or ddof=1.
//   - Stage 2: divide scaled M2 by Count-ddof before converting to float64.
//   - Stage 3: publish a new independent Summary after finite-range checks.
//
// Behavior highlights:
//   - A positive but unrepresentably tiny variance may round to zero.
//   - The extrema are original observations, not model-normalized coordinates.
//
// Inputs: created by Moments.Summary(ddof) or Summarize(values, ddof).
// Returns: independently owned fields, with no retained Moments or slice alias.
// Errors: creation may fail with ErrInsufficientSamples or ErrOverflow.
// Determinism: one final rounding to the specified float64 result fields.
// Complexity: O(1) storage per snapshot and bounded materialization work.
// Notes: the ddof is not retained; the variance is interpreted in call context.
// AI-Hints: never infer a mathematically constant feature solely from Variance==0.
type Summary struct {
	// Count is the exact number of committed observations (uint64).
	Count uint64

	// Mean is the nearest representable float64 arithmetic mean.
	Mean float64

	// Variance is the finite, nonnegative central variance for the chosen ddof.
	Variance float64

	// Min is the smallest original float64 observation.
	Min float64

	// Max is the largest original float64 observation.
	Max float64
}

// ConstantPolicy selects how a zero-spread training column is handled at fit.
// It controls the error-versus-unit-scale decision after the entire input
// matrix has been validated and moments calculated for each feature.
//
// Implementation:
//   - Stage 1: validate the enum before reading training data.
//   - Stage 2: compare a feature's internal compensated M2 with exact zero.
//   - Stage 3: reject constant columns or retain Mean with Scale=1.
//
// Behavior highlights: no tolerance or epsilon is used for this decision.
// Inputs: assigned to ScaleOptions.ConstantPolicy or an explicit fit wrapper.
// Returns: changes fitted-scale policy; inference itself has no such option.
// Errors: an unknown enum yields ErrInvalidOption; rejection yields ErrZeroVariance.
// Determinism: ordered per-column classification, independent of feature names.
// Complexity: O(1) option storage and O(1) decision per fitted feature.
// Notes: the zero-value policy is RejectConstant.
// AI-Hints: a rounded-to-zero public variance may represent real tiny spread.
type ConstantPolicy uint8

const (
	// RejectConstant refuses a zero-variance training column with ErrZeroVariance.
	// This default prevents hiding a feature that cannot be standardized uniquely.
	RejectConstant ConstantPolicy = iota

	// UnitScaleConstant retains its training mean and uses a scale of exactly one.
	// A novel value x then maps to x - trainingMean, not to a forced zero.
	UnitScaleConstant
)

// ScaleOptions configures training-only variance and constant-feature semantics.
// It is passed by value to FitStandardizer, never retained by the immutable
// Standardizer and never consulted again during inference.
//
// Implementation:
//   - Stage 1: check DDOF against 0 and 1, then validate ConstantPolicy.
//   - Stage 2: use both values for each independent training column.
//   - Stage 3: freeze the resulting numeric means/scales in a new model.
//
// Behavior highlights: the zero value means DDOF=0 and RejectConstant.
// Inputs: DDOF and ConstantPolicy are caller-set, with no hidden defaults.
// Returns: determines how FitStandardizer constructs its ScaleFitResult.
// Errors: ErrInvalidOption is returned before touching a matrix on invalid options.
// Determinism: a fixed option value produces a fixed column-wise calculation order.
// Complexity: O(1) validation/storage, separate from O(rows*cols) fitting.
// Notes: changing the original ScaleOptions variable cannot change a fitted model.
// AI-Hints: choose the correct population/sample denominator before fitting.
type ScaleOptions struct {
	// DDOF is 0 for population variance or 1 for sample variance; no other value is valid.
	DDOF int

	// ConstantPolicy decides whether M2==0 rejects fitting or selects unit scale.
	ConstantPolicy ConstantPolicy
}

// Standardizer stores immutable training-column means, scales, and flags.
// It represents a fitted affine transformation; no inference operation updates
// the training parameters or retains a caller-owned matrix/workspace reference.
//
// Implementation:
//   - Stage 1: FitStandardizer copies validated fitted arrays into private storage.
//   - Stage 2: Transform/InverseTransform validate each inference batch.
//   - Stage 3: each call allocates and publishes its own detached matrix.
//
// Behavior highlights:
//   - UnitScaleConstant retains the training mean and maps novel x to x-mean.
//   - A zero-valued or nil Standardizer is unfitted and cannot transform data.
//
// Inputs: Transform expects fitted feature count in Cols, any nonnegative Rows.
// Returns: detached matrix.Matrix, with no model-owned output slice aliases.
// Errors: ErrNilInput for unfitted/nil receiver; ErrShape for wrong features;
//
//	ErrNonFinite and ErrOverflow for invalid/unrepresentable inference values.
//
// Determinism: two row-major passes per inference call; fixed column parameters.
// Complexity: O(features) persistent memory; O(rows*cols) per inference call.
// Notes: concurrent read-only inference is safe when inputs are not mutated.
// AI-Hints:
//   - Reuse this fitted model on heldout or production data; never refit it.
//   - Do not expose or mutate the private arrays through metadata snapshots.
type Standardizer struct {
	// means owns the rounded training means, one for each feature column.
	means []float64

	// scales owns strictly positive, finite training standard deviations.
	scales []float64

	// constant owns the exact-zero training variance mask.
	constant []bool
}

// ScaleFitResult publishes a fitted model and independent parameter snapshots.
// Its exported Means, Scales, and Constant slices are detached from each
// other and from Standardizer's private parameter vectors.
//
// Implementation:
//   - Stage 1: compute one mean, standard deviation, and constant flag per column.
//   - Stage 2: copy all three arrays into model-owned private storage.
//   - Stage 3: return the original independently owned slices to the caller.
//
// Behavior highlights: changing metadata cannot affect Transform output.
// Inputs: created only by successful FitStandardizer or its thin wrappers.
// Returns: non-nil Model and feature-length parameter slices after success.
// Errors: nil fit result on every training error; no partial model is published.
// Determinism: slice indices equal ascending input feature column indices.
// Complexity: O(features) independent metadata plus O(features) model arrays.
// Notes: exported slices are mutable observations, not live model configuration.
// AI-Hints: retain the Model when performing inference; metadata is descriptive.
type ScaleFitResult struct {
	// Model owns all fitted parameters and is safe for concurrent read-only calls.
	Model *Standardizer

	// Means is a caller-owned copy of training column means.
	Means []float64

	// Scales is a caller-owned copy of training column standard deviations.
	Scales []float64

	// Constant is a caller-owned mask of exact-zero training variances.
	Constant []bool
}

// IsNil reports whether the Summary pointer is nil, including when held in an interface.
// It follows core.Nilable without treating a valid zero-valued Summary as absent.
//
// Implementation:
//   - Stage 1: compare the receiver pointer to nil without dereferencing it.
//   - Stage 2: return the boolean without inspecting mutable public fields.
//   - Stage 3: perform no allocation, validation, or state publication.
//
// Behavior highlights:
//   - A non-nil Summary remains present even when Count is zero.
//
// Inputs:
//   - receiver: a possibly nil *Summary; no allocation or ownership transfer.
//
// Returns:
//   - bool: true only for a nil receiver, including a typed-nil interface value.
//
// Errors:
//   - None; this predicate cannot fail or panic for a nil receiver.
//
// Determinism:
//   - Pure pointer comparison, independent of Summary values or goroutine order.
//
// Complexity:
//   - Time O(1), Space O(1), without heap allocation.
//
// Notes:
//   - IsNil checks presence, not validity of the statistical result.
//   - Read-only calls are safe when the caller does not concurrently mutate fields.
//
// AI-Hints:
//   - Do not replace this method with Count == 0: presence and sample sufficiency differ.
//   - Check method errors, not IsNil, to identify failed Summary calculations.
func (r *Summary) IsNil() bool {
	return r == nil
}

// IsNil reports whether the fitted Standardizer pointer is nil.
// It implements core.Nilable and remains safe when a typed-nil model is placed
// behind an interface; an allocated but unfitted model is not classified as nil.
//
// Implementation:
//   - Stage 1: compare the receiver pointer directly with nil.
//   - Stage 2: return the result without accessing fitted parameter slices.
//   - Stage 3: perform no mutation and publish no derived state.
//
// Behavior highlights:
//   - A non-nil zero-valued Standardizer returns false but cannot transform data.
//
// Inputs:
//   - receiver: a possibly nil *Standardizer.
//
// Returns:
//   - bool: true exactly for a nil pointer.
//
// Errors:
//   - None; unfitted model validation belongs to Transform/InverseTransform.
//
// Determinism:
//   - Independent of training data and concurrent read-only inference.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - IsNil is a presence predicate, not a fit-status or parameter-integrity check.
//
// AI-Hints:
//   - Never use IsNil == false to skip validation in Transform.
//   - The zero-valued model has no trained feature count even though it is non-nil.
func (m *Standardizer) IsNil() bool {
	return m == nil
}

// IsNil reports whether the ScaleFitResult pointer is nil.
// The method adopts the core.Nilable convention without using exported slice
// lengths or a caller-mutable Model field as a proxy for result presence.
//
// Implementation:
//   - Stage 1: compare the receiver pointer to nil without dereferencing it.
//   - Stage 2: return the boolean independently of Model and metadata contents.
//   - Stage 3: perform no allocation or state publication.
//
// Behavior highlights:
//   - A non-nil result with a caller-cleared Model field is still non-nil.
//
// Inputs:
//   - receiver: possibly nil *ScaleFitResult, including inside an interface.
//
// Returns:
//   - bool: true only when the receiver is nil.
//
// Errors:
//   - None; parameter or model validity is not checked by this predicate.
//
// Determinism:
//   - Pure pointer comparison, independent of metadata mutation.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The returned result holds caller-owned public slices, not a live model view.
//
// AI-Hints:
//   - Do not infer fit success from IsNil alone; always check the returned error.
//   - Do not inspect Model.IsNil before ensuring Model itself is non-nil.
func (r *ScaleFitResult) IsNil() bool {
	return r == nil
}

// Compile-time assertions protect the same nil-state interface implemented by
// graph and dynamic-programming results across the library.
var (
	_ core.Nilable = (*Summary)(nil)
	_ core.Nilable = (*Standardizer)(nil)
	_ core.Nilable = (*ScaleFitResult)(nil)
)
