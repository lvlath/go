// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package acf

// api.go defines the validated public estimator facade and explicit policy shortcuts.
// These shortcuts select Options and delegate every calculation to Compute.

// Compute estimates autocorrelation at every integer lag from zero through maxLag.
// It fits one observation-window mean (unless NoCenter is selected), then applies
// exactly the estimator convention named in opts to the same deviations at every lag.
//
// Implementation:
//   - Stage 1: validate the estimator and lag, followed by input presence, shape, and finiteness.
//   - Stage 2: delegate centered/scaled preparation and compensated pair scans to direct.
//   - Stage 3: publish the complete detached Result, or nil on any error.
//
// Behavior highlights:
//   - Zero Options selects centered CommonDenominator; maxLag=0 computes lag zero only.
//   - Values[0] is exactly 1 when the centered energy or raw second moment is positive.
//   - LagAdjusted changes the denominator to the available n-k pairs and is not clamped.
//
// Inputs:
//   - values: non-nil, nonempty, finite, caller-owned read-only samples in increasing time order.
//   - maxLag: inclusive maximum lag in sample indices, 0 <= maxLag < len(values).
//   - opts: value-copied normalization and centering policy; no implicit time grid or missing data.
//
// Returns:
//   - *Result: len=maxLag+1, detached caller-owned Values/PairCounts, and fitted Mean.
//   - error: nil on success; every error returns a nil result, without partial coefficients.
//
// Errors:
//   - ErrInvalidOption: unknown Denominator, negative maxLag, or maxLag >= n.
//   - ErrNilInput, ErrEmptyInput: missing and empty observation slices, respectively.
//   - ErrNonFinite: a NaN or infinity in the supplied observations.
//   - ErrZeroVariance: centered energy or uncentered second moment equals zero.
//   - ErrOverflow: a required arithmetic result cannot be represented as finite float64.
//
// Determinism:
//   - Validation scans observations in ascending index order; direct scans lags and pairs ascending.
//   - The same binary64 inputs, options, and Go floating-point environment yield the same result.
//
// Complexity:
//   - Time O(n*(maxLag+1)), Space O(n+maxLag), including validation and owned output slices.
//
// Notes:
//   - Compute never mutates or retains values; concurrent caller mutation during Compute is unsupported.
//   - This is a whole-window fit: appending a sample can change the mean and every previous lag.
//   - Sample spacing, stationarity assumptions, and interpretation of lag units belong to callers.
//
// AI-Hints:
//   - Do not apply LagAdjusted coefficients to a Toeplitz-PSD-dependent PACF solver.
//   - Never recenter each lag separately or clamp LagAdjusted coefficients to [-1, 1].
//   - Use only a fixed training window when preparing regression or forecasting features.
func Compute(values []float64, maxLag int, opts Options) (*Result, error) {
	if err := validateInput(values, maxLag, opts); err != nil {
		return nil, err
	}
	return direct(values, maxLag, opts)
}

// ComputeDefault computes the centered common-denominator sample ACF.
// It is the simplest safe entry point for stationary-series diagnostics and
// preserves the Toeplitz-PSD-compatible convention in exact arithmetic.
//
// Implementation:
//   - Stage 1: select the zero-value Options (centering enabled, CommonDenominator).
//   - Stage 2: delegate all validation, preparation, and direct scans to Compute.
//   - Stage 3: return Compute's entire detached result and unchanged error.
//
// Behavior highlights:
//   - The lag-zero coefficient is exactly one for a nonconstant centered sample.
//   - No pair-adjusted normalization, spectral estimator, or confidence band is implied.
//
// Inputs:
//   - values: finite read-only observations; nil/empty slices are rejected.
//   - maxLag: inclusive upper lag in sample positions, 0 <= maxLag < len(values).
//
// Returns:
//   - *Result: centered CommonDenominator coefficients, pair counts, and fitted mean.
//   - error: identical classification and nil-result behavior to Compute with Options{}.
//
// Errors:
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrNonFinite, ErrZeroVariance, ErrOverflow:
//     precisely the matching Compute error categories and validation precedence.
//
// Determinism:
//   - Delegates to Compute's ascending observation, lag, and pair traversal.
//
// Complexity:
//   - Time O(n*(maxLag+1)), Space O(n+maxLag), including the result.
//
// Notes:
//   - Does not retain the input and does not provide causal prefix invariance.
//
// AI-Hints:
//   - Prefer this wrapper when a downstream Toeplitz-based method requires a common denominator.
func ComputeDefault(values []float64, maxLag int) (*Result, error) {
	return Compute(values, maxLag, Options{})
}

// ComputeLagAdjusted computes a centered sample ACF with pair-count adjustment.
// It uses the same globally centered deviations as ComputeDefault but multiplies
// each nonzero lag by n/(n-k), so values outside [-1, 1] are mathematically valid.
//
// Implementation:
//   - Stage 1: explicitly select LagAdjusted while retaining global centering.
//   - Stage 2: delegate validation and the direct estimator to Compute.
//   - Stage 3: return the complete Result, including its LagAdjusted metadata.
//
// Behavior highlights:
//   - Unlike CommonDenominator, this estimator has no general Toeplitz-PSD guarantee.
//   - Late lags use fewer actual pairs, as recorded in Result.PairCounts.
//
// Inputs:
//   - values: finite read-only sample window of length n >= 1.
//   - maxLag: inclusive maximum nonnegative lag, strictly smaller than n.
//
// Returns:
//   - *Result: centered pair-adjusted coefficients and their exact pair counts.
//   - error: nil on success, otherwise nil result and a matching Compute error.
//
// Errors:
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrNonFinite, ErrZeroVariance, ErrOverflow:
//     same stages and precedence as Compute with Denominator=LagAdjusted.
//
// Determinism:
//   - Same ascending lag/pair order and fixed summation policy as Compute.
//
// Complexity:
//   - Time O(n*(maxLag+1)), Space O(n+maxLag), including output.
//
// Notes:
//   - Does not imply an unbiased estimator of the ACF ratio itself.
//   - The correction applies to lag covariance estimates, not to the mean-fitting window.
//
// AI-Hints:
//   - Never clip values beyond one or feed this output to Levinson-Durbin without a new contract.
func ComputeLagAdjusted(values []float64, maxLag int) (*Result, error) {
	return Compute(values, maxLag, Options{Denominator: LagAdjusted})
}

// ComputeUncentered computes common-denominator autocorrelation of raw second moments.
// It deliberately preserves the observation origin rather than subtracting a
// fitted sample mean, which is useful for physically zero-referenced AC signals.
//
// Implementation:
//   - Stage 1: select NoCenter=true and the common-denominator estimator.
//   - Stage 2: delegate finite validation and scaled raw-product scans to Compute.
//   - Stage 3: publish Mean=0 and the owned lag/pair slices from Compute.
//
// Behavior highlights:
//   - Nonzero constant samples are valid and have Values[k]=(n-k)/n.
//   - The returned Mean is exactly positive zero, not an estimate of the physical DC level.
//
// Inputs:
//   - values: finite, read-only samples where the raw zero origin has domain meaning.
//   - maxLag: inclusive lag in sample indices, 0 <= maxLag < n.
//
// Returns:
//   - *Result: raw-second-moment ACF with CommonDenominator and Mean=0.
//   - error: nil on success; otherwise nil result and matching Compute error.
//
// Errors:
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrNonFinite, ErrZeroVariance, ErrOverflow:
//     same as Compute with NoCenter=true; all-zero samples produce ErrZeroVariance.
//
// Determinism:
//   - Delegates to Compute's increasing observation, lag, and pair order.
//
// Complexity:
//   - Time O(n*(maxLag+1)), Space O(n+maxLag), including output.
//
// Notes:
//   - Unlike centered ACF, this result is not invariant under additive offsets.
//
// AI-Hints:
//   - Do not use on biased ADC signals until their physical DC origin is handled explicitly.
func ComputeUncentered(values []float64, maxLag int) (*Result, error) {
	return Compute(values, maxLag, Options{NoCenter: true})
}

// ComputeUncenteredLagAdjusted computes pair-adjusted raw second-moment correlation.
// It combines explicit absence of sample-mean subtraction with n/(n-k) lag
// scaling, preserving a distinct policy useful for fixed-origin sensor signals.
//
// Implementation:
//   - Stage 1: select NoCenter and LagAdjusted together, with no implicit defaults.
//   - Stage 2: delegate all work to the canonical Compute facade.
//   - Stage 3: return the unchanged result and sentinel-classified failure.
//
// Behavior highlights:
//   - A nonzero constant signal produces exactly one at every admitted lag in exact arithmetic.
//   - Pair adjustment never clamps large positive or negative estimates.
//
// Inputs:
//   - values: finite, caller-owned fixed-origin observations, length n >= 1.
//   - maxLag: inclusive upper lag, 0 <= maxLag < n.
//
// Returns:
//   - *Result: NoCenter=true effective estimator with Mean=0 and LagAdjusted metadata.
//   - error: nil on success, otherwise nil result and matching Compute error.
//
// Errors:
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrNonFinite, ErrZeroVariance, ErrOverflow:
//     exactly the matching Compute error classes and precedence.
//
// Determinism:
//   - Delegates to Compute's increasing lag/pair scan and compensated summation order.
//
// Complexity:
//   - Time O(n*(maxLag+1)), Space O(n+maxLag), including output.
//
// Notes:
//   - The ratio of adjusted second moments is not an ordinary bounded correlation coefficient.
//
// AI-Hints:
//   - Never mistake Mean=0 for a measured zero offset; it indicates centering was disabled.
func ComputeUncenteredLagAdjusted(values []float64, maxLag int) (*Result, error) {
	return Compute(values, maxLag, Options{Denominator: LagAdjusted, NoCenter: true})
}
