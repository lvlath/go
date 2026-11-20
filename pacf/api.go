// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf exposes validated partial-autocorrelation facades.
// This file selects explicit centering policy and delegates to one numerical kernel.
package pacf

import "context"

// Compute estimates a finite-window partial autocorrelation profile with Durbin-Levinson.
// It uses biased sample autocovariance with one denominator n at every lag;
// no Burg, OLS, or pair-adjusted estimator is selected implicitly.
//
// Implementation:
//   - Stage 1: validate context, slice presence, lag shape, and input finiteness.
//   - Stage 2: construct the global centering frame and scan common-denominator covariances.
//   - Stage 3: run the single Levinson kernel and publish a complete detached Result.
//
// Behavior highlights:
//   - Options{} computes only uncentered lag zero; Center=true is explicit.
//   - Values[0] is exactly one; PredictionVariance[0] is gamma[0] in input units.
//   - The recurrence rejects nonpositive prediction variance, never epsilon-clamps it.
//
// Inputs:
//   - ctx: non-nil context; cancellation is polled during each substantial scan.
//   - values: non-nil, nonempty, finite chronological samples; read-only until return.
//   - options: inclusive MaxLag in [0,len(values)-1]; Center fits one full-window mean.
//
// Returns:
//   - *Result: Count=len(values), two independently owned slices of MaxLag+1 elements.
//   - error: nil on success; failures always return a nil *Result with no usable prefix.
//
// Errors:
//   - ErrInvalidOption: nil context or inclusive MaxLag outside the admitted range.
//   - ErrNilInput, ErrEmptyInput: absent or empty sample slice, respectively.
//   - ErrNonFinite: input contains NaN or infinity after successful shape validation.
//   - ErrZeroVariance: selected centered energy or raw second moment equals zero.
//   - ErrSingular: recurrence cannot maintain positive finite prediction variance.
//   - ErrOverflow: unrepresentable workspace byte product, arithmetic, or physical variance.
//   - ctx.Err(): canceled or expired context, including during computation.
//
// Determinism:
//   - Validation scans samples ascending; autocovariance scans k then t ascending.
//   - Levinson processes m then j ascending, reading coefficients from the previous row.
//
// Complexity:
//   - Time O(n*(L+1)+L^2), Space O(L+1), n=len(values), L=MaxLag.
//   - Space includes covariance, ping-pong coefficients, and both returned slices.
//
// Notes:
//   - Neither the input nor any result buffer is retained between calls.
//   - Concurrent calls may share immutable samples; concurrent input mutation is unsupported.
//   - PACF is descriptive: finite-sample coefficients do not prove an AR order or whitening.
//
// AI-Hints:
//   - Never feed lag-adjusted ACF coefficients to this Toeplitz-PSD-dependent recurrence.
//   - Do not substitute in-place coefficient updates or repair singular variance with epsilon.
//   - Use a fixed training window; fitting a new mean on held-out observations leaks information.
func Compute(ctx context.Context, values []float64, options Options) (*Result, error) {
	// Preserve the specified precedence before constructing any numerical workspace.
	if err := validate(ctx, values, options); err != nil {
		return nil, err
	}

	// The frame carries a stable global origin and scale, not an n-element copy.
	// Its scaled second moments remain in a safe arithmetic domain.
	f, err := prepare(ctx, values, options.Center)
	if err != nil {
		return nil, err
	}

	// Every lag reuses that frame and the same n denominator.
	gamma, err := autocov(ctx, values, options.MaxLag, f)
	if err != nil {
		return nil, err
	}

	// Only the canonical solver can publish a Result; intermediate slices never escape.
	return levinson(ctx, gamma, f.scale, len(values))
}

// ComputeCentered estimates PACF after fitting one mean over the entire input window.
// It is the explicit centered counterpart of the zero-value uncentered Compute policy.
//
// Implementation:
//   - Stage 1: select Center=true and the caller-supplied inclusive maximum lag.
//   - Stage 2: delegate validation, scaled covariances, and recursion to Compute.
//   - Stage 3: return the complete result or unchanged error classification.
//
// Behavior highlights:
//   - Nonzero constant series return ErrZeroVariance after mean removal.
//   - Changing the training window can change the fitted mean and every PACF lag.
//
// Inputs:
//   - ctx: non-nil, cancelable context.
//   - values: finite, chronological, caller-owned, read-only window.
//   - maxLag: inclusive order bound in [0,len(values)-1], with zero meaning lag zero only.
//
// Returns:
//   - *Result: detached centered partial correlations and prediction variances.
//   - error: matching Compute failure, always with nil result.
//
// Errors:
//   - All Compute sentinels and ctx.Err(), with identical validation precedence.
//
// Determinism:
//   - Identical to Compute(ctx, values, Options{MaxLag:maxLag, Center:true}).
//
// Complexity:
//   - Time O(n*(L+1)+L^2), Space O(L+1), n=len(values), L=maxLag.
//
// Notes:
//   - Prefer this facade for mean-adjusted AR identification on a fixed training window.
//
// AI-Hints:
//   - Centered PACF is not detrending and does not establish stationarity.
func ComputeCentered(ctx context.Context, values []float64, maxLag int) (*Result, error) {
	return Compute(ctx, values, Options{MaxLag: maxLag, Center: true})
}

// ComputeUncentered estimates PACF from raw second moments about physical zero.
// It preserves the observation origin instead of learning an intercept from the window.
//
// Implementation:
//   - Stage 1: select Center=false and the caller-specified inclusive maximum lag.
//   - Stage 2: delegate all validation and mathematical work to Compute.
//   - Stage 3: propagate the detached Result and error unchanged.
//
// Behavior highlights:
//   - A nonzero constant is valid: it has positive raw second moment.
//   - Unlike centered PACF, adding an offset generally changes the coefficients.
//
// Inputs:
//   - ctx: non-nil, cancelable context.
//   - values: finite read-only samples with a meaningful physical zero.
//   - maxLag: inclusive order bound in [0,len(values)-1].
//
// Returns:
//   - *Result: raw-second-moment PACF profile and physical prediction variances.
//   - error: any Compute error, with no partial result.
//
// Errors:
//   - All Compute sentinels and ctx.Err(), with identical precedence and semantics.
//
// Determinism:
//   - Identical to Compute(ctx, values, Options{MaxLag:maxLag, Center:false}).
//
// Complexity:
//   - Time O(n*(L+1)+L^2), Space O(L+1), n=len(values), L=maxLag.
//
// Notes:
//   - Useful for zero-referenced signed order flow or calibrated sensor residuals.
//
// AI-Hints:
//   - Do not treat Center=false as an optimization that leaves a fitted mean unchanged.
func ComputeUncentered(ctx context.Context, values []float64, maxLag int) (*Result, error) {
	return Compute(ctx, values, Options{MaxLag: maxLag})
}

// ComputeAtLag returns the partial autocorrelation coefficient at one requested lag.
// It computes the full Durbin-Levinson prefix, so intermediate prediction-error
// variances retain the exact same validation and singularity semantics as Compute.
//
// Implementation:
//   - Stage 1: construct explicit MaxLag=lag and Center=center options.
//   - Stage 2: delegate all numerical work and checks to Compute.
//   - Stage 3: project Values[lag], discarding the internally owned complete prefix.
//
// Behavior highlights:
//   - Lag zero returns exactly one only if the selected energy is positive.
//   - The wrapper never bypasses failed lower-order recurrence steps.
//
// Inputs:
//   - ctx: non-nil context, including cancellation support.
//   - values: finite read-only chronological training observations.
//   - lag: integer in [0,len(values)-1], the single requested output lag.
//   - center: true to fit a global mean; false to preserve physical zero.
//
// Returns:
//   - float64: coefficient kappa[lag] (one on lag zero) on success.
//   - error: same class as Compute; returned scalar is meaningless on failure.
//
// Errors:
//   - The same sentinels, validation order, and context errors as Compute.
//
// Determinism:
//   - Exactly the ascending-lag computations of Compute, projected at index lag.
//
// Complexity:
//   - Time O(n*(lag+1)+lag^2), Space O(lag+1), including the discarded prefix.
//
// Notes:
//   - For many lag queries, compute one Result and reuse Values instead of re-running.
//
// AI-Hints:
//   - This is not a direct covariance ratio when lag > 1; it conditions on earlier lags.
func ComputeAtLag(ctx context.Context, values []float64, lag int, center bool) (float64, error) {
	result, err := Compute(ctx, values, Options{MaxLag: lag, Center: center})
	if err != nil {
		return 0, err
	}
	return result.Values[lag], nil
}
