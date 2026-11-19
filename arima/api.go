// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima defines the canonical conditional Gaussian training and forecasting API.
// This file validates public calls, owns model snapshots, and provides policy-only shortcuts.
package arima

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"math"
)

const (
	modelMethodVersion   = "lvlath/arima/conditional-gaussian/v1"
	forecastFloatBuffers = 4 // Predicted differences, levels, marginal variances, impulse response.
	float64Bytes         = 8
)

// Fit estimates a fixed-order univariate ARIMA model by conditional Gaussian MLE.
// The model's intercept is in the differenced equation; the first max(p,q)
// differenced observations condition the innovations and do not enter likelihood.
//
// Implementation:
//   - Stage 1: validate context, order/solver options, input shape and finite samples.
//   - Stage 2: detach d-th differences and run deterministic profile-likelihood L-BFGS.
//   - Stage 3: publish an immutable model and detached accepted objective history.
//
// Behavior highlights:
//   - Order{0,0,0} fits white noise, never selects a model order automatically.
//   - MaxIterations=0 selects 200; Tolerance=0 selects 1e-7.
//   - An exhausted iteration budget yields a valid *FitResult and ErrNotConverged.
//
// Inputs:
//   - ctx: non-nil active context, checked before options and input access.
//   - values: non-nil finite observations, ascending time; read-only and not retained.
//   - options: fixed p<=8, d<=2, q<=8; IncludeDrift explicitly fits c.
//
// Returns:
//   - *FitResult: independent Model, likelihood and accepted optimizer diagnostics.
//   - error: nil on convergence; only ErrNotConverged permits a non-nil result.
//
// Errors:
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrInsufficientData, ErrNonFinite:
//     rejected admission policy, presence, size or observations in that order.
//   - ErrNonStationary, ErrNonInvertible, ErrSingular, ErrOverflow:
//     invalid optimizer candidate/conditional variance or unrepresentable arithmetic.
//   - ErrNotConverged: finite accepted model with no proven gradient convergence.
//   - ctx.Err(): cancellation, without a published partial model.
//
// Determinism:
//   - Samples, lags, gradient coordinates and accepted corrections are processed
//     by increasing index; rejected search steps halve deterministically.
//
// Complexity:
//   - Time O((I+1)*N*(p+q+1)^2), Space O(N*(p+q+1)+I), N=len(values)-d.
//
// Notes:
//   - Inputs cannot be mutated concurrently by their owner during the call.
//   - Conditional MLE is not exact diffuse state-space/Kalman likelihood.
//
// AI-Hints:
//   - Do not interpret Drift as the unconditional level mean when p>0 or d>0.
//   - Check errors.Is(err, ErrNotConverged) before using a partial model for decisions.
func Fit(ctx context.Context, values []float64, options Options) (*FitResult, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	validated, err := finalizeOptions(options)
	if err != nil {
		return nil, err
	}
	if values == nil {
		return nil, ErrNilInput
	}
	if len(values) == 0 {
		return nil, ErrEmptyInput
	}

	burnIn := maxN(validated.Order.AR, validated.Order.MA)
	parameterCount := validated.Order.AR + validated.Order.MA + 1 // Variance is profiled.
	if validated.IncludeDrift {
		parameterCount++
	}
	if len(values)-validated.Order.Difference-burnIn <= parameterCount {
		return nil, ErrInsufficientData
	}
	if err := checkSeries(values, 1); err != nil {
		return nil, err
	}
	differenced, err := difference(values, validated.Order.Difference)
	if err != nil {
		return nil, err
	}
	return fitKernel(ctx, differenced, validated)
}

// FitWithDrift estimates a fixed ARIMA order with an explicit differenced-space intercept.
// It selects IncludeDrift=true without modifying the common estimation kernel.
//
// Implementation:
//   - Stage 1: assemble the fixed order and drift-enabled policy.
//   - Stage 2: delegate all validations and conditional estimation to Fit.
//   - Stage 3: return Fit's complete result and unchanged error classification.
//
// Behavior highlights:
//   - Solver iterations and gradient tolerance use the ordinary Fit defaults.
//
// Inputs:
//   - ctx: non-nil active context; values: read-only finite time-ordered samples.
//   - order: fixed AR/Difference/MA orders; no automatic order selection.
//
// Returns:
//   - *FitResult: detached fit or last accepted fit on ErrNotConverged.
//
// Errors:
//   - Same sentinel categories and validation precedence as Fit with this policy.
//
// Determinism:
//   - Exactly the canonical Fit ordering and L-BFGS acceptance history.
//
// Complexity:
//   - Time O((I+1)*N*(p+q+1)^2), Space O(N*(p+q+1)+I).
//
// Notes:
//   - Use Fit directly to specify a non-default iteration limit or tolerance.
//
// AI-Hints:
//   - With d=1 and AR coefficients, mean increment is c/(1-sum(phi)), not c.
func FitWithDrift(ctx context.Context, values []float64, order Order) (*FitResult, error) {
	return Fit(ctx, values, Options{Order: order, IncludeDrift: true})
}

// FitWithoutDrift estimates a fixed ARIMA order with c fixed to positive zero.
// This represents a real no-intercept constraint, not subtraction of a sample mean.
//
// Implementation:
//   - Stage 1: select the order and IncludeDrift=false.
//   - Stage 2: delegate to the canonical Fit validation and conditional kernel.
//   - Stage 3: forward the complete result and unchanged sentinel error.
//
// Behavior highlights:
//   - The drift parameter is not fitted and is never silently reintroduced.
//
// Inputs:
//   - ctx: active context; values: finite read-only ordered observations.
//   - order: fixed p/d/q, including valid all-zero white-noise order.
//
// Returns:
//   - *FitResult: model with Drift=0, or nil except on ErrNotConverged.
//
// Errors:
//   - Identical to Fit(ctx, values, Options{Order: order}).
//
// Determinism:
//   - Identical to canonical Fit for the selected order.
//
// Complexity:
//   - Time O((I+1)*N*(p+q+1)^2), Space O(N*(p+q+1)+I).
//
// Notes:
//   - Zero-mean in stationary space is not equivalent to centering an integrated level series.
//
// AI-Hints:
//   - Do not use this shortcut when a physical trend/drift is required; select FitWithDrift.
func FitWithoutDrift(ctx context.Context, values []float64, order Order) (*FitResult, error) {
	return Fit(ctx, values, Options{Order: order})
}

// FitARMA estimates a stationary ARMA(p,q) baseline with no differencing.
// It is convenient for spreads, deviations and other already-stationary signals.
//
// Implementation:
//   - Stage 1: construct Order{AR:p, Difference:0, MA:q} explicitly.
//   - Stage 2: delegate data validation and optimization to Fit.
//   - Stage 3: publish Fit's immutable model or its documented partial result.
//
// Behavior highlights:
//   - includeDrift toggles a fitted ARMA intercept, not an automatic centering pass.
//
// Inputs:
//   - ctx: non-nil context; values: finite read-only stationary-domain history.
//   - ar, ma: nonnegative polynomial orders <=8; includeDrift: fit intercept if true.
//
// Returns:
//   - *FitResult: complete conditional ARMA calibration and diagnostics.
//
// Errors:
//   - Identical to Fit for the assembled order and drift option.
//
// Determinism:
//   - Delegates the same causal residual scan and bounded L-BFGS path.
//
// Complexity:
//   - Time O((I+1)*N*(ar+ma+1)^2), Space O(N*(ar+ma+1)+I).
//
// Notes:
//   - Stationarity is imposed on AR parameters, not proved for the observed process.
//
// AI-Hints:
//   - A fitted c implies stationary mean c/(1-sum(phi)); it is not the mean itself.
func FitARMA(ctx context.Context, values []float64, ar, ma int, includeDrift bool) (*FitResult, error) {
	return Fit(ctx, values, Options{Order: Order{AR: ar, MA: ma}, IncludeDrift: includeDrift})
}

// FitIntegrated fits an innovation-driven ARIMA(0,d,0) level or trend baseline.
// It is useful for random walks and local linear trends where autoregressive
// and moving-average lag coefficients are deliberately absent.
//
// Implementation:
//   - Stage 1: select p=q=0 with caller-specified differencing and intercept policy.
//   - Stage 2: let Fit apply the analytic white-noise solution in differenced space.
//   - Stage 3: publish the fitted model and variance with no L-BFGS steps.
//
// Behavior highlights:
//   - d=0 fits white noise; d=1 a random walk; d=2 an integrated trend.
//   - includeDrift fits a constant mean d-th difference, not a level intercept.
//
// Inputs:
//   - ctx: non-nil context; values: finite read-only ordered level observations.
//   - difference: d in [0,2]; includeDrift: fit mean differenced innovation if true.
//
// Returns:
//   - *FitResult: analytic solution with Diagnostics.Iterations=0 on success.
//
// Errors:
//   - Same validation, singularity and overflow categories as canonical Fit.
//
// Determinism:
//   - Differences and squared residuals scan in increasing sample order.
//
// Complexity:
//   - Time O(N), Space O(N), N=len(values), including differencing buffers.
//
// Notes:
//   - This wrapper does not perform auto-order search or seasonal differencing.
//
// AI-Hints:
//   - For d=2, variance growth is cubic-order in horizon even when c=0.
func FitIntegrated(ctx context.Context, values []float64, difference int, includeDrift bool) (*FitResult, error) {
	return Fit(ctx, values, Options{Order: Order{Difference: difference}, IncludeDrift: includeDrift})
}

// New validates and freezes one explicitly parameterized ARIMA model.
// Its coefficients must have the requested lengths, finite values and strictly
// stable/invertible AR/MA polynomials; the supplied innovation variance is positive.
//
// Implementation:
//   - Stage 1: validate shapes, numeric domain and Schur reflection constraints.
//   - Stage 2: own copies of both coefficient slices and hash canonical coefficients.
//   - Stage 3: publish immutable parameters and a deterministic versioned content ID.
//
// Behavior highlights:
//   - No coefficient clipping, estimation, rescaling or history retention occurs.
//   - Signed zeros hash identically because they represent the same parameters.
//
// Inputs:
//   - parameters: complete physical AR/MA coefficients, c, sigma^2 and p/d/q.
//
// Returns:
//   - *Model: immutable snapshot, or nil on every admission error.
//
// Errors:
//   - ErrInvalidOption: inconsistent lengths or polynomial order range.
//   - ErrNonFinite: any nonfinite caller-supplied coefficient or variance.
//   - ErrSingular: nonpositive innovation variance.
//   - ErrNonStationary, ErrNonInvertible: closed or interior polynomial roots.
//
// Determinism:
//   - Hash inputs are written in fixed order: version, p/d/q, AR, MA, c, variance.
//
// Complexity:
//   - Time O(p^2+q^2), Space O(p+q); p,q<=8.
//
// Notes:
//   - A shared model is safe for concurrent Forecast calls with distinct histories.
//
// AI-Hints:
//   - New does not infer IncludeDrift; Drift is already the physical intercept.
func New(parameters Parameters) (*Model, error) {
	if err := checkParameters(parameters); err != nil {
		return nil, err
	}

	owned := parameters
	owned.AR = append([]float64(nil), parameters.AR...)
	owned.MA = append([]float64(nil), parameters.MA...)

	hashState := sha256.New()
	_, _ = hashState.Write([]byte(modelMethodVersion))
	_, _ = fmt.Fprintf(hashState, "/p=%d/d=%d/q=%d/", owned.Order.AR, owned.Order.Difference, owned.Order.MA)
	for _, coefficient := range owned.AR {
		writeCanonicalFloat(hashState, coefficient)
	}
	for _, coefficient := range owned.MA {
		writeCanonicalFloat(hashState, coefficient)
	}
	writeCanonicalFloat(hashState, owned.Drift)
	writeCanonicalFloat(hashState, owned.InnovationVariance)

	return &Model{parameters: owned, id: hex.EncodeToString(hashState.Sum(nil))}, nil
}

// writeCanonicalFloat adds one finite binary64 value to a model-content hash.
// The scalar is already domain-validated; both signs of zero use the same bits.
// Time O(1), Space O(1), with no order-dependent string floating-point formatting.
func writeCanonicalFloat(destination hash.Hash, value float64) {
	if value == 0 {
		value = 0 // Canonical positive zero.
	}
	var bytes [float64Bytes]byte
	binary.LittleEndian.PutUint64(bytes[:], math.Float64bits(value))
	_, _ = destination.Write(bytes[:])
}

// ID returns a deterministic content identifier for a validated immutable model.
// The identifier encodes initialization policy/version, order and canonical coefficients.
//
// Implementation:
//   - Stage 1: handle a nil receiver without dereference.
//   - Stage 2: read the immutable precomputed SHA-256 hexadecimal identifier.
//   - Stage 3: return it without modifying model state.
//
// Behavior highlights:
//   - An uninitialized or nil Model has an empty ID.
//
// Inputs:
//   - m: immutable model receiver; a nil receiver is accepted.
//
// Returns:
//   - string: stable 64-character lowercase hexadecimal hash, or "" for nil/zero model.
//
// Errors:
//   - None.
//
// Determinism:
//   - Hash is constructed once in New with a fixed binary layout and version salt.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - IDs are model fingerprints, not secrecy/authentication claims.
//
// AI-Hints:
//   - Do not assume models from changed initialization/likelihood versions share IDs.
func (m *Model) ID() string {
	if m == nil {
		return ""
	}
	return m.id
}

// Parameters returns the physical coefficients of an immutable fitted or supplied model.
// The AR and MA arrays are detached, so callers can mutate them without altering
// the receiver, forecasts, or the model content identifier.
//
// Implementation:
//   - Stage 1: return a zero Parameters value for a nil receiver.
//   - Stage 2: value-copy order, drift and variance and deep-copy coefficient slices.
//   - Stage 3: return the independent snapshot.
//
// Behavior highlights:
//   - An empty coefficient slice represents a genuine zero-order AR/MA component.
//
// Inputs:
//   - m: immutable receiver, potentially nil.
//
// Returns:
//   - Parameters: detached coefficient vectors plus physical scalar parameters.
//
// Errors:
//   - None.
//
// Determinism:
//   - AR and MA coefficients preserve their original increasing-lag order.
//
// Complexity:
//   - Time O(p+q), Space O(p+q).
//
// Notes:
//   - Nil receiver returns zero Parameters; those values are not a valid New input.
//
// AI-Hints:
//   - Mutation of the returned slices does not tune the model; construct a new Model.
func (m *Model) Parameters() Parameters {
	if m == nil {
		return Parameters{}
	}
	parameters := m.parameters
	parameters.AR = append([]float64(nil), parameters.AR...)
	parameters.MA = append([]float64(nil), parameters.MA...)
	return parameters
}

// Forecast computes conditional future levels and marginal innovation variances.
// Historical residuals follow the same b=max(p,q) initialization as Fit;
// future innovations have zero conditional mean and their response determines variance.
//
// Implementation:
//   - Stage 1: validate context, initialized model, horizon and finite history.
//   - Stage 2: reconstruct residuals and forecast differences using only this origin.
//   - Stage 3: integrate historical anchors and squared integrated impulse responses.
//
// Behavior highlights:
//   - Future observations are never read; the receiver remains immutable.
//   - Forecast variance excludes parameter-estimation and exogenous uncertainty.
//
// Inputs:
//   - ctx: non-nil active context; history: finite read-only consecutive levels.
//   - horizon: number of future samples >=1, bounded by checked slice sizes.
//
// Returns:
//   - *ForecastResult: owned Mean/Variance length horizon; nil on every error.
//
// Errors:
//   - ErrInvalidOption: nil/zero model or nonpositive horizon.
//   - ErrNilInput, ErrEmptyInput, ErrInsufficientData, ErrNonFinite:
//     missing/short/unrepresentable caller history, in that order.
//   - ErrOverflow: differencing, integration, impulse recursion or variance overflow.
//   - ctx.Err(): cancellation, with no usable partial forecast.
//
// Determinism:
//   - Causal history, forecast steps, polynomial lags and impulse indices ascend.
//
// Complexity:
//   - Time O((N+H)*(p+q+d+1)), Space O(N+H), N=history length.
//
// Notes:
//   - Shared Model permits parallel Forecast calls; concurrent mutation of history is unsupported.
//
// AI-Hints:
//   - For d=2, preserve both last level and last first difference; no zero-origin integration.
//   - Interpret Variance as marginal prediction error variance, not parameter confidence.
func (m *Model) Forecast(ctx context.Context, history []float64, horizon int) (*ForecastResult, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if horizon <= 0 || m == nil || m.id == "" {
		return nil, ErrInvalidOption
	}
	if history == nil {
		return nil, ErrNilInput
	}
	if len(history) == 0 {
		return nil, ErrEmptyInput
	}

	parameters := m.parameters
	minHistory := parameters.Order.Difference + maxN(parameters.Order.AR, parameters.Order.MA) + 1
	if len(history) < minHistory {
		return nil, ErrInsufficientData
	}
	if err := checkSeries(history, 1); err != nil {
		return nil, err
	}
	// Four horizon-dependent float buffers are allocated; fail before any partial result.
	if horizon > int(^uint(0)>>1)/(float64Bytes*forecastFloatBuffers) {
		return nil, ErrOverflow
	}
	return forecastKernel(ctx, parameters, history, horizon)
}

// ForecastNext is a one-step conditional forecast using the canonical model kernel.
// It retains both mean and marginal innovation variance in a complete Result.
//
// Implementation:
//   - Stage 1: select horizon=1 without constructing alternative predictions.
//   - Stage 2: delegate validation, residual recomputation and integration to Forecast.
//   - Stage 3: return the same owned result and error unchanged.
//
// Behavior highlights:
//   - Returned Horizon=1 and Mean/Variance have one entry.
//
// Inputs:
//   - ctx: non-nil context; history: non-nil finite history meeting model lag requirements.
//
// Returns:
//   - *ForecastResult: owned next-step level and innovation variance.
//
// Errors:
//   - Exactly Forecast's sentinel and cancellation classifications for horizon 1.
//
// Determinism:
//   - Same causal historical residual and polynomial scan order as Forecast.
//
// Complexity:
//   - Time O(N*(p+q+d+1)), Space O(N), for fixed one-step horizon.
//
// Notes:
//   - This wrapper deliberately does not discard variance or create scalar return values.
//
// AI-Hints:
//   - Use Forecast for longer planning windows; never repeatedly call ForecastNext
//     while using predicted values as if they were observed innovations.
func (m *Model) ForecastNext(ctx context.Context, history []float64) (*ForecastResult, error) {
	return m.Forecast(ctx, history, 1)
}
