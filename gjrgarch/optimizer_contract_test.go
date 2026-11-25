// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test verifies projected BFGS through analytically solvable
// public Fit instances and deterministic cancellation at actual polling points.
package gjrgarch_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

// TestGJR_ConvergenceAtAcceptedBudgetBoundary checks an exact constrained solution.
// The data are zeros; the Gaussian objective reduces to log(Omega/(1-rho))/2
// plus a constant. The minimum is at Omega=floor and all weights zero.
//
// Implementation:
//   - Stage 1: Start with Omega=1 and floor=1/2, so the normalized scale is one.
//   - Stage 2: The unit inverse metric projects the first step onto the vertex;
//     the Omega gradient is 1/2 and the weight gradients are outward normals.
//   - Stage 3: Check the analytic optimum and the final projected-gradient gate.
//
// Complexity: O(T) fixed-size objective work plus bounded active-face solves.
// AI-Hints: A budget of one accepted iteration can still converge. Do not
// report ErrNotConverged merely because the last allowed update was consumed.
func TestGJR_ConvergenceAtAcceptedBudgetBoundary(t *testing.T) {
	const floor = 0.5
	returns := []float64{0, 0, 0, 0}
	options := gjrgarch.Options{Initial: gjrgarch.Parameters{Omega: 1}, VarianceFloor: floor, MaxIterations: 1, Tolerance: 1e-12}
	fit, err := gjrgarch.Fit(context.Background(), returns, options)
	if err != nil || fit == nil || fit.Model == nil {
		t.Fatalf("analytic boundary optimum was not published: %v", err)
	}
	mustEqualBool(t, fit.Diagnostics.Converged, true, "budget-boundary convergence")
	mustEqualInt(t, fit.Diagnostics.Iterations, 1, "accepted updates")
	mustEqualInt(t, len(fit.Diagnostics.Objective), 2, "initial and accepted objectives")
	if fit.Model.Parameters() != (gjrgarch.Parameters{Omega: floor}) {
		t.Fatalf("wrong analytic optimum: %+v", fit.Model.Parameters())
	}
	wantBefore := 0.5 * math.Log(2*math.Pi)
	wantAfter := 0.5 * (math.Log(2*math.Pi) + math.Log(floor))
	mustNearRelative(t, fit.Diagnostics.Objective[0], wantBefore, 1e-13, "initial loss")
	mustNearRelative(t, fit.Diagnostics.Objective[1], wantAfter, 1e-13, "optimal loss")
	mustNearRelative(t, fit.LogLikelihood, -float64(len(returns))*wantAfter, 1e-13, "optimal likelihood")
}

// TestGJR_CancelAcrossOptimizerPollingPoints proves late-cancellation publication.
// The baseline uses the same valid data and explicit policy without cancellation.
// Each replay cancels at one actual Err poll, including dependency polls, without
// assuming a hard-coded number of QR-internal checks.
//
// Implementation:
//   - Stage 1: Obtain the complete reference result and count context polls.
//   - Stage 2: Re-run with cancellation at each observed boundary (or a bounded
//     evenly spaced subset for unusually expensive dependency implementations).
//   - Stage 3: Verify cancellation identity, no false convergence, exact history
//     prefix, and independent original-unit score for any retained candidate.
//
// Complexity: O(min(C,256)*F), where C is the reference poll count and F is one
// bounded one-update fit; no timers, random data, or parallel workers are needed.
// AI-Hints: A context error after the initial valid score can return a model,
// but a context error before that score must return a nil result.
func TestGJR_CancelAcrossOptimizerPollingPoints(t *testing.T) {
	const maxPollTrials = 256
	returns := []float64{0, 0, 0, 0}
	options := gjrgarch.Options{Initial: gjrgarch.Parameters{Omega: 1}, VarianceFloor: 0.5, MaxIterations: 1, Tolerance: 1e-12}
	baselineContext := newPollCancellationContext(t, 0)
	baseline, err := gjrgarch.Fit(baselineContext, returns, options)
	if err != nil || baseline == nil || !baseline.Diagnostics.Converged {
		t.Fatalf("reference fit must converge: %v", err)
	}
	// Always include the early entry/likelihood boundaries and the final polls.
	// Sample the interior if the QR dependency performs many cancellation checks.
	stride := 1 + (baselineContext.polls-1)/maxPollTrials
	var early, late, afterAccepted bool
	for stop := 1; stop <= baselineContext.polls; stop++ {
		if stop > 16 && stop < baselineContext.polls-16 && (stop-1)%stride != 0 {
			continue
		}
		ctx := newPollCancellationContext(t, stop)
		result, fitErr := gjrgarch.Fit(ctx, returns, options)
		ctx.cancel()
		if !errors.Is(fitErr, context.Canceled) {
			t.Fatalf("poll %d/%d: expected context cancellation, got %v", stop, baselineContext.polls, fitErr)
		}
		if result == nil {
			early = true
			// Only entry or initial score can fail without a publishable model.
			if stop > 2 {
				t.Fatalf("poll %d lost the validated initial candidate", stop)
			}
			continue
		}
		late = true
		if result.Model == nil || result.Diagnostics.Converged {
			t.Fatalf("poll %d published an invalid or converged partial result", stop)
		}
		accepted := result.Diagnostics.Iterations
		if accepted < 0 || accepted > baseline.Diagnostics.Iterations || len(result.Diagnostics.Objective) != accepted+1 {
			t.Fatalf("poll %d published inconsistent iteration history", stop)
		}
		afterAccepted = afterAccepted || accepted == 1
		for i, value := range result.Diagnostics.Objective {
			if value != baseline.Diagnostics.Objective[i] {
				t.Fatalf("poll %d changed reference objective prefix at %d", stop, i)
			}
		}
		score, evalErr := gjrgarch.EvaluateGaussianQMLE(context.Background(), returns, result.Model.Parameters(), options.VarianceFloor)
		if evalErr != nil || score == nil {
			t.Fatalf("poll %d returned an infeasible candidate: %v", stop, evalErr)
		}
		mustNearRelative(t, result.LogLikelihood, score.LogLikelihood, 1e-13, "partial likelihood")
	}
	if !early || !late || !afterAccepted {
		t.Fatalf("cancellation sweep missed publication phases: early=%t late=%t accepted=%t", early, late, afterAccepted)
	}
}

// TestGJR_QMLECancelAtSecondScanBlock verifies atomic score publication on
// interruption after the first complete block of valid observations.
// Complexity: O(T) time, O(T) fixture storage; no internal score state is exposed.
func TestGJR_QMLECancelAtSecondScanBlock(t *testing.T) {
	returns := make([]float64, 257)
	for i := range returns {
		returns[i] = float64(i%9-4) / 100
	}
	ctx := newPollCancellationContext(t, 3)
	result, err := gjrgarch.EvaluateGaussianQMLE(ctx, returns, p(), 0)
	mustErrorIs(t, err, context.Canceled)
	mustNilState(t, result, true, "interrupted Gaussian score")
}

// TestGJR_ArmijoBudgetExhaustionRetainsInitial proves an honest line-search stall.
// The explicit scale is one, the residual is 2^-20 and its squared value is
// twice Omega. The variance derivative therefore dominates the directional slope.
// Even the smallest admitted Armijo step demands a loss below the global floor
// bound; no trial is allowed to replace the initial accepted candidate.
//
// Complexity: O(T) per trial, bounded by 21 trials plus the fixed-size projector.
// AI-Hints: Never accept the least-bad trial after exhausting the Armijo budget.
func TestGJR_ArmijoBudgetExhaustionRetainsInitial(t *testing.T) {
	const residual = 0x1p-20
	const floor = 0x1p-41
	initial := gjrgarch.Parameters{Mean: 1 - residual, Omega: floor}
	returns := []float64{1}
	options := gjrgarch.Options{Initial: initial, VarianceFloor: floor, MaxIterations: 1}
	// With normalized scale one, the Omega-only contribution to g*d is -2^80.
	// All other projected components contribute nonpositively in this fixture.
	// The smallest Armijo sufficient-decrease bound is already impossible.
	initialLoss := 0.5 * (math.Log(2*math.Pi) + math.Log(floor) + 2)
	lowerBound := 0.5 * (math.Log(2*math.Pi) + math.Log(floor))
	const smallestStep = 0x1p-20
	const armijo = 1e-4
	if initialLoss-armijo*smallestStep*0x1p80 >= lowerBound {
		t.Fatal("fixture does not force the line search to stall")
	}
	fit, err := gjrgarch.Fit(context.Background(), returns, options)
	mustErrorIs(t, err, gjrgarch.ErrNotConverged)
	if fit == nil || fit.Model == nil || fit.Diagnostics.Converged {
		t.Fatal("line-search stall lost or mislabeled the initial candidate")
	}
	mustEqualInt(t, fit.Diagnostics.Iterations, 0, "accepted updates after Armijo stall")
	mustEqualInt(t, len(fit.Diagnostics.Objective), 1, "rejected trials must not enter history")
	if fit.Model.Parameters() != initial {
		t.Fatal("rejected line trial changed the retained model")
	}
	mustNearRelative(t, fit.Diagnostics.Objective[0], initialLoss, 1e-13, "retained initial loss")
	mustNearRelative(t, fit.LogLikelihood, -initialLoss, 1e-13, "retained log likelihood")
}

// TestGJR_FitPhysicalPublicationOverflow checks two distinct coordinate boundaries.
// A normalized iterate can be feasible even when its original-unit Omega or
// its implied long-run variance cannot be represented as a finite float64.
//
// Implementation:
//   - Stage 1: Use a power-of-two return scale whose square is 2^1024.
//   - Stage 2: Verify the same normalized fit publishes a finite improved model.
//   - Stage 3: Fit the physically scaled inputs; reject the unrepresentable result.
//
// Complexity: Two bounded one-update fits per case, with a singleton observation.
// AI-Hints: A successful normalized objective is not sufficient for publication;
// never return a partially unscaled or nonstationary Model.
func TestGJR_FitPhysicalPublicationOverflow(t *testing.T) {
	const scale = 0x1p512
	const normalizedFloor = 0x1p-24
	const physicalFloor = 0x1p1000
	cases := []struct {
		name              string
		omega             float64
		interceptOverflow bool
	}{
		// The accepted normalized intercept exceeds one.
		{"intercept", 0.25, true},
		// The intercept remains below one, but Omega/(1-rho) exceeds one.
		{"stationary_variance", 0.75, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := gjrgarch.Options{Initial: gjrgarch.Parameters{Omega: tc.omega}, VarianceFloor: normalizedFloor, MaxIterations: 1}
			fit, err := gjrgarch.Fit(context.Background(), []float64{1}, base)
			if err != nil && !errors.Is(err, gjrgarch.ErrNotConverged) {
				t.Fatal(err)
			}
			if fit == nil || fit.Model == nil || fit.Diagnostics.Iterations != 1 {
				t.Fatal("normalized fixture must accept one update")
			}
			candidate := fit.Model.Parameters()
			if (candidate.Omega > 1) != tc.interceptOverflow {
				t.Fatalf("unexpected normalized intercept: %.17g", candidate.Omega)
			}
			longRun, err := fit.Model.UnconditionalVariance()
			if err != nil || longRun <= 1 {
				t.Fatalf("fixture must overflow the physical stationary variance: %g %v", longRun, err)
			}
			// Multiply in two steps; the input Omega is representable even though scale^2 is not.
			physical := base
			physical.Initial.Omega = tc.omega * scale * scale
			physical.VarianceFloor = physicalFloor
			if _, err := gjrgarch.New(physical.Initial); err != nil {
				t.Fatalf("physical initial model is valid: %v", err)
			}
			actual, err := gjrgarch.Fit(context.Background(), []float64{scale}, physical)
			mustErrorIs(t, err, gjrgarch.ErrOverflow)
			mustNilState(t, actual, true, "unrepresentable original-unit candidate")
		})
	}
}

// TestGJR_FitTinyVarianceFloorActiveBoundary checks a stiff physical regime.
// A constant observed level equals the initial mean exactly. The variance score
// in the intercept coordinate can be enormous, but the intercept is already at
// its lower bound; the projected search must not mistake that raw gradient for
// a free direction. The objective is minimized by zero persistence.
// Complexity: A fixed number of one-update fits, each with O(1) observations.
// AI-Hints: Do not assert raw-gradient convergence or move Omega below the floor;
// an unusable BFGS secant must not corrupt the last accepted feasible model.
func TestGJR_FitTinyVarianceFloorActiveBoundary(t *testing.T) {
	cases := []struct {
		name  string
		floor float64
	}{
		{"wide_score", 1e-160},
		{"extreme_score", 1e-240},
		{"near_exponent_limit", 1e-300},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			initial := gjrgarch.Parameters{Mean: 1, Omega: tc.floor, Alpha: 0.1, Gamma: 0.2, Beta: 0.5}
			fit, err := gjrgarch.Fit(context.Background(), []float64{1}, gjrgarch.Options{Initial: initial, VarianceFloor: tc.floor, MaxIterations: 1, Tolerance: 1e-12})
			if err != nil || fit == nil || fit.Model == nil || !fit.Diagnostics.Converged {
				t.Fatalf("active-floor optimum was not published: %v", err)
			}
			mustEqualInt(t, fit.Diagnostics.Iterations, 1, "active-floor update")
			want := gjrgarch.Parameters{Mean: 1, Omega: tc.floor}
			if fit.Model.Parameters() != want {
				t.Fatalf("active-floor optimum drifted: %+v", fit.Model.Parameters())
			}
			mustEqualInt(t, len(fit.Diagnostics.Objective), 2, "active-floor history")
			mustNearRelative(t, fit.Diagnostics.Objective[1], referenceSingletonLoss(want, 1), 1e-13, "active-floor optimum")
		})
	}
}
