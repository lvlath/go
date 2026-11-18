// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion implements the bounded, discrete Poisson count kernel.
// This file owns product inversion, transformed rejection, and stable log mass.
// PTRS is based on W. Hoermann (1993); the constants follow NumPy 2.2.0's
// BSD-3-Clause-licensed distributions.c parameterization.
package jumpdiffusion

import (
	"context"
	"math"

	"github.com/lvlath/go/rng"
)

const (
	// poissonSwitch fixes the reproducibility boundary between inversion and PTRS.
	// Changing this number alters the UniformOpen consumption contract.
	poissonSwitch = 30.0

	// maxPoissonMean bounds accepted per-step intensity times elapsed model time.
	// Even at this maximum, PTRS work depends on rejection attempts, not the mean.
	maxPoissonMean = float64(uint64(1) << 40)

	// maxPoissonCount keeps every accepted integer exactly representable as float64.
	// That exact conversion is required by conditional normal jump aggregation.
	maxPoissonCount = uint64(1)<<53 - 1

	// maxPoissonAttempts bounds RNG calls for either count-sampling regime.
	// An uncooperative source cannot trap a simulation in an endless loop.
	maxPoissonAttempts = 1 << 20

	// poissonSeriesTerms bounds the evaluation of the stable deviance expansion.
	// For |d|<1/8, the omitted geometric tail is negligible over a<=2^40.
	poissonSeriesTerms  = 64
	poissonTaylorRadius = 0.125
	stirlingTableLimit  = 16

	// Hoermann PTRS coefficients form the fixed proposal envelope.
	// These are algorithm constants, not fitted distribution parameters.
	ptrsBaseB          = 0.931
	ptrsRootScale      = 2.53
	ptrsBaseC          = -0.059
	ptrsCScale         = 0.02483
	ptrsBaseAlpha      = 1.1239
	ptrsAlphaScale     = 1.1328
	ptrsAlphaOffset    = 3.4
	ptrsBaseSqueeze    = 0.9277
	ptrsSqueezeScale   = 3.6224
	ptrsSqueezeOffset  = 2.0
	ptrsProposalOffset = 0.43
	ptrsFastSqueezeUs  = 0.07
	ptrsThinRejectUs   = 0.013
)

// samplePoisson samples one discrete Poisson count from an already-owned RNG.
// It is the only dispatch point for the versioned inversion/PTRS selection.
//
// Implementation:
//   - Stage 1: reject unrepresentable means and the unsupported count domain.
//   - Stage 2: return zero without draws, or dispatch to one exact count kernel.
//   - Stage 3: propagate an accepted integer or the original failure category.
//
// Behavior highlights:
//   - The large-mean branch never approximates the count with a Gaussian.
//
// Inputs:
//   - ctx: non-nil context already validated by the public facade.
//   - a: finite mean in [0,2^40], measured per single grid interval.
//   - random: single-goroutine-owned stream, with open uniforms in (0,1).
//
// Returns:
//   - uint64: discrete count <=2^53-1; errors never publish a count.
//
// Errors:
//   - ErrOverflow: nonfinite computed mean; ErrDomain: unsupported mean/count.
//   - ErrNonFinite/ErrDomain: malformed RNG draw; ErrBudget: attempt exhaustion.
//   - ctx.Err(): cancellation, checked at least once per rejection attempt.
//
// Determinism:
//   - Small means draw one uniform per trial; PTRS draws exactly two per attempt.
//
// Complexity:
//   - Time O(B) and Space O(1) worst case, B=2^20; O(a+1) small-mean expected.
//
// Notes:
//   - UniformOpen is caller-owned and must not be shared concurrently.
//
// AI-Hints:
//   - Do not replace the PTRS branch with Knuth, normal rounding, or a new cutoff.
func samplePoisson(ctx context.Context, a float64, random rng.RNG) (uint64, error) {
	if !finite(a) {
		return 0, ErrOverflow
	}
	if a < 0 || a > maxPoissonMean {
		return 0, ErrDomain
	}
	if a == 0 {
		return 0, nil
	}
	if a < poissonSwitch {
		return poissonInversion(ctx, a, random)
	}
	return poissonPTRS(ctx, a, random)
}

// validateOpenUniform rejects external RNG implementations that break (0,1).
// Nonfinite values are classified separately from finite invalid endpoints.
//
// Inputs: draw is a candidate UniformOpen result, not a raw integer word.
// Returns: nil for a strict open uniform; otherwise the matching sentinel.
// Errors: ErrNonFinite for NaN/Inf; ErrDomain for closed/out-of-range samples.
// Determinism: no entropy consumption or mutation; Complexity: O(1) time/space.
// AI-Hints: checking only draw==0 is insufficient for foreign RNG providers.
func validateOpenUniform(draw float64) error {
	if !finite(draw) {
		return ErrNonFinite
	}
	if draw <= 0 || draw >= 1 {
		return ErrDomain
	}
	return nil
}

// poissonInversion samples the small-mean Poisson law by product threshold.
// The terminating index is the number of failures before product<=exp(-a).
//
// Implementation:
//   - Stage 1: establish the positive exponential threshold and unit product.
//   - Stage 2: multiply one validated uniform at a time, checking cancellation.
//   - Stage 3: return the exact discrete index or fail at the attempt budget.
//
// Inputs: a is strictly between zero and 30; ctx and random are validated.
// Returns: nonnegative Poisson count, or zero with error and no published step.
// Errors: ErrNonFinite, ErrDomain, ErrBudget, and ctx.Err().
// Determinism: exactly one UniformOpen per product trial, no Normal calls.
// Complexity: expected O(a+1), worst-case O(B), scratch O(1).
// Notes: exp(-a) cannot underflow in this branch; no floating Gaussian fallback.
// AI-Hints: moving the count increment before the threshold test is off by one.
func poissonInversion(ctx context.Context, a float64, random rng.RNG) (uint64, error) {
	// The a<30 gate ensures this probability is strictly positive in
	// binary64, so product inversion cannot underflow the threshold itself.
	threshold := math.Exp(-a)
	product := 1.0
	var count uint64

	for attempt := 0; attempt < maxPoissonAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		draw := random.UniformOpen()
		if err := validateOpenUniform(draw); err != nil {
			return 0, err
		}
		// Evaluate the stopping inequality before incrementing count: the
		// number of preceding unsuccessful products is the Poisson outcome.
		product *= draw
		if product <= threshold {
			return count, nil
		}
		count++
		if count > maxPoissonCount {
			return 0, ErrDomain
		}
	}
	return 0, ErrBudget
}

// ptrsParameters contains fixed coefficients for one Poisson mean.
// The proposal envelope is calculated once, not once per rejection attempt.
// These private fields never escape a sampling call or retain RNG state.
type ptrsParameters struct {
	mean, b, c, alpha, squeeze float64
}

// newPTRSParameters calculates the transformed-rejection proposal coefficients.
// All denominators are positive for the validated a>=30 domain.
//
// Inputs: mean is finite and within [30,2^40].
// Returns: immutable local PTRS envelope; Errors: none after prevalidation.
// Determinism: fixed operation order; Complexity: O(1) time/space.
// AI-Hints: modifying these coefficients changes acceptance probabilities.
func newPTRSParameters(mean float64) ptrsParameters {
	root := math.Sqrt(mean)
	b := ptrsBaseB + ptrsRootScale*root
	return ptrsParameters{
		mean:    mean,
		b:       b,
		c:       ptrsBaseC + ptrsCScale*b,
		alpha:   ptrsBaseAlpha + ptrsAlphaScale/(b-ptrsAlphaOffset),
		squeeze: ptrsBaseSqueeze - ptrsSqueezeScale/(b-ptrsSqueezeOffset),
	}
}

// poissonPTRS samples a large-mean Poisson count by transformed rejection.
// Each rejection attempt consumes exactly two open uniforms before validation.
//
// Implementation:
//   - Stage 1: derive the fixed proposal envelope from the positive mean.
//   - Stage 2: generate candidate counts and apply squeeze or log-PMF rejection.
//   - Stage 3: accept one exactly representable count or return bounded failure.
//
// Behavior highlights:
//   - A negative proposal is rejected, not clamped to zero.
//   - An oversized nonnegative proposal is ErrDomain, not a wrapped integer.
//
// Inputs: mean in [30,2^40]; ctx and random validated by the caller.
// Returns: accepted count <=2^53-1; a failed attempt never leaks a count.
// Errors: ErrDomain, ErrNonFinite, ErrBudget, ErrOverflow, and ctx.Err().
// Determinism: two UniformOpen calls per attempt; no Normal in this kernel.
// Complexity: O(B) worst-case and O(1) scratch, B=2^20.
// Notes: actual attempt count is stochastic and depends on the supplied uniforms.
// AI-Hints: keep the negative-count test before converting float64 to uint64.
func poissonPTRS(ctx context.Context, mean float64, random rng.RNG) (uint64, error) {
	parameters := newPTRSParameters(mean)
	for attempt := 0; attempt < maxPoissonAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return 0, err
		}

		// Both uniforms belong to one fixed-order PTRS proposal. Consume them
		// before inspecting the first value to keep malformed-stream behavior
		// deterministic and match the validated-input two-draw contract.
		first := random.UniformOpen()
		second := random.UniformOpen()
		if err := validateOpenUniform(first); err != nil {
			return 0, err
		}
		if err := validateOpenUniform(second); err != nil {
			return 0, err
		}

		count, accepted, err := parameters.acceptCandidate(first, second)
		if err != nil {
			return 0, err
		}
		if accepted {
			return count, nil
		}
	}
	return 0, ErrBudget
}

// acceptCandidate applies the PTRS proposal, squeeze, and exact log-mass test.
// It makes no RNG calls; the two candidate uniforms are already validated.
//
// Implementation:
//   - Stage 1: compute the proposal and reject a negative candidate.
//   - Stage 2: enforce exact-integer representability and the squeeze regions.
//   - Stage 3: compare a stable log Poisson mass to the transformed envelope.
//
// Inputs: first and second are validated independent UniformOpen draws.
// Returns: count, accepted flag; rejected proposals return (0,false,nil).
// Errors: ErrDomain for oversized count; ErrOverflow for invalid log mass.
// Determinism: identical proposal arithmetic and fixed branch ordering.
// Complexity: O(1) bounded finite work, O(1) scratch.
// Notes: no acceptance branch is allowed to bypass the count range check.
// AI-Hints: float-to-integer conversion before positivity/range gates is unsafe.
func (p ptrsParameters) acceptCandidate(first, second float64) (uint64, bool, error) {
	// The transformed proposal uses a centered uniform. A zero width means
	// division by zero and is always a rejected candidate, never a sample.
	u := first - 0.5
	us := 0.5 - math.Abs(u)
	if us <= 0 {
		return 0, false, nil
	}

	// The candidate must be checked as float64 before uint64 conversion;
	// conversions of negative or out-of-range values are not a sampling rule.
	proposal := math.Floor((2*p.c/us+p.b)*u + p.mean + ptrsProposalOffset)
	if proposal < 0 {
		return 0, false, nil
	}
	if !finite(proposal) || proposal > float64(maxPoissonCount) {
		return 0, false, ErrDomain
	}
	count := uint64(proposal)

	// The quick acceptance region is justified by PTRS's squeeze bound.
	// Always enforce the integer count domain before this early success.
	if us >= ptrsFastSqueezeUs && second <= p.squeeze {
		return count, true, nil
	}
	// Reject thin-edge proposals without paying for logarithms when the
	// transformed envelope already proves that they cannot be accepted.
	if us < ptrsThinRejectUs && second > us {
		return 0, false, nil
	}

	// All remaining candidates require a stable discrete Poisson log mass;
	// a rounded-normal surrogate would break the acceptance distribution.
	logMass, err := poissonLogPMF(count, p.mean)
	if err != nil {
		return 0, false, err
	}
	logEnvelope := math.Log(second) + math.Log(p.alpha) - math.Log(p.c/(us*us)+p.b)
	return count, logEnvelope <= logMass, nil
}

// poissonLogPMF returns a stable Poisson log probability at an integer count.
// The deviance/stirling representation avoids subtracting enormous lgamma terms.
//
// Implementation:
//   - Stage 1: handle zero exactly as log(P[N=0])=-mean.
//   - Stage 2: compute D(k,mean) with a near-center Taylor expansion.
//   - Stage 3: subtract Gaussian normalization and Stirling error.
//
// Inputs: count <=2^53-1, mean in [30,2^40], both prevalidated.
// Returns: finite log-PMF, or zero with ErrOverflow for invalid arithmetic.
// Errors: ErrOverflow on nonrepresentable computed log probability.
// Determinism: fixed arithmetic order and fixed 64-term maximum expansion.
// Complexity: O(poissonSeriesTerms) bounded time, O(1) scratch.
// Notes: the series removes catastrophic cancellation near count=mean.
// AI-Hints: direct -mean+count*log(mean)-Lgamma(count+1) loses precision.
func poissonLogPMF(count uint64, mean float64) (float64, error) {
	if count == 0 {
		return -mean, nil
	}

	observed := float64(count) // Exact because count <= 2^53-1.
	relative := (observed - mean) / mean
	var deviance float64
	// Near the mode, computing the three large log-factorial terms
	// separately destroys the tiny deviance. The bounded Taylor series
	// retains that information even at mean=2^40.
	if math.Abs(relative) < poissonTaylorRadius {
		deviance = poissonDevianceSeries(mean, relative)
	} else {
		// A large relative deviation is far from the mode, where direct
		// deviance has no near-zero catastrophic subtraction.
		deviance = observed*math.Log(observed/mean) + mean - observed
	}

	mass := -deviance - 0.5*math.Log(2*math.Pi*observed) - stirlingError(count)
	if !finite(mass) {
		return 0, ErrOverflow
	}
	return mass, nil
}

// poissonDevianceSeries computes mean*((1+d)*log1p(d)-d) without cancellation.
// For |d|<1/8, terms have a geometric tail strictly bounded after 64 steps.
//
// Inputs: mean in [30,2^40], relative=(count-mean)/mean with |relative|<1/8.
// Returns: nonnegative deviance (subject to IEEE-754 rounding); Errors: none.
// Determinism: ascending series order, exactly 64 bounded iterations.
// Complexity: O(poissonSeriesTerms) time, O(1) space.
// AI-Hints: direct log1p(d)*(1+d)-d destroys near-mode accuracy.
func poissonDevianceSeries(mean, relative float64) float64 {
	power := relative * relative
	sum := 0.0
	for degree := 2; degree <= poissonSeriesTerms; degree++ {
		term := power / float64(degree*(degree-1))
		sum += term
		power *= -relative
	}
	return mean * sum
}

// stirlingErrors stores high-precision integer factorial correction fixtures.
// Element zero is unused: the Poisson zero-count log mass is handled separately.
var stirlingErrors = [...]float64{
	0,
	0.081061466795327258, 0.041340695955409294, 0.02767792568499834,
	0.020790672103765093, 0.016644691189821193, 0.013876128823070748,
	0.01189670994589177, 0.010411265261972097, 0.009255462182712733,
	0.008330563433362871, 0.007573675487951841, 0.00694284010720953,
	0.006408994188004207, 0.005951370112758848, 0.005554733551962801,
}

// stirlingError returns log(k!) minus Stirling's logarithmic principal term.
// The small-integer table avoids approximation errors near the Poisson threshold.
//
// Inputs: count is a positive count <=2^53-1, already range-checked.
// Returns: positive correction to subtract in log-PMF; Errors: none.
// Determinism: integer table lookup or fixed alternating asymptotic polynomial.
// Complexity: O(1) time and memory.
// Notes: for count>=16 the first omitted Stirling term bounds truncation.
// AI-Hints: never use this asymptotic formula for k=0 or small tabulated k.
func stirlingError(count uint64) float64 {
	// The asymptotic polynomial is inappropriate for small integers; the
	// precomputed exact-log-factorial corrections are used there instead.
	if count < stirlingTableLimit {
		return stirlingErrors[count]
	}
	inverse := 1 / float64(count)
	inverseSquared := inverse * inverse
	return inverse * (1.0/12 + inverseSquared*(-1.0/360+
		inverseSquared*(1.0/1260+inverseSquared*(-1.0/1680+
			inverseSquared*(1.0/1188-inverseSquared*691.0/360360)))))
}
