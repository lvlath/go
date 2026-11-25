// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External test helpers compare public contracts without reflection or
// assertion frameworks. Every helper below is used by independent sax_test
// fixtures to keep sentinel, nil-state, and numerical failures actionable.
package sax_test

import (
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/sax"
)

// mustErrorIs checks a branchable sentinel, never the unstable error string.
func mustErrorIs(t *testing.T, err, sentinel error) {
	t.Helper()
	if !errors.Is(err, sentinel) {
		t.Fatalf("error=%v; expected errors.Is(..., %v)", err, sentinel)
	}
}

// isNilLike follows the same typed-nil core.Nilable protocol as the library.
// It is intentionally not a general reflection-based nil detector.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(core.Nilable); ok {
		return nilable.IsNil()
	}
	return false
}

// mustNilState distinguishes a typed-nil pointer from a non-nil empty result.
func mustNilState(t *testing.T, value any, wantNil bool, operation string) {
	t.Helper()
	if got := isNilLike(value); got != wantNil {
		t.Fatalf("%s: nil-state=%v; want %v", operation, got, wantNil)
	}
}

// mustEqualBool checks deterministic flags such as Degenerate and IsNil.
func mustEqualBool(t *testing.T, got, want bool, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t; want %t", operation, got, want)
	}
}

// mustEqualInt checks sample, alphabet, segment, and symbol dimensions.
func mustEqualInt(t *testing.T, got, want int, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d; want %d", operation, got, want)
	}
}

// mustEqualString checks canonical hashes or audited labels exactly.
func mustEqualString(t *testing.T, got, want, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q; want %q", operation, got, want)
	}
}

// mustEqualSymbols asserts position and shape, not only a vague slice mismatch.
func mustEqualSymbols(t *testing.T, got, want []uint16, operation string) {
	t.Helper()
	mustEqualInt(t, len(got), len(want), operation+" length")
	for index, symbol := range want {
		if got[index] != symbol {
			t.Fatalf("%s[%d]: got %d; want %d", operation, index, got[index], symbol)
		}
	}
}

// mustNear checks finite values with an explicit absolute/relative oracle budget.
func mustNear(t *testing.T, got, want, atol, rtol float64, operation string) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) {
		t.Fatalf("%s: non-finite got=%g want=%g", operation, got, want)
	}
	budget := atol + rtol*math.Max(math.Abs(got), math.Abs(want))
	if math.Abs(got-want) > budget {
		t.Fatalf("%s: got %.17g; want %.17g; allowed %.4g", operation, got, want, budget)
	}
}

// mustEncode performs one public encode and fails immediately on missing output.
// It keeps error paths in individual tests separate from success assertions.
func mustEncode(t *testing.T, values []float64, opts sax.Options) *sax.Result {
	t.Helper()
	result, err := sax.Encode(values, opts)
	if err != nil || result == nil {
		t.Fatalf("Encode(%v): result=%v error=%v", values, result, err)
	}
	return result
}

// exactPopulationZ is an independent oracle on exact original binary64 values.
// The mean and centered population variance are rational, not scaled float64
// approximations. Only sqrt and final coordinate quotients use a 256-bit float.
// Fixtures must be finite and nonconstant. Work is intentionally test-only.
func exactPopulationZ(values []float64) ([]float64, float64, float64) {
	const oraclePrecision = 256
	count := big.NewRat(int64(len(values)), 1)
	mean := new(big.Rat)
	for _, value := range values {
		mean.Add(mean, new(big.Rat).SetFloat64(value))
	}
	mean.Quo(mean, count)
	residuals := make([]*big.Rat, len(values))
	variance := new(big.Rat)
	for index, value := range values {
		residuals[index] = new(big.Rat).Sub(new(big.Rat).SetFloat64(value), mean)
		variance.Add(variance, new(big.Rat).Mul(residuals[index], residuals[index]))
	}
	variance.Quo(variance, count)
	sigma := new(big.Float).SetPrec(oraclePrecision).SetRat(variance)
	sigma.Sqrt(sigma)
	out := make([]float64, len(values))
	for index, residual := range residuals {
		numerator := new(big.Float).SetPrec(oraclePrecision).SetRat(residual)
		out[index], _ = numerator.Quo(numerator, sigma).Float64()
	}
	mean64, _ := mean.Float64()
	sigma64, _ := sigma.Float64()
	return out, mean64, sigma64
}

// exactAffineFixture proves the requested translation and power-of-two scale
// occurred exactly in binary64 before testing normalization invariance.
// A failed fixture is a test construction error, not a SAX failure.
func exactAffineFixture(t *testing.T, values []float64, offset float64, exponent int) []float64 {
	t.Helper()
	factor := new(big.Rat).SetFloat64(math.Ldexp(1, exponent))
	origin := new(big.Rat).SetFloat64(offset)
	out := make([]float64, len(values))
	for index, value := range values {
		out[index] = math.Ldexp(offset+value, exponent)
		want := new(big.Rat).Add(origin, new(big.Rat).SetFloat64(value))
		want.Mul(want, factor)
		if new(big.Rat).SetFloat64(out[index]).Cmp(want) != 0 {
			t.Fatalf("fixture %d rounded before encoding: offset=%g exponent=%d", index, offset, exponent)
		}
	}
	return out
}

// assertPopulationOracle checks a public identity-PAA fit against exact moments.
// No relative tolerance is used for dimensionless Z coordinates.
func assertPopulationOracle(t *testing.T, result *sax.Result, values []float64) {
	t.Helper()
	z, mean, sigma := exactPopulationZ(values)
	mustEqualInt(t, len(result.PAA), len(z), "identity PAA length")
	mustEqualBool(t, result.Degenerate, false, "nonconstant window")
	mustNear(t, result.Mean, mean, 0, 2e-15, "oracle mean")
	mustNear(t, result.StdDev, sigma, 0, 2e-15, "oracle population sigma")
	for index, value := range z {
		mustNear(t, result.PAA[index], value, 1e-12, 0, "oracle Z coordinate")
	}
}

// rationalPAA reproduces each half-open segment's exact weighted integral of
// binary64 input samples, using integer grid coordinates and arbitrary-precision
// rationals. It is intentionally independent of the package's PAA implementation.
func rationalPAA(values []float64, segments int) []*big.Rat {
	length := len(values)
	out := make([]*big.Rat, segments)
	for j := range out {
		sum := new(big.Rat)
		left, right := j*length, (j+1)*length
		for i, value := range values {
			lo := max(left, i*segments)
			hi := min(right, (i+1)*segments)
			if hi <= lo {
				continue
			}
			weight := big.NewRat(int64(hi-lo), 1)
			term := new(big.Rat).SetFloat64(value)
			term.Mul(term, weight)
			sum.Add(sum, term)
		}
		out[j] = sum.Quo(sum, big.NewRat(int64(length), 1))
	}
	return out
}

// rationalEuclideanSquared treats the supplied binary64 observations as exact
// rational numbers, then sums their squared differences with no rounding.
func rationalEuclideanSquared(left, right []float64) *big.Rat {
	sum := new(big.Rat)
	for i, x := range left {
		diff := new(big.Rat).Sub(new(big.Rat).SetFloat64(x), new(big.Rat).SetFloat64(right[i]))
		sum.Add(sum, diff.Mul(diff, diff))
	}
	return sum
}

// rationalPAASquared integrates exact squared differences of fractional PAA
// means; the segment duration factor is n/w, NOT floor(n/w).
func rationalPAASquared(left, right []*big.Rat, originalLength int) *big.Rat {
	result := new(big.Rat)
	for i := range left {
		diff := new(big.Rat).Sub(left[i], right[i])
		result.Add(result, diff.Mul(diff, diff))
	}
	result.Mul(result, big.NewRat(int64(originalLength), int64(len(left))))
	return result
}

// rationalSquareRoot converts an exact positive rational into a high-precision
// float before square-rooting; this avoids an intermediate float64 overflow.
func rationalSquareRoot(x *big.Rat) float64 {
	if x.Sign() == 0 {
		return 0
	}
	hi := new(big.Float).SetPrec(256).SetRat(x)
	hi.Sqrt(hi)
	value, _ := hi.Float64()
	return value
}

// referenceZ fits in original units using a separate two-pass population
// calculation. Moderately scaled test inputs avoid its naive extreme-domain
// limitations; the production scaled kernel is separately tested at extremes.
func referenceZ(values []float64) ([]float64, float64, float64) {
	mean := 0.0
	for _, value := range values {
		mean += value
	}
	mean /= float64(len(values))
	squared := 0.0
	for _, value := range values {
		delta := value - mean
		squared += delta * delta
	}
	sigma := math.Sqrt(squared / float64(len(values)))
	out := make([]float64, len(values))
	for i, value := range values {
		out[i] = (value - mean) / sigma
	}
	return out, mean, sigma
}
