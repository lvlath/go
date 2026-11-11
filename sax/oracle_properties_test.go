// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External mathematical oracles use exact rational fractional geometry and
// independent two-pass population fitting. They do not call paa.Transform,
// private sax helpers, or the production metric to calculate expected values.
package sax_test

import (
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/lvlath/go/sax"
)

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

func TestPAA_RationalNearestEvenFractionalOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(20261111))
	for iteration := 0; iteration < 1200; iteration++ {
		length := 2 + rng.Intn(20)
		segments := 1 + rng.Intn(length)
		data := make([]float64, length)
		for i := range data {
			// Dyadic fractions ensure every input is exactly representable.
			data[i] = float64(rng.Intn(61)-30) / 8
		}
		opts := sax.Options{Segments: segments, Alphabet: 3,
			Normalization: sax.Raw, Breakpoints: []float64{-0.25, 0.25},
			BreakpointID: "rational-fixture", ReturnPAA: true}
		result, err := sax.Encode(data, opts)
		if err != nil {
			t.Fatalf("iteration=%d: %v", iteration, err)
		}
		oracle := rationalPAA(data, segments)
		for j := range oracle {
			expected, _ := oracle[j].Float64()
			if math.Float64bits(result.PAA[j]) != math.Float64bits(expected) {
				t.Fatalf("iter=%d n=%d w=%d j=%d PAA=%.17g want nearest-even %.17g",
					iteration, length, segments, j, result.PAA[j], expected)
			}
		}
	}
}

func TestMindist_ExactRationalJensenOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(20261112))
	for iteration := 0; iteration < 1200; iteration++ {
		length := 3 + rng.Intn(24)
		segments := 1 + rng.Intn(length)
		left, right := make([]float64, length), make([]float64, length)
		for i := range left {
			left[i] = float64(rng.Intn(201)-100) / 16
			right[i] = float64(rng.Intn(201)-100) / 16
		}
		opts := sax.Options{Segments: segments, Alphabet: 5,
			Normalization: sax.Raw, Breakpoints: []float64{-2, -0.5, 0.5, 2},
			BreakpointID: "dyadic", ReturnPAA: true}
		encoder, err := sax.NewEncoder(opts)
		if err != nil {
			t.Fatal(err)
		}
		leftWord, err := encoder.Encode(left)
		if err != nil {
			t.Fatal(err)
		}
		rightWord, err := encoder.Encode(right)
		if err != nil {
			t.Fatal(err)
		}
		mindist, err := encoder.Mindist(leftWord, rightWord)
		if err != nil {
			t.Fatal(err)
		}
		exactPAA := rationalSquareRoot(rationalPAASquared(rationalPAA(left, segments), rationalPAA(right, segments), length))
		exactEuclidean := rationalSquareRoot(rationalEuclideanSquared(left, right))
		budget := 3e-13 * math.Max(1, exactEuclidean)
		if mindist > exactPAA+budget || exactPAA > exactEuclidean+budget {
			t.Fatalf("iter=%d n=%d w=%d: mindist %.17g, exact PAA %.17g, exact Euclidean %.17g",
				iteration, length, segments, mindist, exactPAA, exactEuclidean)
		}
	}
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

func TestMindist_IndependentPopulationZOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(20261113))
	for iteration := 0; iteration < 900; iteration++ {
		length := 5 + rng.Intn(30)
		segments := 1 + rng.Intn(length)
		left, right := make([]float64, length), make([]float64, length)
		for i := range left {
			left[i] = float64(rng.Intn(1000)-500)/10 + 12
			right[i] = float64(rng.Intn(1000)-500)/10 - 7
		}
		opts := sax.DefaultOptions(segments)
		opts.ReturnPAA = true
		encoder, err := sax.NewEncoder(opts)
		if err != nil {
			t.Fatal(err)
		}
		leftWord, err := encoder.Encode(left)
		if err != nil {
			t.Fatal(err)
		}
		rightWord, err := encoder.Encode(right)
		if err != nil {
			t.Fatal(err)
		}
		bound, err := encoder.Mindist(leftWord, rightWord)
		if err != nil {
			t.Fatal(err)
		}
		leftZ, leftMean, leftStd := referenceZ(left)
		rightZ, rightMean, rightStd := referenceZ(right)
		mustNear(t, leftWord.Mean, leftMean, 5e-13, 5e-13, "independent mean")
		mustNear(t, leftWord.StdDev, leftStd, 5e-13, 5e-13, "independent pop sigma")
		mustNear(t, rightWord.Mean, rightMean, 5e-13, 5e-13, "independent mean right")
		mustNear(t, rightWord.StdDev, rightStd, 5e-13, 5e-13, "independent pop sigma right")
		paad := rationalSquareRoot(rationalPAASquared(rationalPAA(leftZ, segments), rationalPAA(rightZ, segments), length))
		euclidean := rationalSquareRoot(rationalEuclideanSquared(leftZ, rightZ))
		budget := 5e-11 * math.Max(1, euclidean)
		if bound > paad+budget || paad > euclidean+budget {
			t.Fatalf("Z iteration=%d n=%d w=%d: MINDIST=%g PAA=%g Euclidean=%g", iteration, length, segments, bound, paad, euclidean)
		}
	}
}
