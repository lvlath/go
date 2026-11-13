// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap_test fixes the public nilability and unbiased-number laws.
// The finite-word oracle is independent of any production RNG implementation.
package bootstrap_test

import (
	"testing"

	"github.com/lvlath/go/bootstrap"
)

// TestResult_IsNilInterface checks the explicit Result nil-receiver protocol.
func TestResult_IsNilInterface(t *testing.T) {
	var absent *bootstrap.Result
	var nilable interface{ IsNil() bool } = absent
	mustNilState(t, nilable, true, "typed nil Result behind interface")
	mustNilState(t, &bootstrap.Result{}, false, "allocated Result")
	mustNilState(t, nil, true, "nil interface")
	mustEqualBool(t, (&bootstrap.Result{}).IsNil(), false, "non-nil receiver")
}

// TestRNG_UnbiasedMappingExhaustive proves equal accepted residues for every
// valid bound of an eight-bit uniform source, independently of any RNG fixture.
func TestRNG_UnbiasedMappingExhaustive(t *testing.T) {
	const cardinality uint64 = 1 << 8
	for bound := uint64(1); bound < cardinality; bound++ {
		// The unsigned threshold represents cardinality modulo the target bound.
		threshold := (cardinality - bound) % bound
		counts := make([]int, int(bound))
		for word := uint64(0); word < cardinality; word++ {
			if word >= threshold {
				counts[word%bound]++
			}
		}

		// The accepted interval has a length divisible exactly by bound.
		want := int((cardinality - threshold) / bound)
		for remainder, got := range counts {
			if got != want {
				t.Fatalf("bound=%d residue=%d got=%d want=%d", bound, remainder, got, want)
			}
		}
	}
}

// TestRNG_BiasedMappingCounterexample anchors the defect that naive modulo
// would introduce and prevents a flawed finite-word oracle from passing vacuously.
func TestRNG_BiasedMappingCounterexample(t *testing.T) {
	var biased [3]int
	for word := 0; word < 256; word++ {
		biased[word%3]++
	}
	mustEqualInt(t, biased[0], 86, "biased remainder zero")
	mustEqualInt(t, biased[1], 85, "biased remainder one")
	mustEqualInt(t, biased[2], 85, "biased remainder two")
}

// TestStationary_UniformInvariantSmallChain uses exact rational counting to
// verify the uniform stationary distribution of the reset-or-advance Markov law.
func TestStationary_UniformInvariantSmallChain(t *testing.T) {
	// Every output column j receives one continuation from j-1 and an equally
	// weighted restart from every input i; both parts are counted as integers.
	for rows := 1; rows <= 9; rows++ {
		for blockLength := 1; blockLength <= rows; blockLength++ {
			for destination := 0; destination < rows; destination++ {
				incomingNumerator := 0
				for source := 0; source < rows; source++ {
					// Common denominator rows*blockLength: every restart contributes 1.
					incomingNumerator++
					if (source+1)%rows == destination {
						incomingNumerator += (blockLength - 1) * rows
					}
				}
				// The sum is exactly rows*blockLength, hence each column sum is 1.
				mustEqualInt(t, incomingNumerator, rows*blockLength, "Markov column stochasticity")
			}
		}
	}
}
