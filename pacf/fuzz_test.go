// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test fuzzes public contracts with arbitrary floating-point bit patterns.
// The properties defend finite successful output, atomic errors, and affine invariance.
package pacf_test

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/pacf"
)

// FuzzCompute checks that nonfinite or extreme finite inputs never produce an
// invalid successful Result, mutate the samples, or expose a partial prefix.
func FuzzCompute(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, uint8(4), true)
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0}, uint8(0), false)
	f.Add([]byte{255, 255, 255, 255, 255, 255, 239, 127}, uint8(0), true)
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 127}, uint8(0), false)
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0}, uint8(0), false)

	f.Fuzz(func(t *testing.T, raw []byte, lag uint8, center bool) {
		if len(raw) < 8 || len(raw) > 512 {
			return
		}
		n := len(raw) / 8
		values := make([]float64, n)
		original := make([]uint64, n)
		nonfinite := false
		for i := range values {
			original[i] = binary.LittleEndian.Uint64(raw[i*8:])
			values[i] = math.Float64frombits(original[i])
			if math.IsNaN(values[i]) || math.IsInf(values[i], 0) {
				nonfinite = true
			}
		}
		maxLag := int(lag) % n
		result, err := pacf.Compute(context.Background(), values, pacf.Options{MaxLag: maxLag, Center: center})

		for i, value := range values {
			if math.Float64bits(value) != original[i] {
				t.Fatalf("input bit pattern at index %d was modified", i)
			}
		}
		if nonfinite {
			mustNilState(t, result, true, "nonfinite fuzzed input")
			mustErrorIs(t, err, pacf.ErrNonFinite)
			return
		}
		if err != nil {
			mustNilState(t, result, true, "failed fuzzed input")
			if !errors.Is(err, pacf.ErrZeroVariance) && !errors.Is(err, pacf.ErrSingular) && !errors.Is(err, pacf.ErrOverflow) {
				t.Fatalf("unclassified error from finite samples: %v", err)
			}
			return
		}
		if result.IsNil() || result.Count != n || len(result.Values) != maxLag+1 ||
			len(result.PredictionVariance) != maxLag+1 || result.Values[0] != 1 {
			t.Fatal("successful result violates canonical shape and lag-zero law")
		}
		for i, coefficient := range result.Values {
			variance := result.PredictionVariance[i]
			if math.IsNaN(coefficient) || math.IsInf(coefficient, 0) ||
				math.IsNaN(variance) || math.IsInf(variance, 0) || variance <= 0 {
				t.Fatalf("invalid finite success at lag %d", i)
			}
			if i > 0 && math.Abs(coefficient) >= 1 {
				t.Fatalf("reflection coefficient escaped strictly bounded domain at lag %d", i)
			}
		}
	})
}

// FuzzComputeAffineEquivariance verifies the exact centered estimand under
// integer-valued scaling and translation that preserve binary64 input precision.
func FuzzComputeAffineEquivariance(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9}, uint8(5))
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0}, uint8(7))
	f.Add([]byte{255, 0, 255, 0, 255, 0, 255, 0}, uint8(3))

	f.Fuzz(func(t *testing.T, raw []byte, lag uint8) {
		if len(raw) == 0 || len(raw) > 48 {
			return
		}
		values := make([]float64, len(raw))
		transformed := make([]float64, len(raw))
		for i, value := range raw {
			values[i] = float64(int(value) - 128)
			transformed[i] = 1024 - 4*values[i]
		}
		maxLag := int(lag) % len(values)
		original, firstErr := pacf.ComputeCentered(context.Background(), values, maxLag)
		affine, secondErr := pacf.ComputeCentered(context.Background(), transformed, maxLag)
		if firstErr != nil || secondErr != nil {
			if !errors.Is(firstErr, pacf.ErrZeroVariance) || !errors.Is(secondErr, pacf.ErrZeroVariance) ||
				original != nil || affine != nil {
				t.Fatalf("affine centered errors disagree: original=%v transformed=%v", firstErr, secondErr)
			}
			return
		}
		for i := range original.Values {
			if !closeEnough(original.Values[i], affine.Values[i], 5e-11) ||
				!closeEnough(16*original.PredictionVariance[i], affine.PredictionVariance[i], 5e-11) {
				t.Fatalf("centered affine invariance failed at lag %d", i)
			}
		}
	})
}
