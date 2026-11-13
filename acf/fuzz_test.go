// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package acf_test fuzzes public estimator validation and result ownership.
// The bound on decoded sample count keeps quadratic direct scans tractable.
package acf_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/lvlath/go/acf"
)

// FuzzComputeBoundaries checks non-panicking validation and result invariants
// across arbitrary binary64 samples, invalid enums, and invalid lag bounds.
// It intentionally exercises only the public Compute facade and does not
// compare implementation-specific normalized workspaces.
func FuzzComputeBoundaries(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64}, uint8(0), int(1))
	f.Add([]byte{0, 0, 0, 0, 0, 0, 248, 127}, uint8(0), int(0))
	f.Add([]byte{}, uint8(255), int(-1))
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 191}, uint8(1), int(0))

	f.Fuzz(func(t *testing.T, data []byte, rawEstimator uint8, maxLag int) {
		const maximumFuzzSamples = 20
		const bytesPerSample = 8
		count := len(data) / bytesPerSample
		if count > maximumFuzzSamples {
			count = maximumFuzzSamples
		}

		// Decode arbitrary IEEE-754 bits without removing NaN or Inf. Invalid
		// observations must be classified by Compute rather than silently skipped.
		values := make([]float64, count)
		for i := range values {
			bits := binary.LittleEndian.Uint64(data[i*bytesPerSample:])
			values[i] = math.Float64frombits(bits)
		}
		result, err := acf.Compute(values, maxLag, acf.Options{Denominator: acf.Denominator(rawEstimator)})
		if err != nil {
			if result != nil {
				t.Fatal("failure published a partial result")
			}
			return
		}

		if result == nil || result.IsNil() || result.Denominator != acf.Denominator(rawEstimator) {
			t.Fatal("successful result has invalid estimator/nil-state metadata")
		}
		if len(result.Values) != maxLag+1 || len(result.PairCounts) != maxLag+1 || result.Values[0] != 1 {
			t.Fatal("successful result shape or lag-zero identity was corrupted")
		}
		for lag, coefficient := range result.Values {
			if math.IsNaN(coefficient) || math.IsInf(coefficient, 0) || result.PairCounts[lag] != len(values)-lag {
				t.Fatalf("invalid coefficient/pair metadata at lag %d", lag)
			}
		}
	})
}
