// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels_test exercises only the public sessionlevels facade.
// This file fuzzes bounded public inputs and exact error/result invariants.
package sessionlevels_test

import (
	"context"
	"errors"
	"testing"
)

func FuzzCompute(f *testing.F) {
	for _, seed := range fuzzSeeds() {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		input, wantErr := fuzzFixture(data)
		before := cloneFixture(input)
		result, err := computeFixture(context.Background(), input)
		requireUnchanged(t, input, before)
		if wantErr != nil {
			if result != nil || !errors.Is(err, wantErr) {
				t.Fatalf("result=%v error=%v, want nil and %v", result, err, wantErr)
			}
			return
		}
		if err != nil || result == nil {
			t.Fatalf("valid fixture rejected: %v", err)
		}
		requireResult(t, result, oracle(input))
		requireResult(t, result, mustCompute(t, input))
	})
}
