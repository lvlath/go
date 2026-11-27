// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap_test protects batch contracts and exact mathematical behavior.
// This file checks numerical expectations using exact bits or an independent rational oracle.
package vwap_test

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/lvlath/go/vwap"
)

// TestVWAP_ZeroSeedGolden protects zero-volume warmup and the first positive-weight seed.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: compute the [10,20,30]/[0,2,1] history.
//   - Stage 3: require masked +0, then 20, then the rounded 70/3 prefix.
//
// Behavior highlights:
//   - This test protects zero-volume warmup and the first positive-weight seed without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - A zero price must not be used as a readiness test.
func TestVWAP_ZeroSeedGolden(t *testing.T) {
	fixture := zeroSeedFixture(false)
	p, v, r := fixture.price, fixture.volume, fixture.reset
	result := mustCompute(t, p, v, r)
	requireResult(t, result, []vwap.Value{{}, {Value: 20, Valid: true}, {Value: 70.0 / 3, Valid: true}})
	verifyParity(t, p, v, r)
}

// TestVWAP_ResetBeforeRow protects reset-before-inclusion ordering.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: reset on the third row of the zero-seed history.
//   - Stage 3: require the third mean to equal 30 rather than a blend with the old segment.
//
// Behavior highlights:
//   - This test protects reset-before-inclusion ordering without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Resetting after inclusion silently assigns the boundary row to the wrong segment.
func TestVWAP_ResetBeforeRow(t *testing.T) {
	fixture := zeroSeedFixture(true)
	p, v, r := fixture.price, fixture.volume, fixture.reset
	requireResult(t, mustCompute(t, p, v, r), []vwap.Value{{}, {Value: 20, Valid: true}, {Value: 30, Valid: true}})
	verifyParity(t, p, v, r)
}

// TestVWAP_HugeProductsNoOverflow protects representable means of unrepresentable floating products.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: repeat positive and negative extreme constant prices with huge finite weights.
//   - Stage 3: require each output to retain the exact price bits.
//
// Behavior highlights:
//   - This test protects representable means of unrepresentable floating products without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Do not permit a float price*volume intermediate to manufacture an overflow.
func TestVWAP_HugeProductsNoOverflow(t *testing.T) {
	for _, price := range []float64{1e308, -1e308, math.MaxFloat64, -math.MaxFloat64} {
		p, v, r := []float64{price, price}, []float64{1e308, math.MaxFloat64}, []bool{true, false}
		want := []vwap.Value{{Value: price, Valid: true}, {Value: price, Valid: true}}
		requireResult(t, mustCompute(t, p, v, r), want)
		verifyParity(t, p, v, r)
	}
}

// TestVWAP_CancellationPrecision protects low-order information across extreme signed cancellation.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: include 1e308, one, and -1e308 with equal weights.
//   - Stage 3: require the final mean to equal RN64(1/3) and every prefix to match the rational oracle.
//
// Behavior highlights:
//   - This test protects low-order information across extreme signed cancellation without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - A compensated floating approximation is not a substitute for the exact continuation sum.
func TestVWAP_CancellationPrecision(t *testing.T) {
	fixture := cancellationFixture()
	p, v, r := fixture.price, fixture.volume, fixture.reset
	result := mustCompute(t, p, v, r)
	requireValue(t, vwap.Value{Value: result.Values[2], Valid: result.Valid[2]}, vwap.Value{Value: 1.0 / 3, Valid: true})
	verifyParity(t, p, v, r)
}

// TestVWAP_AllZeroVolume protects masked positive-zero payloads across zero-weight resets.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: submit finite signed prices with both volume zero signs.
//   - Stage 3: require every prefix to remain undefined and exactly +0.
//
// Behavior highlights:
//   - This test protects masked positive-zero payloads across zero-weight resets without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Zero-volume input still has finite-price validation and a segment count.
func TestVWAP_AllZeroVolume(t *testing.T) {
	fixture := zeroVolumeFixture()
	p, v, r := fixture.price, fixture.volume, fixture.reset
	requireResult(t, mustCompute(t, p, v, r), make([]vwap.Value, len(p)))
	verifyParity(t, p, v, r)
}

// TestVWAP_AnchorErrorFirstRow protects the mandatory explicit first anchor.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: attempt unanchored nonempty batch and initial streaming calls.
//   - Stage 3: require ErrAnchor, no partial output, and an unchanged empty checkpoint.
//
// Behavior highlights:
//   - This test protects the mandatory explicit first anchor without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - A failed first observation must not establish Started.
func TestVWAP_AnchorErrorFirstRow(t *testing.T) {
	result, err := vwap.Compute(context.Background(), []float64{math.NaN()}, []float64{-1}, []bool{false}, vwap.Options{})
	requireError(t, err, vwap.ErrAnchor)
	if result != nil {
		t.Fatal("partial result on anchor failure")
	}
	stream := newStream(t)
	before := stream.Snapshot()
	value, err := stream.Push(context.Background(), 12, 1, false)
	requireError(t, err, vwap.ErrAnchor)
	requireValue(t, value, vwap.Value{})
	requireUnchanged(t, stream, before)
}

// TestVWAP_ValidationAndPrecedence protects context, shape, anchor, and per-row domain precedence.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: submit fixed combinations of conflicting invalid inputs.
//   - Stage 3: classify sentinel identities and require nil results after rejection.
//
// Behavior highlights:
//   - This test protects context, shape, anchor, and per-row domain precedence without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Do not copy a global non-finite priority from a different package.
func TestVWAP_ValidationAndPrecedence(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	deadline, cancelDeadline := expiredContext()
	defer cancelDeadline()
	cases := []struct {
		name    string
		ctx     context.Context
		p, v    []float64
		r       []bool
		want    error
		alsoNil bool
	}{
		{"nil-context", nil, []float64{1}, nil, nil, vwap.ErrNilContext, false},
		{"canceled", canceled, []float64{1}, nil, nil, context.Canceled, false},
		{"deadline", deadline, []float64{1}, nil, nil, context.DeadlineExceeded, false},
		{"price-nil", context.Background(), nil, []float64{1}, []bool{false}, vwap.ErrShape, true},
		{"volume-nil", context.Background(), []float64{1}, nil, []bool{false}, vwap.ErrShape, true},
		{"reset-nil", context.Background(), []float64{1}, []float64{1}, nil, vwap.ErrShape, true},
		{"shape-first", context.Background(), []float64{math.NaN()}, []float64{}, []bool{false}, vwap.ErrShape, false},
		{"reset-shape", context.Background(), []float64{1}, []float64{1}, []bool{}, vwap.ErrShape, false},
		{"anchor-first", context.Background(), []float64{math.NaN()}, []float64{-1}, []bool{false}, vwap.ErrAnchor, false},
		{"nan-price", context.Background(), []float64{math.NaN()}, []float64{-1}, []bool{true}, vwap.ErrNonFinite, false},
		{"infinite-price", context.Background(), []float64{math.Inf(-1)}, []float64{0}, []bool{true}, vwap.ErrNonFinite, false},
		{"nan-volume", context.Background(), []float64{1}, []float64{math.NaN()}, []bool{true}, vwap.ErrNonFinite, false},
		{"infinite-volume", context.Background(), []float64{1}, []float64{math.Inf(1)}, []bool{true}, vwap.ErrNonFinite, false},
		{"negative-infinite-volume", context.Background(), []float64{1}, []float64{math.Inf(-1)}, []bool{true}, vwap.ErrNonFinite, false},
		{"negative-volume", context.Background(), []float64{-7}, []float64{-1}, []bool{true}, vwap.ErrVolume, false},
		{"ordered-row-errors", context.Background(), []float64{1, math.NaN()}, []float64{-1, 1}, []bool{true, false}, vwap.ErrVolume, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := vwap.Compute(tc.ctx, tc.p, tc.v, tc.r, vwap.Options{})
			requireError(t, err, tc.want)
			if tc.alsoNil {
				requireError(t, err, vwap.ErrNilInput)
			}
			if result != nil {
				t.Fatal("partial result on failure")
			}
		})
	}
}

// TestVWAP_EmptyAndOwnership protects empty-input success and detached result channels.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: compute nil/empty inputs and reuse independent ordinary histories.
//   - Stage 3: require non-nil empty channels and no mutation aliasing.
//
// Behavior highlights:
//   - This test protects empty-input success and detached result channels without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - A shallow Result copy is not an independent slice clone.
func TestVWAP_EmptyAndOwnership(t *testing.T) {
	for mask := 0; mask < 8; mask++ {
		var p, v []float64
		var r []bool
		if mask&1 != 0 {
			p = []float64{}
		}
		if mask&2 != 0 {
			v = []float64{}
		}
		if mask&4 != 0 {
			r = []bool{}
		}
		result := mustCompute(t, p, v, r)
		if result.Values == nil || result.Valid == nil || len(result.Values) != 0 || len(result.Valid) != 0 {
			t.Fatalf("empty channels not canonical: %+v", result)
		}
	}
	p, v, r := []float64{-3, 0, 9}, []float64{2, 1, 3}, []bool{true, false, false}
	originalP, originalV, originalR := slices.Clone(p), slices.Clone(v), slices.Clone(r)
	first := mustCompute(t, p, v, r)
	second := mustCompute(t, p, v, r)
	first.Values[0], first.Valid[0] = 999, false
	if second.Values[0] != -3 || !second.Valid[0] {
		t.Fatal("results alias")
	}
	if !slices.Equal(p, originalP) || !slices.Equal(v, originalV) || !slices.Equal(r, originalR) {
		t.Fatal("result aliases or mutates inputs")
	}
	p[0], v[0], r[0] = 0, 0, false
	if second.Values[0] != -3 || !second.Valid[0] {
		t.Fatal("input mutations changed output")
	}
}

// TestResult_IsNilInterface protects pointer-only nilability behind a structural interface.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: compare typed-nil, non-nil empty, and successful empty results.
//   - Stage 3: require IsNil to depend only on pointer presence.
//
// Behavior highlights:
//   - This test protects pointer-only nilability behind a structural interface without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Do not redefine absence as empty or unready output.
func TestResult_IsNilInterface(t *testing.T) {
	type nilable interface{ IsNil() bool }
	var pointer *vwap.Result
	var view nilable = pointer
	if !view.IsNil() {
		t.Fatal("typed nil must be nil")
	}
	view = &vwap.Result{}
	if view.IsNil() {
		t.Fatal("non-nil empty result must not be nil")
	}
	if mustCompute(t, nil, nil, nil).IsNil() {
		t.Fatal("successful empty result is nil")
	}
}

// TestVWAP_RationalOracle checks broad finite binary64 histories against an independent rational oracle.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: generate fixed-seed histories with resets, zeros, and extreme values.
//   - Stage 3: compare all batch and streaming prefix bits.
//
// Behavior highlights:
//   - This test checks broad finite binary64 histories against an independent rational oracle without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Changing seeds or widening tolerances after a failure hides numerical defects.
func TestVWAP_RationalOracle(t *testing.T) {
	for seed := int64(1); seed <= 24; seed++ {
		p, v, r := randomHistory(seed, 257)
		verifyParity(t, p, v, r)
	}
}

// TestVWAP_ExactRoundingBoundaries protects nearest-even ties and normal/subnormal transitions.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: replay independently constructed exact boundary fixtures.
//   - Stage 3: require batch and streaming bit parity with rational rounding.
//
// Behavior highlights:
//   - This test protects nearest-even ties and normal/subnormal transitions without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Signed underflow zero is observable and must not be normalized indiscriminately.
func TestVWAP_ExactRoundingBoundaries(t *testing.T) {
	for _, fixture := range roundingFixtures() {
		verifyParity(t, fixture.price, fixture.volume, fixture.reset)
	}
}

// TestVWAP_ScalePermutationAndPrefixCausality protects exact scaling, final permutation invariance, and prefix causality.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: scale representable weights, permute complete segments, and truncate future rows.
//   - Stage 3: compare the appropriate final or prefix values without tolerance.
//
// Behavior highlights:
//   - This test protects exact scaling, final permutation invariance, and prefix causality without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Permutation preserves only the final segment mean, not every chronological prefix.
func TestVWAP_ScalePermutationAndPrefixCausality(t *testing.T) {
	fixture := scaleFixture()
	p, v, r := fixture.price, fixture.volume, fixture.reset
	baseline := mustCompute(t, p, v, r)
	for _, power := range []int{-1000, -100, 100, 900} {
		scaled := make([]float64, len(v))
		for i := range v {
			scaled[i] = math.Ldexp(v[i], power)
		}
		result := mustCompute(t, p, scaled, r)
		for i := range p {
			requireValue(t, vwap.Value{Value: result.Values[i], Valid: result.Valid[i]},
				vwap.Value{Value: baseline.Values[i], Valid: baseline.Valid[i]})
		}
	}
	rng := rand.New(rand.NewSource(19))
	for repeat := 0; repeat < 16; repeat++ {
		order := rng.Perm(len(p))
		pp, vv := make([]float64, len(p)), make([]float64, len(p))
		for i, from := range order {
			pp[i], vv[i] = p[from], v[from]
		}
		result := mustCompute(t, pp, vv, r)
		requireValue(t, vwap.Value{Value: result.Values[len(p)-1], Valid: result.Valid[len(p)-1]},
			vwap.Value{Value: baseline.Values[len(p)-1], Valid: true})
	}
	for end := 1; end <= len(p); end++ {
		prefix := mustCompute(t, p[:end], v[:end], r[:end])
		for i := range prefix.Values {
			if math.Float64bits(prefix.Values[i]) != math.Float64bits(baseline.Values[i]) || prefix.Valid[i] != baseline.Valid[i] {
				t.Fatal("future rows changed an earlier prefix")
			}
		}
	}
}

// TestVWAP_DeterministicCancellation protects rejection at every batch polling stage.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: use counted contexts spanning validation, computation, and final publication.
//   - Stage 3: require context.Canceled and no partial result.
//
// Behavior highlights:
//   - This test protects rejection at every batch polling stage without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Do not replace deterministic poll cancellation with scheduler-sensitive sleeps.
func TestVWAP_DeterministicCancellation(t *testing.T) {
	p, v, r := benchmarkHistory(2050, 64)
	// Polls: entry, validation rows 0/1024/2048, post-validation,
	// computation rows 0/1024/2048, and the publication boundary.
	for at := 1; at <= 9; at++ {
		ctx := newCountedContext(at)
		result, err := vwap.Compute(ctx, p, v, r, vwap.Options{})
		requireError(t, err, context.Canceled)
		if result != nil {
			t.Fatalf("partial result at cancellation poll %d", at)
		}
		if ctx.calls != at {
			t.Fatalf("poll count=%d; wanted %d", ctx.calls, at)
		}
	}
	ctx := newCountedContext(10)
	result, err := vwap.Compute(ctx, p, v, r, vwap.Options{})
	if err != nil || result == nil {
		t.Fatalf("unexpected cancellation: %v", err)
	}
	for at := 2; at <= 3; at++ {
		result, err := vwap.Compute(newCountedContext(at), nil, nil, nil, vwap.Options{})
		if !errors.Is(err, context.Canceled) || result != nil {
			t.Fatal("empty publication ignored cancellation")
		}
	}
}
