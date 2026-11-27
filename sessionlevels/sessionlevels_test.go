// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels_test exercises only the public sessionlevels facade.
// This file protects validation, causality, ownership, numeric selection, and cancellation.
package sessionlevels_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/lvlath/go/sessionlevels"
)

func TestSessionLevels_TwoSessionsGolden(t *testing.T) {
	requireResult(t, mustCompute(t, goldenFixture()), goldenResult())
}

func TestSessionLevels_FutureMutationCausality(t *testing.T) {
	f := goldenFixture()
	before := mustCompute(t, f)
	f.high[3] = 1000
	requireResult(t, mustCompute(t, f), before)
}

func TestSessionLevels_RevisionHorizon(t *testing.T) {
	f := generatedFixture(97, 80)
	before := mustCompute(t, f)
	for k, session := range f.sessions {
		for _, channel := range []string{"high", "low", "close"} {
			t.Run(fmt.Sprintf("session=%d/%s", k, channel), func(t *testing.T) {
				changed := cloneFixture(f)
				i := session.End - 1
				switch channel {
				case "high":
					changed.high[i] = 1000
				case "low":
					changed.low[i] = -1000
				case "close":
					changed.closePrices[i] = changed.low[i]
				}
				after := mustCompute(t, changed)
				requireResult(t, resultPrefix(after, session.End), resultPrefix(before, session.End))
				requireResult(t, after, oracle(changed))
				if k+2 < len(f.sessions) {
					start := f.sessions[k+2].Start
					for j := start; j < len(f.high); j++ {
						if math.Float64bits(after.PreviousHigh[j]) != math.Float64bits(before.PreviousHigh[j]) ||
							math.Float64bits(after.PreviousLow[j]) != math.Float64bits(before.PreviousLow[j]) ||
							math.Float64bits(after.PreviousClose[j]) != math.Float64bits(before.PreviousClose[j]) {
							t.Fatalf("revision leaked beyond the next session at %d", j)
						}
					}
				}
			})
		}
	}
}

func TestSessionLevels_SingleSessionMasked(t *testing.T) {
	for _, complete := range []bool{false, true} {
		for _, n := range []int{1, 7, 2049} {
			f := sizedFixture(n, n)
			f.sessions[0].Complete = complete
			requireMasked(t, mustCompute(t, f), n)
		}
	}
}

func TestSessionLevels_PreviousCloseNotExtreme(t *testing.T) {
	f := goldenFixture()
	r := mustCompute(t, f)
	for i := 2; i < 4; i++ {
		if r.PreviousClose[i] != 3 || r.PreviousClose[i] == r.PreviousHigh[i] || r.PreviousClose[i] == r.PreviousLow[i] {
			t.Fatalf("row %d did not preserve the final close", i)
		}
	}
}

func TestSessionLevels_CoverageErrors(t *testing.T) {
	for _, tc := range coverageFailures() {
		t.Run(tc.name, func(t *testing.T) {
			f := goldenFixture()
			f.sessions = tc.sessions
			requireError(t, f, sessionlevels.ErrSession)
		})
	}
}

func TestSessionLevels_EarlyIncompleteError(t *testing.T) {
	for _, lastComplete := range []bool{false, true} {
		f := goldenFixture()
		f.sessions[0].Complete, f.sessions[1].Complete = false, lastComplete
		requireError(t, f, sessionlevels.ErrSession)
	}
	f := sizedFixture(9, 3)
	f.sessions[1].Complete = false
	requireError(t, f, sessionlevels.ErrSession)
}

func TestSessionLevels_EmptyInput(t *testing.T) {
	for _, nilSessions := range []bool{false, true} {
		f := sizedFixture(0, 1)
		if nilSessions {
			f.sessions = nil
		}
		r := mustCompute(t, f)
		requireMasked(t, r, 0)
		if r.IsNil() {
			t.Fatal("an empty successful result is not nil")
		}
	}
	f := sizedFixture(0, 1)
	f.sessions = goldenFixture().sessions
	requireError(t, f, sessionlevels.ErrSession)
}

func TestSessionLevels_NilAndShape(t *testing.T) {
	for channel := 0; channel < 3; channel++ {
		f := goldenFixture()
		channels := []*[]float64{&f.high, &f.low, &f.closePrices}
		*channels[channel] = nil
		requireError(t, f, sessionlevels.ErrNilInput)
		f = goldenFixture()
		channels = []*[]float64{&f.high, &f.low, &f.closePrices}
		*channels[channel] = (*channels[channel])[:1]
		requireError(t, f, sessionlevels.ErrShape)
	}
	requireError(t, fixture{}, sessionlevels.ErrNilInput)
	f := sizedFixture(0, 1)
	f.low = nil
	requireError(t, f, sessionlevels.ErrNilInput)
}

func TestSessionLevels_SignedPrices(t *testing.T) {
	f := signedFixture()
	r := mustCompute(t, f)
	for i := 2; i < 4; i++ {
		if !r.Valid[i] || r.PreviousHigh[i] != -2 || r.PreviousLow[i] != -10 || r.PreviousClose[i] != -4 {
			t.Fatalf("wrong signed summary at %d", i)
		}
	}
	requireResult(t, r, oracle(f))
}

func TestSessionLevels_WrongBarsErrBar(t *testing.T) {
	for kind := 0; kind < 3; kind++ {
		for _, index := range []int{0, 3} {
			f := goldenFixture()
			switch kind {
			case 0:
				f.low[index] = f.high[index] + 1
			case 1:
				f.closePrices[index] = f.low[index] - 1
			case 2:
				f.closePrices[index] = f.high[index] + 1
			}
			requireError(t, f, sessionlevels.ErrBar)
		}
	}
}

func TestSessionLevels_NonFinite(t *testing.T) {
	for channel := 0; channel < 3; channel++ {
		for _, index := range []int{0, 3} {
			for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
				f := goldenFixture()
				channels := [][]float64{f.high, f.low, f.closePrices}
				channels[channel][index] = value
				requireError(t, f, sessionlevels.ErrNonFinite)
				f.sessions = f.sessions[:1]
				f.sessions[0].End = len(f.high)
				requireError(t, f, sessionlevels.ErrNonFinite)
			}
		}
	}
}

func TestSessionLevels_ErrorPrecedence(t *testing.T) {
	f := goldenFixture()
	f.closePrices[0], f.high[3] = 100, math.NaN()
	requireError(t, f, sessionlevels.ErrNonFinite)
	f.sessions[1].Start++
	requireError(t, f, sessionlevels.ErrSession)
	f.low = f.low[:1]
	requireError(t, f, sessionlevels.ErrShape)
	f.high = nil
	requireError(t, f, sessionlevels.ErrNilInput)
	r, err := computeFixture(nil, f)
	if r != nil || !errors.Is(err, sessionlevels.ErrNilContext) {
		t.Fatal("nil context must precede every input error")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r, err = computeFixture(ctx, f)
	if r != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("entry cancellation must precede every input error")
	}
	deadline, stop := context.WithDeadline(context.Background(), time.Unix(1, 0))
	defer stop()
	r, err = computeFixture(deadline, f)
	if r != nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("an expired deadline must retain its context error identity")
	}
}

func TestResult_IsNilInterface(t *testing.T) {
	type nilable interface{ IsNil() bool }
	var pointer *sessionlevels.Result
	var value nilable = pointer
	if value == nil || !value.IsNil() {
		t.Fatal("a non-nil interface holding a nil result must report IsNil")
	}
	for _, r := range []*sessionlevels.Result{{}, mustCompute(t, sizedFixture(0, 1)), mustCompute(t, goldenFixture())} {
		value = r
		if value.IsNil() {
			t.Fatal("a non-nil result must not report IsNil")
		}
	}
}

func TestSessionLevels_SignedZeroTiePolicy(t *testing.T) {
	for _, negativeFirst := range []bool{false, true} {
		f := zeroFixture(negativeFirst)
		r := mustCompute(t, f)
		requireResult(t, resultPrefix(r, 2), resultPrefix(oracle(f), 2))
		if !r.Valid[2] || math.Float64bits(r.PreviousHigh[2]) != math.Float64bits(f.high[0]) ||
			math.Float64bits(r.PreviousLow[2]) != math.Float64bits(f.low[0]) ||
			math.Float64bits(r.PreviousClose[2]) != math.Float64bits(f.closePrices[1]) {
			t.Fatal("ready signed-zero extrema or closing-price bits changed")
		}
		requireMasked(t, resultPrefix(r, 2), 2)
	}
}

func TestSessionLevels_ExtremeFiniteSelection(t *testing.T) {
	f := extremeFixture()
	r := mustCompute(t, f)
	requireResult(t, r, oracle(f))
	if r.PreviousHigh[1] != math.MaxFloat64 || r.PreviousLow[1] != -math.MaxFloat64 ||
		r.PreviousClose[1] != math.SmallestNonzeroFloat64 {
		t.Fatal("selection must preserve representable extreme values without arithmetic")
	}
}

func TestSessionLevels_AdmissiblePrefixes(t *testing.T) {
	f := generatedFixture(192, 71)
	full := mustCompute(t, f)
	for cut := 0; cut <= len(f.high); cut++ {
		prefix := cutFixture(f, cut)
		requireResult(t, mustCompute(t, prefix), resultPrefix(full, cut))
	}
}

func TestSessionLevels_Ownership(t *testing.T) {
	f := goldenFixture()
	before := cloneFixture(f)
	r := mustCompute(t, f)
	other := mustCompute(t, f)
	requireUnchanged(t, f, before)
	r.PreviousHigh[2] = 777
	if r.PreviousLow[2] != 1 || r.PreviousClose[2] != 3 || !r.Valid[2] {
		t.Fatal("numerical output channels alias")
	}
	r.PreviousLow[2], r.PreviousClose[2], r.Valid[2] = -777, 888, false
	r.PreviousHigh = append(r.PreviousHigh, 999)
	requireUnchanged(t, f, before)
	requireResult(t, other, goldenResult())
	f.high[1], f.low[0], f.closePrices[1], f.sessions[0].End = 400, -10, 200, 1
	requireResult(t, other, goldenResult())
}

func TestSessionLevels_AliasedInputs(t *testing.T) {
	f := sizedFixture(9, 3)
	f.low, f.closePrices = f.high, f.high
	before := cloneFixture(f)
	r := mustCompute(t, f)
	requireResult(t, r, oracle(f))
	r.PreviousLow[3] = 123
	requireUnchanged(t, f, before)
}

func TestSessionLevels_RejectionDoesNotMutate(t *testing.T) {
	f := goldenFixture()
	f.closePrices[3] = -100
	before := cloneFixture(f)
	requireError(t, f, sessionlevels.ErrBar)
	requireUnchanged(t, f, before)
}

func TestSessionLevels_CancellationAtEveryCheckpoint(t *testing.T) {
	for _, f := range []fixture{sizedFixture(0, 1), sizedFixture(3073, 3073), sizedFixture(3073, 1)} {
		counter := newPollCancelContext(0)
		r, err := computeFixture(counter, f)
		counter.cancel()
		if err != nil || r == nil {
			t.Fatalf("counting run failed: %v", err)
		}
		polls := counter.calls.Load()
		minimum := int64(3 + 2*((len(f.high)+1023)/1024) + (len(f.sessions)+1023)/1024)
		if polls < minimum {
			t.Fatalf("only %d polls; contract requires at least %d for these scans", polls, minimum)
		}
		before := cloneFixture(f)
		for at := int64(1); at <= polls; at++ {
			ctx := newPollCancelContext(at)
			got, callErr := computeFixture(ctx, f)
			ctx.cancel()
			if got != nil || !errors.Is(callErr, context.Canceled) {
				t.Fatalf("poll %d/%d: result=%v error=%v", at, polls, got, callErr)
			}
			requireUnchanged(t, f, before)
		}
	}
}

func TestSessionLevels_ConcurrentIndependentCalls(t *testing.T) {
	f := generatedFixture(38, 127)
	want := oracle(f)
	type outcome struct {
		result *sessionlevels.Result
		err    error
	}
	const workers = 8
	results := make(chan outcome, workers)
	for i := 0; i < workers; i++ {
		go func() {
			r, err := computeFixture(context.Background(), f)
			results <- outcome{result: r, err: err}
		}()
	}
	for i := 0; i < workers; i++ {
		got := <-results
		if got.err != nil {
			t.Fatal(got.err)
		}
		requireResult(t, got.result, want)
	}
}

func TestSessionLevels_OutputOnlyAllocations(t *testing.T) {
	for _, width := range []int{1, 4097} {
		f := sizedFixture(4097, width)
		var result *sessionlevels.Result
		var err error
		allocations := testing.AllocsPerRun(10, func() {
			result, err = computeFixture(context.Background(), f)
		})
		if err != nil || result == nil {
			t.Fatalf("allocation fixture failed: %v", err)
		}
		if allocations > 5 {
			t.Fatalf("%.0f allocations: expected at most four channels and one result", allocations)
		}
	}
}
