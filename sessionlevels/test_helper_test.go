// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels_test exercises only the public sessionlevels facade.
// This file owns procedural fixtures, exact assertions, and cancellation controls.
package sessionlevels_test

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/sessionlevels"
)

type fixture struct {
	high, low, closePrices []float64
	sessions               []sessionlevels.Session
}

func goldenFixture() fixture {
	return fixture{
		high:        []float64{2, 4, 9, 8},
		low:         []float64{1, 2, 6, 5},
		closePrices: []float64{1.5, 3, 7, 6},
		sessions: []sessionlevels.Session{
			{Start: 0, End: 2, Complete: true},
			{Start: 2, End: 4, Complete: false},
		},
	}
}

func goldenResult() *sessionlevels.Result {
	return &sessionlevels.Result{
		PreviousHigh:  []float64{0, 0, 4, 4},
		PreviousLow:   []float64{0, 0, 1, 1},
		PreviousClose: []float64{0, 0, 3, 3},
		Valid:         []bool{false, false, true, true},
	}
}

func sizedFixture(n, sessionBars int) fixture {
	f := fixture{
		high: make([]float64, n), low: make([]float64, n), closePrices: make([]float64, n),
		sessions: make([]sessionlevels.Session, 0),
	}
	if n > 0 {
		f.sessions = make([]sessionlevels.Session, 0, (n-1)/sessionBars+1)
	}
	for i := range f.high {
		base := float64(i%37) - 18
		f.high[i], f.low[i], f.closePrices[i] = base+2, base-3, base-1
	}
	for start := 0; start < n; {
		end := start + min(sessionBars, n-start)
		f.sessions = append(f.sessions, sessionlevels.Session{Start: start, End: end, Complete: end < n})
		start = end
	}
	return f
}

func generatedFixture(seed int64, n int) fixture {
	rng := rand.New(rand.NewSource(seed))
	f := sizedFixture(n, 1)
	for i := range f.high {
		l := rng.Intn(201) - 100
		width := rng.Intn(16)
		f.low[i], f.high[i] = float64(l), float64(l+width)
		f.closePrices[i] = float64(l + rng.Intn(width+1))
	}
	f.sessions = f.sessions[:0]
	for start := 0; start < n; {
		end := start + min(1+rng.Intn(9), n-start)
		f.sessions = append(f.sessions, sessionlevels.Session{Start: start, End: end, Complete: end < n})
		start = end
	}
	return f
}

func cloneFixture(f fixture) fixture {
	return fixture{
		high: slices.Clone(f.high), low: slices.Clone(f.low), closePrices: slices.Clone(f.closePrices),
		sessions: slices.Clone(f.sessions),
	}
}

func cutFixture(f fixture, end int) fixture {
	out := fixture{
		high: f.high[:end], low: f.low[:end], closePrices: f.closePrices[:end],
		sessions: make([]sessionlevels.Session, 0, len(f.sessions)),
	}
	for _, session := range f.sessions {
		if session.Start >= end {
			break
		}
		if session.End > end {
			session.End, session.Complete = end, false
		}
		out.sessions = append(out.sessions, session)
	}
	return out
}

func resultPrefix(r *sessionlevels.Result, n int) *sessionlevels.Result {
	return &sessionlevels.Result{
		PreviousHigh: r.PreviousHigh[:n], PreviousLow: r.PreviousLow[:n],
		PreviousClose: r.PreviousClose[:n], Valid: r.Valid[:n],
	}
}

func computeFixture(ctx context.Context, f fixture) (*sessionlevels.Result, error) {
	return sessionlevels.Compute(ctx, f.high, f.low, f.closePrices, f.sessions, sessionlevels.Options{})
}

func mustCompute(t testing.TB, f fixture) *sessionlevels.Result {
	t.Helper()
	r, err := computeFixture(context.Background(), f)
	if err != nil || r == nil {
		t.Fatalf("Compute: result=%v error=%v", r, err)
	}
	return r
}

func requireError(t testing.TB, f fixture, want error) {
	t.Helper()
	r, err := computeFixture(context.Background(), f)
	if r != nil || !errors.Is(err, want) {
		t.Fatalf("result=%v error=%v; want nil and %v", r, err, want)
	}
}

func requireFloats(t testing.TB, name string, got, want []float64) {
	t.Helper()
	if (got == nil) != (want == nil) || len(got) != len(want) {
		t.Fatalf("%s nilness/length mismatch: got=%v want=%v", name, got, want)
	}
	for i := range want {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("%s[%d]=%g (%016x), want %g (%016x)", name, i,
				got[i], math.Float64bits(got[i]), want[i], math.Float64bits(want[i]))
		}
	}
}

func requireResult(t testing.TB, got, want *sessionlevels.Result) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatal("result comparison requires two non-nil results")
	}
	requireFloats(t, "PreviousHigh", got.PreviousHigh, want.PreviousHigh)
	requireFloats(t, "PreviousLow", got.PreviousLow, want.PreviousLow)
	requireFloats(t, "PreviousClose", got.PreviousClose, want.PreviousClose)
	if got.Valid == nil || len(got.Valid) != len(want.Valid) {
		t.Fatal("Valid must be non-nil with the expected length")
	}
	for i := range want.Valid {
		if got.Valid[i] != want.Valid[i] {
			t.Fatalf("Valid[%d]=%t, want %t", i, got.Valid[i], want.Valid[i])
		}
	}
}

func requireUnchanged(t testing.TB, got, want fixture) {
	t.Helper()
	requireFloats(t, "input high", got.high, want.high)
	requireFloats(t, "input low", got.low, want.low)
	requireFloats(t, "input close", got.closePrices, want.closePrices)
	if !slices.Equal(got.sessions, want.sessions) {
		t.Fatalf("input sessions changed: got=%v want=%v", got.sessions, want.sessions)
	}
}

func requireMasked(t testing.TB, r *sessionlevels.Result, n int) {
	t.Helper()
	want := &sessionlevels.Result{
		PreviousHigh: make([]float64, n), PreviousLow: make([]float64, n),
		PreviousClose: make([]float64, n), Valid: make([]bool, n),
	}
	requireResult(t, r, want)
}

func coverageFailures() []struct {
	name     string
	sessions []sessionlevels.Session
} {
	maxInt := int(^uint(0) >> 1)
	return []struct {
		name     string
		sessions []sessionlevels.Session
	}{
		{"nil", nil},
		{"empty", []sessionlevels.Session{}},
		{"first_not_zero", []sessionlevels.Session{{Start: 1, End: 4}}},
		{"negative_start", []sessionlevels.Session{{Start: -1, End: 4}}},
		{"negative_end", []sessionlevels.Session{{Start: 0, End: -1}}},
		{"empty_segment", []sessionlevels.Session{{Start: 0, End: 0}}},
		{"end_above_n", []sessionlevels.Session{{Start: 0, End: 5}}},
		{"uncovered_tail", []sessionlevels.Session{{Start: 0, End: 3}}},
		{"gap", []sessionlevels.Session{{Start: 0, End: 1, Complete: true}, {Start: 2, End: 4}}},
		{"overlap", []sessionlevels.Session{{Start: 0, End: 3, Complete: true}, {Start: 2, End: 4}}},
		{"inverted", []sessionlevels.Session{{Start: 0, End: 3, Complete: true}, {Start: 3, End: 2}}},
		{"duplicate", []sessionlevels.Session{{Start: 0, End: 4, Complete: true}, {Start: 0, End: 4}}},
		{"max_end", []sessionlevels.Session{{Start: 0, End: maxInt}}},
		{"max_start", []sessionlevels.Session{{Start: maxInt, End: maxInt}}},
		{"min_start", []sessionlevels.Session{{Start: -maxInt - 1, End: maxInt}}},
	}
}

func signedFixture() fixture {
	return fixture{
		high: []float64{-8, -2, -1, -3}, low: []float64{-10, -7, -6, -9},
		closePrices: []float64{-9, -4, -3, -5},
		sessions:    goldenFixture().sessions,
	}
}

func zeroFixture(firstNegative bool) fixture {
	negativeZero := math.Copysign(0, -1)
	first, second := 0.0, negativeZero
	if firstNegative {
		first, second = negativeZero, 0
	}
	return fixture{
		high: []float64{first, second, 0}, low: []float64{first, second, 0},
		closePrices: []float64{first, second, 0},
		sessions:    []sessionlevels.Session{{Start: 0, End: 2, Complete: true}, {Start: 2, End: 3}},
	}
}

func extremeFixture() fixture {
	tiny := math.SmallestNonzeroFloat64
	return fixture{
		high:        []float64{math.MaxFloat64, -tiny, tiny, 0},
		low:         []float64{-math.MaxFloat64, -2 * tiny, tiny, 0},
		closePrices: []float64{tiny, -tiny, tiny, 0},
		sessions: []sessionlevels.Session{
			{Start: 0, End: 1, Complete: true}, {Start: 1, End: 2, Complete: true},
			{Start: 2, End: 3, Complete: true}, {Start: 3, End: 4},
		},
	}
}

// dailyFixture models a caller's synthetic NYSE regular-session manifest.
// The adapter has already selected 390 one-minute bars for 09:30-16:00 and a live cut.
func dailyFixture() fixture {
	const previousBars = 390
	f := sizedFixture(previousBars+3, previousBars)
	for i := 0; i < previousBars; i++ {
		f.high[i], f.low[i], f.closePrices[i] = 102, 98, 100
	}
	f.high[100], f.low[200], f.closePrices[previousBars-1] = 103, 97, 101
	copy(f.high[previousBars:], []float64{103, 105, 106})
	copy(f.low[previousBars:], []float64{100, 102, 103})
	copy(f.closePrices[previousBars:], []float64{102, 104, 105})
	return f
}

// weeklyFixture uses daily bars: a four-day holiday week, a five-day week, and a live cut.
// These are supplied index ranges, not a calendar calculation performed by the package.
func weeklyFixture() fixture {
	return fixture{
		high:        []float64{110, 112, 111, 115, 116, 118, 117, 119, 121, 122, 123},
		low:         []float64{100, 102, 101, 104, 112, 113, 110, 115, 117, 119, 120},
		closePrices: []float64{108, 109, 110, 113, 115, 117, 116, 118, 120, 121, 122},
		sessions: []sessionlevels.Session{
			{Start: 0, End: 4, Complete: true}, {Start: 4, End: 9, Complete: true}, {Start: 9, End: 11},
		},
	}
}

// pollCancelContext cancels a real, concurrency-safe context on a chosen Err poll.
// Unlike sleeping or inspecting the call stack, this deterministically reaches each checkpoint.
type pollCancelContext struct {
	context.Context
	cancel context.CancelFunc
	limit  int64
	calls  atomic.Int64
}

func newPollCancelContext(limit int64) *pollCancelContext {
	ctx, cancel := context.WithCancel(context.Background())
	return &pollCancelContext{Context: ctx, cancel: cancel, limit: limit}
}

func (c *pollCancelContext) Err() error {
	if c.calls.Add(1) >= c.limit && c.limit > 0 {
		c.cancel()
	}
	return c.Context.Err()
}

func fuzzFixture(data []byte) (fixture, error) {
	if len(data) < 2 {
		return sizedFixture(0, 1), nil
	}
	n := 1 + int(data[0])%32
	f := sizedFixture(n, 1+int(data[0])%7)
	switch data[1] % 16 {
	case 1:
		f.high = nil
		return f, sessionlevels.ErrNilInput
	case 2:
		f.low = append(f.low, 0)
		return f, sessionlevels.ErrShape
	case 3:
		f.sessions = nil
		return f, sessionlevels.ErrSession
	case 4:
		f.sessions[0].Start = -1
		return f, sessionlevels.ErrSession
	case 5:
		f.sessions[0].End = n + 1
		return f, sessionlevels.ErrSession
	case 6, 14:
		f.sessions = []sessionlevels.Session{{Start: 0, End: 1}, {Start: 1, End: n, Complete: true}}
		return f, sessionlevels.ErrSession
	case 7:
		f.high[n-1] = math.NaN()
		return f, sessionlevels.ErrNonFinite
	case 8:
		f.closePrices[0] = f.high[0] + 1
		return f, sessionlevels.ErrBar
	case 9:
		if len(data) >= 10 {
			value := math.Float64frombits(binary.LittleEndian.Uint64(data[2:10]))
			f.high[0], f.low[0], f.closePrices[0] = value, value, value
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return f, sessionlevels.ErrNonFinite
			}
		}
	case 10:
		f.low[0], f.high[n-1] = f.high[0]+1, math.Inf(1)
		return f, sessionlevels.ErrNonFinite
	case 11:
		f.sessions[len(f.sessions)-1].Complete = true
	case 12:
		f.closePrices[n-1] = math.Inf(-1)
		return f, sessionlevels.ErrNonFinite
	case 13:
		return sizedFixture(0, 1), nil
	case 15:
		f.sessions[len(f.sessions)-1].End--
		return f, sessionlevels.ErrSession
	}
	return f, nil
}

func fuzzSeeds() [][]byte {
	seeds := make([][]byte, 0, 24)
	seeds = append(seeds, []byte{}, []byte{0})
	for mode := byte(0); mode < 16; mode++ {
		seeds = append(seeds, []byte{31, mode})
	}
	for _, value := range []float64{math.MaxFloat64, math.SmallestNonzeroFloat64, math.Copysign(0, -1), math.NaN(), math.Inf(1)} {
		seed := make([]byte, 10)
		seed[0], seed[1] = 31, 9
		binary.LittleEndian.PutUint64(seed[2:], math.Float64bits(value))
		seeds = append(seeds, seed)
	}
	return seeds
}
