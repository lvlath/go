// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
package montecarlo_test

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

func TestMC04_SlowPathZeroBoundedWindow(t *testing.T) {
	entered := make(chan struct{}, 1)
	eighth := make(chan struct{}, 1)
	release := make(chan struct{})
	var started atomic.Int64
	var illegal atomic.Bool
	s := &funcSampler{1, func(ctx context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if started.Add(1) == 8 {
			eighth <- struct{}{}
		}
		if id >= 8 {
			illegal.Store(true)
		}
		if id == 0 {
			entered <- struct{}{}
			select {
			case <-release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		return []float64{float64(id)}, nil
	}}
	done := make(chan error, 1)
	go func() {
		_, err := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 256, Workers: 4}, s)
		done <- err
	}()
	<-entered
	// Eight accepted jobs with path 0 blocked demonstrate the full 2W window.
	select {
	case <-eighth:
	case <-time.After(10 * time.Second):
		close(release)
		t.Fatal("scheduler did not fill bounded window")
	}
	if count := started.Load(); count != 8 || illegal.Load() {
		close(release)
		t.Fatalf("unbounded dispatch: %d", count)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
func TestMC07_CancelledPrefix(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered := make(chan struct{}, 1)
	gate := make(chan struct{})
	s := &funcSampler{1, func(ctx context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == 2 {
			entered <- struct{}{}
			select {
			case <-gate:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		return []float64{float64(id)}, nil
	}}
	// Workers=1 guarantees 0 and 1 are reduced before path2 starts.
	outcome := make(chan struct {
		r *montecarlo.Result
		e error
	}, 1)
	go func() {
		r, e := montecarlo.Run(ctx, montecarlo.Plan{Paths: 12, Workers: 1, RetainSamples: true, Quantiles: []float64{.5}}, s)
		outcome <- struct {
			r *montecarlo.Result
			e error
		}{r, e}
	}()
	<-entered
	cancel()
	got := <-outcome
	mustErrorIs(t, got.e, context.Canceled)
	if got.r == nil || got.r.Complete || got.r.Completed != 2 || got.r.Samples.Rows() != 2 || got.r.Quantiles != nil {
		t.Fatalf("invalid canceled prefix: %+v", got.r)
	}
}

func TestPartialPrefixMatrixBoundsAndClone(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	entered := make(chan struct{}, 1)
	s := &funcSampler{2, func(ctx context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == 2 {
			entered <- struct{}{}
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return []float64{float64(id), 2 * float64(id)}, nil
	}}
	outcome := make(chan *montecarlo.Result, 1)
	go func() {
		r, _ := montecarlo.Run(ctx, montecarlo.Plan{Paths: 12, Workers: 1, RetainSamples: true}, s)
		outcome <- r
	}()
	<-entered
	cancel()
	r := <-outcome
	if r.Completed != 2 {
		t.Fatalf("expected 2, got %d", r.Completed)
	}
	if _, err := r.Samples.At(2, 0); err == nil {
		t.Fatal("out of prefix read accepted")
	}
	if err := r.Samples.Set(2, 0, 10); err == nil {
		t.Fatal("out of prefix write accepted")
	}
	cloned := r.Samples.Clone()
	if cloned.Rows() != 2 || cloned.Cols() != 2 {
		t.Fatal("wrong clone shape")
	}
	if err := cloned.Set(0, 0, 999); err != nil {
		t.Fatal(err)
	}
	v, _ := r.Samples.At(0, 0)
	if v == 999 {
		t.Fatal("clone aliases prefix")
	}
}
