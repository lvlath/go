// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo coordinates bounded execution and ordered publication.
// This file owns the sliding window, interruption policy, and result lifecycle.
package montecarlo

import (
	"context"
	"errors"
	"sync"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/rng"
)

// driverState belongs exclusively to the coordinator goroutine. Its mutable
// state is never accessed by workers: workers own RNG streams and send values
// through outcomes; one coordinator owns the ordered statistical transition.
// No locks are required on driverState because there is exactly one writer.
type driverState struct {
	config     configuration
	factory    *rng.Factory
	child      context.Context
	cancel     context.CancelFunc
	jobs       chan uint64
	outcomes   chan sampleOutcome
	workers    sync.WaitGroup
	moments    *reducer
	result     *Result
	retained   *matrix.Dense
	raw        []float64
	pending    map[uint64]sampleOutcome
	nextIssue  uint64
	nextReduce uint64
	inFlight   int
	row        int
	failure    error
	stopped    bool
}

// drive is the canonical execution kernel for a validated experiment.
// It allocates optional full-sized storage before launching any sampler calls,
// then joins every accepted job before publishing the completed prefix.
//
// Implementation:
//   - Stage 1: initialize factory, summaries and budget-checked backing storage.
//   - Stage 2: start fixed worker ownership and run the bounded coordinator loop.
//   - Stage 3: close dispatch, join workers, finalize moments and optional quantiles.
//
// Inputs: ctx is a live context at validation; c contains checked sizes and
//
//	detached levels; sampler meets fixed-dimension/concurrency requirements.
//
// Returns: a Result for completed or interrupted computation; allocation errors
//
//	before worker launch return nil result and an error.
//
// Determinism: only nextReduce advances moment state, always in ascending path ID.
// Complexity: O(Paths*d + d*Paths*log(Paths)) with quantiles; streaming memory
//
//	O(d + W*d) when neither samples nor quantiles are requested.
//
// AI-Hints: never replace the single coordinator reducer with worker-local sums.
func drive(ctx context.Context, c configuration, sampler Sampler) (*Result, error) {
	state, err := newDriverState(ctx, c)
	if err != nil {
		return nil, err
	}
	defer state.cancel()

	state.startWorkers(sampler)
	state.dispatch(ctx)
	state.joinWorkers()
	return state.finalize()
}

// newDriverState allocates only buffers proven addressable by the preflight.
// A retained Dense is the sole raw owner for samples+quantiles mode; quantiles
// without retention use one flat raw buffer, never a duplicate matrix.
//
// Inputs: ctx is the parent execution context; c already passed checkBudget.
// Returns: initialized coordinator, or nil and a wrapped allocation error.
// Determinism: metadata and buffer contents have a stable zero-value origin.
// Complexity: Time/Space O(Paths*d) only if raw storage was requested; else O(d+W).
// AI-Hints: do not allocate a second Paths*d buffer alongside retained Dense.
func newDriverState(ctx context.Context, c configuration) (*driverState, error) {
	child, cancel := context.WithCancel(ctx)
	factory := rng.NewFactory()
	state := &driverState{
		config:   c,
		factory:  factory,
		child:    child,
		cancel:   cancel,
		jobs:     make(chan uint64),
		outcomes: make(chan sampleOutcome, c.workers),
		moments:  newReducer(c.dimension),
		result: &Result{
			Requested:      c.plan.Paths,
			Seed:           c.plan.Seed,
			RNG:            factory.Version(),
			QuantileLevels: append([]float64(nil), c.levels...),
		},
		pending: make(map[uint64]sampleOutcome, c.workers),
	}
	if !c.keepRaw {
		return state, nil
	}

	if c.plan.RetainSamples {
		// The matrix constructor owns the backing array and its finite-value policy.
		retained, err := matrix.NewDense(c.pathRows, c.dimension)
		if err != nil {
			cancel()
			return nil, errors.Join(ErrOverflow, err)
		}
		state.retained = retained
	} else {
		// c.pathRows*c.dimension was checked against MaxInt before this allocation.
		state.raw = make([]float64, c.pathRows*c.dimension)
	}
	return state, nil
}

// startWorkers gives each goroutine its own stream lifecycle and jobs channel.
// It never assigns a worker-specific seed or shares mutable RNG objects.
//
// Inputs: sampler must be safe for up to config.workers concurrent Sample calls.
// Returns: no value; the WaitGroup counts exactly the created workers.
// Determinism: worker selection is arbitrary, but stream identity is pathID-only.
// Complexity: Time O(W), Space O(W) goroutine/channel metadata, excluding vectors.
// AI-Hints: each worker emits exactly one outcome for every received job.
func (d *driverState) startWorkers(sampler Sampler) {
	for workerIndex := 0; workerIndex < d.config.workers; workerIndex++ {
		d.workers.Add(1)
		go func() {
			defer d.workers.Done()
			worker(d.child, sampler, d.factory, d.config.plan.Seed,
				d.config.dimension, d.jobs, d.outcomes)
		}()
	}
}

// dispatch sends only path IDs in [nextReduce, nextReduce+2W).
// It drains already accepted jobs after failure or cancellation; the child
// context stops additional useful sampler work, while worker completion is joined.
//
// Inputs: parent context may be cancelled at any point during dispatch.
// Returns: no value; mutations stay inside the single coordinator goroutine.
// Determinism: dispatch may reorder completion, but reduceReady fixes publication.
// Complexity: O(Paths) channel transitions plus O(Paths*d) reductions;
//
//	the pending map contains at most 2W path outcomes at all times.
//
// AI-Hints: disable the done select case after cancellation to avoid a busy loop.
func (d *driverState) dispatch(parent context.Context) {
	for d.inFlight > 0 || (!d.stopped && d.nextReduce < d.config.plan.Paths) {
		// A cancellation observed between selects must disable all new sends.
		if err := parent.Err(); err != nil {
			d.stop(err)
		}

		// Nil channels disable select arms without a separate active/passive loop.
		var dispatch chan uint64
		if !d.stopped && d.nextIssue < d.config.plan.Paths &&
			d.nextIssue-d.nextReduce < d.config.window {
			dispatch = d.jobs
		}
		var cancelled <-chan struct{}
		if !d.stopped {
			cancelled = parent.Done()
		}

		select {
		case dispatch <- d.nextIssue:
			// A send commits one path to exactly one worker and the bounded window.
			d.nextIssue++
			d.inFlight++
		case outcome := <-d.outcomes:
			// All accepted jobs emit outcomes, including skipped work after cancel.
			d.inFlight--
			d.acceptOutcome(outcome)
		case <-cancelled:
			d.stop(parent.Err())
		}
	}
}

// stop records only the first observed terminal cause and cancels child work.
// Subsequent context/sampler errors may arrive during the required drain.
//
// Inputs: cause is a non-nil sampler, validation, numeric or context error.
// Returns: no value; the original cause remains available to errors.Is.
// Determinism: the first observed error is retained; partial length is timing-dependent.
// Complexity: Time O(1), Space O(1).
// AI-Hints: do not let a later context.Canceled mask an earlier sampler cause.
func (d *driverState) stop(cause error) {
	if cause == nil {
		return
	}
	if d.failure == nil {
		d.failure = cause
	}
	d.stopped = true
	d.cancel()
}

// acceptOutcome buffers a valid completed path until it reaches nextReduce.
// Invalid outcomes stop dispatch immediately but can never fill a missing row.
//
// Inputs: outcome is one unique path ID issued by dispatch, with detached values.
// Returns: no value; only reduceReady advances Completed's candidate prefix.
// Determinism: arrival order affects early-stop timing, never floating reduction order.
// Complexity: O(d) per newly reducible path, O(W*d) bounded pending storage.
// AI-Hints: retain earlier pending successes after a later path fails, so the
//
//	final partial Result can still contain the whole valid preceding prefix.
func (d *driverState) acceptOutcome(outcome sampleOutcome) {
	if outcome.err != nil {
		d.stop(outcome.err)
	} else if outcome.id >= d.nextReduce {
		// Each logical ID is dispatched once, so no replacement policy is needed.
		d.pending[outcome.id] = outcome
	}
	d.reduceReady()
}

// reduceReady consumes every available successive outcome starting at nextReduce.
// It never folds a success located beyond an uncompleted or failed lower path.
//
// Inputs: all pending vectors have worker-validated dimension and finite values.
// Returns: no value; on storage/moments errors stop preserves prefix integrity.
// Determinism: exact ascending path IDs and ascending column order on every row.
// Complexity: amortized O(d) per reduced path, O(1) work on a missing prefix.
// AI-Hints: advancing nextReduce before both storage and moments commit would
//
//	publish an invalid prefix on a rare dependency error.
func (d *driverState) reduceReady() {
	for {
		outcome, ok := d.pending[d.nextReduce]
		if !ok {
			return
		}
		delete(d.pending, d.nextReduce)
		if err := d.commitRow(outcome.values); err != nil {
			d.stop(err)
			return
		}
		d.nextReduce++
	}
}

// commitRow records one validated row and updates its vector moments.
// The index advances only after both operations succeeded; retained rows after
// a failed transition are never made visible through the partial prefix view.
//
// Inputs: row contains exactly dimension finite values and owns detached storage.
// Returns: nil on commit, or an error classifying the failed dependency stage.
// Determinism: one fixed ascending column scan in each subordinate operation.
// Complexity: Time O(d), Space O(1) additional per path.
// AI-Hints: keep this path as the only place that mutates samples or moments.
func (d *driverState) commitRow(row []float64) error {
	if err := d.storeRow(row); err != nil {
		return err
	}
	if err := d.moments.add(row); err != nil {
		return err
	}
	return nil
}

// storeRow writes to at most one raw backing buffer after budget preflight.
// The dense path uses Set for the original matrix's numeric/storage policy;
// the quantiles-only path uses a checked row-major flat slice.
//
// Inputs: row is one finite vector, and d.row is the next checked row index.
// Returns: nil or ErrDomain joined with unexpected matrix storage failure.
// Determinism: row-major input order, with no sorting or worker contribution.
// Complexity: Time O(d), Space O(1) extra.
// AI-Hints: do not convert uint64 path IDs to int in the hot loop.
func (d *driverState) storeRow(row []float64) error {
	if !d.config.keepRaw {
		return nil
	}
	if d.retained != nil {
		for column, value := range row {
			if err := d.retained.Set(d.row, column, value); err != nil {
				return errors.Join(ErrDomain, err)
			}
		}
	} else {
		// The complete element count passed checkedInt in the preflight stage.
		base := d.row * d.config.dimension
		copy(d.raw[base:base+d.config.dimension], row)
	}
	d.row++
	return nil
}

// joinWorkers closes dispatch only after all accepted outcomes have been drained.
// The worker WaitGroup then guarantees no sampler call outlives Run's return.
//
// Inputs: dispatch has reached its completion, cancellation, or error exit.
// Returns: no value; closes jobs and waits for every worker goroutine.
// Determinism: shutdown timing is not observable in a successful numeric result.
// Complexity: O(W) coordination; join latency depends on sampler cancellation.
// AI-Hints: Go cannot forcibly stop an arbitrary Sampler ignoring context.
func (d *driverState) joinWorkers() {
	close(d.jobs)
	d.workers.Wait()
}

// finalize publishes the reduced prefix, moments and exact quantiles.
// Summary vectors are all-or-nothing; on numeric failure their fields remain nil
// rather than exposing a partially filled coordinate vector.
//
// Inputs: the pool has already joined; no worker can mutate state or data.
// Returns: complete Result, or incomplete Result and the first terminal error.
// Determinism: summary columns and quantiles are published in ascending order.
// Complexity: O(d) for moments, plus O(Paths*d*log(Paths)) for quantile sorting.
// AI-Hints: do not calculate quantiles on an incomplete run, even for a long prefix.
func (d *driverState) finalize() (*Result, error) {
	// Cancellation during the final barrier still prevents a success marker.
	if err := d.child.Err(); err != nil {
		d.stop(err)
	}
	d.result.Completed = d.nextReduce
	if err := d.moments.summarize(d.result); err != nil {
		d.stop(err)
	}

	if d.config.plan.RetainSamples && d.nextReduce != 0 {
		if d.nextReduce == d.config.plan.Paths {
			d.result.Samples = d.retained
		} else {
			// Expose only actual reduced rows without duplicating the budgeted array.
			d.result.Samples = &prefixMatrix{
				owner: d.retained, rows: d.row, cols: d.config.dimension,
			}
		}
	}

	if d.failure == nil && d.nextReduce == d.config.plan.Paths &&
		len(d.config.levels) != 0 {
		quantiles, err := quantileMatrix(d.child, d.config.pathRows,
			d.config.dimension, d.config.levels, d.valueAt)
		if err != nil {
			d.stop(err)
		} else if err := d.child.Err(); err != nil {
			d.stop(err)
		} else {
			d.result.Quantiles = quantiles
		}
	}

	d.result.Complete = d.failure == nil && d.nextReduce == d.config.plan.Paths
	return d.result, d.failure
}

// valueAt returns an already-stored finite component for exact column sorting.
// It reads from the unique raw owner: either retained Dense or flat scratch.
//
// Inputs: row and column come from checked quantile iteration bounds.
// Returns: the scalar or the backend's indexing error for explicit propagation.
// Determinism: row-major storage and ascending iteration, no mutable alias escape.
// Complexity: Time O(1), Space O(1).
// AI-Hints: a closure that discards matrix.At errors hides backend contract drift.
func (d *driverState) valueAt(row, column int) (float64, error) {
	if d.retained != nil {
		return d.retained.At(row, column)
	}
	return d.raw[row*d.config.dimension+column], nil
}
