// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm implements one canonical IFGO recurrent forward kernel.
// This file owns compensated affine arithmetic, exact overflow rescue and masked steps.
package lstm

import (
	"context"
	"math"
	"math/big"
)

const (
	// Gate rows are inspected for cancellation at a bounded interval inside a long step.
	gateContextPollInterval = 64
	// Extreme finite affine results require exact classification near MaxFloat64.
	// This guard prevents an unrepresentable MaxFloat64+epsilon from rounding down
	// to a finite MaxFloat64 and escaping the overflow classification.
	exactAffineThreshold = math.MaxFloat64 / 2
)

// affine computes one IFGO preactivation from finite weights, input and previous hidden state.
// The fast path uses Neumaier-style compensation; rare overflow risks use exact rational sums.
//
// Implementation:
//   - Stage 1: initialize the accumulator from the stored combined bias.
//   - Stage 2: add Input row terms, then Recurrent row terms in ascending column order.
//   - Stage 3: add compensation and rescue if a product, sum, correction or large
//     boundary value may hide an unrepresentable exact result.
//
// Behavior highlights:
//   - A temporary Inf is not a valid activated gate; exact cancellation may be finite.
//   - The float64 fast path contains no heap allocation per multiplication.
//
// Inputs:
//   - row: IFGO row index in [0,4H); x: current D finite input features.
//   - hidden: previous H hidden values, read-only for this entire gate pass.
//
// Returns:
//   - float64: finite preactivation rounded according to the selected numerical path.
//
// Errors:
//   - ErrOverflow: exact affine magnitude exceeds the largest finite float64.
//
// Determinism:
//   - Bias, ascending Input columns, ascending Recurrent columns; no parallel reduction.
//
// Complexity:
//   - Usual time O(D+H), space O(1); exact rescue additionally uses big.Rat memory.
//
// Notes:
//   - Exact rescue is a correctness branch, not a second choice of gate equations.
//   - Compensated summation mitigates rounding but cannot make all CPU math identical.
//
// AI-Hints:
//   - Never replace ErrOverflow by sigmoid(Inf), even if the resulting bit is 1.
//   - In particular, a finite rounded MaxFloat64 can conceal a true out-of-range sum.
func (m *Model) affine(row int, x, hidden []float64) (float64, error) {
	sum := m.bias[row]
	compensation := float64(0)

	// Keep the accumulator stable for conditioned values without allocating big.Rat.
	add := func(term float64) bool {
		if !finite(term) {
			return false // Product overflow: exact cancellation may still be possible.
		}
		next := sum + term
		if !finite(next) {
			return false // Intermediate sum overflow needs a whole-dot check.
		}
		if math.Abs(sum) >= math.Abs(term) {
			compensation += (sum - next) + term
		} else {
			compensation += (term - next) + sum
		}
		sum = next
		return finite(compensation)
	}

	// Input coefficients are contiguous rows of length D in privately owned storage.
	for col, weight := range m.wi[row*m.d : (row+1)*m.d] {
		if !add(weight * x[col]) {
			return m.affineExact(row, x, hidden)
		}
	}
	// Recurrent coefficients use the previous hidden vector, never an updated gate value.
	for col, weight := range m.wr[row*m.h : (row+1)*m.h] {
		if !add(weight * hidden[col]) {
			return m.affineExact(row, x, hidden)
		}
	}

	result := sum + compensation
	if !finite(result) || math.Abs(result) >= exactAffineThreshold {
		// The boundary guard also catches a true MaxFloat64+tiny rounded to MaxFloat64.
		return m.affineExact(row, x, hidden)
	}
	return result, nil
}

// affineExact re-evaluates the complete row as exact products and an exact rational sum.
// It distinguishes a truly unrepresentable result from finite cancellation of huge terms.
//
// Implementation:
//   - Stage 1: convert the finite combined bias exactly from IEEE-754 to big.Rat.
//   - Stage 2: multiply and add every exact input and recurrent term in fixed order.
//   - Stage 3: compare magnitude against exact MaxFloat64 before final binary64 rounding.
//
// Behavior highlights:
//   - Intermediate products are arbitrary precision, so +Inf/-Inf never appear.
//   - Exact magnitudes exceeding MaxFloat64 cause ErrOverflow even if rounding
//     would otherwise yield a still-finite MaxFloat64 at the boundary.
//
// Inputs:
//   - row: one checked IFGO coefficient row; x and hidden: finite immutable vectors.
//
// Returns:
//   - float64: finite nearest binary64 result for a representable-magnitude sum.
//
// Errors:
//   - ErrOverflow: exact sum is outside [-MaxFloat64, +MaxFloat64].
//
// Determinism:
//   - Same term order as affine; arbitrary-precision rational arithmetic is exact.
//
// Complexity:
//   - O(D+H) rational operations with operand-size-dependent time and heap space;
//     invoked only on high-dynamic-range exceptional inputs.
//
// Notes:
//   - Ordinary binary64 underflow follows standard rounding and is not overflow.
//
// AI-Hints:
//   - Do not use big.Float with insufficient precision as a substitute for this path.
func (m *Model) affineExact(row int, x, hidden []float64) (float64, error) {
	total := new(big.Rat).SetFloat64(m.bias[row])
	left := new(big.Rat)
	right := new(big.Rat)
	product := new(big.Rat)

	for col, weight := range m.wi[row*m.d : (row+1)*m.d] {
		left.SetFloat64(weight)
		right.SetFloat64(x[col])
		product.Mul(left, right)
		total.Add(total, product)
	}
	for col, weight := range m.wr[row*m.h : (row+1)*m.h] {
		left.SetFloat64(weight)
		right.SetFloat64(hidden[col])
		product.Mul(left, right)
		total.Add(total, product)
	}

	// Float64() alone is insufficient here: MaxFloat64+1 rounds to MaxFloat64.
	limit := new(big.Rat).SetFloat64(math.MaxFloat64)
	if new(big.Rat).Abs(total).Cmp(limit) > 0 {
		return 0, ErrOverflow
	}
	result, _ := total.Float64()
	if !finite(result) {
		return 0, ErrOverflow
	}
	return result, nil
}

// recurrentCell applies the standard LSTM cell-memory transition with range checks.
// It protects exact overflow hidden by rounded binary64 addition at MaxFloat64.
//
// Implementation:
//   - Stage 1: compute forgetGate*previous + inputGate*candidate in binary64.
//   - Stage 2: accept ordinary conditioned finite values without heap allocation.
//   - Stage 3: near range limits, form two exact rational products and compare
//     the exact sum's magnitude to the finite binary64 maximum before rounding.
//
// Behavior highlights:
//   - Underflow is ordinary binary64 rounding; exact range overflow is ErrOverflow.
//   - The fallback is rare and uses the identical recurrence equation.
//
// Inputs:
//   - forgetGate/inputGate: finite sigmoids in [0,1].
//   - previous: finite initial or preceding cell value; candidate: finite tanh in [-1,1].
//
// Returns:
//   - float64: checked finite next cell state, or zero with an error.
//
// Errors:
//   - ErrOverflow: exact cell sum is outside the binary64 finite magnitude range.
//
// Determinism:
//   - The same fixed forget product, input product and sum order for every call.
//
// Complexity:
//   - Time/Space O(1) on the fast path; rare fixed-size big.Rat allocation.
//
// Notes:
//   - Only validated internal gate values reach this function.
//
// AI-Hints:
//   - A finite rounded nextCell == MaxFloat64 does not prove exact representability.
func recurrentCell(forgetGate, previous, inputGate, candidate float64) (float64, error) {
	next := forgetGate*previous + inputGate*candidate
	if finite(next) && math.Abs(next) < exactAffineThreshold {
		return next, nil
	}

	left := new(big.Rat).Mul(new(big.Rat).SetFloat64(forgetGate), new(big.Rat).SetFloat64(previous))
	right := new(big.Rat).Mul(new(big.Rat).SetFloat64(inputGate), new(big.Rat).SetFloat64(candidate))
	total := new(big.Rat).Add(left, right)
	limit := new(big.Rat).SetFloat64(math.MaxFloat64)
	if new(big.Rat).Abs(total).Cmp(limit) > 0 {
		return 0, ErrOverflow
	}
	value, _ := total.Float64()
	if !finite(value) {
		return 0, ErrOverflow
	}
	return value, nil
}

// kernel advances all valid timesteps and writes optional sequence features in-place.
// Padded timesteps do not execute gates or touch cell/hidden state; outputs remain +0.
//
// Implementation:
//   - Stage 1: allocate one shared-per-call IFGO vector (only when a valid step exists).
//   - Stage 2: for each batch row and valid timestep, calculate all 4H affines
//     from the previous hidden state, then compute candidate cell and hidden values.
//   - Stage 3: copy completed hidden vectors into the optional detached output tensor.
//
// Behavior highlights:
//   - Both old hidden and old cell remain visible until each corresponding step update.
//   - No per-step/per-gate heap storage is allocated on the normal numeric path.
//   - Caller-visible atomicity is implemented by Infer publishing only after success.
//
// Inputs:
//   - ctx: active context; input: checked [B,T,D] float64 tensor.
//   - lengths: nil or checked local valid lengths; h/c: private mutable [B,H] vectors.
//   - output: nil for final-only mode or zero-filled private [B,T,H] buffer.
//
// Returns:
//   - error: nil on success, otherwise no result is exposed by Infer.
//
// Errors:
//   - ctx.Err(): cancellation at an active step or within long gate vectors.
//   - ErrOverflow: unrepresentable affine, cell, or hidden value.
//
// Determinism:
//   - Ascending batch->time->IFGO row->input/recurrent column->hidden element.
//
// Complexity:
//   - Time O(B*T*H*(D+H)), Space O(4H) transient gates; output is preallocated.
//
// Notes:
//   - This function is private: shape, ownership and input finiteness were checked.
//
// AI-Hints:
//   - All 4H affines MUST be evaluated before overwriting any h element.
//   - Never copy the last valid h into padded output positions.
func (m *Model) kernel(ctx context.Context, input Tensor3, lengths []int, h, c, output []float64) error {
	var gates []float64
	for batch := 0; batch < input.Batch; batch++ {
		limit := input.Time
		if lengths != nil {
			limit = lengths[batch]
		}
		if limit == 0 {
			continue // Neither a gate allocation nor a state update is needed.
		}
		if gates == nil {
			gates = make([]float64, gateBlocks*m.h)
		}
		base := batch * m.h

		for timestep := 0; timestep < limit; timestep++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			offset := tensorOffset(input, batch, timestep)
			features := input.Data[offset : offset+m.d]
			previousHidden := h[base : base+m.h]

			// Compute every preactivation from the SAME previous state before mutation.
			for row := range gates {
				if row%gateContextPollInterval == 0 {
					if err := ctx.Err(); err != nil {
						return err
					}
				}
				preactivation, err := m.affine(row, features, previousHidden)
				if err != nil {
					return err
				}
				gates[row] = preactivation
			}

			// Gate order is IFGO, not the IFOG ordering used by some other runtimes.
			for hiddenIndex := 0; hiddenIndex < m.h; hiddenIndex++ {
				inputGate := sigmoid(gates[inputGateBlock*m.h+hiddenIndex])
				forgetGate := sigmoid(gates[forgetGateBlock*m.h+hiddenIndex])
				candidateGate := math.Tanh(gates[candidateGateBlock*m.h+hiddenIndex])
				outputGate := sigmoid(gates[outputGateBlock*m.h+hiddenIndex])

				// Cell memory and visible state follow the standard LSTM recurrence.
				nextCell, err := recurrentCell(forgetGate, c[base+hiddenIndex], inputGate, candidateGate)
				if err != nil {
					return err
				}
				nextHidden := outputGate * math.Tanh(nextCell)
				if !finite(nextHidden) {
					return ErrOverflow
				}
				c[base+hiddenIndex] = nextCell
				h[base+hiddenIndex] = nextHidden
			}

			if output != nil {
				// The allocation was zero-filled; only valid timestep outputs are written.
				start := (batch*input.Time + timestep) * m.h
				copy(output[start:start+m.h], h[base:base+m.h])
			}
		}
	}
	return nil
}
