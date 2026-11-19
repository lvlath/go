// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test benchmarks the canonical inference API on reproducible
// dense finite inputs across latency, long-history, batch and output regimes.
// All fixture construction is excluded from measured time and reported allocations.
package lstm_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/lvlath/go/lstm"
	"github.com/lvlath/go/matrix"
)

// BenchmarkInfer measures real affine/gate computation instead of an error path.
// It exercises B=1/16, T=0/32/1024, D=8/64, H=16/128 and both output modes.
//
// Implementation:
//   - Stage 1: create finite nonzero dense IFGO matrices and nonzero telemetry once.
//   - Stage 2: import the model, verify the fully known tensor geometry, and reset time.
//   - Stage 3: invoke only canonical Infer and check its error inside the timed loop.
//
// Behavior highlights:
//   - T=0 measures state/publication overhead; T=32 measures short online packets.
//   - T=1024 measures history processing; B=16 exposes independent batch throughput.
//
// Determinism:
//   - All fixtures are generated using integer modular patterns, never global RNG.
//
// Complexity:
//   - Each invocation is O(B*T*H*(D+H)); optional output O(B*T*H).
//
// AI-Hints:
//   - Do not use all-zero weights here: they conceal the actual numerical regime.
//   - Setup and error-path measurement must not contaminate allocs/op or ns/op.
func BenchmarkInfer(b *testing.B) {
	shapes := []struct {
		batch, steps, inputWidth, hidden int
	}{
		{1, 0, 8, 16}, {16, 0, 8, 16},
		{1, 32, 8, 16}, {16, 32, 8, 16},
		{1, 1024, 8, 16}, {16, 1024, 8, 16},
		{1, 32, 64, 128}, {16, 32, 64, 128},
		{1, 1024, 64, 128}, {16, 1024, 64, 128},
	}
	for _, shape := range shapes {
		for _, sequence := range []bool{false, true} {
			name := fmt.Sprintf("B%d_T%d_D%d_H%d_Sequence%t", shape.batch, shape.steps, shape.inputWidth, shape.hidden, sequence)
			b.Run(name, func(b *testing.B) {
				rows := 4 * shape.hidden
				inputWeights, err := matrix.NewDense(rows, shape.inputWidth)
				if err != nil {
					b.Fatal(err)
				}
				recurrentWeights, err := matrix.NewDense(rows, shape.hidden)
				if err != nil {
					b.Fatal(err)
				}
				for row := 0; row < rows; row++ {
					for col := 0; col < shape.inputWidth; col++ {
						value := .005 * float64((row*7+col*11)%17-8)
						if err = inputWeights.Set(row, col, value); err != nil {
							b.Fatal(err)
						}
					}
					for col := 0; col < shape.hidden; col++ {
						value := .003 * float64((row*5+col*3)%13-6)
						if err = recurrentWeights.Set(row, col, value); err != nil {
							b.Fatal(err)
						}
					}
				}
				bias := make([]float64, rows)
				for index := range bias {
					bias[index] = .01 * float64(index%5-2)
				}
				model, err := lstm.Import(lstm.Weights{Input: inputWeights, Recurrent: recurrentWeights, Bias: bias, Hidden: shape.hidden})
				if err != nil {
					b.Fatal(err)
				}
				data := make([]float64, shape.batch*shape.steps*shape.inputWidth)
				for index := range data {
					data[index] = .02 * float64(index%31-15)
				}
				input := lstm.Tensor3{Data: data, Batch: shape.batch, Time: shape.steps, Features: shape.inputWidth}
				options := lstm.Options{ReturnSequence: sequence}
				ctx := context.Background()

				b.ReportAllocs()
				b.ResetTimer()
				for iteration := 0; iteration < b.N; iteration++ {
					if _, err = model.Infer(ctx, input, nil, nil, options); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
