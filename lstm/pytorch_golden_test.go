// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
// Frozen external reference captured from PyTorch 2.10.0+cpu torch.nn.LSTM,
// float64 CPU, 1 layer, bidirectional=false, proj_size=0, batch_first=true,
// eval(), deterministic algorithms enabled, MKLDNN disabled, one CPU thread.
// The source numbers and expected outputs are committed directly as Go literals;
// no Python interpreter, PyTorch package, serializer, or exporter runs in CI.
// AI-Hints: this is an independent external oracle, not a Go-computed snapshot.
package lstm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/lstm"
)

func TestPyTorchDoubleCPUGolden(t *testing.T) {
	wi := []float64{-0.22199999999999998, 0.036999999999999998, 0.185, -0.036999999999999998, 0.11099999999999999, -0.11099999999999999, 0.036999999999999998, -0.185, -0.036999999999999998, 0.22199999999999998, -0.11099999999999999, 0.14799999999999999, -0.185, 0.073999999999999996, 0.22199999999999998, 0}
	wr := []float64{-0.14500000000000002, -0.058000000000000003, 0, 0.087000000000000008, 0.14500000000000002, -0.087000000000000008, -0.029000000000000001, 0.058000000000000003, 0.11600000000000001, -0.11600000000000001, -0.058000000000000003, 0.029000000000000001, 0.087000000000000008, -0.14500000000000002, -0.087000000000000008, 0}
	bias := []float64{-0.089999999999999997, -0.034000000000000002, 0.021999999999999999, -0.012999999999999999, 0.042999999999999997, 0.0080000000000000036, 0.064000000000000001, 0.029000000000000005}
	x := []float64{-0.64800000000000002, -0.081000000000000003, 0.48599999999999999, -0.32400000000000001, 0.24299999999999999, -0.56700000000000006, 0, 0.56700000000000006, -0.24299999999999999, 0.32400000000000001, -0.48599999999999999, 0.081000000000000003}
	seq := []float64{0.013660286173635943, 0.014777492162501701, -0.0037795816030461778, -0.017303330244298368, -0.022475350494597545, -0.035985765669772317, 0.042424027062736881, 0.022592903663089543, 0.053903536695805949, 0.02955714018517018, 0.048919443164907629, 0.030089638668012168}
	hidden := []float64{-0.022475350494597545, -0.035985765669772317, 0.048919443164907629, 0.030089638668012168}
	cell := []float64{-0.04545281830043333, -0.069206483351841042, 0.090842075512978668, 0.062885072264854458}
	m, e := lstm.Import(lstm.Weights{Input: mustMatrix(t, 8, 2, wi), Recurrent: mustMatrix(t, 8, 2, wr), Bias: bias, Hidden: 2})
	if e != nil {
		t.Fatal(e)
	}
	if m.ID() != "d521b810acc028f8666cf7f6970f2a9b3f675710489d4c32be96fead620a1658" {
		t.Fatalf("canonical ModelID mismatch: %s", m.ID())
	}
	r, e := m.Infer(context.Background(), lstm.Tensor3{Data: x, Batch: 2, Time: 3, Features: 2}, nil, nil, lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	for i, want := range seq {
		if math.Abs(r.Sequence.Data[i]-want) > 1e-12 {
			t.Fatalf("PyTorch seq[%d]: %.17g versus %.17g", i, r.Sequence.Data[i], want)
		}
	}
	for i, want := range hidden {
		if math.Abs(at(t, r.Final.Hidden, i/2, i%2)-want) > 1e-12 {
			t.Fatalf("PyTorch hidden[%d] mismatch", i)
		}
	}
	for i, want := range cell {
		if math.Abs(at(t, r.Final.Cell, i/2, i%2)-want) > 1e-12 {
			t.Fatalf("PyTorch cell[%d] mismatch", i)
		}
	}
}
