// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File fuzz_test.go probes untrusted price bytes and checkpoint damage.
package rsi_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/lvlath/go/rsi"
)

// FuzzStreamParityAndRollback varies price magnitude and malformed inputs under deterministic fuzz bytes.
// Every accepted price must match batch bits and every rejected price must preserve state.
func FuzzStreamParityAndRollback(f *testing.F) {
	f.Add([]byte{1, 2, 0, 4, 1, 8})
	f.Add([]byte{0, 0, 0, 0})
	f.Add([]byte{255, 4, 255, 0, 3})
	f.Fuzz(func(t *testing.T, bytes []byte) {
		if len(bytes) > 256 {
			return
		}
		period := 1 + len(bytes)%17
		s, err := rsi.New(rsi.Options{Period: period})
		if err != nil {
			t.Fatal(err)
		}
		closes := make([]float64, 0, len(bytes))
		for i, code := range bytes {
			before := s.Snapshot()
			// Include invalid input and a high dynamic range without randomization.
			var price float64
			switch code % 11 {
			case 0:
				price = 0
			case 1:
				price = math.NaN()
			case 2:
				price = math.Inf(1)
			default:
				price = float64(code+1) * math.Ldexp(1, (i%9)*90-360)
			}
			v, valid, pushErr := s.Push(price)
			if pushErr != nil {
				assertSnapshotEqual(t, before, s.Snapshot())
				continue
			}
			closes = append(closes, price)
			batch, e := rsi.Compute(closes, rsi.Options{Period: period})
			if e != nil {
				t.Fatal(e)
			}
			mustSameFloat(t, v, batch.Values[len(closes)-1])
			mustEqualBool(t, valid, batch.Valid[len(closes)-1], "fuzz parity")
			restored, e := rsi.New(rsi.Options{Period: period})
			if e != nil {
				t.Fatal(e)
			}
			if e = restored.Restore(s.Snapshot()); e != nil {
				t.Fatal(e)
			}
			assertSnapshotEqual(t, s.Snapshot(), restored.Snapshot())
		}
	})
}

// FuzzSnapshotRejectsBadConfiguration mutates external metadata while retaining an otherwise valid shape.
// A mismatched identifier or digest must reject atomically and preserve the original checkpoint.
func FuzzSnapshotRejectsBadConfiguration(f *testing.F) {
	f.Add([]byte("incorrect"))
	f.Add([]byte{0, 1, 2})
	f.Fuzz(func(t *testing.T, payload []byte) {
		s, e := rsi.New(rsi.Options{Period: 7})
		if e != nil {
			t.Fatal(e)
		}
		before := s.Snapshot()
		candidate := s.Snapshot()
		if len(payload)%2 == 0 {
			candidate.Config = string(payload)
		} else {
			candidate.Algorithm = string(payload)
		}
		// A matching literal is a legitimate no-op, not an invalid input.
		if candidate.Config == before.Config && candidate.Algorithm == before.Algorithm {
			return
		}
		if len(payload) >= 8 && payload[0] == 255 {
			candidate.Count = binary.LittleEndian.Uint64(payload[:8])
		}
		mustErrorIs(t, s.Restore(candidate), rsi.ErrStateVersion)
		assertSnapshotEqual(t, before, s.Snapshot())
	})
}
