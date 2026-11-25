// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Gaussian regression fixtures fingerprint the original binary64 table through
// public Encoder results. No private table inspector or reflection is used.
package sax_test

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"testing"

	"github.com/lvlath/go/sax"
)

func TestGaussian_OriginalBinaryTableIdentityAllAlphabets(t *testing.T) {
	// Golden digests use the original offline quantile bytes and the v2 version.
	// They were independently serialized, not captured from NewEncoder output.
	const expectedAll = "4ea3a319048e111d5e04caf2f6feac60a754c6fd1417bfdcfb0e980d11cb45b5"
	goldenIDs := map[int]string{
		2:   "58d0338ff1522e8e7b78d9933f3f1fd3065900b7d26de7425af48a7e523f0a62",
		3:   "4ff569dbab019c8e27d84b2dbcb68195f11848e227ee4fd644c71d7adbebe86b",
		4:   "81d21c1bf08ba39344874c22e7e3935bd576908ab88707bd2a9f415184f0445b",
		8:   "5f474b57e9b4400911c97ec63452d25dc3daeb98d19ecf04ce85b6de4269a405",
		16:  "72b6c82514f757327923151e550c95e69cee06b1b9df9648d50df19127a755dd",
		32:  "4ceeec652817aa5be21ff77139b4316b3cccc25a9bf387162e67e729252dcb13",
		64:  "34e86e8d5fead870636f2bcc0c183620d15ca735f111f1cf2308c3c0cdbf5700",
		128: "e930c8f792e958231c782f79d81530948f346a4e1b1434f6d6825bfbdf3ecc96",
		256: "d242c7638bc68fd045c77ec1813f76db1287017553e5a095a5f6381bc43b14a9",
	}
	hasher := sha256.New()
	for alphabet := 2; alphabet <= 256; alphabet++ {
		opts := sax.DefaultOptions(4)
		opts.Normalization = sax.Raw
		opts.Alphabet = alphabet
		enc, err := sax.NewEncoder(opts)
		if err != nil {
			t.Fatalf("alphabet %d: %v", alphabet, err)
		}
		word, err := enc.Encode([]float64{-100, -1, 1, 100})
		if err != nil {
			t.Fatalf("alphabet %d: %v", alphabet, err)
		}
		if expected, exists := goldenIDs[alphabet]; exists {
			mustEqualString(t, word.EncodingID, expected, "pinned identity")
		}
		_, _ = hasher.Write([]byte(word.EncodingID))
	}
	mustEqualString(t, hex.EncodeToString(hasher.Sum(nil)), expectedAll, "all Gaussian alphabet identities")
}

func TestGaussian_NeighboringThresholdBoundaryULPs(t *testing.T) {
	// The published a=3 NDTRI values are deliberately asymmetric by one ULP
	// due to the binary64 evaluation of p and 1-p; replacing with -x breaks IDs.
	lower, upper := -0.43072729929545756, 0.43072729929545744
	values := []float64{
		math.Nextafter(lower, math.Inf(-1)), lower, math.Nextafter(lower, math.Inf(1)),
		math.Nextafter(upper, math.Inf(-1)), upper, math.Nextafter(upper, math.Inf(1)),
	}
	encoded, err := sax.EncodeRaw(values, len(values), 3)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualSymbols(t, encoded.Symbols, []uint16{0, 1, 1, 1, 2, 2}, "nearest-boundary ULP behavior")
}

func TestGaussian_DefaultZeroThresholdGoesRight(t *testing.T) {
	result, err := sax.EncodeRaw([]float64{-1, math.Copysign(0, -1), 0, 1}, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualSymbols(t, result.Symbols, []uint16{0, 1, 1, 1}, "signed zero Gaussian bin")
}
