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
	// Golden digest computed independently from the original offline NDTRI
	// binary64 table, before moving it from Base64 into an embed asset.
	const expectedAll = "9eac826bb2dc3ed036bc1166951fd459b9cd95aac057ee0592e3cdf580ec60cc"
	goldenIDs := map[int]string{
		2:   "7db0086c581a57e0bb37e559b23ba8520601f5659ba8391a88369c63534dc5df",
		3:   "b68dd98b9f9a551ec4756c8791965b6a9afd08a8b85c655c85c4b2251f5d5ffe",
		4:   "5acf22d804cd41393ec8e92b5108baf75f956ef7a5255541d533f99e52f7f3ed",
		8:   "e29bad5afbed999bef9d2b0a643876767a5b15e9e6555955beef8ed73f7cc1b4",
		16:  "38e27d71f96cd450ebe5e408fe792e0d075471ad17b6b2d7107600c4e4503bdf",
		32:  "9e5d708e741fbed3d0eb34697245d9f1c37ec6245324abe1ac0c0c3fd9e33cc8",
		64:  "3f21edd47f23d593815b0d9ec72cfd8ce15bd4d229672126fa4a4b2ce2ec515f",
		128: "0d2b097b2300c03ea39daeead41764d10e452e84618446b466b83c42ba6926e6",
		256: "618cc660c20575e8df5278d76342000cfcfcb6494745506faad1e432fbbc652a",
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
