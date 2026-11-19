// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test validates causal alignment between batch and streamed variance.
// Suffix changes must not rewrite earlier conditional variance estimates.
package gjrgarch_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

func TestCausality_ChangingCurrentReturnCannotChangeCurrentVariance(t *testing.T) {
	m, e := gjrgarch.New(p())
	if e != nil {
		t.Fatal(e)
	}
	first := []float64{-.1, .2, .3, -.7, .4}
	second := append([]float64(nil), first...)
	second[3] = 100
	a, e := m.Filter(context.Background(), first, 1)
	if e != nil {
		t.Fatal(e)
	}
	b, e := m.Filter(context.Background(), second, 1)
	if e != nil {
		t.Fatal(e)
	}
	for i := 0; i <= 3; i++ {
		near(t, a[i], b[i])
	}
	if a[4] == b[4] {
		t.Fatal("the previous shock must affect the following variance")
	}
}
