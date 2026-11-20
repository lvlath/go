// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax_test checks frozen query-set identity under sorting and value drift.
// Query hashes capture semantics but never claim to authenticate temporal provenance.
package tfsax_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/tfsax"
)

func TestQueriesHash_ReorderingAndContentMismatch(t *testing.T) {
	queries := []tfsax.NamedQuery{
		{ID: 5, Family: tfsax.FamilyMotif, Query: tfsax.Query{Values: []float64{0, 1, 2}, TopK: 2}},
		{ID: 3, Family: tfsax.FamilyPhase, Query: tfsax.Query{Values: []float64{3, 2, 1}, TopK: 2}},
	}
	first, err := tfsax.QueriesHash(queries)
	if err != nil {
		t.Fatal(err)
	}
	reversed := []tfsax.NamedQuery{queries[1], queries[0]}
	second, err := tfsax.QueriesHash(reversed)
	if err != nil || second != first {
		t.Fatalf("equivalent query-set order changed identity: %s vs %s, err=%v", first, second, err)
	}
	modified := []tfsax.NamedQuery{queries[0], queries[1]}
	modified[0].Query.Values = append([]float64(nil), queries[0].Query.Values...)
	modified[0].Query.Values[0] = math.Copysign(0, -1)
	third, err := tfsax.QueriesHash(modified)
	if err != nil || third == first {
		t.Fatalf("changed float64 bits failed to change query hash: %v", err)
	}
	modified[0].Query.TopK = 4
	fourth, err := tfsax.QueriesHash(modified)
	if err != nil || fourth == third {
		t.Fatalf("changed TopK failed to change query hash: %v", err)
	}
}

func TestQueriesHash_DuplicateAndMalformed(t *testing.T) {
	_, err := tfsax.QueriesHash(nil)
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
	queries := []tfsax.NamedQuery{{ID: 2, Family: tfsax.FamilyPhase,
		Query: tfsax.Query{Values: []float64{1, 2}, TopK: 1}}}
	queries = append(queries, queries[0])
	_, err = tfsax.QueriesHash(queries)
	mustErrorIs(t, err, tfsax.ErrDuplicateID)
	queries = queries[:1]
	queries[0].Query.Values = []float64{1, math.NaN()}
	_, err = tfsax.QueriesHash(queries)
	mustErrorIs(t, err, tfsax.ErrNonFinite)
	queries[0].Query.Values = []float64{1, 2}
	queries[0].Family = "invalid"
	if _, err = tfsax.QueriesHash(queries); !errors.Is(err, tfsax.ErrInvalidInput) {
		t.Fatal(err)
	}
}
