// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax_test checks frozen query-set identity under sorting and value drift.
// Query hashes capture semantics but never claim to authenticate temporal provenance.
package tfsax_test

import (
	"errors"
	"math"
	"strconv"
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

// TestQueriesHash_CanonicalTopKWord protects the versioned eight-byte TopK
// encoding against changes in host integer width or internal serialization.
// The fixed hashes were calculated independently from the documented LE64
// byte layout, rather than by calling the implementation under test.
func TestQueriesHash_CanonicalTopKWord(t *testing.T) {
	const (
		ordinaryDigest = "c3079533c100ce2e6c21cab9a69796dbfb34e4a39e7003a2db6ba33ef7bacae9"
		maxInt32Digest = "58e39aceb9659c29cb662a2b0af775c3788eb665bb48407123a93483fcd39018"
		maxInt64Digest = "9c30a7644b5685ce271ac5c80c5f70980333daae20ffbe518d27c2daf6826c3e"
	)
	maxIntDigest := maxInt32Digest
	if strconv.IntSize == 64 {
		maxIntDigest = maxInt64Digest
	}

	cases := []struct {
		name   string
		topK   int
		digest string
	}{
		{name: "ordinary", topK: 2, digest: ordinaryDigest},
		{name: "platform_maximum", topK: int(^uint(0) >> 1), digest: maxIntDigest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			query := tfsax.NamedQuery{
				ID:     5,
				Family: tfsax.FamilyMotif,
				Query:  tfsax.Query{Values: []float64{0, 1, 2}, TopK: tc.topK},
			}
			got, err := tfsax.QueriesHash([]tfsax.NamedQuery{query})
			if err != nil {
				t.Fatalf("valid TopK rejected: %v", err)
			}
			if got != tc.digest {
				t.Fatalf("query digest changed: got %s, want %s", got, tc.digest)
			}
			if query.Query.TopK != tc.topK {
				t.Fatal("hashing mutated the caller's query")
			}
		})
	}
}

// TestQueriesHash_NonpositiveTopK verifies that invalid signed counts never
// become successful unsigned identities, regardless of machine integer width.
func TestQueriesHash_NonpositiveTopK(t *testing.T) {
	for _, topK := range []int{0, -1, -int(^uint(0)>>1) - 1} {
		query := tfsax.NamedQuery{
			ID:     5,
			Family: tfsax.FamilyMotif,
			Query:  tfsax.Query{Values: []float64{0, 1, 2}, TopK: topK},
		}
		got, err := tfsax.QueriesHash([]tfsax.NamedQuery{query})
		if !errors.Is(err, tfsax.ErrInvalidInput) || got != "" {
			t.Fatalf("TopK %d: got digest=%q err=%v, want empty digest and ErrInvalidInput", topK, got, err)
		}
	}
}
