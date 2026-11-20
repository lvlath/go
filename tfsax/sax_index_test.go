// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax_test independently checks the public immutable SAX comparator.
// These tests protect ranking-only DTW semantics, no-alias ownership and strict errors.
package tfsax_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/tfsax"
)

func TestSAXIndex_DeepCopyAndDeterministicParity(t *testing.T) {
	corpus, err := tfsax.GenerateFamily(tfsax.FamilyMotif, 902, 32, 48)
	if err != nil {
		t.Fatal(err)
	}
	queryValues := append([]float64(nil), corpus[9].Values...)
	query := tfsax.Query{Values: queryValues, TopK: 5}
	hash, err := tfsax.CorpusHash(corpus)
	if err != nil {
		t.Fatal(err)
	}
	index, err := tfsax.PrepareSAXIndex(corpus, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	if index.CorpusDigest() != hash || index.IndexBytes() != 32*(8*2+8) {
		t.Fatalf("wrong index digest/payload %s %d", index.CorpusDigest(), index.IndexBytes())
	}
	expected, err := index.Search(context.Background(), query, 4, 16)
	if err != nil {
		t.Fatal(err)
	}
	oneShot, err := tfsax.ScanSAX(context.Background(), corpus, query, 4, 16, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	mustIDs(t, oneShot.Candidates, expected.Candidates)
	if expected.Diagnostics.Exact || expected.Diagnostics.Examined != 16 || expected.Diagnostics.Pruned != 16 {
		t.Fatalf("heuristic proof fabricated: %+v", expected.Diagnostics)
	}

	// The prepared index owns both symbols and exact-refinement raw data.
	corpus[9].Values[0] += 300
	corpus[9].ID = 9999
	again, err := index.Search(context.Background(), query, 4, 16)
	if err != nil {
		t.Fatal(err)
	}
	mustIDs(t, again.Candidates, expected.Candidates)
	if index.CorpusDigest() != hash {
		t.Fatal("caller mutation changed prepared index identity")
	}
	again.Candidates[0] = 12345
	third, err := index.Search(context.Background(), query, 4, 16)
	if err != nil {
		t.Fatal(err)
	}
	mustIDs(t, third.Candidates, expected.Candidates)
}

func TestSAXIndex_ErrorClassification(t *testing.T) {
	corpus := []tfsax.Candidate{
		{ID: 1, Values: []float64{0, 1, 0, 1}},
		{ID: 2, Values: []float64{1, 0, 1, 0}},
	}
	query := tfsax.Query{Values: []float64{0, 1, 0, 1}, TopK: 1}
	if _, err := tfsax.PrepareSAXIndex(corpus, 5, 8); !errors.Is(err, tfsax.ErrInvalidOption) {
		t.Fatal(err)
	}
	if _, err := tfsax.PrepareSAXIndex(corpus, 2, 257); !errors.Is(err, tfsax.ErrInvalidOption) {
		t.Fatal(err)
	}
	index, err := tfsax.PrepareSAXIndex(corpus, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	var absent *tfsax.SAXIndex
	if absent.IndexBytes() != 0 || absent.CorpusDigest() != "" {
		t.Fatal("nil prepared index must have no payload/digest")
	}
	_, err = absent.Search(context.Background(), query, 0, 1)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	_, err = index.Search(nil, query, 0, 1)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	_, err = index.Search(context.Background(), query, -2, 1)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	_, err = index.Search(context.Background(), query, 0, 3)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	_, err = index.Search(context.Background(), tfsax.Query{Values: []float64{0, math.NaN(), 0, 1}, TopK: 1}, 0, 1)
	mustErrorIs(t, err, tfsax.ErrNonFinite)
	_, err = tfsax.ScanSAXWithOptions(context.Background(), corpus, query, tfsax.SAXSearchOptions{Segments: 0, Alphabet: 8, Window: 0, Shortlist: 1})
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	_, err = tfsax.ScanSAXWithOptions(nil, corpus, query, tfsax.SAXSearchOptions{})
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
}

func TestSAXIndex_ConcurrentReadOnlySearch(t *testing.T) {
	corpus, err := tfsax.GenerateFamily(tfsax.FamilyPhase, 93, 20, 32)
	if err != nil {
		t.Fatal(err)
	}
	index, err := tfsax.PrepareSAXIndex(corpus, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	query := tfsax.Query{Values: append([]float64(nil), corpus[5].Values...), TopK: 4}
	baseline, err := index.Search(context.Background(), query, 4, 10)
	if err != nil {
		t.Fatal(err)
	}
	var workers sync.WaitGroup
	errorsFromWorkers := make(chan error, 8)
	for worker := 0; worker < 8; worker++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			result, searchErr := index.Search(context.Background(), query, 4, 10)
			if searchErr != nil {
				errorsFromWorkers <- searchErr
				return
			}
			if len(result.Candidates) != len(baseline.Candidates) {
				errorsFromWorkers <- tfsax.ErrInvalidInput
				return
			}
			for i, candidate := range result.Candidates {
				if candidate != baseline.Candidates[i] {
					errorsFromWorkers <- tfsax.ErrInvalidInput
					return
				}
			}
		}()
	}
	workers.Wait()
	close(errorsFromWorkers)
	for searchErr := range errorsFromWorkers {
		t.Fatalf("concurrent immutable read disagreed: %v", searchErr)
	}
}
