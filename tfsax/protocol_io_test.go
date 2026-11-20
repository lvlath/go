// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax_test exercises strict protocol parsing and atomic artifact publication.
// These tests ensure that a malformed research plan is rejected before measurement.
package tfsax_test

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/lvlath/go/tfsax"
)

func TestProtocol_JSONStrictAndAtomic(t *testing.T) {
	protocol := tfsax.DefaultProtocol()
	var marshalled bytes.Buffer
	path := filepath.Join(t.TempDir(), "protocol.json")
	if err := tfsax.WriteJSON(path, protocol); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	marshalled.Write(content)
	parsed, err := tfsax.ParseProtocol(content)
	if err != nil || parsed != protocol {
		t.Fatalf("strict round trip failed: %v / %+v", err, parsed)
	}
	_, err = tfsax.ParseProtocol(nil)
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
	_, err = tfsax.ParseProtocol(append(marshalled.Bytes(), []byte(" true")...))
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
	_, err = tfsax.ParseProtocol([]byte(`{"schema":"unknown","surprise":3}`))
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
	_, err = tfsax.ParseProtocol([]byte(`{"schema":"unknown"}`))
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	mustErrorIs(t, tfsax.WriteJSON("", protocol), tfsax.ErrInvalidOption)
	if err = tfsax.WriteJSON(path, protocol); err != nil {
		t.Fatal(err)
	}
	second, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(content, second) {
		t.Fatalf("non-reproducible JSON bytes: %v", err)
	}
}

func TestGateReport_EvidenceAndErrorProtection(t *testing.T) {
	d := tfsax.UnresolvedDefinition()
	if err := tfsax.ValidateDefinition(d); err != nil {
		t.Fatal(err)
	}
	r := tfsax.EvaluateGate(d, nil)
	if r.Passed || r.CompleteComparison || r.Reason != "definition is unresolved" {
		t.Fatalf("invalid gate report %+v", r)
	}
	d.Approved = true
	mustErrorIs(t, tfsax.ValidateDefinition(d), tfsax.ErrInvalidOption)
	r = tfsax.EvaluateGate(d, nil)
	if r.Passed || r.Reason != "definition metadata is invalid" {
		t.Fatalf("unreviewed approval accepted %+v", r)
	}
	path := filepath.Join(t.TempDir(), "report.md")
	if err := tfsax.WriteNegativeReport(path, r); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(path)
	if err != nil || !bytes.Contains(content, []byte("NOT ADMITTED")) {
		t.Fatalf("negative report not published: %v", err)
	}
	r.Passed = true
	mustErrorIs(t, tfsax.WriteNegativeReport(path, r), tfsax.ErrInvalidOption)
	unchanged, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(content, unchanged) {
		t.Fatal("rejected report unexpectedly overwrote the file")
	}
}

func TestCSV_SchemaAndFileFailure(t *testing.T) {
	entry := tfsax.Observation{QueryID: 2, Family: tfsax.FamilyPhase, Branch: "exact",
		Result: tfsax.RetrievalResult{Complete: true, Diagnostics: tfsax.SearchDiagnostics{Exact: true, Examined: 1}},
		Recall: tfsax.MetricValue{Applicable: true, Value: 1}, MissedBest: tfsax.MetricValue{Applicable: true}}
	file := filepath.Join(t.TempDir(), "observations.csv")
	if err := tfsax.WriteCSV(file, []tfsax.Observation{entry}); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(file)
	if err != nil || !bytes.Contains(b, []byte("representation_version,examined,pruned,available,shortlist")) {
		t.Fatalf("incomplete audit fields: %v", err)
	}
	mustErrorIs(t, tfsax.WriteCSV("", []tfsax.Observation{entry}), tfsax.ErrInvalidOption)
	if err := tfsax.WriteCSV(filepath.Join(t.TempDir(), "missing", "raw.csv"), nil); err == nil {
		t.Fatal("nonexistent directory unexpectedly accepted")
	}
}

func TestProtocol_BudgetsCountAllQueries(t *testing.T) {
	protocol := tfsax.DefaultProtocol()
	cellCost := uint64(protocol.Length * protocol.Length * protocol.CorpusSize)
	protocol.MaxCells = cellCost
	if err := tfsax.ValidateProtocol(protocol); !errors.Is(err, tfsax.ErrBudget) {
		t.Fatalf("Q>1 did not contribute to work budget: %v", err)
	}
	protocol.MaxCells = cellCost * uint64(protocol.QueryCount)
	if err := tfsax.ValidateProtocol(protocol); err != nil {
		t.Fatal(err)
	}
	protocol.Horizon = math.MaxInt
	mustErrorIs(t, tfsax.ValidateProtocol(protocol), tfsax.ErrInvalidOption)
}

// failedAuditWriter forces a concrete I/O error on flushing buffered CSV bytes.
// It exercises error propagation without filesystem permission or clock races.
type failedAuditWriter struct{ cause error }

func (writer failedAuditWriter) Write([]byte) (int, error) { return 0, writer.cause }

func TestCSVStream_ErrorsAndStableBytes(t *testing.T) {
	mustErrorIs(t, tfsax.WriteCSVStream(nil, nil), tfsax.ErrInvalidOption)
	cause := errors.New("audit sink refused bytes")
	if err := tfsax.WriteCSVStream(failedAuditWriter{cause: cause}, nil); !errors.Is(err, cause) {
		t.Fatalf("writer error lost: %v", err)
	}
	obs := []tfsax.Observation{{QueryID: 5, Family: tfsax.FamilyConstant,
		Branch: "exact", Result: tfsax.RetrievalResult{Complete: true}}}
	var output bytes.Buffer
	if err := tfsax.WriteCSVStream(&output, obs); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(output.Bytes(), []byte("5,constant-step,exact")) {
		t.Fatal("writer lost ordered observation fields")
	}
}

func TestEvaluateGate_CompletelyNamedMetadataStillBlocked(t *testing.T) {
	definition := tfsax.Definition{Schema: "lvlath-tfsax-definition-v1", Name: "Explicit new proposal",
		Source: "authored-not-peer-reviewed", Domain: "scalar", TimeGrid: "uniform",
		Normalization: "frozen-train", Segmentation: "explicit", Alphabet: "explicit",
		Boundaries: "explicit", OutputShape: "explicit", MissingPolicy: "finite-only",
		Score: "explicit", Ordering: "distance-then-ID", PruningProof: "none",
		Ownership: "detached", Approved: true}
	if err := tfsax.ValidateDefinition(definition); err != nil {
		t.Fatal(err)
	}
	noData := tfsax.EvaluateGate(definition, nil)
	if noData.Passed || noData.Reason != "no observations" {
		t.Fatalf("empty evidence was promoted: %+v", noData)
	}
	withData := tfsax.EvaluateGate(definition, []tfsax.Observation{{Branch: "exact"}})
	if withData.Passed || withData.CompleteComparison || withData.Reason != "four-way independent certification not completed" {
		t.Fatalf("unreviewed data promoted: %+v", withData)
	}
}
