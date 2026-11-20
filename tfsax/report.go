// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax exports primary benchmark observations and explicit research-gate refusals.
// This file never turns an incomplete comparison into certified production evidence.
package tfsax

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
)

const researchContractVersion = "lvlath-spec-29-v1" // Version governing admission evidence, not a production encoder.

// WriteCSV exports raw query/branch measurements with explicit NA and proof columns.
// The completed file is atomically replaced only after the CSV writer has succeeded.
//
// Implementation:
//   - Stage 1: serialize fixed-schema headers and individual observations in given order.
//   - Stage 2: check buffered CSV errors before publishing bytes.
//   - Stage 3: use a same-directory atomic rename of a mode-0600 temporary file.
//
// Behavior highlights:
//   - Recall and missed-best applicability are separate booleans; NA remains distinguishable.
//
// Inputs:
//   - path: destination file in an existing writable parent directory.
//   - observations: read-only complete raw measurements, no reordering or aggregation.
//
// Returns:
//   - nil after a completed file replacement, error otherwise; no partial target overwrite.
//
// Errors:
//   - Original csv/file errors; ErrInvalidOption for empty output path.
//
// Determinism:
//   - Row order is exactly the input observation order; float output uses 17 digits.
//
// Complexity:
//   - O(number of observations * (TopK+1)) time for record processing;
//   - O(total serialized output bytes) additional buffering for atomic publication.
//
// Notes:
//   - Search timings have inherent runtime variance; do not hash raw CSV as semantic identity.
//
// AI-Hints:
//   - Never omit applicability columns: a zero recall with no reachable target is not a measurement.
func WriteCSV(path string, observations []Observation) error {
	var buffer bytes.Buffer
	if err := WriteCSVStream(&buffer, observations); err != nil {
		return err
	}
	return writeAtomically(path, buffer.Bytes())
}

// WriteCSVStream writes canonical observation rows to a caller-provided byte stream.
// It is suitable for an audit pipeline that owns the destination and its lifecycle.
//
// Implementation:
//   - Stage 1: reject a nil destination and write the fixed report column header.
//   - Stage 2: serialize all rows in input order without inventing missing metrics.
//   - Stage 3: flush CSV buffers and return any underlying io.Writer failure.
//
// Behavior highlights:
//   - Applicability columns make zero scores distinguishable from absent denominators.
//
// Inputs:
//   - destination: caller-owned non-nil io.Writer; never closed by the function.
//   - observations: read-only raw query/branch measurements; no mutation or reordering.
//
// Returns:
//   - nil after a successful CSV flush, or the original writer error.
//
// Errors:
//   - ErrInvalidOption for nil writer; original io.Writer/csv errors otherwise.
//
// Determinism:
//   - Column and row ordering are fixed; no wall-clock metadata is synthesized.
//
// Complexity:
//   - Time O(number of observations * fixed report fields),
//   - Space O(1) internal buffered CSV storage excluding caller-owned writer buffers.
//
// Notes:
//   - Streaming cannot roll back a partially written caller-owned destination on error.
//
// AI-Hints:
//   - Prefer WriteCSV for atomic local files; prefer WriteCSVStream for controlled audit streams.
func WriteCSVStream(destination io.Writer, observations []Observation) error {
	if destination == nil {
		return ErrInvalidOption
	}
	writer := csv.NewWriter(destination)
	header := []string{
		"query_id", "family", "branch", "complete", "exact", "representation_version",
		"examined", "pruned", "available", "shortlist", "recall_applicable", "recall",
		"missed_best_applicable", "missed_best", "duration_ns", "index_bytes",
	}
	if err := writer.Write(header); err != nil {
		return err
	}
	for _, observation := range observations {
		row := []string{
			strconv.FormatUint(observation.QueryID, 10), string(observation.Family), observation.Branch,
			strconv.FormatBool(observation.Result.Complete), strconv.FormatBool(observation.Result.Diagnostics.Exact),
			observation.Result.Diagnostics.RepresentationVersion,
			strconv.Itoa(observation.Result.Diagnostics.Examined), strconv.Itoa(observation.Result.Diagnostics.Pruned),
			strconv.Itoa(observation.Available), strconv.Itoa(observation.Shortlist),
			strconv.FormatBool(observation.Recall.Applicable),
			strconv.FormatFloat(observation.Recall.Value, 'g', 17, 64),
			strconv.FormatBool(observation.MissedBest.Applicable),
			strconv.FormatFloat(observation.MissedBest.Value, 'g', 17, 64),
			strconv.FormatInt(observation.DurationNS, 10), strconv.FormatUint(observation.IndexBytes, 10),
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return err
	}
	return nil
}

// GateReport records the decision to admit or refuse a proposed symbolic representation.
// The current gate cannot approve production because TFSAX definition and proof are absent.
//
// Implementation:
//   - EvaluateGate constructs a negative decision from definition validity and available evidence; no production promotion is inferred.
//
// Behavior highlights:
//   - Passed=false is a concluded admission refusal, not proof that all future proposals fail.
//   - CompleteComparison requires actual independent measurements from all four branches.
//
// Inputs:
//   - Produced by EvaluateGate or supplied as a negative record to WriteNegativeReport.
//
// Returns:
//   - Deterministic metadata fields excluding wall-clock measurements.
//
// Errors:
//   - EvaluateGate represents refusal in Reason; WriteNegativeReport rejects Passed=true.
//
// Determinism:
//   - Reason strings are fixed by verified gate categories.
//
// Complexity:
//   - O(1) carrier storage plus reason text.
//
// Notes:
//   - A false Passed value does not by itself diagnose poor accuracy: the reason distinguishes missing evidence.
//
// AI-Hints:
//   - Never set Passed based on user-modifiable Approved alone.
type GateReport struct {
	ContractVersion    string `json:"contract_version"`    // Research contract governing admission.
	DefinitionApproved bool   `json:"definition_approved"` // Declared metadata state, not independent review proof.
	CompleteComparison bool   `json:"complete_comparison"` // Whether all required branches were independently verified.
	Passed             bool   `json:"passed"`              // Production admission decision; always false in this stage.
	Reason             string `json:"reason"`              // Fixed human-readable refusal category.
}

// EvaluateGate refuses production promotion in the absence of validated four-way evidence.
// It never infers a mathematical TFSAX implementation from observation names or metadata.
//
// Implementation:
//   - Stage 1: validate scientific definition metadata and its Approved claim.
//   - Stage 2: distinguish absent measurements from unavailable independent certification.
//   - Stage 3: publish an explicit negative decision without inventing outcome metrics.
//
// Behavior highlights:
//   - The current implementation always refuses; metadata and observation count only select an honest reason.
//
// Inputs:
//   - definition: proposed metadata, read-only.
//   - observations: recorded measurements, not sufficient to authenticate a missing encoder.
//
// Returns:
//   - GateReport with Passed=false and a stable refusal reason.
//
// Errors:
//   - None; invalid definition is reported rather than exposed as a partial valid admission.
//
// Determinism:
//   - Decision depends on exact metadata and whether evidence exists, not timing.
//
// Complexity:
//   - O(1) time/space; content is not interpreted as an unverified scientific proof.
//
// Notes:
//   - A future approved implementation requires an independent versioned gate procedure.
//
// AI-Hints:
//   - Do not add an unconditional success branch based on four arbitrary Branch strings.
func EvaluateGate(definition Definition, observations []Observation) GateReport {
	report := GateReport{ContractVersion: researchContractVersion, DefinitionApproved: definition.Approved}
	if err := ValidateDefinition(definition); err != nil {
		report.Reason = "definition metadata is invalid"
		return report
	}
	if !definition.Approved {
		report.Reason = "definition is unresolved"
		return report
	}
	if len(observations) == 0 {
		report.Reason = "no observations"
		return report
	}
	report.Reason = "four-way independent certification not completed"
	return report
}

// WriteNegativeReport publishes the gate refusal as an auditable Markdown artifact.
// A caller cannot use this function to publish an admitted production decision.
//
// Implementation:
//   - Stage 1: reject a Passed=true input rather than mislabel it as negative.
//   - Stage 2: format version, definition, comparison state and stable reason.
//   - Stage 3: atomically replace the requested report file.
//
// Behavior highlights:
//   - An existing negative result is exported as markdown; it is never reclassified as a successful research gate.
//
// Inputs:
//   - path: writable target filename in an existing directory.
//   - report: read-only negative gate decision; no timing is fabricated.
//
// Returns:
//   - nil after successful publication, or classified/filesystem error.
//
// Errors:
//   - ErrInvalidOption for Passed=true or empty path, original filesystem errors otherwise.
//
// Determinism:
//   - Report body is stable for an identical GateReport; no wall-clock values are added.
//
// Complexity:
//   - Time/space O(output bytes) for atomic publication.
//
// Notes:
//   - A negative gate means not admitted with current evidence, not universally disproven.
//
// AI-Hints:
//   - Keep empirical non-superiority separate from missing-definition refusal reasons.
func WriteNegativeReport(path string, report GateReport) error {
	if report.Passed {
		return ErrInvalidOption
	}
	content := fmt.Sprintf("# TFSAX research gate\n\nContract: %s\n\nDefinition approved: %t\n\nFour-way complete: %t\n\nDecision: NOT ADMITTED\n\nReason: %s\n\nThis decision is not evidence that every possible TFSAX formulation fails.\n",
		report.ContractVersion, report.DefinitionApproved, report.CompleteComparison, report.Reason)
	return writeAtomically(path, []byte(content))
}
