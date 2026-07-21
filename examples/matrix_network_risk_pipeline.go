// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package main

import (
	"fmt"
	"math"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

const (
	// ---- Generic tolerances: examples should be stable, not fragile. ----
	constRTol = 1e-10
	constATol = 1e-12

	// ---- Jacobi eigen controls: deterministic spectral decomposition cap. ----
	constEigenTol     = 1e-10
	constEigenMaxIter = 200
)

// ExampleMatrix_networkRiskPipeline demonstrates an end-to-end matrix workflow
// for service-network risk analytics.
//
// Scenario:
//
//	A platform team operates a payment/search backend with a small but realistic
//	service graph:
//
//	  edge   -> auth/profile/search/db/backup flow through directed calls;
//	  api    -> search and a self-loop representing internal retry pressure;
//	  db     -> backup contains a real zero-weight replication link;
//	  backup -> edge models emergency traffic re-entry during failover.
//
//	The incident-response problem is not just "find a path". The team needs one
//	reproducible pipeline that combines:
//
//	  - graph topology;
//	  - zero-weight edge semantics;
//	  - shortest failover distances;
//	  - edge-column structure;
//	  - dirty telemetry ingestion;
//	  - statistical normalization;
//	  - correlation/spectral analysis;
//	  - small dense risk system solving.
//
//	This is the kind of workflow an infrastructure architect or reliability team
//	can face when building deterministic post-incident analysis, capacity planning,
//	or failover simulation tooling.
//
// Playground: https://go.dev/play/p/pMqxpUoHlVx
//
// What this example proves:
//
//  1. Graph semantics survive matrix conversion.
//     A finite zero in adjacency can mean "real free replication edge", not absence.
//
//  2. Distance semantics are explicit.
//     Metric closure is not exported back as topology. It is a distance artifact.
//
//  3. Dirty telemetry must be sanitized before statistics.
//     NaN and +Inf are ingested only through an explicit raw-policy matrix and then
//     converted into finite values before covariance/correlation.
//
//  4. Linear algebra is used as a verification layer.
//     Inverse, LU, QR, MatVec, Product, and AllClose are not decorative calls:
//     they form consistency checks around the risk system.
//
// Complexity:
//
//	Let V be service count, E edge count, S telemetry samples, and F telemetry features.
//
//	  - Adjacency / incidence build: O(V + E) plus dense matrix storage.
//	  - Metric closure: O(V^3).
//	  - Centering / normalization: O(S·F).
//	  - Covariance / correlation: O(S·F + S·F^2).
//	  - Dense inverse / LU / QR / eigen: O(F^3) for this risk feature system.
func ExampleMatrix_networkRiskPipeline() {
	// NewGraph creates the source service topology. It is directed because service
	// calls have direction, weighted because communication costs matter, and loop
	// capable because "api -> api" models internal retry/self-pressure.
	g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted(), core.WithLoops())
	if err != nil {
		fmt.Println(err)
		return
	}

	// Static example data: all edges are compatible with the graph capabilities.
	// Ignoring AddEdge errors is acceptable only in this deterministic example.
	// Production code must check each graph mutation error.
	_, _ = g.AddEdge("edge", "auth", 2)
	_, _ = g.AddEdge("edge", "api", 3)
	_, _ = g.AddEdge("auth", "profile", 1)
	_, _ = g.AddEdge("api", "search", 4)
	_, _ = g.AddEdge("profile", "db", 2)
	_, _ = g.AddEdge("search", "db", 1)
	_, _ = g.AddEdge("db", "backup", 0) // real zero-weight replication link
	_, _ = g.AddEdge("backup", "edge", 5)
	_, _ = g.AddEdge("api", "api", 0) // loop; allowed by graph and matrix policy

	// NewMatrixOptions freezes how graph semantics are projected into matrices:
	// directed cells, preserved weights, and retained loops.
	graphOpts, err := matrix.NewMatrixOptions(
		matrix.WithDirected(),
		matrix.WithWeighted(),
		matrix.WithAllowLoops(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	// NewAdjacencyMatrix creates the dense service-call operator and deterministic
	// VertexIndex mapping from service IDs to matrix coordinates.
	am, err := matrix.NewAdjacencyMatrix(g, graphOpts)
	if err != nil {
		fmt.Println(err)
		return
	}

	// DegreeVector gives a compact local topology signal. For directed adjacency
	// this is the outgoing row mass under the selected matrix policy.
	degree, err := am.DegreeVector()
	if err != nil {
		fmt.Println(err)
		return
	}

	// BuildMetricClosure computes all-pairs shortest service-call distances. This
	// is the failover/routing view: +Inf for unreachable pairs, zero diagonal, and
	// Floyd-Warshall closure under the hood.
	closure, err := matrix.BuildMetricClosure(g, graphOpts)
	if err != nil {
		fmt.Println(err)
		return
	}

	// NewIncidenceMatrix builds the edge-column topology representation. Unlike
	// adjacency, incidence preserves one column per effective edge and is useful
	// for conservation-style or structural edge participation analysis.
	inc, err := matrix.NewIncidenceMatrix(g, graphOpts)
	if err != nil {
		fmt.Println(err)
		return
	}

	edgeIdx := am.VertexIndex["edge"]
	dbIdx := am.VertexIndex["db"]
	backupIdx := am.VertexIndex["backup"]

	// At reads the shortest closed distance edge -> db from the metric closure.
	// This answers: "How far is the database from the ingress edge under service calls?"
	distEdgeDB, err := closure.Mat.At(edgeIdx, dbIdx)
	if err != nil {
		fmt.Println(err)
		return
	}

	// At reads the raw adjacency cell db -> backup. A finite zero here is not
	// absence; it is the real zero-weight replication link.
	zeroReplication, err := am.Mat.At(dbIdx, backupIdx)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("vertices=%d\n", len(am.VertexIndex))
	fmt.Printf("incidenceEdges=%d\n", len(inc.Edges))
	fmt.Printf("degree(edge)=%.0f\n", degree[edgeIdx])
	fmt.Printf("distance(edge,db)=%.0f\n", distEdgeDB)
	fmt.Printf("zeroReplicationWeight=%.0f\n", zeroReplication)

	// NewPreparedDense allocates a raw-ingestion matrix with NaN/Inf validation
	// disabled. This is intentional: external telemetry can be dirty, but that
	// dirt must be contained and sanitized before statistics.
	raw, err := matrix.NewPreparedDense(6, 4, matrix.WithNoValidateNaNInf())
	if err != nil {
		fmt.Println(err)
		return
	}

	// Fill loads deterministic row-major service telemetry:
	// latency, saturation, errors, retry pressure.
	if err = raw.Fill([]float64{
		12, 0.20, 1, 3,
		14, 0.25, 1, 4,
		18, 0.40, 2, 5,
		21, 0.55, 3, 8,
		math.NaN(), 0.65, 4, 13,
		30, math.Inf(1), 5, 21,
	}); err != nil {
		fmt.Println(err)
		return
	}

	// ReplaceInfNaN converts dirty telemetry into finite values. This prevents
	// undefined numeric propagation in covariance, correlation, inverse, LU, and QR.
	clean, err := matrix.ReplaceInfNaN(raw, 0)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Clip enforces a bounded feature domain. Here it is a simple risk-engineering
	// guardrail: after sanitation, values must stay inside [0,100].
	clipped, err := matrix.Clip(clean, 0, 100)
	if err != nil {
		fmt.Println(err)
		return
	}

	// CenterColumns removes per-feature baselines. This separates service-behavior
	// movement from absolute feature scale.
	centered, means, err := matrix.CenterColumns(clipped)
	if err != nil {
		fmt.Println(err)
		return
	}

	// NormalizeRowsL2 converts each observation into a comparable direction vector.
	// Degenerate rows remain stable by package policy.
	unitRows, norms, err := matrix.NormalizeRowsL2(centered)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Correlation builds the Pearson feature-correlation matrix. Degenerate columns
	// are handled explicitly by the matrix package instead of silently producing NaN.
	corr, _, _, err := matrix.Correlation(clipped)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Symmetrize repairs tiny asymmetry drift and makes the matrix suitable for
	// symmetric spectral analysis.
	corrSym, err := matrix.Symmetrize(corr)
	if err != nil {
		fmt.Println(err)
		return
	}

	// T computes the transpose used to verify the symmetry contract.
	corrT, err := matrix.T(corrSym)
	if err != nil {
		fmt.Println(err)
		return
	}

	// AllClose checks Corr ≈ Corrᵀ using explicit tolerance instead of brittle
	// direct equality.
	okCorrSym, err := matrix.AllClose(corrSym, corrT, constRTol, constATol)
	if err != nil {
		fmt.Println(err)
		return
	}

	// EigenSym extracts deterministic spectral information from the correlation
	// operator. In a production pipeline, this can drive clustering, anomaly
	// detection, or factor compression.
	_, _, err = matrix.EigenSym(corrSym, constEigenTol, constEigenMaxIter)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Covariance over normalized rows produces a compact feature-risk system.
	cov, _, err := matrix.Covariance(unitRows)
	if err != nil {
		fmt.Println(err)
		return
	}

	// IdentityLike creates I with the same square dimension as covariance. This is
	// used for ridge stabilization.
	I, err := matrix.IdentityLike(cov)
	if err != nil {
		fmt.Println(err)
		return
	}

	// ScaleBy computes λI. The small ridge term protects the dense inverse/solve
	// workflow from near-singular covariance.
	ridgeI, err := matrix.ScaleBy(I, 1e-3)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Sum forms the regularized system A = Cov + λI.
	system, err := matrix.Sum(cov, ridgeI)
	if err != nil {
		fmt.Println(err)
		return
	}

	// InverseOf computes A⁻¹. The example does not trust it blindly: the result is
	// immediately checked by multiplying back against A.
	inv, err := matrix.InverseOf(system)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Product computes A*A⁻¹.
	prod, err := matrix.Product(system, inv)
	if err != nil {
		fmt.Println(err)
		return
	}

	// AllClose verifies A*A⁻¹ ≈ I and publishes a compact sanity signal.
	okInverse, err := matrix.AllClose(prod, I, constRTol, 1e-8)
	if err != nil {
		fmt.Println(err)
		return
	}

	// LUDecompose gives a deterministic no-pivot factorization. It is useful when
	// reproducibility and auditability matter more than adaptive pivot behavior.
	L, U, err := matrix.LUDecompose(system)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Product reconstructs L*U so the LU contract is verified explicitly.
	LU, err := matrix.Product(L, U)
	if err != nil {
		fmt.Println(err)
		return
	}

	// AllClose verifies the LU reconstruction against the original system.
	okLU, err := matrix.AllClose(LU, system, constRTol, 1e-8)
	if err != nil {
		fmt.Println(err)
		return
	}

	// QRDecompose gives the Householder QR factorization under the package contract
	// A ≈ Qᵀ*R.
	Q, R, err := matrix.QRDecompose(system)
	if err != nil {
		fmt.Println(err)
		return
	}

	// T computes Qᵀ for the documented reconstruction shape.
	QT, err := matrix.T(Q)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Product reconstructs Qᵀ*R.
	QTR, err := matrix.Product(QT, R)
	if err != nil {
		fmt.Println(err)
		return
	}

	// AllClose verifies the QR reconstruction. This protects the example from
	// demonstrating a factorization without checking its mathematical contract.
	okQR, err := matrix.AllClose(QTR, system, constRTol, 1e-8)
	if err != nil {
		fmt.Println(err)
		return
	}

	// MatVecMul applies the final risk system to a deterministic service-risk
	// weighting vector. The output is not interpreted here; its length proves the
	// system is shaped and consumable.
	score, err := matrix.MatVecMul(system, []float64{1, 0.5, 0.25, 0.125})
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("meansLen=%d\n", len(means))
	fmt.Printf("normsLen=%d\n", len(norms))
	fmt.Printf("okCorrelationSymmetric=%v\n", okCorrSym)
	fmt.Printf("okInverse=%v\n", okInverse)
	fmt.Printf("okLU=%v\n", okLU)
	fmt.Printf("okQR=%v\n", okQR)
	fmt.Printf("scoreLen=%d\n", len(score))

	// Output:
	// vertices=7
	// incidenceEdges=9
	// degree(edge)=2
	// distance(edge,db)=4
	// zeroReplicationWeight=0
	// meansLen=4
	// normsLen=6
	// okCorrelationSymmetric=true
	// okInverse=true
	// okLU=true
	// okQR=true
	// scoreLen=4
}
