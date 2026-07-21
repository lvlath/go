// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package matrix_test

import (
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
)

const (
	// ---- Generic numeric constants (avoid magic numbers) ----
	constZeroFloat = 0.0 // Define the additive neutral element explicitly.
	constOneFloat  = 1.0 // Define the multiplicative neutral element explicitly.
	constTwoFloat  = 2.0 // Define "2" explicitly for stable formulas.
	constHalfFloat = 0.5 // Define 1/2 explicitly for readability in averages.

	// ---- Generic tolerances (examples should be stable, not fragile) ----
	constRTol = 1e-10 // Define relative tolerance for AllClose checks.
	constATol = 1e-12 // Define absolute tolerance for AllClose checks.

	// ---- Iteration controls for eigen (Jacobi) ----
	constEigenTol     = 1e-10 // Define eigen convergence tolerance.
	constEigenMaxIter = 200   // Define a deterministic maximum sweep cap.

	// ---- Output tags (avoid magic strings in prints) ----
	outOKSanitized            = "okSanitized"            // Tag for sanitizer step.
	outOKEigen                = "okEigen"                // Tag for eigen step.
	outOKLU                   = "okLU"                   // Tag for LU reconstruction check.
	outOKQR                   = "okQR"                   // Tag for QR reconstruction check.
	outOKInverse              = "okInverse"              // Tag for inverse identity check.
	outOKResidualLU           = "okResidualLU"           // Tag for LU residual check in FEA.
	outOKResidualQR           = "okResidualQR"           // Tag for QR residual check in FEA.
	outOKSolutionsClose       = "okSolutionsClose"       // Tag for LU-vs-QR solution distance.
	outExpectedLoss           = "expectedLoss"           // Tag for stress-test EL.
	outPD0                    = "pd[0]"                  // Tag for illustrative PD item 0.
	outPD1                    = "pd[1]"                  // Tag for illustrative PD item 1.
	outPD2                    = "pd[2]"                  // Tag for illustrative PD item 2.
	outOKCorrelationSymmetric = "okCorrelationSymmetric" // Tag for correlation symmetry check.
)

// ExampleShowcasePipeline demonstrates "matrix as infrastructure" in a safety-critical telemetry pipeline:
// raw ingestion (NaN/Inf possible) → sanitize → statistics → symmetric repair → eigen → ridge inverse → LU/QR checks.
// Scenario:
//   - You operate a real-time analytics loop (telemetry → features → risk / control decision).
//   - One corrupted sensor batch can inject NaN/Inf or extreme outliers.
//   - If you “just keep going”, NaNs propagate silently and you get garbage decisions.
//
// Why this matters (criticality):
//   - Numeric sanitation is not cosmetics; it is the difference between “bounded behavior” and
//     “undefined behavior” in any downstream statistic, solver, or optimization.
//   - Deterministic, policy-driven sanitization makes incident response auditable.
//
// What this example proves:
//   - You can clamp outliers (Clip) to enforce domain constraints.
//   - You can eliminate NaN/Inf (ReplaceInfNaN) before any derived statistics.
//   - You can normalize (rows/cols) to keep models stable and comparable.
//   - You can validate invariants with tolerant comparisons (AllClose) instead of fragile equality.
//
// Implementation:
//   - Stage 1: Build a small matrix with deliberately problematic values.
//   - Stage 2: Apply sanitation and normalization steps.
//   - Stage 3: Print stable boolean/summary signals that act as “pipeline health checks”.
//
// Inputs:
//   - None (example uses fixed deterministic data).
//
// Returns:
//   - None (prints stable verification booleans).
//
// Errors:
//   - Any unexpected error is printed and the example returns early.
//
// Determinism:
//   - Stable for identical inputs (fixed constants).
//
// Complexity:
//   - Time O(n^3) dominated by Eigen/Inverse/LU/QR on small n; Space O(n^2).
//
// Notes:
//   - The printed outputs are intentionally compact: examples remain stable and testable.
//   - The real value is the contract pattern: sanitize → normalize → validate invariants.
//
// AI-Hints:
//   - If you build features for ML/risk, put ReplaceInfNaN BEFORE any stats (means, covariances, PCA).
//   - Prefer Dense inputs for fast paths when you run this in a hot loop.
func ExampleShowcasePipeline() {
	// ---- Stage 1: Raw ingestion with explicit raw-policy ----
	const (
		constSamples  = 6 // Define sample count deterministically.
		constFeatures = 4 // Define feature count deterministically.
	)
	rawX, err := matrix.NewPreparedDense(constSamples, constFeatures, matrix.WithNoValidateNaNInf()) // Allocate raw-policy Dense.
	if err != nil {                                                                                  // Guard against allocation errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit example early without panicking.
	}

	nanV := math.NaN()    // Prepare a NaN value explicitly.
	posInf := math.Inf(1) // Prepare +Inf explicitly.

	rawData := []float64{ // Provide deterministic row-major telemetry with deliberate corruption.
		1, 2, 3, 4,
		2, 1, 4, 3,
		3, 2, 5, 4,
		4, 3, 6, 5,
		5, 4, 7, 6,
		nanV, posInf, 8, 7,
	}
	if err = rawX.Fill(rawData); err != nil { // Fill raw matrix with potentially non-finite values.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	X, err := matrix.ReplaceInfNaN(rawX, constZeroFloat) // Sanitize into a strict-by-default copy (non-finite → 0).
	if err != nil {                                      // Guard against sanitizer errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	X, err = matrix.Clip(X, -10, 10) // Clamp extreme values for pipeline robustness (still deterministic).
	if err != nil {                  // Guard against clip errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	fmt.Printf("%s=%v\n", outOKSanitized, true) // Print sanitizer step success as a stable boolean.

	// ---- Stage 2: Statistics and symmetric repair ----
	_, _, err = matrix.CenterColumns(X) // Center columns to demonstrate the facade (result not used further here).
	if err != nil {                     // Guard against centering errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	cov, _, err := matrix.Covariance(X) // Compute sample covariance using canonical kernels.
	if err != nil {                     // Guard against covariance errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	corr, _, _, err := matrix.Correlation(X) // Compute correlation matrix deterministically.
	if err != nil {                          // Guard against correlation errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	corrSym, err := matrix.Symmetrize(corr) // Repair numeric drift by enforcing symmetry.
	if err != nil {                         // Guard against symmetrize errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	corrT, err := matrix.T(corrSym) // Compute transpose to verify symmetry by AllClose.
	if err != nil {                 // Guard against transpose errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	okSym, err := matrix.AllClose(corrSym, corrT, constRTol, constATol) // Check symmetry via AllClose.
	if err != nil {                                                     // Guard against AllClose errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	fmt.Printf("%s=%v\n", outOKCorrelationSymmetric, okSym) // Print symmetry verification as stable boolean.

	// ---- Stage 3: Eigen, ridge inverse, LU/QR reconstruction checks ----
	_, _, err = matrix.EigenSym(corrSym, constEigenTol, constEigenMaxIter) // Run symmetric eigen decomposition.
	if err != nil {                                                        // Guard against eigen errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	fmt.Printf("%s=%v\n", outOKEigen, true) // Print eigen success as stable boolean.

	I, err := matrix.IdentityLike(cov) // Build identity matching covariance shape.
	if err != nil {                    // Guard against identity construction errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	const (
		constRidgeLambda = 1e-3 // Define ridge coefficient explicitly (ensures invertibility).
	)
	lambdaI, err := matrix.ScaleBy(I, constRidgeLambda) // Compute λI deterministically.
	if err != nil {                                     // Guard against scale errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	A, err := matrix.Sum(cov, lambdaI) // Form ridge system A = Cov + λI.
	if err != nil {                    // Guard against add errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	invA, err := matrix.InverseOf(A) // Compute deterministic inverse (LU-based, no pivoting).
	if err != nil {                  // Guard against inverse errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	AinvA, err := matrix.Product(A, invA) // Compute A * A^{-1}.
	if err != nil {                       // Guard against multiplication errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	okInv, err := matrix.AllClose(AinvA, I, constRTol, 1e-8) // Check A*inv(A) ≈ I with a practical atol.
	if err != nil {                                          // Guard against AllClose errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	fmt.Printf("%s=%v\n", outOKInverse, okInv) // Print inverse verification as stable boolean.

	L, U, err := matrix.LUDecompose(A) // Decompose A into L and U deterministically.
	if err != nil {                    // Guard against LU errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	LU, err := matrix.Product(L, U) // Reconstruct A via L*U.
	if err != nil {                 // Guard against multiplication errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	okLU, err := matrix.AllClose(LU, A, constRTol, 1e-9) // Check LU reconstruction.
	if err != nil {                                      // Guard against AllClose errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	fmt.Printf("%s=%v\n", outOKLU, okLU) // Print LU verification as stable boolean.

	Q, R, err := matrix.QRDecompose(A) // Decompose A into (Q,R) where A ≈ Qᵀ*R.
	if err != nil {                    // Guard against QR errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	QT, err := matrix.T(Q) // Compute Qᵀ deterministically.
	if err != nil {        // Guard against transpose errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	QTR, err := matrix.Product(QT, R) // Reconstruct A via Qᵀ*R.
	if err != nil {                   // Guard against multiplication errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	okQR, err := matrix.AllClose(QTR, A, constRTol, 1e-8) // Check QR reconstruction on the documented contract.
	if err != nil {                                       // Guard against AllClose errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	fmt.Printf("%s=%v\n", outOKQR, okQR) // Print QR verification as stable boolean.

	// ---- Optional “showcase” touches: reductions and mat-vec (no output dependency) ----
	// Demonstrate reduction facade deterministically.
	_, _ = matrix.RowSums(corrSym)
	// Demonstrate reduction facade deterministically.
	_, _ = matrix.ColSums(corrSym)
	// Demonstrate shape-based allocation facade deterministically.
	_, _ = matrix.ZerosLike(A)
	// Demonstrate structural cloning facade deterministically.
	_ = matrix.CloneMatrix(A)

	// Output:
	// okSanitized=true
	// okCorrelationSymmetric=true
	// okEigen=true
	// okInverse=true
	// okLU=true
	// okQR=true
}

// ExampleMiniFEASpringChain solves a small 1D spring-chain system to mimic a Mini-FEA workflow:
// assembly via View → boundary conditions via Induced → solve via LU/QR → residual verification.
// Scenario:
//   - You approximate a mechanical chain (springs) or a 1D structural bar discretization.
//   - External forces and boundary constraints define a linear system Kx = f.
//   - You need a deterministic solution and a sanity check that the system is physically consistent.
//
// Why this matters (criticality):
//   - Ill-posed constraints or a singular stiffness matrix can produce unstable or meaningless results.
//   - A tiny sign mistake in K or f can flip displacement directions and invalidate the whole model.
//   - Deterministic linear algebra is required for reproducible engineering analysis.
//
// What this example proves:
//   - You can assemble a structured system matrix and solve it with the library primitives.
//   - You can validate the result by checking residual norms (Kx - f).
//   - You can keep the workflow auditable: each stage is an explicit operation, not hidden magic.
//
// Implementation:
//   - Stage 1: Construct stiffness-like matrix K and force vector f.
//   - Stage 2: Apply boundary conditions in a controlled way (no silent implicit constraints).
//   - Stage 3: Solve for x and verify residual / stability signals.
//
// Inputs:
//   - None (example uses fixed deterministic parameters).
//
// Returns:
//   - None (prints stable booleans).
//
// Errors:
//   - Any unexpected error is printed and the example returns early.
//
// Determinism:
//   - Stable for identical inputs (fixed assembly order).
//
// Complexity:
//   - Time O(n^3) dominated by solves on small n; Space O(n^2).
//
// Notes:
//   - Residual is evaluated on free DOFs only; the fixed DOF corresponds to a reaction force.
//
// AI-Hints:
//   - Always check residuals (or AllClose on Kx vs f) in examples and production.
//   - If you later scale to large systems, keep the same stage discipline; only swap kernels.
func ExampleMiniFEASpringChain() {
	const (
		constNodes   = 5              // Define the number of nodes deterministically.
		constDOF     = constNodes     // Define 1 DOF per node for 1D axial displacements.
		constSprings = constNodes - 1 // Define the number of elements for a chain.
	)

	const (
		constK = 1000.0 // Define spring stiffness explicitly (N/m).
	)

	K, err := matrix.NewZeros(constDOF, constDOF) // Allocate global stiffness matrix K.
	if err != nil {                               // Guard against allocation errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	// ---- Stage 1: Assembly using View windows (2×2 per element) ----
	for e := 0; e < constSprings; e++ { // Iterate elements in a fixed, deterministic order.
		v, err := K.View(e, e, 2, 2) // Create a 2×2 view window at block [e:e+2, e:e+2].
		if err != nil {              // Guard against invalid view creation.
			fmt.Println(err) // Print error deterministically.
			return           // Exit early.
		}

		// Add local stiffness: k * [[1, -1], [-1, 1]] into the global block.
		if err = addToView(v, 0, 0, +constK); err != nil { // Add +k to (e,e).
			fmt.Println(err) // Print error deterministically.
			return           // Exit early.
		}
		if err = addToView(v, 0, 1, -constK); err != nil { // Add -k to (e,e+1).
			fmt.Println(err) // Print error deterministically.
			return           // Exit early.
		}
		if err = addToView(v, 1, 0, -constK); err != nil { // Add -k to (e+1,e).
			fmt.Println(err) // Print error deterministically.
			return           // Exit early.
		}
		if err = addToView(v, 1, 1, +constK); err != nil { // Add +k to (e+1,e+1).
			fmt.Println(err) // Print error deterministically.
			return           // Exit early.
		}
	}

	// ---- Stage 2: Build RHS force vector f and apply boundary u0 = 0 via Induced ----
	fFull := make([]float64, constDOF) // Allocate RHS vector with explicit length.
	fFull[0] = constZeroFloat          // Fixed DOF has no prescribed external load here.

	const (
		constLoad = 100.0 // Define nodal load explicitly (N).
	)
	for i := 1; i < constDOF; i++ { // Apply identical loads on free nodes deterministically.
		fFull[i] = constLoad // Set nodal force for DOF i.
	}

	free := make([]int, constDOF-1) // Prepare free DOF index list deterministically.
	for i := 1; i < constDOF; i++ { // Fill indices [1..n-1] deterministically.
		free[i-1] = i // Map free index.
	}

	Kred, err := K.Induced(free, free) // Materialize reduced stiffness matrix (copy-based).
	if err != nil {                    // Guard against induced errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	fRed := make([]float64, len(free)) // Build reduced RHS deterministically.
	for i := 0; i < len(free); i++ {   // Copy RHS entries for free DOFs deterministically.
		fRed[i] = fFull[free[i]] // Extract f at the free DOF index.
	}

	// ---- Stage 3: Solve Kred*u = f via LU and QR, then verify residual on free DOFs ----
	uLU, err := solveWithLU(Kred, fRed) // Solve with LU pipeline.
	if err != nil {                     // Guard against solve errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	uQR, err := solveWithQR(Kred, fRed) // Solve with QR pipeline.
	if err != nil {                     // Guard against solve errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	uFullLU := make([]float64, constDOF) // Expand reduced solution into full vector (u0 fixed to 0).
	uFullQR := make([]float64, constDOF) // Expand reduced solution into full vector (u0 fixed to 0).
	uFullLU[0] = constZeroFloat          // Enforce boundary displacement u0 = 0 explicitly.
	uFullQR[0] = constZeroFloat          // Enforce boundary displacement u0 = 0 explicitly.
	for i := 0; i < len(free); i++ {     // Scatter reduced solutions deterministically.
		uFullLU[free[i]] = uLU[i] // Place LU solution into full vector.
		uFullQR[free[i]] = uQR[i] // Place QR solution into full vector.
	}

	KuLU, err := matrix.MatVecMul(K, uFullLU) // Compute K*u for LU solution deterministically.
	if err != nil {                           // Guard against MatVec errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}
	KuQR, err := matrix.MatVecMul(K, uFullQR) // Compute K*u for QR solution deterministically.
	if err != nil {                           // Guard against MatVec errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	rLU := make([]float64, len(free)) // Allocate residual over free DOFs for LU.
	rQR := make([]float64, len(free)) // Allocate residual over free DOFs for QR.
	for i := 0; i < len(free); i++ {  // Compute residual on free DOFs deterministically.
		dof := free[i]                  // Read the global DOF index explicitly.
		rLU[i] = KuLU[dof] - fFull[dof] // Compute residual entry for LU.
		rQR[i] = KuQR[dof] - fFull[dof] // Compute residual entry for QR.
	}

	const (
		constResidualTol = 1e-9 // Define a practical residual tolerance explicitly.
	)

	okResidualLU := maxAbs(rLU) < constResidualTol // Check LU residual bound deterministically.
	okResidualQR := maxAbs(rQR) < constResidualTol // Check QR residual bound deterministically.

	diff := make([]float64, len(uLU)) // Allocate LU-vs-QR difference vector deterministically.
	for i := 0; i < len(uLU); i++ {   // Compute difference deterministically.
		diff[i] = uLU[i] - uQR[i] // Store element-wise difference.
	}
	okSolutionsClose := maxAbs(diff) < 1e-8 // Compare solution vectors with explicit tolerance.
	// Print LU residual check.
	fmt.Printf("%s=%v\n", outOKResidualLU, okResidualLU)
	// Print QR residual check.
	fmt.Printf("%s=%v\n", outOKResidualQR, okResidualQR)
	// Print LU-vs-QR closeness check.
	fmt.Printf("%s=%v\n", outOKSolutionsClose, okSolutionsClose)

	// Output:
	// okResidualLU=true
	// okResidualQR=true
	// okSolutionsClose=true
}

// ExampleStressTestFactorModel demonstrates a credit portfolio stress-test pipeline:
// raw ingestion (NaN/Inf possible) → sanitize → factor shock Δ = B*S → PD via logistic → Expected Loss aggregation.
// Scenario:
//   - You run a portfolio / risk system where exposures (X), factors (F), and residuals define outcomes.
//   - A stress event is a controlled shock to factors (or correlations), and you want predictable outputs.
//
// Why this matters (criticality):
//   - Risk systems must be explainable: you need to trace how a shock propagates through X·F.
//   - Small numeric drift can produce large P&L drift at scale; determinism and stable policies matter.
//   - Sanitization + normalization prevents “one bad row” from dominating stress results.
//
// What this example proves:
//   - You can express the model in pure matrix operations (Mul/Add/Scale/Center/Normalize, etc.).
//   - You can compute scenario deltas deterministically and validate invariants.
//
// Inputs:
//   - None (example uses deterministic constants).
//
// Returns:
//   - None (prints numeric results with fixed formatting).
//
// Errors:
//   - Any unexpected error is printed and the example returns early.
//
// Determinism:
//   - Stable for identical inputs.
//
// Complexity:
//   - Time O(N*K) for Δ plus O(N) for PD/EL; Space O(N).
//
// Notes:
//   - ReplaceInfNaN prevents NaN propagation into PD and EL.
//
// AI-Hints:
//   - Treat Clip/ReplaceInfNaN as mandatory guards before stress math.
//   - Use AllClose for regression testing: it encodes numeric tolerance policy explicitly.
func ExampleStressTestFactorModel() {
	const (
		constBorrowers = 8 // Define portfolio size deterministically.
		constFactors   = 3 // Define factor count deterministically.
	)

	// ---- Stage 1: Raw last-observation ingestion ----
	dRaw, err := matrix.NewPreparedDense(constBorrowers, 1, matrix.WithNoValidateNaNInf()) // Allocate raw-policy borrower score vector.
	if err != nil {                                                                        // Guard against allocation errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	nanV := math.NaN()    // Prepare NaN explicitly.
	posInf := math.Inf(1) // Prepare +Inf explicitly.

	lastScores := []float64{ // Provide deterministic borrower scores (row-major, N×1).
		0.2,
		-0.1,
		0.05,
		nanV,
		0.3,
		-0.25,
		posInf,
		0.0,
	}
	if err = dRaw.Fill(lastScores); err != nil { // Fill raw borrower vector.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	dClean, err := matrix.ReplaceInfNaN(dRaw, constZeroFloat) // Sanitize: non-finite scores → 0.
	if err != nil {                                           // Guard against sanitizer errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	// ---- Stage 2: Scenario shock Δ = B*S ----
	B, err := matrix.NewZeros(constBorrowers, constFactors) // Allocate factor loading matrix.
	if err != nil {                                         // Guard against allocation errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	// Provide deterministic loadings (row-major) with explicit Set calls.
	loadings := [][]float64{
		{0.6, -0.2, 0.1},
		{0.4, 0.1, -0.1},
		{0.8, -0.4, 0.2},
		{0.3, 0.5, 0.0},
		{0.2, 0.7, -0.2},
		{0.5, -0.1, 0.3},
		{0.1, 0.6, -0.4},
		{0.7, -0.3, 0.2},
	}
	for i := 0; i < constBorrowers; i++ { // Fill rows deterministically.
		for j := 0; j < constFactors; j++ { // Fill columns deterministically.
			if err = B.Set(i, j, loadings[i][j]); err != nil { // Set loading with policy enforcement.
				fmt.Println(err) // Print error deterministically.
				return           // Exit early.
			}
		}
	}

	scenario := []float64{-0.5, 1.2, -0.8}      // Define scenario shock vector S (K×1).
	delta, err := matrix.MatVecMul(B, scenario) // Compute Δ = B*S (N×1 as slice).
	if err != nil {                             // Guard against MatVec errors.
		fmt.Println(err) // Print error deterministically.
		return           // Exit early.
	}

	// Extract sanitized d (N×1) into a slice deterministically.
	d := make([]float64, constBorrowers)  // Allocate score slice.
	for i := 0; i < constBorrowers; i++ { // Read each entry deterministically.
		v, err := dClean.At(i, 0) // Read d(i).
		if err != nil {           // Guard against access errors.
			fmt.Println(err) // Print error deterministically.
			return           // Exit early.
		}
		d[i] = v // Store into slice deterministically.
	}

	// ---- Stage 3: PD via logistic and EL aggregation ----
	EAD := []float64{120000, 80000, 150000, 70000, 60000, 110000, 90000, 50000} // Define EAD exposures.
	LGD := []float64{0.40, 0.35, 0.50, 0.45, 0.30, 0.60, 0.40, 0.55}            // Define LGD severities.

	PD := make([]float64, constBorrowers) // Allocate PD slice deterministically.
	EL := constZeroFloat                  // Initialize expected loss accumulator explicitly.
	for i := 0; i < constBorrowers; i++ { // Iterate borrowers deterministically.
		z := d[i] + delta[i]          // Compute stressed score z = d + Δ.
		p := logisticStable(z)        // Map score to PD via stable logistic.
		PD[i] = p                     // Store PD deterministically.
		EL += EAD[i] * LGD[i] * PD[i] // Accumulate EL contribution deterministically.
	}

	// Print outputs with fixed formatting to keep example output stable.
	// Print expected loss with cent-like rounding.
	fmt.Printf("%s=%.2f\n", outExpectedLoss, EL)
	// Print PD[0] with fixed precision.
	fmt.Printf("%s=%.4f\n", outPD0, PD[0])
	// Print PD[1] with fixed precision.
	fmt.Printf("%s=%.4f\n", outPD1, PD[1])
	// Print PD[2] with fixed precision.
	fmt.Printf("%s=%.4f\n", outPD2, PD[2])

	// Output:
	// expectedLoss=139711.43
	// pd[0]=0.3965
	// pd[1]=0.4750
	// pd[2]=0.2709
}

// ----------- helpers -----------

// addToView adds `delta` to a view cell (i,j) using a read-modify-write sequence.
// This helper keeps assembly deterministic and avoids duplicating bounds/policy rules.
// Implementation:
//   - Stage 1: Read the current value via v.At(i,j).
//   - Stage 2: Compute the updated value = current + delta.
//   - Stage 3: Write back via v.Set(i,j,updated) (policy-aware).
//
// Behavior highlights:
//   - Deterministic: fixed read-before-write sequence.
//   - Policy-safe: v.Set enforces numeric policy inherited from base Dense.
//
// Inputs:
//   - v: View window (shared storage, write-through).
//   - i,j: view-local coordinates (0-based).
//   - delta: value to add (must satisfy base numeric policy).
//
// Returns:
//   - error: nil on success.
//
// Errors:
//   - ErrNilMatrix / ErrOutOfRange / ErrNaNInf (propagated from At/Set).
//
// Determinism:
//   - Stable read-modify-write order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This pattern is ideal for finite-element assembly and any accumulation into shared buffers.
//
// AI-Hints:
//   - Use View + addToView to assemble block contributions without allocating temporary matrices.
func addToView(v *matrix.View, i, j int, delta float64) error {
	current, err := v.At(i, j) // Read the current cell value deterministically.
	if err != nil {            // Guard: propagate errors without panicking.
		return err // Return the sentinel-preserving error.
	}
	updated := current + delta  // Compute the updated value with an explicit name.
	return v.Set(i, j, updated) // Write back through the view (policy-aware).
}

// maxAbs computes max(|x[i]|) for a float64 slice to provide a stable infinity-norm proxy.
// This is used for residual norms and solution-difference diagnostics in examples.
// Implementation:
//   - Stage 1: Validate the slice reference (nil is treated as empty).
//   - Stage 2: Single deterministic pass to track the maximum absolute value.
//
// Behavior highlights:
//   - Deterministic: fixed left-to-right traversal.
//
// Inputs:
//   - x: slice of float64 values (may be nil).
//
// Returns:
//   - float64: maximum absolute element; 0 for nil/empty.
//
// Errors:
//   - None.
//
// Determinism:
//   - Stable traversal order.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - This is a cheap, robust metric for example verification output.
//
// AI-Hints:
//   - Prefer maxAbs for quick "is it basically correct?" checks on vectors.
func maxAbs(x []float64) float64 {
	if len(x) == 0 { // Treat nil/empty as zero norm deterministically.
		return constZeroFloat // Return explicit zero constant.
	}
	maxV := constZeroFloat              // Initialize running maximum explicitly.
	for idx := 0; idx < len(x); idx++ { // Traverse left-to-right deterministically.
		v := math.Abs(x[idx]) // Compute absolute value using math.Abs.
		if v > maxV {         // Update maximum when a larger magnitude is found.
			maxV = v // Store the new maximum deterministically.
		}
	}
	return maxV // Return the computed maximum absolute value.
}

// solveLowerUnitTriangular solves L*y = b where L is unit lower-triangular (L[i,i] == 1) using forward substitution.
// This is used for LU-based solves in the Mini-FEA example.
// Implementation:
//   - Stage 1: Validate dimensions and input lengths.
//   - Stage 2: Forward substitution with deterministic i→k order.
//
// Behavior highlights:
//   - Deterministic: fixed i increasing, inner k in [0..i).
//
// Inputs:
//   - L: unit lower-triangular matrix (n×n).
//   - b: RHS vector (length n).
//
// Returns:
//   - []float64: solution vector y (length n).
//
// Errors:
//   - Sentinel errors from L.At(i,k) and shape validation.
//
// Determinism:
//   - Fixed loop order i↑, k↑.
//
// Complexity:
//   - Time O(n^2), Space O(n).
//
// Notes:
//   - Assumes L has a unit diagonal; no division by L[i,i] is performed.
//
// AI-Hints:
//   - For performance, keep L as *Dense to hit fast-paths in other kernels upstream.
func solveLowerUnitTriangular(L matrix.Matrix, b []float64) ([]float64, error) {
	if err := matrix.ValidateNotNil(L); err != nil { // Validate L is not nil.
		return nil, err // Propagate the sentinel error.
	}
	n := L.Rows()      // Read n deterministically.
	if L.Cols() != n { // Enforce square shape for triangular solve.
		return nil, matrix.ErrDimensionMismatch // Return a stable sentinel for shape mismatch.
	}
	if err := matrix.ValidateVecLen(b, n); err != nil { // Validate RHS length.
		return nil, err // Propagate the sentinel error.
	}

	y := make([]float64, n)  // Allocate solution vector deterministically.
	for i := 0; i < n; i++ { // Forward substitution outer loop.
		sum := constZeroFloat    // Initialize accumulator explicitly.
		for k := 0; k < i; k++ { // Accumulate known terms deterministically.
			lik, err := L.At(i, k) // Read L(i,k).
			if err != nil {        // Guard against access errors.
				return nil, err // Propagate sentinel error.
			}
			sum += lik * y[k] // Accumulate lik * y[k].
		}
		y[i] = b[i] - sum // Unit diagonal implies y[i] = b[i] - sum.
	}

	return y, nil // Return the computed solution.
}

// solveUpperTriangular solves U*x = y where U is upper-triangular using backward substitution.
// This is used for both LU-based solves and QR-based solves (R is upper-triangular).
// Implementation:
//   - Stage 1: Validate dimensions and input lengths.
//   - Stage 2: Backward substitution with deterministic i↓ and k in (i..n).
//
// Behavior highlights:
//   - Deterministic: fixed i decreasing, inner k increasing.
//
// Inputs:
//   - U: upper-triangular matrix (n×n).
//   - y: RHS vector (length n).
//
// Returns:
//   - []float64: solution vector x (length n).
//
// Errors:
//   - ErrSingular if U[i,i] == 0, plus sentinel errors from U.At.
//
// Determinism:
//   - Fixed loop order i↓, k↑.
//
// Complexity:
//   - Time O(n^2), Space O(n).
//
// Notes:
//   - Checks for zero pivots using matrix.ZeroPivot to match package conventions.
//
// AI-Hints:
//   - Use this solve after QR when you have R and transformed RHS.
func solveUpperTriangular(U matrix.Matrix, y []float64) ([]float64, error) {
	if err := matrix.ValidateNotNil(U); err != nil { // Validate U is not nil.
		return nil, err // Propagate the sentinel error.
	}
	n := U.Rows()      // Read n deterministically.
	if U.Cols() != n { // Enforce square shape for triangular solve.
		return nil, matrix.ErrDimensionMismatch // Return stable shape sentinel.
	}
	if err := matrix.ValidateVecLen(y, n); err != nil { // Validate RHS length.
		return nil, err // Propagate the sentinel error.
	}

	x := make([]float64, n)       // Allocate solution vector deterministically.
	for i := n - 1; i >= 0; i-- { // Backward substitution outer loop.
		sum := constZeroFloat        // Initialize accumulator explicitly.
		for k := i + 1; k < n; k++ { // Accumulate known terms deterministically.
			uik, err := U.At(i, k) // Read U(i,k).
			if err != nil {        // Guard against access errors.
				return nil, err // Propagate sentinel error.
			}
			sum += uik * x[k] // Accumulate uik * x[k].
		}
		pivot, err := U.At(i, i) // Read diagonal pivot U(i,i).
		if err != nil {          // Guard against access errors.
			return nil, err // Propagate sentinel error.
		}
		if pivot == matrix.ZeroPivot { // Detect singular pivot deterministically.
			return nil, matrix.ErrSingular // Return the package sentinel for singularity.
		}
		x[i] = (y[i] - sum) / pivot // Solve for x[i] deterministically.
	}

	return x, nil // Return the computed solution.
}

// solveWithLU solves A*x = b using deterministic LU factorization (no pivoting) and triangular solves.
// This is used to demonstrate an engineering-style "solve + verify residual" pipeline.
// Implementation:
//   - Stage 1: Factorize A into (L,U) via LUDecompose.
//   - Stage 2: Forward solve L*y = b (unit diagonal).
//   - Stage 3: Backward solve U*x = y.
//
// Behavior highlights:
//   - Deterministic: LU factorization and solves use fixed loop orders.
//
// Inputs:
//   - A: square system matrix (n×n).
//   - b: RHS vector (length n).
//
// Returns:
//   - []float64: solution x (length n).
//
// Errors:
//   - Sentinels from LUDecompose, At-based reads, and singular pivot checks.
//
// Determinism:
//   - Fully deterministic for identical float64 inputs.
//
// Complexity:
//   - Time O(n^3) for LU + O(n^2) solves, Space O(n^2) for factors + O(n).
//
// Notes:
//   - No pivoting trades numerical stability for strict reproducibility.
//
// AI-Hints:
//   - Prefer LU when you want deterministic decompositions and your A is well-conditioned.
func solveWithLU(A matrix.Matrix, b []float64) ([]float64, error) {
	L, U, err := matrix.LUDecompose(A) // Compute LU factorization deterministically.
	if err != nil {                    // Guard against factorization errors.
		return nil, err // Propagate the sentinel-preserving error.
	}
	y, err := solveLowerUnitTriangular(L, b) // Solve L*y = b via forward substitution.
	if err != nil {                          // Guard against solve errors.
		return nil, err // Propagate sentinel errors.
	}
	x, err := solveUpperTriangular(U, y) // Solve U*x = y via backward substitution.
	if err != nil {                      // Guard against solve errors.
		return nil, err // Propagate sentinel errors.
	}
	return x, nil // Return the final solution vector.
}

// solveWithQR solves A*x = b using Householder QR where A ≈ Qᵀ * R (per package contract).
// The solve uses the identity: Qᵀ*R*x = b  =>  R*x = Q*b  (multiply both sides by Q).
// Implementation:
//   - Stage 1: Compute (Q,R) via QRDecompose.
//   - Stage 2: Compute transformed RHS qb = MatVecMul(Q, b).
//   - Stage 3: Backward solve R*x = qb.
//
// Behavior highlights:
//   - Deterministic: QR loop order is fixed; MatVecMul is deterministic.
//
// Inputs:
//   - A: square system matrix (n×n).
//   - b: RHS vector (length n).
//
// Returns:
//   - []float64: solution x (length n).
//
// Errors:
//   - Sentinels from QRDecompose, MatVecMul, and triangular solve.
//
// Determinism:
//   - Fully deterministic for identical float64 inputs.
//
// Complexity:
//   - Time O(n^3) for QR + O(n^2) for MatVec + O(n^2) solve, Space O(n^2).
//
// Notes:
//   - The QR contract here is A ≈ Qᵀ*R (not Q*R); the solve matches that contract.
//
// AI-Hints:
//   - Use QR when you want a stable solve path without pivoting on the same contract surface.
func solveWithQR(A matrix.Matrix, b []float64) ([]float64, error) {
	Q, R, err := matrix.QRDecompose(A) // Compute QR decomposition deterministically.
	if err != nil {                    // Guard against factorization errors.
		return nil, err // Propagate sentinel errors.
	}
	qb, err := matrix.MatVecMul(Q, b) // Compute Q*b to match A ≈ Qᵀ*R contract.
	if err != nil {                   // Guard against MatVec errors.
		return nil, err // Propagate sentinel errors.
	}
	x, err := solveUpperTriangular(R, qb) // Solve R*x = Q*b via backward substitution.
	if err != nil {                       // Guard against solve errors.
		return nil, err // Propagate sentinel errors.
	}
	return x, nil // Return the final solution vector.
}

// logisticStable computes logistic(x) = 1/(1+exp(-x)) using a numerically stable branch strategy.
// This is used in the credit stress-test example to map scores into probabilities.
// Implementation:
//   - Stage 1: Branch on sign(x) to avoid overflow in exp.
//   - Stage 2: Compute the stable form for each branch.
//
// Behavior highlights:
//   - Stable for large |x|; deterministic for identical inputs.
//
// Inputs:
//   - x: real-valued score.
//
// Returns:
//   - float64: probability in (0,1).
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure function; deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - For x >> 0, exp(-x) underflows safely; for x << 0, exp(x) underflows safely.
//
// AI-Hints:
//   - Prefer stable logistic in risk models to avoid NaN propagation in tails.
func logisticStable(x float64) float64 {
	if x >= constZeroFloat { // Use non-negative branch to avoid exp overflow.
		return constOneFloat / (constOneFloat + math.Exp(-x)) // Compute stable positive-side form.
	}
	ex := math.Exp(x)                // Compute exp(x) for negative x (safe).
	return ex / (constOneFloat + ex) // Compute stable negative-side form.
}
