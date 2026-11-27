// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package drawdown_test demonstrates executable risk-audit workflows over observed equity.
// All fixtures are deterministic, engineered offline records, not exchange-audited histories.
// AI-Hints: admit NAV, source revisions and timestamps outside the kernel; use the exact
// witness rather than a rounded argmax, and keep risk policy outside the package.
package drawdown_test

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/lvlath/go/drawdown"
)

// ExampleCompute demonstrates a fund administrator's cashflow-neutral NAV audit.
// The risk committee needs to distinguish a true investment loss from a decline
// in the total account balance caused by investor redemptions.
//
// Scenario (medium):
//   - A fund starts with 1,000 units and USD 100,000 marked equity.
//   - Subscriptions and redemptions execute at the previous mark's NAV per unit;
//     the ledger supplies the cash amount and the subsequent marked balance.
//   - An investor redeems during a losing episode and another after recovery.
//
// Process:
//   - Reconcile unit count and derive NAV per unit externally from admitted marks.
//   - Compute the exact investment drawdown witness from NAV, not raw balances.
//   - Compare the balance-only statistic to expose the cashflow distortion and
//     report the peak, trough, recovery and ending NAV to the committee.
//
// Complexity:
//   - Time O(n), Space O(n) for ledger channels and the two batch computations.
//
// Notes:
//   - These engineered values make all subscriptions and redemptions exact in units.
//   - A production fund administrator handles fees, dealing prices, valuation times,
//     rounding, corporate actions and reconciliation; the example is not a ledger API.
func ExampleCompute() {
	const initialUnits = 1000.0
	// A positive cashflow buys units; a negative cashflow redeems units at prior NAV.
	ledger := []struct {
		id                string
		cashflow, balance float64
	}{
		{"M00", 0, 100000}, {"M01", 0, 104000}, {"M02", 0, 110000},
		{"M03", 55000, 165000}, {"M04", 0, 150000}, {"M05", 0, 135000},
		{"M06", -18000, 117000}, {"M07", 0, 123500}, {"M08", 0, 130000},
		{"M09", 0, 143000}, {"M10", 0, 149500}, {"M11", -23000, 126500},
		{"M12", 0, 132000},
	}
	// Construct the two equally indexed representations, retaining ledger IDs.
	nav := make([]float64, len(ledger))
	balances := make([]float64, len(ledger))
	units := initialUnits
	for i, mark := range ledger {
		if i > 0 && mark.cashflow != 0 {
			// A dealing price of zero requires a separate recapitalization policy.
			if nav[i-1] <= 0 {
				fmt.Println("invalid dealing NAV:", mark.id)
				return
			}
			units += mark.cashflow / nav[i-1] // Deal at the previous admitted NAV.
		}
		if units <= 0 || mark.balance < 0 {
			fmt.Println("invalid ledger mark:", mark.id)
			return
		}
		balances[i] = mark.balance
		nav[i] = mark.balance / units // Use the post-flow unit count at the new mark.
	}
	ctx := context.Background()
	result, err := drawdown.Compute(ctx, nav)
	if err != nil {
		fmt.Println("NAV audit:", err)
		return
	}
	// The scalar projection is sufficient for the intentionally different balance metric.
	raw, err := drawdown.ComputeMaximum(ctx, balances)
	if err != nil {
		fmt.Println("balance audit:", err)
		return
	}
	fmt.Printf("NAV MDD=%.2f%%; balance MDD=%.2f%%; final NAV=%.2f USD/unit\n", 100*result.Maximum, 100*raw, nav[len(nav)-1])
	fmt.Printf("peak=%s trough=%s", ledger[result.PeakIndex].id, ledger[result.TroughIndex].id)
	if result.RecoveryIndex >= 0 {
		fmt.Printf(" recovery=%s\n", ledger[result.RecoveryIndex].id)
	} else {
		fmt.Println(" recovery=unobserved")
	}
	// Output:
	// NAV MDD=18.18%; balance MDD=29.09%; final NAV=120.00 USD/unit
	// peak=M02 trough=M05 recovery=M09
}

// ExampleCompute_backtest demonstrates an independent release gate for a systematic strategy.
// A positive terminal P&L is insufficient evidence of acceptable historical risk.
//
// Scenario (medium+):
//   - Audit 500 synthetic weekday valuation marks in USD, beginning at 100,000.
//   - The engineered history includes an early correction, a multi-stage severe
//     decline, an eventual recovery and two later smaller corrections.
//   - A caller-owned policy requests review if sampled MDD exceeds 20 percent.
//
// Process:
//   - Keep valuation dates aligned with the exact equity indices; no inferred gaps.
//   - Run one Compute, map witness indices back to the report rows, and compare
//     terminal performance with the observed maximum loss.
//   - Report observation intervals separately from dates and wall-clock time.
//
// Complexity:
//   - Time O(n), Space O(n) including dates, fixture and detached depth output.
//
// Notes:
//   - The fixture calendar is a weekday schedule, not an exchange holiday calendar.
//   - This risk screen is not a model approval or claim of strategy profitability.
func ExampleCompute_backtest() {
	const reviewLimit = 0.20
	equity := backtestFixture()
	// Build an explicit, aligned schedule for the offline audit.
	dates := make([]time.Time, len(equity))
	day := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	for i := range dates {
		for day.Weekday() == time.Saturday || day.Weekday() == time.Sunday {
			day = day.AddDate(0, 0, 1)
		}
		dates[i] = day
		day = day.AddDate(0, 0, 1)
	}
	r, err := drawdown.Compute(context.Background(), equity)
	if err != nil {
		fmt.Println("audit:", err)
		return
	}
	// Preserve the independent terminal-return and peak-relative loss definitions.
	terminalReturn := (equity[len(equity)-1] - equity[0]) / equity[0]
	fmt.Printf("marks=%d terminal-return=%.0f%% MDD=%.0f%% review=%t\n", len(equity), 100*terminalReturn, 100*r.Maximum, r.Maximum > reviewLimit)
	fmt.Printf("peak=%s USD %.0f; trough=%s USD %.0f\n", dates[r.PeakIndex].Format(time.DateOnly), equity[r.PeakIndex], dates[r.TroughIndex].Format(time.DateOnly), equity[r.TroughIndex])
	// The absence of a recovery is a meaningful successful result, never a negative duration.
	if r.RecoveryIndex >= 0 {
		fmt.Printf("recovery=%s; decline=%d intervals; recovery=%d intervals; total=%d intervals\n",
			dates[r.RecoveryIndex].Format(time.DateOnly), r.TroughIndex-r.PeakIndex,
			r.RecoveryIndex-r.TroughIndex, r.RecoveryIndex-r.PeakIndex)
	} else {
		fmt.Println("recovery=unobserved at report cutoff")
	}
	// Output:
	// marks=500 terminal-return=52% MDD=25% review=true
	// peak=2024-05-20 USD 120000; trough=2024-08-12 USD 90000
	// recovery=2024-11-04; decline=60 intervals; recovery=60 intervals; total=120 intervals
}

// ExampleCompute_trailingLimit demonstrates a prop-desk sampled risk dashboard.
// Current loss and the worst loss in the account's supplied history answer different questions.
//
// Scenario (special):
//   - An admitted account starts at USD 10,000, builds a high-water mark, then
//     crosses the desk's inclusive 10% rounded-depth review threshold.
//   - After recovery, the current mark is safe under this illustrative threshold
//     even though the historical maximum still records the breach.
//
// Process:
//   - Recompute at selected checkpoints using only the history available then.
//   - Use ComputeCurrentDepth for current status and ComputeMaximum for the final audit.
//   - Print review status only; no order, liquidation or automatic reset is inferred.
//
// Complexity:
//   - O(n) time and O(n) transient output for one prefix; repeated calls cost
//     O(sum of prefix lengths), not constant-time streaming.
//
// AI-Hints:
//   - This is a sampled rounded-depth threshold policy, not exact-rational
//     certification of a prop firm's trailing balance, intraday or equity rules.
func ExampleCompute_trailingLimit() {
	const reviewLimit = 0.10
	equity := []float64{10000, 10200, 10400, 10800, 10600, 10000, 9720, 9600, 9900, 10800, 11000, 10450}
	ctx := context.Background()
	// Simulate periodic as-of review cuts. Later values must not enter an earlier decision.
	for _, i := range []int{4, 6, 7, 9, 11} {
		current, err := drawdown.ComputeCurrentDepth(ctx, equity[:i+1])
		if err != nil {
			fmt.Println("current audit:", err)
			return
		}
		fmt.Printf("mark=%02d equity=%.0f current=%.2f%% review=%t\n", i, equity[i], 100*current, current >= reviewLimit)
	}
	historical, err := drawdown.ComputeMaximum(ctx, equity)
	if err != nil {
		fmt.Println("historical audit:", err)
		return
	}
	fmt.Printf("historical MDD=%.2f%%\n", 100*historical)
	// Output:
	// mark=04 equity=10600 current=1.85% review=false
	// mark=06 equity=9720 current=10.00% review=true
	// mark=07 equity=9600 current=11.11% review=true
	// mark=09 equity=10800 current=0.00% review=false
	// mark=11 equity=10450 current=5.00% review=false
	// historical MDD=11.11%
}

// ExampleCompute_crisisRecovery compares two stress-recovery shapes and a censored history.
// A portfolio risk team cares about time underwater as well as maximum severity.
//
// Scenario (medium+):
//   - A long, slow-recovery stress path resembles a financial-crisis scenario;
//     a rapid decline/rebound resembles a pandemic-shock scenario.
//   - These 2008-style and 2020-style paths are engineered weekly NAV marks,
//     not historical prices. A third cut ends before recovery is observed.
//
// Process:
//   - Audit each complete or explicitly truncated history independently.
//   - Count underwater observations using the causal Depth profile.
//   - Compare maximum loss and elapsed observation intervals; keep censored
//     recovery distinct from a completed duration.
//
// Complexity:
//   - Time O(sum of scenario lengths), Space O(max scenario length) in active output.
//
// Notes:
//   - Use identical sampling and adjustment conventions for real comparisons.
//   - Underwater observation counts are not calendar durations; never use -1 in subtraction.
func ExampleCompute_crisisRecovery() {
	// A normalized NAV index is valid equity because its origin is positive.
	slow := []float64{100, 105, 110, 100, 90, 75, 66, 70, 80, 90, 85, 92, 100, 105, 109, 110}
	fast := []float64{100, 105, 110, 88, 77, 88, 99, 110, 112, 115, 118, 120, 116, 119, 122, 125}
	for _, scenario := range []struct {
		name   string
		equity []float64
	}{
		{"2008-style", slow}, {"2020-style", fast}, {"slow-cut", slow[:10]},
	} {
		r, err := drawdown.Compute(context.Background(), scenario.equity)
		if err != nil {
			fmt.Println("stress audit:", err)
			return
		}
		// Count all underwater marks, not just those within the selected witness.
		underwater := 0
		for _, depth := range r.Depth {
			if depth > 0 {
				underwater++
			}
		}
		fmt.Printf("%s: MDD=%.0f%% underwater=%d/%d final=%.2f%%", scenario.name, 100*r.Maximum, underwater, len(r.Depth), 100*r.Depth[len(r.Depth)-1])
		if r.RecoveryIndex < 0 {
			fmt.Println(" peak-to-recovery=unobserved")
			continue
		}
		fmt.Printf(" peak-to-recovery=%d intervals\n", r.RecoveryIndex-r.PeakIndex)
	}
	// Output:
	// 2008-style: MDD=40% underwater=12/16 final=0.00% peak-to-recovery=13 intervals
	// 2020-style: MDD=30% underwater=6/16 final=0.00% peak-to-recovery=5 intervals
	// slow-cut: MDD=40% underwater=7/10 final=18.18% peak-to-recovery=unobserved
}

// ExampleCompute_precision demonstrates a numerical release test for a risk-reporting adapter.
// An auditor must preserve evidence even when display rounding hides a loss or a tie.
//
// Scenario (special):
//   - A normalized test ledger contains a one-ULP loss, two nearly total losses
//     at different peaks, and a later true zero balance in a replay extension.
//   - The display rounds both near-total depths to 100%, but their exact residual
//     ratios differ and true zero must win over either positive residual.
//
// Process:
//   - Run Compute on the admitted prefix; print exact small depth and witness indices.
//   - Extend with a real zero and recompute; use original equity at TroughIndex
//     to distinguish a total loss from a rounded 100% measurement.
//
// Complexity:
//   - O(n) time and O(n) owning output per batch; no arbitrary-precision production workspace.
//
// AI-Hints:
//   - This adversarial binary64 fixture is numerical evidence, not a forecast of insolvency.
func ExampleCompute_precision() {
	const residual = 0x1p-55
	equity := []float64{1, math.Nextafter(1, 0), residual, 2, residual, 2, 0}
	ctx := context.Background()
	prefix, err := drawdown.Compute(ctx, equity[:len(equity)-1])
	if err != nil {
		fmt.Println("prefix audit:", err)
		return
	}
	fmt.Printf("small depth=%g; rounded collision=%t\n", prefix.Depth[1], prefix.Depth[2] == prefix.Depth[4])
	fmt.Printf("prefix witness=%d/%d/%d; true total loss=%t\n", prefix.PeakIndex, prefix.TroughIndex, prefix.RecoveryIndex, equity[prefix.TroughIndex] == 0)
	full, err := drawdown.Compute(ctx, equity)
	if err != nil {
		fmt.Println("full audit:", err)
		return
	}
	fmt.Printf("extended witness=%d/%d/%d; true total loss=%t\n", full.PeakIndex, full.TroughIndex, full.RecoveryIndex, equity[full.TroughIndex] == 0)
	// Output:
	// small depth=1.1102230246251565e-16; rounded collision=true
	// prefix witness=3/4/5; true total loss=false
	// extended witness=3/6/-1; true total loss=true
}
