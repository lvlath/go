<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  File: docs/ARIMA.md
  Package: github.com/lvlath/go/arima
  Purpose: Mathematics, public API, numerical contracts, integration and operational examples.
  Contract status: Implemented fixed-order conditional Gaussian ARIMA baseline.
  Scope: Nonseasonal ARIMA(p,d,q), p/q 0..8, d 0..2, conditional estimation,
         anchored level forecast and marginal future-innovation variance.
  License: AGPL-3.0-only. See ../LICENSE.
-->

# ARIMA: conditional Gaussian modeling, anchored integration and forecasting

> A fixed-order, one-dimensional statistical forecasting baseline. It estimates a
> *conditional* ARMA law on differenced observations, retains the original series
> origin, and publishes both future levels and marginal innovation variances.

## 1. Select the correct mathematical problem

Use `github.com/lvlath/go/arima` when observations are ordered, uniformly spaced,
finite scalar measurements, a nonseasonal order `(p,d,q)` has been selected by the
application, and you need an interpretable parametric forecast. Typical uses are
capacity/backlog projections, demand-index scenarios, operational-spread baselines,
and industrial consumption forecasts. A successful model fit is **not** evidence
of stable future relationships or calibrated decisions.

| Task                       | Surface                             | What it does not do                              |
|----------------------------|-------------------------------------|--------------------------------------------------|
| Estimate a fixed model     | `Fit(ctx, values, Options)`         | No order search or holdout fitting               |
| Construct a reviewed model | `New(Parameters)`                   | No estimation or root clipping                   |
| Forecast future levels     | `model.Forecast(ctx, history, H)`   | No synthetic future observations                 |
| Forecast one step          | `model.ForecastNext(ctx, history)`  | No different one-step model                      |
| Inspect a model            | `model.Parameters()` / `model.ID()` | No mutable parameters or authenticated signature |

The package does **not** provide seasonal terms, regressors, automatic frequency
inference, gaps or missing-value repair, automatic model selection, online
retraining, exact diffuse state-space likelihood, prediction covariance matrices,
parameter standard errors, calibration guarantees or a domain-specific alert policy.
The application must establish timestamp cadence and a training/holdout boundary.

### 1.1. Public types and entry points

```go
package arima

type Order struct { AR, Difference, MA int }
type Options struct {
    Order         Order
    IncludeDrift  bool
    MaxIterations int
    Tolerance     float64
}
type Parameters struct {
    AR, MA                    []float64
    Drift, InnovationVariance float64
    Order                     Order
}
type Diagnostics struct {
    Iterations int
    Converged  bool
    Objective  []float64
    EffectiveN int
}
type FitResult struct {
    Model         *Model
    Diagnostics   Diagnostics
    LogLikelihood float64
}
type ForecastResult struct {
    Mean, Variance []float64
    Horizon        int
}
```

The opaque `Model` owns validated coefficients. `FitResult` and
`ForecastResult` implement `IsNil() bool`: it returns `true` **only** for a
nil receiver pointer, not for a non-nil result whose fields were modified.

```go
func Fit(ctx context.Context, values []float64, opts Options) (*FitResult, error)
func FitWithDrift(ctx context.Context, values []float64, order Order) (*FitResult, error)
func FitWithoutDrift(ctx context.Context, values []float64, order Order) (*FitResult, error)
func FitARMA(ctx context.Context, values []float64, ar, ma int, includeDrift bool) (*FitResult, error)
func FitIntegrated(ctx context.Context, values []float64, difference int, includeDrift bool) (*FitResult, error)
func New(parameters Parameters) (*Model, error)
func (m *Model) ID() string
func (m *Model) Parameters() Parameters
func (m *Model) Forecast(ctx context.Context, history []float64, horizon int) (*ForecastResult, error)
func (m *Model) ForecastNext(ctx context.Context, history []float64) (*ForecastResult, error)
```

These signatures are a map of the public surface; see
[`arima/api.go`](../arima/api.go) for executable GoDoc. Convenience functions
select **only** options and then delegate to `Fit` or `Forecast`:

| Facade                         | Equivalent canonical call                                          | Appropriate use                                        |
|--------------------------------|--------------------------------------------------------------------|--------------------------------------------------------|
| `FitWithDrift(ctx,x,order)`    | `Fit(ctx,x,Options{Order:order,IncludeDrift:true})`                | Differenced intercept is a justified model assumption  |
| `FitWithoutDrift(ctx,x,order)` | `Fit(ctx,x,Options{Order:order})`                                  | Intercept is physically constrained to zero            |
| `FitARMA(ctx,x,p,q,drift)`     | `Fit(ctx,x,Options{Order:Order{AR:p,MA:q},IncludeDrift:drift})`    | Stationary series without differencing                 |
| `FitIntegrated(ctx,x,d,drift)` | `Fit(ctx,x,Options{Order:Order{Difference:d},IncludeDrift:drift})` | Differenced white noise with a possible intercept      |
| `m.ForecastNext(ctx,x)`        | `m.Forecast(ctx,x,1)`                                              | Single-step inference retaining both mean and variance |

`Options{}` is *exactly* white noise without drift, not an ARIMA order selector.
`MaxIterations==0` selects 200 accepted iterations and `Tolerance==0` selects
`1e-7`. A negative iteration limit, negative/nonfinite tolerance or an
out-of-range order is rejected. The drift choice is not inferred from `d`.

## 2. Data admission and the differenced-space convention

For original levels `x[0],...,x[n-1]`, define `y=Delta^d x`, with
`p=Order.AR`, `d=Order.Difference`, `q=Order.MA`:

![ARIMA differenced-space equation](https://latex.codecogs.com/svg.image?y_t%3D%5CDelta%5Ed%20x_t%3Dc%2B%5Csum_%7Bi%3D1%7D%5Ep%5Cphi_i%20y_%7Bt-i%7D%2B%5Cvarepsilon_t%2B%5Csum_%7Bj%3D1%7D%5Eq%5Ctheta_j%5Cvarepsilon_%7Bt-j%7D)

```text
Delta x[t]   = x[t+1] - x[t]               (length n-1)
Delta^2 x[t] = x[t+2] - 2*x[t+1] + x[t]  (length n-2, conceptual identity)
N            = n - d
b            = max(p, q)
EffectiveN   = N - b
```

The implementation calculates second differences by **two sequential first
differences**, rather than forming `2*x[t+1]` in one expression. Each applied
operator creates an owned buffer and rejects an unrepresentable result with
`ErrOverflow`. Data values are finite `float64`; they need not be positive.

Supported orders are `0 <= p,q <= 8`, `0 <= d <= 2`. Fitting requires

```text
EffectiveN > p + q + I(IncludeDrift) + 1
```

The final `1` accounts for the estimated innovation variance. This strict
admission rule applies to `Fit`, **not** to `New` or forecasting. Forecasting
requires at least `d + max(p,q) + 1` input levels to establish the same
conditioning history and every required integration anchor.

**Cadence law:** the time index is an equally spaced sample position. An input
of 36 observations does not reveal whether their spacing is seconds, days or
quarters. Use `series.Validate` with a caller-selected timestamp policy before
passing a window to ARIMA. If gaps are reported, segment or reject the window;
ARIMA does not repair it.

### 2.1. Drift is not automatically the mean

The model uses `Parameters.Drift = c` as the **intercept in the differenced
ARMA equation**. For a stationary differenced process,

![Stationary mean in differenced coordinates](https://latex.codecogs.com/svg.image?%5Cmathbb%7BE%7D%5By_t%5D%3D%5Cfrac%7Bc%7D%7B1-%5Csum_%7Bi%3D1%7D%5Ep%5Cphi_i%7D)

| Order     | Interpretation                                                                          |
|-----------|-----------------------------------------------------------------------------------------|
| `d=0,p=0` | `c` is the conditional mean of stationary white noise                                   |
| `d=0,p>0` | The stationary mean is `c/(1-sum(phi))`, not generally `c`                              |
| `d=1,p=0` | The expected one-step increment is `c`                                                  |
| `d=1,p>0` | Long-run increment is `c/(1-sum(phi))` when the stationary ARMA component has that mean |
| `d=2,p=0` | A nonzero `c` is an expected acceleration; expected levels have quadratic growth        |

For example, `p=1,phi=0.5,c=1` has long-run differenced mean `2`, not `1`.
`IncludeDrift=false` fixes `c=+0` in `Fit`. `New` instead accepts the explicit
physical `Drift` supplied by the caller; it has no `IncludeDrift` argument.

## 3. Conditional innovations and the fitted estimand

This is **conditional Gaussian maximum likelihood with profile variance**,
sometimes called conditional sum-of-squares likelihood; it is not exact
likelihood under diffuse or stationary Kalman state initialization.

For `t < b`, set the residual `epsilon[t]` to zero and treat `y[t]` as
conditioning data. For `t >= b`:

![Conditional ARMA innovation](https://latex.codecogs.com/svg.image?%5Cvarepsilon_t%3Dy_t-c-%5Csum_%7Bi%3D1%7D%5Ep%5Cphi_i%20y_%7Bt-i%7D-%5Csum_%7Bj%3D1%7D%5Eq%5Ctheta_j%20%5Cvarepsilon_%7Bt-j%7D)

```text
SSE           = sum_{t=b}^{N-1} epsilon[t]^2
sigmaSquared  = SSE / EffectiveN
loss          = EffectiveN/2 * (log(2*pi) + 1 + log(sigmaSquared))
LogLikelihood = -loss
```

`Diagnostics.Objective` records **negative total** conditional
log-likelihood values of accepted candidates; it is not the SSE, a per-sample
average or a holdout score. The first `b` observations never enter `SSE`.
They still supply actual lagged `y` inputs for the first effective residual.
Their residuals are zero by policy rather than inferred from a hidden
presample state.

The scalar score uses a scaled sum of squares to avoid overflowing a raw
intermediate `epsilon^2` unnecessarily. An exactly zero effective SSE has no
positive Gaussian variance MLE in this contract and yields `ErrSingular`.
Even when all input values are finite, a required difference, derivative,
variance or forecast may be unrepresentable; that is `ErrOverflow`.
This is not an arbitrary-precision likelihood promise.

### 3.1. Independent white-noise oracle

For `x=[1,2,3]`, `Order{}` and `IncludeDrift=true`, the exact MLEs are
`c=2` and `sigmaSquared=((1-2)^2+(2-2)^2+(3-2)^2)/3=2/3`.

![Profile conditional Gaussian log likelihood](https://latex.codecogs.com/svg.image?%5Cell_%5Cmathrm%7Bcond%7D%3D-%5Cfrac%7BN_%5Cmathrm%7Beff%7D%7D%7B2%7D%5Cleft%28%5Clog%282%5Cpi%29%2B1%2B%5Clog%28%5Cmathrm%7BSSE%7D%2FN_%5Cmathrm%7Beff%7D%29%5Cright%29)

No gradient descent is needed in the `p=q=0` case. For an ordinary
stationary ARMA model with nonzero orders, the same formula is evaluated
recursively; the conditional initialization must be aligned before comparing
output with R, statsmodels or other numerical software.

## 4. Stationarity and invertibility: exact domain, no clipping

The sign convention deliberately distinguishes the two polynomials:

```text
AR:  A(z) = 1 - phi[1]*z - ... - phi[p]*z^p
MA:  B(z) = 1 + theta[1]*z + ... + theta[q]*z^q
```

Every polynomial root must be **strictly outside** the unit circle. The
implementation uses Schur step-down coefficients instead of relying on
arbitrary tolerances in a complex polynomial-root solver.

At degree `m`, write the current stable-polynomial coefficients as `a[1..m]`,
extract `k_m=a[m]`, reject if `abs(k_m)>=1`, and recover degree `m-1` through

```text
previous[j] = (current[j] + k_m*current[m-j]) / (1-k_m*k_m)
```

(indices in that identity are one-based). Repeat until the constant
polynomial. Failure is `ErrNonStationary` for AR and `ErrNonInvertible` for
MA. For MA, the check is performed on the *negative* MA coefficients to
match `B(z)=1+sum(theta*z^j)`.

Fitting maps unrestricted optimizer coordinates `u` to strict reflection
coefficients `k_m=tanh(u_m)` and constructs the polynomial by Levinson's
step-up recursion:

```text
a_m[j] = a_(m-1)[j] - k_m * a_(m-1)[m-j]   for 1 <= j < m
a_m[m] = k_m
phi = a_p
(theta) = -a_q
```

For illustration, `k_1=0.4,k_2=0.2` produces `a=[0.32,0.2]`. The corresponding
AR polynomial uses `phi=[0.32,0.2]`; the stable-coefficient construction
for an MA polynomial uses `theta=[-0.32,-0.2]`. Do not interchange the signs.
When floating-point `tanh` rounds to exactly `-1` or `+1`, the candidate is
rejected: it is **never** clipped into feasibility or accepted on a tolerance.

A stable/invertible ARMA polynomial does not make `d>0` level forecasts
stationary. Differencing is precisely the integration that introduces unit
roots into the **level** process; the strict root gate applies to the ARMA
component in `y` coordinates.

## 5. Deterministic estimation and optimizer diagnostics

`Fit` validates its public inputs, differences the data, builds a deterministic
zero-reflection initial candidate (`phi=0`, `theta=0`), sets its initial drift
to the differenced sample mean when requested, and profiles innovation
variance from effective residuals. For `p=q=0`, the fit is analytical.

For nonzero AR/MA order, the solver evaluates an analytic likelihood gradient
in physical `[AR...,MA...,Drift]` coordinates by differentiating residuals
through the MA recursion. It then applies the reflection step-up Jacobians,
including the **negative sign** for MA (`theta=-a`), to obtain optimizer
coordinates. This is not a numerical finite-difference gradient.

L-BFGS retains up to eight accepted correction pairs. Direction construction
uses deterministic newest/oldest traversal. A capped backtracking search
tries at most twenty Armijo step halvings; rejected candidates are never
published. A strict objective decrease and the scaled-gradient condition
must hold for reported convergence; numerical plateau alone is not success.

```text
Initial score -> Objective[0]
Accept step 1 -> Objective[1]
Accept step 2 -> Objective[2]
...
Diagnostics.Iterations = number of accepted solver steps
Diagnostics.Converged  = true only after the gradient criterion succeeds
Diagnostics.EffectiveN = len(training) - d - max(p,q)
```

If the limit or line search is exhausted without gradient convergence,
`Fit` returns a **last accepted immutable candidate plus `ErrNotConverged`**.
It is a diagnostic artifact, not an approved production model. A canceled
context returns `nil,ctx.Err()`, and arithmetic or validation failures
produce no usable fit. Convergence is local and does not certify global
optimality, good model identification or predictive skill.

## 6. Forecasting means: two historical anchors for d=2

A `Model` contains coefficients and a content identifier, **not** an advancing
historical stream. Each `Forecast` invocation accepts its own complete
observed history and reconstructs conditional innovations with the exact same
`b=max(p,q)` rule used in training. The observation origin is the last index
of *that* history, not the model's fitting endpoint.

For every future differenced observation:

```text
yHat[t] = c
          + sum(phi[i] * (observed-or-predicted y[t-i]))
          + sum(theta[j] * known-history epsilon[t-j])
```

All future innovations have conditional expectation zero. Already known
residuals continue to contribute to the MA part until their lag window
expires. The forecast never reads future observations supplied by a different
call; after rolling the origin forward, the application may supply a longer
*observed* prefix to another call.

### 6.1. Anchor recovery, including the off-by-one boundary

For a series `x[0..n-1]`, the first predicted point is **at index n**.

```text
d=0:
  xHat[n+h] = yHat[n+h]

d=1:
  level = x[n-1]
  for every h=0,1,...: level += yHat[n+h]; publish level

d=2:
  level = x[n-1]
  slope = x[n-1] - x[n-2]
  for every h=0,1,...:
    slope += yHat[n+h]
    level += slope
    publish level
```

The `d=2` state must contain **both** the last observed level and the last
observed first difference. For `history=[1,3,6]` and a zero second-difference
forecast, the slope at origin is `3` and the level forecasts are `[9,12,15]`.
Starting the slope at zero instead would produce the wrong trajectory.

| Oracle           | Model and history                            | Mean at horizons 1..3 |
|------------------|----------------------------------------------|-----------------------|
| White noise      | `p=q=d=0,c=2`                                | `[2,2,2]`             |
| AR(1)            | `phi=0.5,c=1,last y=10`                      | `[6,4,3]`             |
| Random walk      | `d=1,c=0,last x=5`                           | `[5,5,5]`             |
| Twice integrated | `d=2,c=0,history=[1,3,6]`                    | `[9,12,15]`           |
| MA(1)            | `theta=0.5`, `history=[0,2]`, `epsilon[1]=2` | `[1,0,0]`             |

The MA(1) example is intentional: with `b=1`, `epsilon[0]=0`, and the last
known `epsilon[1]=2`, the next mean is `theta*epsilon[1]=1`, not `-1`.

## 7. Forecasting uncertainty in O(H) working memory

With the signed convention above, compute the stationary ARMA impulse
response without inverting or materializing a covariance matrix:

![ARMA impulse response](https://latex.codecogs.com/svg.image?%5Cpsi_0%3D1%2C%5Cqquad%5Cpsi_j%3D%5Ctheta_j%2B%5Csum_%7Bi%3D1%7D%5E%5Cmin%28p%2Cj%29%5Cphi_i%5Cpsi_%7Bj-i%7D)

Treat `theta_j=0` past `q`. Apply zero, one or two running cumulative sums
according to `d` to obtain the **integrated response** `Psi_j`:

```text
d=0: Psi[j] = psi[j]
d=1: Psi[j] = sum_{k=0}^j psi[k]
d=2: Psi[j] = sum_{k=0}^j sum_{ell=0}^k psi[ell]
```

![Marginal innovation forecast variance](https://latex.codecogs.com/svg.image?%5Coperatorname%7BVar%7D%5Cleft%28x_%7Bn%2Bh-1%7D%20%7C%20%5Cmathcal%7BF%7D_%7Bn-1%7D%5Cright%29%3D%5Csigma%5E2%5Csum_%7Bj%3D0%7D%5E%7Bh-1%7D%5CPsi_j%5E2)

Only future innovation uncertainty appears. These marginal variances do
**not** include estimated-parameter uncertainty, future external shocks,
misspecified distributions or covariance between different forecast horizons.
An application can form an illustrative conditional Gaussian one-sided
planning bound `mean[h-1] + z*sqrt(variance[h-1])`, but calibration,
multiple-horizon dependence and the operational action are external.

| Oracle                 | `sigma^2` | Marginal variances, h=1..3 |
|------------------------|----------:|----------------------------|
| White noise `p=q=d=0`  |         3 | `[3,3,3]`                  |
| AR(1), `phi=0.5`       |         4 | `[4,5,5.25]`               |
| Random walk `d=1`      |         2 | `[2,4,6]`                  |
| Twice integrated `d=2` |         1 | `[1,5,14]`                 |
| MA(1), `theta=0.5`     |         1 | `[1,1.25,1.25]`            |

The `d=2` weights are `Psi=[1,2,3,...]`, hence the horizon-three variance
is `1^2+2^2+3^2=14` in the unit-innovation case.

### 7.1. Actual complexity accounting

Let `N=len(history)-d`, `H=horizon`, `k=p+q+I(IncludeDrift)` and `I` be
the number of accepted optimizer steps. The permitted polynomial orders are
small but the lengths `N` and `H` are not fixed.

| Operation      | Time                                                   | Additional space | Main allocation                                               |
|----------------|--------------------------------------------------------|------------------|---------------------------------------------------------------|
| `New`          | `O(p^2+q^2)`                                           | `O(p+q)`         | owned coefficients / model hash                               |
| `Fit`          | `O((I+1)*N*(p+q+1)^2)` with bounded line search factor | `O(N*(p+q+1)+I)` | residual/derivative workspace; accepted loss history          |
| `Forecast`     | `O((N+H)*(p+q+d+1))`                                   | `O(N+H)`         | historical differences/residuals, prediction/response buffers |
| `ForecastNext` | `O(N*(p+q+d+1))`                                       | `O(N)`           | same one-step kernel and historical workspace                 |
| `Parameters`   | `O(p+q)`                                               | `O(p+q)`         | detached AR/MA slices                                         |
| `ID` / `IsNil` | `O(1)`                                                 | `O(1)`           | none                                                          |

`Forecast` needs **no dense `H x H` covariance matrix**. Small Jacobians used
during fitting are polynomial-order-sized and are unrelated to horizon
forecast covariance. An extremely large `H` can still exhaust available RAM;
size arithmetic is checked, but the implementation does not convert OS OOM
into a Go error.

## 8. Errors, precedence and partial-result law

Sentinel checks use `errors.Is(err, arima.ErrXxx)`, never error-message
matching. At each context-based facade, an invalid (`nil`) context returns
`ErrInvalidOption`; an already canceled context returns `ctx.Err()` before
examining other inputs.

**Fit precedence** (where conditions coexist): context -> options -> nil
samples -> empty samples -> strict effective-size test -> complete nonfinite
sample scan -> differencing -> likelihood and optimizer.

**Forecast precedence:** context -> model initialized/horizon positive -> nil
history -> empty history -> history length -> nonfinite scan -> horizon buffer
admission -> differencing/innovation/forecast. `New` separately validates
order and coefficient lengths, then finiteness, positive variance, AR and MA
Schur conditions. A zero-value or nil `Model` cannot forecast.

| Condition                                                                                   | Error                 | Publication                                               |
|---------------------------------------------------------------------------------------------|-----------------------|-----------------------------------------------------------|
| Nil context, invalid order, horizon <= 0, unsupported option, uninitialized model           | `ErrInvalidOption`    | nil result                                                |
| Nil observation/history slice                                                               | `ErrNilInput`         | nil result                                                |
| Non-nil, zero-length observation/history                                                    | `ErrEmptyInput`       | nil result                                                |
| Fit effective size too small; Forecast insufficient history                                 | `ErrInsufficientData` | nil result                                                |
| NaN/Inf **caller input**                                                                    | `ErrNonFinite`        | nil result                                                |
| AR polynomial on/within unit circle                                                         | `ErrNonStationary`    | nil model                                                 |
| MA polynomial on/within unit circle                                                         | `ErrNonInvertible`    | nil model                                                 |
| Zero fitted SSE; nonpositive supplied variance                                              | `ErrSingular`         | nil result                                                |
| Nonrepresentable differencing, parameter derivatives, integration, result or buffer product | `ErrOverflow`         | nil result                                                |
| Local search budget/line-search exhausted                                                   | `ErrNotConverged`     | last accepted `*FitResult`, `Diagnostics.Converged=false` |
| Cancellation                                                                                | `ctx.Err()`           | nil result                                                |

No `Forecast` result is published on any error. A non-nil result returned
with `ErrNotConverged` is **not** a successful fit. In particular,
`FitResult.IsNil()` means pointer presence, not convergence, validation or
model-approval state.

## 9. Ownership, identity, determinism and concurrency

`New` deep-copies AR/MA inputs; `Parameters()` deep-copies both coefficient
slices. Returned `Diagnostics.Objective`, `ForecastResult.Mean` and
`ForecastResult.Variance` are detached. A `Model` is immutable after
publication, and independent `Forecast` calls use their own workspaces.
Input slices must remain unchanged during an active call; concurrent
caller-side mutation is unsupported.

`Model.ID()` is a lowercase SHA-256 hexadecimal fingerprint of the versioned
conditional-Gaussian method, the order, and binary64 coefficients, intercept
and innovation variance. Signed zeros use the same canonical float encoding.
The hash is **not** a signed model artifact, authorization token, persistence
format or data-lineage proof. A nil model's ID is empty. Its
`Parameters()` returns the zero `Parameters` value (which is not a valid
supplied model because the variance is zero).

The model does not retain any timestamp, training history, feature grid or
last innovation. Two calls with the same model and *different* histories have
different conditioning information by design. For a fixed input and options,
traversal is sample-index ascending, lag-index ascending and optimizer
correction order fixed; there is no RNG, map-order tie, network request or
clock-based optimization branch.

## 10. Runnable operational examples

All repository examples use **audited synthetic engineering fixtures** with
plausible units, not verified government GDP releases, exchange quotes or
live production feeds. They demonstrate correct data/control wiring, not
statistical suitability. Each output is deterministic under the fixed package
contract.

### 10.1. Day-by-day service-capacity review: train, freeze, roll origin

The following program separates **training**, **forecast origin** and
**subsequently observed** values. A model is fitted on the training prefix
only. One-step forecasts at later origins use the then-observed prefix but
never refit on future data or append model predictions as observations.

```go
package main

import (
    "context"
    "fmt"
    "math"

    "github.com/lvlath/go/arima"
)

func main() {
    // Illustrative daily incident backlog with one observation per day.
    backlog := []float64{
        180, 184, 189, 191, 195, 199, 201, 206,
        210, 212, 215, 218, 222, 226, 229, 233,
        235, 239, 242, 246, 250, 252, 257, 260,
        263, 267, 270, 272, 277, 281, 284, 288,
    }
    const trainingDays = 25
    const reviewCapacity = 290.0
    const oneSided95 = 1.645
    const twoSided95 = 1.96

    ctx := context.Background()
    fit, err := arima.FitIntegrated(ctx, backlog[:trainingDays], 1, true)
    if err != nil {
        // Never silently deploy a nonconverged or otherwise invalid candidate.
        fmt.Println("fit error:", err)
        return
    }
    if !fit.Diagnostics.Converged {
        fmt.Println("candidate requires review")
        return
    }

    modelID := fit.Model.ID()
    breaches := 0
    covered := 0
    for end := trainingDays; end < len(backlog); end++ {
        // This origin cannot see backlog[end] until after forecast publication.
        prediction, forecastErr := fit.Model.ForecastNext(ctx, backlog[:end])
        if forecastErr != nil {
            fmt.Println("forecast error:", forecastErr)
            return
        }
        upper := prediction.Mean[0] + oneSided95*math.Sqrt(prediction.Variance[0])
        if upper > reviewCapacity {
            breaches++ // A planning flag, not automatic staffing activation.
        }
        if math.Abs(backlog[end]-prediction.Mean[0]) <=
            twoSided95*math.Sqrt(prediction.Variance[0]) {
            covered++ // Simple diagnostic; not a confidence guarantee.
        }
    }
    fmt.Printf("holdout=%d covered=%d capacity-reviews=%d model-frozen=%t\n",
        len(backlog)-trainingDays, covered, breaches, fit.Model.ID() == modelID)
}
```

Run inside the repository: `go run ./examples/arima` for a multi-horizon
service-queue planning report. The package-level examples include GDP index
planning, liquidity-spread review, day-ahead service backlog, industrial
energy acceleration, a stationary data-center correction model and
model-fingerprint governance in [`arima/example_test.go`](../arima/example_test.go).

### 10.2. Applied decision boundaries

| Domain                 | Model question                    | Application must still provide                                          |
|------------------------|-----------------------------------|-------------------------------------------------------------------------|
| Quarterly output index | Fixed-order growth trajectory     | Observed release cadence, revisions, trend breaks, holdout scoring      |
| Bid-ask spread         | Mean-reverting spread baseline    | Quote-quality checks, intraday calendar, jumps, liquidity budget policy |
| Service queue          | Future unresolved workload        | Holidays, interventions, staff capacity and acceptable false alarms     |
| Industrial electricity | Acceleration of daily MWh         | Weather/shift covariates, realistic capacity reserve policy             |
| Data-center telemetry  | Stationary correction around zero | Sensor health, calibration, alert hold time and operational response    |
| Model rollout          | Immutable parameter identity      | Authenticated storage, deployment approval and input lineage            |

## 11. Interoperability within lvlath

- **`series`** validates explicit timestamp cadence and can compute
  mask-aware differences. ARIMA computes and owns its own `d`-fold
  differencing and retains historical anchors; do not feed already
  differenced data into a model with `Difference>0`.
- **`acf`** computes retrospective fixed-window sample autocorrelation,
  useful for examining training/holdout residual dependence when the
  application supplies the residuals. ARIMA does not export innovation
  history as a public artifact; do not claim a built-in ACF residual audit.
- **`ols` / `matrix`** can solve unrelated full-rank linear problems.
  Analytical white-noise with drift resembles a constant-only least-squares
  estimate, but fitting general ARMA by ordinary QR is **not** equivalent
  to the conditional ARMA likelihood recursion.
- **`gjrgarch`** models conditional heteroskedastic variance, while ARIMA
  estimates a constant innovation variance for a conditional mean model.
  Coupling mean residuals to GJR-GARCH requires an explicit, separately
  validated application pipeline; no implicit ARIMA-GARCH joint fit exists.
- **`gbm` / `jumpdiffusion` / `montecarlo`** generate stochastic scenarios
  under different assumptions. ARIMA forecasts are not automatically GBM
  price paths, risk-neutral measures or tail-event simulation policies.

## 12. Verification oracles and regression defenses

These are **mathematical** checkpoints, not merely comparisons between two
routes through the same implementation:

| ID   | Independent check                                                                    | Defect prevented                   |
|------|--------------------------------------------------------------------------------------|------------------------------------|
| AR01 | White noise `c=2,sigma2=3` predicts mean `2`, variance `3` at every H                | Mistaken horizon accumulation      |
| AR02 | AR(1) `phi=0.5,c=1,sigma2=4`, last `10`: mean `[6,4,3]`, variance `[4,5,5.25]`       | Wrong AR/psi recursion             |
| AR03 | Random walk `d=1,c=0`, last `5`: mean `[5,5,5]`, variance `[2,4,6]` at sigma2=2      | Missing level anchor               |
| AR04 | `d=2`, `[1,3,6]`: `[9,12,15]`, variance `[1,5,14]` at sigma2=1                       | Missing slope anchor               |
| AR05 | MA(1) theta `+0.5`, history `[0,2]`: first means `[1,0]`, first variances `[1,1.25]` | MA sign/burn-in                    |
| AR06 | Fit white noise with drift on `[1,2,3]`: `c=2`, `sigma2=2/3`                         | Variance denominator and intercept |
| AR07 | Exact `phi=1` or `theta=1` rejects with separate sentinels                           | Silent root clipping               |
| AR08 | Forecast at a fixed origin unchanged by any **unread** future values                 | Look-ahead / mutable model         |
| AR09 | Constant data returns `ErrSingular`                                                  | Undefined `log(0)`                 |

Use `go test ./arima -count=1`, `go test -race ./arima -count=1`,
`go test ./... -count=1`, `go vet ./...` and the repository-pinned
`golangci-lint run` in the real checkout. Tests must live in `package
arima_test`, validate errors with `errors.Is`, and use independent numerical
oracles. Passing fit diagnostics alone never proves a model is appropriate
for a particular operational decision.

## 13. Pitfalls and field checklist

1. Verify timestamps or source cadence **before** passing a scalar slice.
2. Fix `p,d,q` from the application model policy, not from the zero Options value.
3. Keep `Drift` in the differenced-space equation; never silently reinterpret it.
4. Preserve the last actual level and, for `d=2`, the last actual first difference.
5. Exclude all burn-in residuals from SSE and use `EffectiveN`, not raw `n`.
6. Reject unit-circle roots; no coefficient clipping or silent alternative fitting.
7. Treat `ErrNotConverged` as a failed production gate even with a candidate model.
8. Distinguish marginal **innovation** variance from parameter-estimation uncertainty.
9. Preserve independent training and forecast origins; no look-ahead during rolling evaluation.
10. Inspect numeric errors and finite output before using any planning bound.
11. Refit only by deliberate application policy and record the new model fingerprint.
12. Cross-check analytic fixtures and a chronological holdout before deployment.

Further reading: [`arima/doc.go`](../arima/doc.go),
[`arima/api.go`](../arima/api.go),
[`arima/example_test.go`](../arima/example_test.go),
[`examples/arima/main.go`](../examples/arima/main.go),
[`TUTORIAL.md`](TUTORIAL.md), [`FAQ_&_TIPS.md`](FAQ_%26_TIPS.md),
[`SERIES.md`](SERIES.md) and [`ACF.md`](ACF.md).
