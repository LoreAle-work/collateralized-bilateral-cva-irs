# Collateralized Bilateral CVA/DVA for an Interest Rate Swap under Hull-White Monte Carlo with Wrong-Way Risk and EVT Tail Exposure

## 1. Project Overview

This project implements a transparent counterparty credit risk engine for a collateralized plain-vanilla interest rate swap. The model starts from an initial EUR zero curve, simulates future interest-rate scenarios using a one-factor Hull-White model, reprices an at-market interest rate swap along Monte Carlo paths, computes exposure profiles, applies collateral mechanics, converts stylized credit spreads into hazard curves, and calculates unilateral and bilateral first-to-default CVA/DVA.

The project then extends the base XVA engine with stress testing, wrong-way risk, and Extreme Value Theory tail exposure diagnostics.

The full modeling pipeline is:

$$
\text{ECB zero curve}
\rightarrow
\text{Hull-White simulation}
\rightarrow
\text{swap valuation}
\rightarrow
\text{exposure profiles}
\rightarrow
\text{collateralized exposure}
\rightarrow
\text{hazard curves}
\rightarrow
\text{CVA/DVA}
\rightarrow
\text{stress testing}
\rightarrow
\text{WWR and EVT extensions}.
$$

The goal is not to produce a production-grade bank XVA system. The goal is to build a modular, mathematically explicit, and reproducible research implementation that demonstrates the mechanics of exposure simulation, collateralized counterparty risk, bilateral CVA/DVA, wrong-way risk, and tail exposure estimation.

## 2. Main Features

The project includes:

- ECB zero curve loading and interpolation.
- Historical Hull-White parameter estimation.
- One-factor Hull-White Monte Carlo simulation.
- Conditional Hull-White zero-coupon bond pricing.
- At-market interest rate swap construction using the par swap rate.
- Pathwise swap valuation across Monte Carlo scenarios.
- Expected Exposure, Expected Negative Exposure, and Potential Future Exposure.
- Collateralized exposure with threshold, minimum transfer amount, bilateral posting, and margin lag.
- Stylized credit spread curves converted into piecewise-constant hazard curves.
- Survival probabilities and marginal default probabilities.
- Unilateral CVA and DVA.
- Bilateral first-to-default CVA and DVA.
- Credit and collateral stress scenarios.
- Wrong-way risk through stochastic counterparty intensity correlated with interest-rate shocks.
- EVT-based high-quantile PFE estimation using Peaks-over-Threshold and GPD fitting.
- Model validation checks and output tables.

## 3. Repository Structure

Recommended project structure:

```text
project_root/
├── data/
│   ├── raw/
│   │   ├── ecb_aaa_spot_yield_curve.csv
│   │   └── credit_spread_scenarios.csv
│   └── processed/
│       └── vasicek_short_rate_data.csv
│
├── notebooks/
│   └── 02_hull_white_simulation.ipynb
│
├── outputs/
│   ├── exposure_summary.csv
│   ├── collateral_exposure_summary.csv
│   ├── credit_key_summary.csv
│   ├── xva_summary.csv
│   ├── xva_stress_summary.csv
│   ├── wrong_way_risk_summary.csv
│   ├── evt_pfe_995_profile.csv
│   ├── evt_pfe_999_profile.csv
│   ├── evt_pfe_comparison.csv
│   └── final_results.csv
│
├── src/
│   ├── rates/
│   │   ├── calibration.py
│   │   ├── curves.py
│   │   ├── hull_white.py
│   │   └── vasicek.py
│   │
│   ├── derivatives/
│   │   └── swap.py
│   │
│   ├── collateral/
│   │   └── margining.py
│   │
│   ├── credit/
│   │   └── hazard_curve.py
│   │
│   ├── xva/
│   │   ├── cva_dva.py
│   │   └── wrong_way_risk.py
│   │
│   └── evt/
│       └── tail_exposure.py
│
├── tests/
│   ├── test_swap.py
│   ├── test_collateral.py
│   ├── test_credit.py
│   ├── test_xva.py
│   └── test_evt.py
│
├── README.md
└── requirements.txt
```

## 4. Data Inputs

### 4.1 Interest Rate Data

The interest-rate input is an ECB AAA zero-coupon yield curve. The curve is used to construct:

- zero rates $R(0,T)$,
- discount factors $P(0,T)$,
- instantaneous forward rates $f(0,T)$,
- the Hull-White drift term $\theta(t)$.

The zero-coupon discount factor is computed as:

$$
P(0,T)=\exp(-R(0,T)T),
$$

where $R(0,T)$ is the continuously compounded zero rate for maturity $T$.

### 4.2 Credit Spread Data

The credit data is stylized rather than market CDS data. The project uses a scenario file containing counterparty and own spread curves in basis points.

Example structure:

| Maturity | Counterparty Base | Counterparty Stress | Self Base | Self Stress |
|---:|---:|---:|---:|---:|
| 1Y | 50 bps | 120 bps | 40 bps | 100 bps |
| 3Y | 80 bps | 180 bps | 65 bps | 150 bps |
| 5Y | 120 bps | 250 bps | 100 bps | 220 bps |
| 7Y | 150 bps | 320 bps | 130 bps | 280 bps |
| 10Y | 180 bps | 400 bps | 160 bps | 350 bps |

The spreads are converted into decimal form before being transformed into hazard rates.

## 5. Interest Rate Modeling

## 5.1 From Vasicek to Hull-White

A Vasicek model was first estimated as a transparent Gaussian mean-reverting benchmark:

$$
dr_t = a(b-r_t)dt + \sigma dW_t.
$$

The Vasicek model is easy to estimate through an AR(1) representation, but it has important limitations:

- it does not fit the initial term structure by construction,
- it assumes a constant long-run mean $b$,
- residual diagnostics show non-normality, volatility clustering, and regime dependence,
- the full-sample calibration can produce unrealistically slow mean reversion.

The final exposure engine therefore uses the one-factor Hull-White model:

$$
dr_t = [\theta(t)-ar_t]dt + \sigma dW_t.
$$

The Hull-White model extends Vasicek by replacing the constant long-run mean with a time-dependent drift $\theta(t)$ chosen to fit the initial zero curve.

### Variable Definitions

| Symbol | Meaning |
|---|---|
| $r_t$ | short rate at time $t$ |
| $a$ | mean-reversion speed |
| $\sigma$ | short-rate volatility |
| $\theta(t)$ | time-dependent drift fitted to the initial curve |
| $W_t$ | Brownian motion |
| $P(0,T)$ | initial discount factor to maturity $T$ |
| $f(0,t)$ | instantaneous forward rate at time $t$ |

## 5.2 Hull-White Drift

Under the one-factor Hull-White model, the drift term is constructed from the initial forward curve:

$$\theta(t)=
\frac{\partial f(0,t)}{\partial t}
+
a f(0,t)
+
\frac{\sigma^2}{2a}\left(1-e^{-2at}\right).
$$

This ensures consistency with the initial term structure.

## 5.3 Calibration Window Selection

Hull-White dynamic parameters $a$ and $\sigma$ are estimated historically using a short-rate proxy from ECB data. Several windows were compared.

| Calibration Window | $a$ | $\sigma$ | Half-Life |
|---|---:|---:|---:|
| Full sample | 0.031719 | 0.004372 | 21.85 years |
| Last 10Y | 0.003964 | 0.004009 | 174.86 years |
| Last 5Y | 0.296758 | 0.004991 | 2.34 years |
| Post-2022 | 0.549645 | 0.005222 | 1.26 years |

The final baseline uses the Last 5Y calibration window:

| Parameter | Value |
|---|---:|
| $a$ | 0.296758 |
| $\sigma$ | 0.004991 |
| $r_0$ | 0.022823 |
| Half-life | 2.34 years |

The half-life of a mean-reverting process is:

$$
t_{1/2}=\frac{\ln(2)}{a}.
$$

The Last 5Y window is selected because it provides a more plausible mean-reversion speed for a 5Y swap exposure horizon than the full-sample or 10Y estimates.

## 5.4 Hull-White Simulation

The short rate is simulated using an Euler scheme:

$$r_{t+\Delta t}=
r_t + [\theta(t)-ar_t]\Delta t + \sigma\sqrt{\Delta t}Z_t,
$$

where:

$$
Z_t \sim N(0,1).
$$

The simulation uses:

| Setting | Value |
|---|---:|
| Horizon | 5 years |
| Time steps | 1260 |
| Paths | 10,000 |
| Step size | approximately daily |
| Random seed | 42 |

The Gaussian shocks are stored and later reused in the wrong-way risk extension.

## 6. Swap Pricing Methodology

## 6.1 Swap Definition

The derivative is a plain-vanilla fixed-for-floating payer interest rate swap.

A payer swap means:

$$
\text{pay fixed, receive floating}.
$$

The swap value from the bank perspective is:

$$
V(t)=V_{\text{float}}(t)-V_{\text{fixed}}(t).
$$

The modeled swap has:

| Swap Feature | Value |
|---|---:|
| Notional | 10,000,000 |
| Maturity | 5 years |
| Payment frequency | Semiannual |
| Swap type | Payer |

## 6.2 Par Swap Rate

The fixed rate is set to the par swap rate so that the initial risk-free value is zero.

The par rate is:

$$K_{\text{par}}=
\frac{1-P(0,T_n)}{\sum_{i=1}^{n}\alpha_iP(0,T_i)}.
$$

where:

| Symbol | Meaning |
|---|---|
| $K_{\text{par}}$ | par fixed swap rate |
| $P(0,T_i)$ | discount factor to payment date $T_i$ |
| $T_n$ | final maturity |
| $\alpha_i$ | accrual factor for period $i$ |

For a par payer swap:

$$V_0=N\left[(1-P(0,T_n))-K_{\text{par}}\sum_i \alpha_iP(0,T_i)\right]=
0.
$$

## 6.3 Conditional Hull-White Bond Pricing

At each future simulation date, the swap is repriced using conditional Hull-White zero-coupon prices:

$$
P(t,T)=A(t,T)e^{-B(t,T)r_t},
$$

with:

$$
B(t,T)=\frac{1-e^{-a(T-t)}}{a}.
$$

The initial condition is enforced exactly:

$$
P(0,T)=\text{initial curve discount factor}.
$$

This is important because using realized pathwise discount factors instead of conditional bond prices would make the time-zero swap value incorrectly vary across Monte Carlo paths.

## 6.4 Swap Pricing Sanity Checks

For the par swap:

| Check | Result |
|---|---:|
| Initial mean value | 0.000000 |
| Initial standard deviation across paths | 0.000000 |

An off-market test was also run by increasing the fixed rate by 50 bps. For a payer swap, paying above-market fixed should make the swap negative at inception.

| Off-Market Test | Result |
|---|---:|
| Initial mean value | -233,415.35 |
| Initial min value | -233,415.35 |
| Initial max value | -233,415.35 |
| Initial standard deviation | 0.000000 |

This confirms that the model does not artificially force all initial swap values to zero. Only the at-market par swap starts at zero.

## 7. Exposure Simulation

For each Monte Carlo path $m$ and time $t_i$, the model computes a simulated swap value:

$$
V_m(t_i).
$$

## 7.1 Positive Exposure

Positive exposure is:

$$
E_m(t_i)=\max(V_m(t_i),0).
$$

Positive exposure matters for CVA because the bank loses if the counterparty defaults while the trade has positive value to the bank.

## 7.2 Negative Exposure

Negative exposure is:

$$
NE_m(t_i)=\max(-V_m(t_i),0).
$$

Negative exposure matters for DVA because the bank owes value to the counterparty when the trade is negative.

## 7.3 Expected Exposure

Expected Exposure is the Monte Carlo average of positive exposure:

$$
EE(t_i)=\mathbb{E}[E(t_i)].
$$

## 7.4 Expected Negative Exposure

Expected Negative Exposure is:

$$
ENE(t_i)=\mathbb{E}[NE(t_i)].
$$

## 7.5 Potential Future Exposure

Potential Future Exposure at confidence level $q$ is the cross-sectional quantile:

$$
PFE_q(t_i)=Q_q(E_m(t_i)).
$$

## 7.6 Uncollateralized Exposure Results

| Metric | Value |
|---|---:|
| Initial swap value | 0.000000 |
| Average EE | 19,500.05 |
| Average ENE | 56,792.35 |
| Maximum positive exposure | 435,876.29 |
| Maximum negative exposure | 580,806.84 |
| Peak PFE 95% | 202,340.97 |
| Peak PFE 99% | 267,424.12 |

The expected negative exposure is larger than the expected positive exposure under the simulated dynamics. This later explains why DVA is larger than CVA in the base case.

## Suggested Plot: Exposure Profile

Recommended figure:

- x-axis: time in years,
- y-axis: exposure,
- lines: EE, PFE 95%, PFE 99%.

Commentary:

The plot shows the expected and tail positive exposure of the swap through time. Exposure is low at inception because the swap starts at-market, then grows as rates move and the swap acquires value. Exposure eventually declines as the swap approaches maturity.

## 8. Collateral Modeling

## 8.1 Motivation

Collateral reduces counterparty exposure. Without collateral:

$$
E(t)=\max(V(t),0).
$$

With collateral:

$$
E_c(t)=\max(V(t)-C(t),0),
$$

where $C(t)$ is the collateral balance.

## 8.2 Collateral Sign Convention

| Collateral Balance | Meaning |
|---:|---|
| $C(t)>0$ | collateral held by the bank |
| $C(t)<0$ | collateral posted by the bank |
| $C(t)=0$ | no collateral balance |

## 8.3 Collateral Agreement

The base collateral agreement is:

| Parameter | Value | Interpretation |
|---|---:|---|
| Threshold | 100,000 | first 100k exposure remains unsecured |
| Minimum transfer amount | 10,000 | collateral moves only if change exceeds 10k |
| Margin lag | 5 steps | collateral is based on stale mark-to-market |
| Bilateral | True | both parties post collateral |

The collateral target is based on lagged mark-to-market. If the lag is $h$ steps, collateral at time $t_i$ is based on $V(t_{i-h})$.

For positive lagged value:

$$
C(t_i)=\max(V(t_{i-h})-H,0),
$$

where $H$ is the threshold.

For negative lagged value under bilateral collateralization:

$$
C(t_i)=-\max(-V(t_{i-h})-H,0).
$$

## 8.4 Collateralized Exposures

Collateralized positive exposure is:

$$
E_c(t)=\max(V(t)-C(t),0).
$$

Collateralized negative exposure is:

$$
NE_c(t)=\max(-V(t)+C(t),0).
$$

These become the exposure inputs for CVA and DVA.

## 8.5 Collateral Results

| Metric | Before Collateral | After Collateral |
|---|---:|---:|
| Average EE | 19,500.05 | 16,603.19 |
| Average ENE | 56,792.35 | 45,832.65 |
| Peak PFE 95% | 202,340.97 | 198,506.08 |
| Peak PFE 99% | 267,424.12 | 231,106.03 |

Collateral reduces exposure but does not eliminate it because the agreement includes a threshold, minimum transfer amount, and margin lag.

## 8.6 Perfect Collateral Sanity Check

A perfect collateral test was run with:

| Parameter | Value |
|---|---:|
| Threshold | 0 |
| Minimum transfer amount | 0 |
| Margin lag | 0 |
| Bilateral | True |

Result:

| Check | Value |
|---|---:|
| Max perfect collateral positive exposure | 0.00000000 |
| Max perfect collateral negative exposure | 0.00000000 |

This confirms that remaining exposure under the base collateral agreement is caused by collateral frictions rather than a coding error.

## Suggested Plot: Collateral Impact

Recommended figures:

1. Uncollateralized EE vs collateralized EE.
2. Uncollateralized ENE vs collateralized ENE.
3. Uncollateralized PFE vs collateralized PFE.

Commentary:

Collateral reduces both expected and tail exposure. The reduction is larger in regions where exposure exceeds the threshold. Since the average EE is below the 100k threshold, collateral has a limited effect on average positive exposure but a more visible effect on high-percentile exposure.

## 9. Credit Curve Construction

## 9.1 Spread-to-Hazard Approximation

Credit spreads are converted into approximate hazard rates using:

$$
\lambda \approx \frac{s}{1-R},
$$

where:

| Symbol | Meaning |
|---|---|
| $\lambda$ | hazard rate or default intensity |
| $s$ | credit spread in decimal form |
| $R$ | recovery rate |
| $1-R$ | loss given default |

The baseline recovery rate is:

$$
R=40\%,
$$

so:

$$
LGD=1-R=60\%.
$$

This is a simplified approximation rather than full CDS bootstrapping.

## 9.2 Piecewise-Constant Hazard Rates

The hazard curve is piecewise constant:

$$
\lambda(t)=\lambda_i,
\quad
T_{i-1}<t\leq T_i.
$$

This means the hazard rate is constant inside each maturity bucket and jumps at bucket boundaries.

## 9.3 Survival Probability

Survival probability is:

$$
Q(0,t)=\exp\left(-\int_0^t \lambda(u)du\right).
$$

For piecewise-constant hazard rates:

$$
Q(0,t_n)=\exp\left(-\sum_i \lambda_i\Delta t_i\right).
$$

## 9.4 Cumulative Default Probability

Cumulative default probability is:

$$
PD(0,t)=1-Q(0,t).
$$

## 9.5 Marginal Default Probability

The marginal default probability over interval $[t_{i-1},t_i]$ is:

$$
PD(t_{i-1},t_i)=Q(0,t_{i-1})-Q(0,t_i).
$$

This is the default probability used in the CVA/DVA sums.

## 9.6 Hazard Rate Results

Counterparty hazard rates:

| Maturity | Spread | Hazard Rate |
|---:|---:|---:|
| 1Y | 50 bps | 0.008333 |
| 3Y | 80 bps | 0.013333 |
| 5Y | 120 bps | 0.020000 |
| 7Y | 150 bps | 0.025000 |
| 10Y | 180 bps | 0.030000 |

Self hazard rates:

| Maturity | Spread | Hazard Rate |
|---:|---:|---:|
| 1Y | 40 bps | 0.006667 |
| 3Y | 65 bps | 0.010833 |
| 5Y | 100 bps | 0.016667 |
| 7Y | 130 bps | 0.021667 |
| 10Y | 160 bps | 0.026667 |

## 9.7 Credit Curve Results

| Time | Counterparty Cumulative PD | Self Cumulative PD |
|---:|---:|---:|
| 1Y | 0.8299% | 0.6644% |
| 3Y | 3.4395% | 2.7936% |
| 5Y | 7.2257% | 5.9804% |

Sanity checks:

| Check | Result |
|---|---:|
| Counterparty survival at $t=0$ | 1.000000 |
| Counterparty survival at final time | 0.927743 |
| Counterparty cumulative PD at final time | 0.072257 |
| Minimum marginal PD | 0.0000000000 |
| Maximum marginal PD | 0.0000766323 |

## Suggested Plot: Survival and Default Probability Curves

Recommended figures:

1. Counterparty and self survival probabilities.
2. Counterparty and self cumulative default probabilities.

Commentary:

Survival probabilities decline over time, while cumulative default probabilities increase. The counterparty cumulative PD is higher than the self cumulative PD because counterparty spreads are larger in the base credit scenario.

## 10. CVA and DVA Methodology

## 10.1 CVA Intuition

Credit Valuation Adjustment is the expected discounted loss caused by counterparty default.

CVA uses positive exposure because the bank loses only when the trade has positive value and the counterparty defaults.

## 10.2 DVA Intuition

Debit Valuation Adjustment is the valuation effect of the bank's own default risk.

DVA uses negative exposure because the bank benefits, from a valuation perspective, when it may default while owing money to the counterparty.

This effect is economically controversial but standard in bilateral valuation.

## 10.3 Unilateral CVA

Unilateral CVA is:

$$
CVA
=
(1-R_C)
\sum_i
DF(0,t_i)
EE_c(t_i)
PD_C(t_{i-1},t_i),
$$

where:

| Symbol | Meaning |
|---|---|
| $R_C$ | counterparty recovery rate |
| $DF(0,t_i)$ | discount factor to time $t_i$ |
| $EE_c(t_i)$ | collateralized expected positive exposure |
| $PD_C(t_{i-1},t_i)$ | counterparty marginal default probability |

## 10.4 Unilateral DVA

Unilateral DVA is:

$$DVA=
(1-R_B)
\sum_i
DF(0,t_i)
ENE_c(t_i)
PD_B(t_{i-1},t_i),
$$

where:

| Symbol | Meaning |
|---|---|
| $R_B$ | bank or self recovery rate |
| $ENE_c(t_i)$ | collateralized expected negative exposure |
| $PD_B(t_{i-1},t_i)$ | bank or self marginal default probability |

## 10.5 Bilateral First-to-Default Adjustment

In a bilateral setting, only the first default matters. If the bank defaults first, the counterparty cannot later default and generate CVA for the bank. If the counterparty defaults first, the bank cannot later default and generate DVA.

The first-to-default CVA formula is:

$$
CVA_{FTD}=
(1-R_C)
\sum_i
DF(0,t_i)
EE_c(t_i)
PD_C(t_{i-1},t_i)
Q_B(0,t_i).
$$

The first-to-default DVA formula is:

$$
DVA_{FTD}=
(1-R_B)
\sum_i
DF(0,t_i)
ENE_c(t_i)
PD_B(t_{i-1},t_i)
Q_C(0,t_i).
$$

where:

| Symbol | Meaning |
|---|---|
| $Q_B(0,t_i)$ | survival probability of the bank |
| $Q_C(0,t_i)$ | survival probability of the counterparty |

The survival term of the other party prevents double counting after first default.

## 10.6 Adjusted Value

The CVA/DVA-adjusted value is:

$$
V_{adjusted}=V_{risk\text{-}free}-
CVA_{FTD}
+
DVA_{FTD}.
$$

Since the swap is initially at-market:

$$
V_{risk\text{-}free}=0.
$$

Therefore:

$$
V_{adjusted}=-CVA_{FTD}+DVA_{FTD}.
$$

## 11. Base XVA Results

## 11.1 Unilateral Results

| Metric | Value |
|---|---:|
| Unilateral CVA | 644.45 |
| Unilateral DVA | 1,552.89 |

## 11.2 First-to-Default Results

| Metric | Value |
|---|---:|
| CVA | 630.33 |
| DVA | 1,492.87 |
| Net XVA adjustment $(-CVA+DVA)$ | 862.54 |
| Risk-free value | 0.00 |
| CVA/DVA-adjusted value | 862.54 |

## 11.3 Unilateral vs First-to-Default

| Metric | Value |
|---|---:|
| Unilateral CVA | 644.45 |
| First-to-default CVA | 630.33 |
| Unilateral DVA | 1,552.89 |
| First-to-default DVA | 1,492.87 |
| FTD CVA reduction | 14.12 |
| FTD DVA reduction | 60.02 |

The first-to-default adjustments are lower than the unilateral adjustments because each party's default contribution is weighted by the survival probability of the other party.

## 11.4 Interpretation

DVA is larger than CVA because collateralized negative exposure is much larger than collateralized positive exposure.

| Exposure Metric | Average Value |
|---|---:|
| Collateralized EE | 16,603.19 |
| Collateralized ENE | 45,832.65 |

Even though the bank's own spreads are lower than the counterparty spreads, the negative exposure profile dominates. Therefore:

$$
DVA > CVA,
$$

and:

$$
-CVA + DVA > 0.
$$

## Suggested Plot: Cumulative CVA and DVA

Recommended figure:

- x-axis: time in years,
- y-axis: cumulative value adjustment,
- lines: cumulative CVA and cumulative DVA.

Commentary:

The cumulative contribution plot shows how CVA and DVA build through time. Each daily contribution is small, but the cumulative adjustment increases over the life of the swap. DVA accumulates more than CVA because expected negative exposure is larger than expected positive exposure.

## 12. Stress Testing

## 12.1 Stress Scenario Design

The project compares base and stressed assumptions for credit spreads and collateral terms.

| Scenario | Counterparty Curve | Self Curve | Collateral Terms |
|---|---|---|---|
| Base | Base | Base | Base CSA |
| Counterparty credit stress | Stress | Base | Base CSA |
| Self credit stress | Base | Stress | Base CSA |
| Joint credit stress | Stress | Stress | Base CSA |
| Collateral stress | Base | Base | Stressed CSA |
| Joint credit + collateral stress | Stress | Stress | Stressed CSA |

The base CSA is:

| Parameter | Value |
|---|---:|
| Threshold | 100,000 |
| Minimum transfer amount | 10,000 |
| Margin lag | 5 steps |

The stressed CSA is:

| Parameter | Value |
|---|---:|
| Threshold | 250,000 |
| Minimum transfer amount | 25,000 |
| Margin lag | 10 steps |

## 12.2 Stress Results

| Scenario | Avg Coll. EE | Avg Coll. ENE | Peak Coll. PFE 99% | CVA | DVA | Net XVA | Adjusted Value |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 16,603.19 | 45,832.65 | 231,106.03 | 630.33 | 1,492.87 | 862.54 | 862.54 |
| Counterparty credit stress | 16,603.19 | 45,832.65 | 231,106.03 | 1,348.23 | 1,423.18 | 74.96 | 74.96 |
| Self credit stress | 16,603.19 | 45,832.65 | 231,106.03 | 611.97 | 3,254.12 | 2,642.15 | 2,642.15 |
| Joint credit stress | 16,603.19 | 45,832.65 | 231,106.03 | 1,310.31 | 3,106.99 | 1,796.68 | 1,796.68 |
| Collateral stress | 19,367.04 | 56,369.79 | 266,432.79 | 733.66 | 1,810.14 | 1,076.48 | 1,076.48 |
| Joint credit + collateral stress | 19,367.04 | 56,369.79 | 266,432.79 | 1,528.96 | 3,792.90 | 2,263.94 | 2,263.94 |

## 12.3 Stress Impacts Relative to Base

| Scenario | Delta CVA | Delta DVA | Delta Net XVA | Delta Adjusted Value |
|---|---:|---:|---:|---:|
| Base | 0.00 | 0.00 | 0.00 | 0.00 |
| Counterparty credit stress | 717.89 | -69.69 | -787.58 | -787.58 |
| Self credit stress | -18.36 | 1,761.25 | 1,779.62 | 1,779.62 |
| Joint credit stress | 679.98 | 1,614.12 | 934.14 | 934.14 |
| Collateral stress | 103.33 | 317.27 | 213.94 | 213.94 |
| Joint credit + collateral stress | 898.63 | 2,300.03 | 1,401.40 | 1,401.40 |

## 12.4 Interpretation

Counterparty credit stress increases CVA because counterparty marginal default probabilities increase.

Self credit stress increases DVA because own marginal default probabilities increase.

Collateral stress increases both CVA and DVA because worse collateral terms increase both collateralized positive and negative exposure.

Some cross-effects appear because of the first-to-default adjustment. Counterparty stress lowers counterparty survival probabilities, which can reduce DVA because DVA requires the counterparty to survive until the bank defaults. Self stress lowers bank survival probabilities, which can reduce CVA because CVA requires the bank to survive until the counterparty defaults.

## Suggested Plots: Stress Scenarios

Recommended figures:

1. CVA by scenario.
2. DVA by scenario.
3. Adjusted value by scenario.
4. Delta CVA and delta DVA relative to base.

Commentary:

These plots show that counterparty spread widening mainly affects CVA, own spread widening mainly affects DVA, and collateral deterioration increases exposure-driven valuation adjustments.

## 13. Wrong-Way Risk Extension

## 13.1 Motivation

The base CVA model assumes exposure and counterparty default risk are independent:

$$
EE_c(t_i) \times PD_C(t_{i-1},t_i).
$$

Wrong-way risk relaxes this assumption. It appears when exposure is high in scenarios where the counterparty is also more likely to default.

Under wrong-way risk, the relevant quantity becomes:

$$
\mathbb{E}\left[E_{c,m}(t_i)PD_{C,m}(t_{i-1},t_i)\right].
$$

This is not generally equal to:

$$
\mathbb{E}[E_c(t_i)]\mathbb{E}[PD_C(t_{i-1},t_i)].
$$

## 13.2 Stochastic Counterparty Intensity

The project models path-dependent counterparty hazard rates:

$$\lambda_{C,m}(t_i)=\lambda_C(t_i)\exp\left(\eta\sqrt{\Delta t}Z^{\lambda}_{m,i}-
\frac{1}{2}\eta^2\Delta t
\right),
$$

where:

| Symbol | Meaning |
|---|---|
| $\lambda_{C,m}(t_i)$ | pathwise counterparty hazard rate |
| $\lambda_C(t_i)$ | deterministic base hazard rate |
| $\eta$ | hazard volatility |
| $Z^{\lambda}_{m,i}$ | credit intensity shock |

The credit shock is correlated with the interest-rate shock:

$$Z^{\lambda}_{m,i}=
\rho Z^r_{m,i}
+
\sqrt{1-\rho^2}Z^{\perp}_{m,i}.
$$

where:

| Symbol | Meaning |
|---|---|
| $\rho$ | correlation between rate shocks and credit intensity shocks |
| $Z^r$ | Hull-White rate shock |
| $Z^{\perp}$ | independent standard normal shock |

## 13.3 Pathwise Survival and Default Probabilities

Pathwise survival is:

$$
Q_{C,m}(0,t_i)=
\exp\left(-\sum_{j=1}^{i}\lambda_{C,m}(t_j)\Delta t_j\right).
$$

Pathwise marginal default probability is:

$$
PD_{C,m}(t_{i-1},t_i)=
Q_{C,m}(0,t_{i-1})-Q_{C,m}(0,t_i).
$$

## 13.4 Wrong-Way CVA Formula

The wrong-way CVA approximation is:

$$
CVA^{WWR}=
(1-R_C)
\sum_i
DF(0,t_i)
\mathbb{E}\left[
E_{c,m}(t_i)PD_{C,m}(t_{i-1},t_i)
\right]
Q_B(0,t_i).
$$

## 13.5 Wrong-Way Risk Results

For $\rho=0.50$:

| Metric | Value |
|---|---:|
| Base first-to-default CVA | 630.33 |
| Wrong-way-risk CVA | 635.33 |
| Incremental WWR CVA vs deterministic base | 4.99 |
| Relative WWR impact vs deterministic base | 0.7923% |

## 13.6 Correlation Sensitivity

| Correlation | Base CVA | WWR CVA | Incremental vs Base | Relative vs Base | Incremental vs Zero Corr | Relative vs Zero Corr |
|---:|---:|---:|---:|---:|---:|---:|
| -0.75 | 630.33 | 627.20 | -3.14 | -0.4978% | -5.59 | -0.8836% |
| -0.50 | 630.33 | 629.18 | -1.16 | -0.1835% | -3.61 | -0.5706% |
| -0.25 | 630.33 | 631.06 | 0.73 | 0.1153% | -1.73 | -0.2730% |
| 0.00 | 630.33 | 632.79 | 2.45 | 0.3893% | 0.00 | 0.0000% |
| 0.25 | 630.33 | 634.26 | 3.93 | 0.6231% | 1.47 | 0.2329% |
| 0.50 | 630.33 | 635.33 | 4.99 | 0.7923% | 2.54 | 0.4014% |
| 0.75 | 630.33 | 635.67 | 5.33 | 0.8460% | 2.88 | 0.4549% |

## 13.7 Interpretation

The deterministic base CVA is 630.33. When counterparty intensity is stochastic but uncorrelated with rate shocks, CVA becomes 632.79. This difference reflects stochastic-intensity effects, not pure wrong-way risk.

To isolate the correlation effect, the project compares each scenario to the zero-correlation stochastic-intensity case:

$$
\Delta CVA_{\rho}=CVA(\rho)-CVA(0).
$$

Positive correlations increase CVA relative to the zero-correlation stochastic baseline. Negative correlations reduce CVA. Therefore, in this swap setup, positive rate-credit correlation produces wrong-way risk and negative correlation produces right-way risk.

## Suggested Plot: WWR Correlation Sensitivity

Recommended figure:

- x-axis: correlation $\rho$,
- y-axis: WWR CVA,
- horizontal line: deterministic base CVA.

Commentary:

The plot shows that CVA increases monotonically with the rate-credit correlation parameter. The zero-correlation stochastic-intensity CVA is slightly above the deterministic CVA, while positive correlations add pure wrong-way risk.

## 14. EVT Tail Exposure Extension

## 14.1 Motivation

Empirical Monte Carlo PFE becomes noisy at very high confidence levels. With 10,000 paths, the 99.9% empirical quantile is determined by roughly the top 10 observations.

Extreme Value Theory provides a tail-smoothing method by fitting a Generalized Pareto Distribution to exceedances above a high threshold.

## 14.2 Peaks-over-Threshold Setup

At each time $t_i$, define collateralized positive exposure observations:

$$
E_{c,1}(t_i),E_{c,2}(t_i),\ldots,E_{c,M}(t_i).
$$

Choose a high threshold:

$$
u(t_i)=Q_{0.95}(E_c(t_i)).
$$

Define exceedances:

$$
Y_m(t_i)=E_{c,m}(t_i)-u(t_i)
\quad \text{conditional on} \quad E_{c,m}(t_i)>u(t_i).
$$

The exceedances are modeled with a GPD:

$$
Y \sim GPD(\xi,\beta),
$$

where:

| Symbol | Meaning |
|---|---|
| $\xi$ | GPD shape parameter |
| $\beta$ | GPD scale parameter |
| $u$ | threshold |

## 14.3 EVT Quantile Formula

Let $p_u$ be the empirical probability of exceeding the threshold:

$$
p_u=P(E_c>u).
$$

For target quantile $q$, the EVT quantile is:

$$
x_q=
u+
\frac{\beta}{\xi}
\left[
\left(\frac{1-q}{p_u}\right)^{-\xi}-1
\right].
$$

When $\xi \approx 0$, the exponential limit is used:

$$
x_q=
u+
\beta\log\left(\frac{p_u}{1-q}\right).
$$

## 14.4 EVT Results

| Target Quantile | Peak Empirical PFE | Peak EVT PFE | EVT Minus Empirical |
|---:|---:|---:|---:|
| 99.5% | 239,159.53 | 241,445.97 | 2,286.45 |
| 99.9% | 252,244.88 | 257,267.07 | 5,022.19 |

For 99.9% PFE, EVT gives a tail estimate approximately 5,022 higher than the empirical Monte Carlo estimate, or about 2.0% higher.

## 14.5 GPD Shape Interpretation

The fitted GPD shape parameter is mostly negative:

$$
\xi < 0.
$$

Interpretation:

| Shape | Tail Type |
|---:|---|
| $\xi>0$ | heavy tail |
| $\xi=0$ | exponential-type tail |
| $\xi<0$ | bounded tail |

The mostly negative shape is plausible because the exposure comes from a finite-maturity interest rate swap under Gaussian one-factor Hull-White dynamics with collateralization.

## 14.6 Payment-Date Effects

The empirical and EVT PFE profiles show spikes around semiannual payment dates:

$$
0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0.
$$

These spikes align with the swap cashflow schedule. They occur because the remaining fixed and floating cashflows change discretely at payment dates.

EVT is fitted independently at each time point, so it preserves these exposure discontinuities rather than smoothing them away.

## Suggested Plots: EVT

Recommended figures:

1. Empirical vs EVT PFE at 99.5%.
2. Empirical vs EVT PFE at 99.9%.
3. GPD shape parameter over time.
4. Empirical vs EVT 99.9% PFE with vertical payment-date markers.

Commentary:

The EVT curves are slightly above empirical PFE at the peak, giving a modestly more conservative tail estimate. The payment-date overlay confirms that sharp PFE movements are driven by swap cashflow mechanics rather than numerical instability.

## 15. Model Validation Checks

The notebook includes validation checks for the core modeling logic.

| Validation Check | Expected Result |
|---|---|
| Par swap initial value close to zero | True |
| Initial value identical across paths | True |
| Perfect collateral eliminates positive exposure | True |
| Perfect collateral eliminates negative exposure | True |
| Counterparty survival starts at one | True |
| Counterparty survival decreases by maturity | True |
| First-to-default CVA below unilateral CVA | True |
| First-to-default DVA below unilateral DVA | True |
| Counterparty stress increases CVA | True |
| Self stress increases DVA | True |
| Positive WWR correlation increases CVA vs zero correlation | True |

These checks help ensure that the numerical implementation is consistent with the intended financial logic.

## 16. Final Results Summary

| Category | Metric | Value |
|---|---|---:|
| Swap | Risk-free initial value | 0.00 |
| Exposure | Average uncollateralized EE | 19,500.05 |
| Exposure | Average collateralized EE | 16,603.19 |
| Exposure | Peak collateralized PFE 99% | 231,106.03 |
| XVA | Base first-to-default CVA | 630.33 |
| XVA | Base first-to-default DVA | 1,492.87 |
| XVA | Base adjusted value | 862.54 |
| WWR | WWR CVA at $\rho=0.50$ | 635.33 |
| WWR | Pure WWR impact at $\rho=0.50$ | 2.54 |
| EVT | Peak EVT PFE 99.9% | 257,267.07 |

## 17. How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```text
notebooks/02_hull_white_simulation.ipynb
```

Run tests:

```bash
pytest tests
```

The notebook saves key output tables into:

```text
outputs/
```

## 18. Model Limitations

This project is a transparent research implementation rather than a production XVA system.

### 18.1 Credit Curves

Credit curves are based on stylized spread scenarios and the approximation:

$$
\lambda \approx \frac{s}{1-R}.
$$

They are not bootstrapped from market CDS quotes.

### 18.2 Interest Rate Dynamics

The exposure engine uses a one-factor Hull-White model with historically estimated parameters. A production implementation would typically calibrate to swaptions or caps/floors and may require multi-factor dynamics.

### 18.3 Swap Valuation

The swap is priced in a simplified single-curve framework. A production swap pricer would use OIS discounting, forward curves, full schedule generation, day-count conventions, fixing rules, and business-day calendars.

### 18.4 Collateral Modeling

The collateral module includes threshold, minimum transfer amount, bilateral posting, and margin lag. It does not include initial margin, collateral haircuts, dispute periods, eligible collateral schedules, rehypothecation, or collateral remuneration.

### 18.5 CVA/DVA Methodology

CVA/DVA are computed on a discrete time grid using simplified first-to-default formulas. The model does not include detailed closeout mechanics, settlement delays, default-time interpolation, or portfolio-level netting.

### 18.6 Wrong-Way Risk

Wrong-way risk is modeled through a stylized lognormal stochastic counterparty intensity correlated with interest-rate shocks. The hazard volatility and correlation are scenario assumptions rather than calibrated market parameters.

### 18.7 EVT Tail Exposure

EVT tail estimates depend on the threshold choice and are fitted independently at each time point. Future improvements could include threshold stability diagnostics, mean excess plots, and bootstrap confidence intervals.

## 19. Future Research Directions

The following extensions would improve the realism and research depth of the project.

## 19.1 CDS Curve Bootstrapping

Replace the spread-to-hazard approximation with full CDS curve bootstrapping.

This would model:

- premium leg,
- protection leg,
- accrual on default,
- discounting,
- CDS payment dates,
- recovery assumptions,
- survival curve bootstrapping.

This is the most important credit-modeling upgrade.

## 19.2 Swaption-Calibrated Hull-White Model

Replace historical calibration of $a$ and $\sigma$ with calibration to market-implied swaption volatilities.

This would make the exposure simulation more consistent with derivative market prices.

## 19.3 Multi-Curve Swap Pricing

Extend the swap pricer to separate:

- OIS discount curve,
- Euribor forward curve,
- basis spreads.

This would be closer to modern collateralized IRS valuation.

## 19.4 Initial Margin Modeling

Add a stylized initial margin model:

$$
E_{c,IM}(t)=\max(V(t)-VM(t)-IM(t),0).
$$

Possible approaches:

- VaR-style initial margin,
- expected shortfall initial margin,
- MPOR-based exposure changes,
- simplified SIMM-style proxy.

## 19.5 Netting Set Portfolio

Extend from a single swap to a portfolio of swaps and compare:

$$
\sum_j \max(V_j(t),0)
$$

with:

$$
\max\left(\sum_j V_j(t),0\right).
$$

This would demonstrate netting benefits, which are central to real CVA.

## 19.6 Two-Factor Interest Rate Models

Upgrade from one-factor Hull-White to G2++:

$$
r_t=x_t+y_t+\phi(t).
$$

This would better capture level, slope, and curvature movements in the yield curve.

## 19.7 Stochastic Recovery

Replace fixed recovery with stochastic recovery:

$$
R_t \sim \text{Beta}(\alpha,\beta),
$$

or make recovery negatively correlated with default intensity.

## 19.8 EVT Robustness Diagnostics

Add:

- threshold sensitivity,
- mean excess plots,
- GPD parameter stability plots,
- bootstrap confidence intervals,
- EVT expected shortfall.

## 19.9 Regulatory CVA Capital

Implement a simplified regulatory CVA capital calculation, such as a stylized SA-CVA approximation, and compare valuation CVA with capital-oriented CVA risk measures.

## 19.10 Software Engineering Improvements

Future engineering improvements include:

- fuller unit test coverage,
- configuration files,
- command-line scripts,
- automated report generation,
- continuous integration,
- reproducible environment files.

## 20. Interpretation of the Project

This project should be interpreted as a transparent quantitative research implementation of collateralized bilateral CVA/DVA for an interest rate swap. It is modular enough to be extended, explicit enough to be audited, and broad enough to demonstrate the interaction between interest-rate exposure, collateral, credit curves, first-to-default mechanics, stress testing, wrong-way risk, and tail exposure estimation.

It is not a production XVA engine, but it provides the mathematical and computational foundation for one.
