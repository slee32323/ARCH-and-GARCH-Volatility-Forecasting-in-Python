# ARCH and GARCH Volatility Forecasting in Python

**An Empirical Time Series Analysis of Monthly Revenue Growth, Conditional Heteroskedasticity, and Out-of-Sample Volatility Forecasting**

## 1. Project Overview

This project investigates monthly revenue growth volatility using Autoregressive Conditional Heteroskedasticity (ARCH) and Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models.

The primary objective is to evaluate whether time-varying volatility models provide more accurate forecasts than a constant-variance benchmark.

Using monthly revenue data from January 2016 through December 2025, the analysis examines stationarity, conditional heteroskedasticity, volatility persistence, and out-of-sample forecasting performance.

The project also generates six-month-ahead volatility forecasts for January through June 2026.

**Research objectives:**

- Examine the statistical properties of monthly revenue growth.
- Test for stationarity using the Augmented Dickey-Fuller (ADF) test.
- Investigate ARCH effects using the ARCH-LM test.
- Estimate ARCH(1) and GARCH(1,1) volatility models.
- Evaluate in-sample model fit using AIC and BIC.
- Compare out-of-sample variance forecasts using mean squared error (MSE).
- Assess whether GARCH models outperform a constant-variance benchmark.
- Generate six-month volatility forecasts for 2026.

---

## 2. Dataset

The dataset contains monthly revenue observations denominated in Egyptian pounds (EGP).

| Variable | Description |
|---|---|
| Date | Monthly observation date |
| Revenue_EGP | Monthly revenue in Egyptian pounds |
| Orders | Number of customer orders |
| Marketing_Spend_EGP | Monthly marketing expenditures |
| Returns_Count | Number of returned orders |
| Promotion | Promotional activity indicator |

**Sample characteristics:**

| Item | Description |
|---|---|
| Original analysis period | January 2016 – December 2025 |
| Data frequency | Monthly |
| Cleaned revenue observations | 120 |
| Log return observations | 119 |
| Training period | February 2016 – December 2024 |
| Training observations | 107 |
| Testing period | January – December 2025 |
| Testing observations | 12 |

Although the original dataset contains multiple business variables, the volatility models are estimated using monthly revenue log returns.

### Data Preprocessing

Data preparation involved:

1. Standardizing inconsistent date formats.
2. Removing duplicate monthly observations.
3. Converting revenue and other numerical columns into numeric data types.
4. Handling missing numerical observations using linear interpolation.
5. Organizing the dataset chronologically.
6. Constructing monthly log returns from cleaned revenue observations.

Monthly percentage log returns are calculated as:

\[
r_t = 100 \times \ln\left(\frac{Revenue_t}{Revenue_{t-1}}\right)
\]

This transformation allows the analysis to focus on proportional monthly revenue changes rather than revenue levels.

---

## 3. Methodology

### 3.1 Stationarity Testing

The Augmented Dickey-Fuller (ADF) test is applied to the training sample of monthly log returns.

**Hypotheses:**

- H0: The series contains a unit root.
- H1: The series does not contain a unit root.

**Results:**

| Statistic | Value |
|---|---:|
| ADF Statistic | -9.5401 |
| p-value | 2.74 × 10⁻¹⁶ |

The null hypothesis of a unit root is rejected at the 5% significance level.

The results provide strong statistical evidence against a unit root in the training return series, supporting the use of the return transformation for subsequent volatility modeling.

### 3.2 ARCH-LM Test

The ARCH-Lagrange Multiplier test examines whether lagged squared innovations help explain conditional variance.

The test is performed on demeaned training returns using six lags.

| Statistic | Value |
|---|---:|
| LM Statistic | 2.1421 |
| LM p-value | 0.9062 |
| F Statistic | 0.3395 |
| F p-value | 0.9143 |

At the 5% significance level, the null hypothesis of no ARCH effects is not rejected.

Therefore, the initial diagnostic test does not provide statistically significant evidence of ARCH effects in the training sample.

ARCH and GARCH models are nevertheless estimated to compare their empirical fit and forecasting performance.

---

## 4. Volatility Models

### 4.1 ARCH(1)

The ARCH(1) model expresses conditional variance as a function of the previous period's squared innovation:

\[
\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2
\]

Where:

- \(\sigma_t^2\): Conditional variance
- \(\omega\): Constant variance component
- \(\alpha_1\): Sensitivity to the previous squared innovation
- \(\epsilon_{t-1}^2\): Previous period's squared innovation

### 4.2 GARCH(1,1)

The GARCH(1,1) specification extends ARCH by incorporating lagged conditional variance:

\[
\sigma_t^2 =
\omega +
\alpha_1 \epsilon_{t-1}^2 +
\beta_1 \sigma_{t-1}^2
\]

Where:

- \(\alpha_1\): ARCH coefficient measuring the influence of recent shocks
- \(\beta_1\): GARCH coefficient measuring volatility persistence
- \(\alpha_1+\beta_1\): A commonly used indicator of volatility persistence

Both models use a constant conditional mean and normally distributed innovations.

### Training-Sample Parameter Estimates

| Parameter | ARCH(1) | GARCH(1,1) |
|---|---:|---:|
| Mean (μ) | 1.9411 | 0.6294 |
| Omega (ω) | 42.4101 | 0.1200 |
| Alpha (α₁) | 0.3857 | ≈ 0 |
| Beta (β₁) | N/A | 0.9861 |

The GARCH model exhibits a high estimated persistence coefficient. However, its ARCH coefficient is approximately zero, indicating little estimated sensitivity to new squared innovations.

The original ARCH-LM test also did not find significant ARCH effects, so the estimated persistence should be interpreted cautiously.

---

## 5. Model Diagnostics

Standardized residuals from both fitted models were evaluated using ARCH-LM tests with six lags.

| Model | LM Statistic | p-value |
|---|---:|---:|
| ARCH(1) | 2.8425 | 0.8283 |
| GARCH(1,1) | 8.7030 | 0.1910 |

Neither model exhibits statistically significant residual ARCH effects at the 5% level.

However, because the original training returns also lacked significant detected ARCH effects, this result alone does not establish that either model successfully removed conditional heteroskedasticity.

---

## 6. Out-of-Sample Forecasting

The dataset is divided into training and testing samples to evaluate forecasting performance without using future observations in model estimation.

**Training:** February 2016 – December 2024

**Testing:** January – December 2025

ARCH(1) and GARCH(1,1) are fitted using only the training observations.

Each model produces 12-month-ahead conditional variance forecasts from a fixed forecasting origin at the end of December 2024.

Forecast volatility is obtained by taking the square root of the predicted conditional variance:

\[
\hat{\sigma}_{t+h}
=
\sqrt{\hat{\sigma}_{t+h}^2}
\]

### Forecast Behavior

The ARCH(1) model forecasts volatility rising from approximately 6.69 percentage points in January 2025 to 8.31 percentage points in December.

The GARCH(1,1) model forecasts a gradual decline from approximately 5.40 to 5.12 percentage points over the same period.

These forecasts describe the predicted dispersion of monthly revenue log returns, not the expected direction of revenue growth.

### Constant-Variance Benchmark

A constant-variance forecasting benchmark is constructed using the variance of the training sample.

| Measure | Estimate |
|---|---:|
| Constant Variance | 61.8157 |
| Constant Volatility | 7.8623 percentage points |

The benchmark predicts the same variance for each month in the 2025 testing period.

Its purpose is to determine whether more complex volatility models provide meaningful improvements over a simple, time-invariant forecasting approach.

---

## 7. Forecast Evaluation

Conditional variance is not directly observed.

Therefore, out-of-sample evaluation uses squared deviations of realized monthly returns from the training-sample mean as a proxy:

\[
RV_t=(r_t-\bar{r}_{train})^2
\]

The variance forecast loss is evaluated using mean squared error:

\[
MSE = \frac{1}{T}
\sum_{t=1}^{T}
(RV_t-\hat{\sigma}_t^2)^2
\]

where \(T=12\) represents the 12 monthly test observations.

Lower MSE indicates smaller average squared forecast errors relative to the selected variance proxy.

### Model Comparison

| Model | AIC | BIC | Out-of-Sample MSE |
|---|---:|---:|---:|
| ARCH(1) | 749.2898 | 757.3083 | 1986.9221 |
| **GARCH(1,1)** | **742.2965** | **752.9878** | **508.5217** |
| Constant Variance | N/A | N/A | 1736.4969 |

### Key Findings

**GARCH(1,1) achieves the lowest out-of-sample MSE among the three forecasting approaches.**

Its MSE is substantially lower than those of both ARCH(1) and the constant-variance benchmark.

GARCH(1,1) also achieves lower AIC and BIC than ARCH(1), indicating superior training-sample information-criterion performance between these two fitted models.

Interestingly, the constant-variance benchmark outperforms ARCH(1) in the out-of-sample evaluation.

This demonstrates that additional model complexity does not necessarily lead to better forecasting performance.

Nevertheless, these conclusions are specific to the evaluation period and loss function used in this study.

---

## 8. Final GARCH(1,1) Estimation

Following the out-of-sample evaluation, GARCH(1,1) is selected as the preferred forecasting model.

The model is re-estimated using all 119 monthly log returns from February 2016 through December 2025.

### Full-Sample Parameter Estimates

| Parameter | Estimate | p-value |
|---|---:|---:|
| Mean (μ) | 0.5471 | 0.383 |
| Omega (ω) | 0.1040 | 0.909 |
| Alpha (α₁) | ≈ 0 | 1.000 |
| Beta (β₁) | 0.9863 | < 0.001 |

The estimated GARCH coefficient indicates highly persistent fitted conditional variance, while the near-zero ARCH coefficient suggests minimal estimated dependence on the previous squared innovation.

The full-sample estimation is used exclusively for forecasting the subsequent six-month period.

---

## 9. January–June 2026 Volatility Forecast

Six-month-ahead conditional volatility forecasts are generated using the GARCH(1,1) model refitted through December 2025.

| Month | Forecast Volatility (Approximate Percentage Points) |
|---|---:|
| January 2026 | 5.058 |
| February 2026 | 5.033 |
| March 2026 | 5.009 |
| April 2026 | 4.985 |
| May 2026 | 4.961 |
| June 2026 | 4.938 |

The forecast suggests a gradual decline in the conditional standard deviation of monthly revenue log returns.

This pattern is consistent with a slowly adjusting conditional variance process under the fitted GARCH specification.

These values are forecasts generated using information available through December 2025, not observed realized volatility for 2026.

---

## 10. Limitations

Several limitations should be considered when interpreting the findings.

**Limited evidence of ARCH effects:** The initial ARCH-LM test does not identify statistically significant ARCH effects in the training sample.

**Near-zero ARCH coefficient:** The estimated GARCH(1,1) specification places almost no weight on the previous squared innovation, limiting evidence of time-varying shock responsiveness.

**Short testing period:** The out-of-sample evaluation includes only 12 monthly observations.

**Noisy variance proxy:** Squared monthly return deviations provide an imperfect proxy for unobserved conditional variance.

**Distributional assumptions:** Both volatility models assume normally distributed innovations.

**Single forecasting origin:** The evaluation uses a fixed-origin, 12-month forecasting exercise rather than repeated rolling-origin validation.

Future research could investigate alternative innovation distributions, expanding-window validation, additional volatility specifications, and larger datasets.

---

## 11. Technologies and Libraries

- Python
- JupyterLab
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- ARCH (`arch`)
- OpenPyXL

### Installation

```bash
pip install pandas numpy matplotlib statsmodels arch openpyxl
```

### Repository Files

```text
ARCH-and-GARCH-Volatility-Forecasting-in-Python/
│
├── ARCH_GARCH_Volatility_Forecasting.ipynb
├── Time_Series_Dataset.xlsx
└── README.md
```

### How to Run

1. Download or clone the repository.
2. Install the required Python libraries.
3. Open the Jupyter notebook in JupyterLab or Jupyter Notebook.
4. Ensure the input dataset is available locally.
5. Update the dataset file path in the notebook if necessary.
6. Execute the notebook cells sequentially.

The notebook documents the empirical workflow, fitted models, statistical test results, forecast comparisons, and volatility visualizations.

---

## 12. Conclusion

This project demonstrates an end-to-end empirical volatility forecasting workflow using monthly business revenue data.

Both ARCH(1) and GARCH(1,1) models were estimated and evaluated through diagnostic testing, information criteria, and out-of-sample variance forecasting.

GARCH(1,1) achieved the lowest AIC and BIC among the two conditional heteroskedasticity models and the lowest out-of-sample MSE among all three forecasting approaches.

After full-sample re-estimation, the selected model predicted a gradual decline in monthly revenue growth volatility during the first half of 2026.

However, the lack of significant initial ARCH effects, near-zero estimated ARCH coefficient, short testing period, and imperfect variance proxy limit the strength and generalizability of the conclusions.

Overall, the findings illustrate the importance of combining model estimation, statistical diagnostics, out-of-sample testing, and simple forecasting benchmarks when evaluating volatility models.

---

**Project Focus:** Time Series Econometrics | Conditional Volatility | ARCH/GARCH | Forecast Evaluation | Python
