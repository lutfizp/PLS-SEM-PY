## 1. Convergent Validity & Reliability

These metrics ensure that indicators meant to measure a single latent variable (construct) actually correlate strongly with each other.

**Code Location:** Results from `plspm_calc.unidimensionality()` and `plspm_calc.inner_summary()` are combined and displayed in `tab1`.

```python
# Inside main() function
rel = pd.concat([unidim[['cronbach_alpha', 'dillon_goldstein_rho']], inner_sum['ave']], axis=1)
rel.columns = ["Cronbach's Alpha", "Composite Reliability", "AVE"]
st.dataframe(rel.round(3).style.format("{:.3f}"))
```

**How it Works:**
- **`cronbach_alpha` (Cronbach's Alpha):** Calculated by the `plspm` library. This is the most common measure of internal consistency reliability. A good value is typically above 0.7.
- **`dillon_goldstein_rho` (Composite Reliability - CR):** Also calculated by `plspm`. This is preferred over Cronbach's Alpha in PLS-SEM because it doesn't assume all indicators have the same weight. The formula is based on the sum of squared outer loadings and error variance. A good value is typically above 0.7.
- **`ave` (Average Variance Extracted - AVE):** Calculated by `plspm`. It measures how much variance of the indicators is explained by the latent variable. The formula is the average of the squared outer loadings for each construct. Values should be above 0.5.

## 2. Outer Loadings

*Outer loadings* are the correlations between each indicator and its latent variable. This value shows how strongly each indicator represents the construct it measures.

**Code Location:** Results from `plspm_calc.outer_model()` are processed and displayed in `tab1`.

```python
# Inside main() function
loadings = plspm_calc.outer_model()
# ... (table pivot code for formatting)
outer_loadings_pivoted = loadings.pivot_table(index=loadings.index, columns='Variable', values='loading')
# ...
st.dataframe(outer_loadings_pivoted.round(3).style.applymap(style_primary_loading).format("{:.3f}", na_rep=""), use_container_width=True)
```

**How it Works:**
- The `plspm` library calculates outer loadings as part of the core PLS algorithm.
- The code takes these results and reformats them into a pivot table (`pivot_table`) for readability, where rows are indicators and columns are latent variables.
- The `style_primary_loading` function is added to apply coloring: yellow if loading ≥ 0.708 (considered highly valid) and red if below.

## 3. Discriminant Validity (Cross-Loadings)

*Cross-loadings* compare the loading of an indicator on its own construct vs. its loading on other constructs. The loading on its own construct must be higher than any other loading.

**Code Location:** Calculated manually in `main()` and displayed in `tab1`.

```python
# Inside main() function
all_data_for_corr = pd.concat([df[all_indicators], scores], axis=1)
correlations = all_data_for_corr.corr().loc[all_indicators, all_lvs]
correlations.to_csv('cross_loadings.csv')
```

**How it Works:**
- This code takes the original data (`df`) containing all indicators and merges it with `scores` (latent variable values calculated by PLS).
- Then, a correlation matrix is calculated for all merged data.
- Results are filtered to show only correlations between indicators (rows) and latent variables (columns). This is the *cross-loadings* table.
- Ideally, the highest correlation value in each row should be in the column corresponding to that indicator's construct.

## 4. Discriminant Validity (HTMT Ratio)

The Heterotrait-Monotrait (HTMT) Ratio is a more modern and rigorous method for measuring discriminant validity.

**Code Location:** Custom function `calculate_htmt(data, mv_map)`.

```python
def calculate_htmt(data, mv_map):
    # ... (matrix initialization)
    corr_matrix = data[model_indicators].corr().abs()

    for i in lvs:
        for j in lvs:
            # ...
            # Get indicators for construct i and j
            indicators_i = [k for k, v in mv_map.items() if v == i]
            indicators_j = [k for k, v in mv_map.items() if v == j]
            
            # Average heterotrait correlation
            r_ij = corr_matrix.loc[indicators_i, indicators_j].values.mean()
            
            # Average monotrait correlation for construct i
            vals_i = corr_matrix.loc[indicators_i, indicators_i].values
            r_ii = vals_i[~np.eye(vals_i.shape[0], dtype=bool)].mean()
            
            # Average monotrait correlation for construct j
            # ... (similar to r_ii)
            
            # HTMT formula
            htmt_val = r_ij / np.sqrt(r_ii * r_jj)
            htmt_matrix.loc[i, j] = htmt_val
    return htmt_matrix
```

**How it Works:**
1.  **`corr_matrix`**: First, the absolute correlation matrix of all indicators in the model is calculated.
2.  **`r_ij`**: For each pair of constructs (e.g., `PE` and `BI`), the code calculates the average of all correlations between `PE` indicators and `BI` indicators. This is the *Heterotrait* part.
3.  **`r_ii` and `r_jj`**: The code calculates the average correlation between indicators within the same construct (e.g., correlation between `PE1`, `PE2`, etc.). This is the *Monotrait* part.
4.  **`htmt_val`**: The HTMT formula is then applied: `average heterotrait correlation` divided by the `geometric mean of monotrait correlations`.
5.  Good HTMT values should generally be below 0.85 or 0.90.

## 5. Prediction Quality (R-Square & f-Square)

-   **R-Square (R²)**: Shows the percentage of variance in a dependent latent variable explained by the independent latent variables influencing it.
-   **f-Square (f²)**: Measures the impact (effect size) of an independent variable on a dependent variable.

**Code Location:**
-   **R-Square**: Pulled from `plspm_calc.inner_summary()['r_squared']`.
-   **f-Square**: Calculated manually in the `run_analysis` function.

```python
# f-Square calculation in run_analysis
orig_r2 = inner_sum['r_squared']
for i, (src, dst) in enumerate(path_definitions):
    # Create temp model without the path being tested
    temp_paths = [p for p in path_definitions if p != (src, dst)]
    # ...
    temp_calc = Plspm(df, temp_config, Scheme.PATH, ...)
    
    # R-Square of full model and model without path
    r2_incl = orig_r2.get(dst, 0)
    r2_excl = temp_calc.inner_summary()['r_squared'].get(dst, 0)
    
    # f-Square formula
    f2 = (r2_incl - r2_excl) / (1 - r2_incl) if (1 - r2_incl) != 0 else 0
    f2_values.append({'Path': f"{src} -> {dst}", 'f2': abs(f2)})
```

**How it Works:**
-   **R-Square**: This value is automatically calculated by the `plspm` library for each dependent (endogenous) variable.
-   **f-Square**: Calculated by re-running the PLS analysis. For each path (e.g., `PE -> BI`), the code:
    1.  Stores the R² of `BI` in the full model (`r2_incl`).
    2.  Re-runs the PLS model *without* the `PE -> BI` path.
    3.  Stores the R² of `BI` in this reduced model (`r2_excl`).
    4.  Calculates f² with the formula: `(R²_incl - R²_excl) / (1 - R²_incl)`.

## 6. Prediction Quality (Q-Square)

Q-Square (Q²) or Stone-Geisser's Q² measures the predictive relevance of the model for each indicator of the dependent latent variables.

**Code Location:** Custom function `calculate_indicator_q2(...)`.

```python
def calculate_indicator_q2(data, scores, structure_paths, mv_map):
    # ...
    for lv in endogenous_lvs:
        # ...
        X = scores[predictors].values
        
        for indicator in indicators:
            y = data[indicator].values
            # Blindfolding procedure using K-Fold Cross-Validation
            kf = KFold(n_splits=7, shuffle=True, random_state=42)
            sse, sso = 0, 0
            
            for train_index, test_index in kf.split(X):
                # ... (split data train and test)
                model = LinearRegression().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Sum of Squares of Prediction Errors (SSE)
                sse += np.sum((y_test - y_pred)**2)
                # Sum of Squares Total (SSO)
                sso += np.sum((y_test - np.mean(y_train))**2)
                
            # Q-Square formula
            q2_val = 1 - (sse / sso) if sso > 0 else 0
            q2_results.append({'Variable': indicator, 'Q²': q2_val})
            
    return pd.DataFrame(q2_results)
```

**How it Works (Blindfolding):**
1.  This function focuses on dependent variables (`endogenous_lvs`).
2.  For each indicator of the dependent variable, a *blindfolding* procedure (implemented here using `KFold` cross-validation) is performed. `n_splits=7` means data is divided into 7 parts.
3.  Iteratively, 6 parts of the data are used to train a regression model (`LinearRegression`) predicting the indicator value, and the remaining 1 part is used for testing.
4.  **SSE**: The squared difference between the actual indicator value (`y_test`) and its prediction (`y_pred`) is accumulated.
5.  **SSO**: The squared difference between the actual indicator value (`y_test`) and the mean indicator value on training data (`y_train`) is accumulated.
6.  **Q²** is calculated with the formula: `1 - (SSE / SSO)`. A Q² value greater than 0 indicates the model has predictive relevance.

## 7. Collinearity Between Indicators (VIF)

The Variance Inflation Factor (VIF) is used to detect multicollinearity, which is a correlation that is too high between indicators within the same construct.

**Code Location:** Custom function `calculate_indicator_vif(data, mv_map)`.

```python
def calculate_indicator_vif(data, mv_map):
    # ...
    for lv, indicators in lv_to_indicators.items():
        if len(indicators) > 1:
            for i, indicator_name in enumerate(indicators):
                # The indicator being tested becomes the y variable
                y = data[indicator_name]
                # Other indicators in the same construct become the X variables
                x_indicators = [ind for ind in indicators if ind != indicator_name]
                X = data[x_indicators]
                # ...
                
                # Create OLS regression model
                model = sm.OLS(y, X).fit()
                r_squared = model.rsquared
                
                # VIF formula
                vif = 1 / (1 - r_squared) if r_squared < 1.0 else np.inf
                vif_results.append({'Indicator': indicator_name, 'VIF': vif})
    
    return pd.DataFrame(vif_results)
```

**How it Works:**
1.  This function runs for every construct that has more than one indicator.
2.  For each indicator within a construct (e.g., `PE1` in construct `PE`), the code does the following:
    a.  `PE1` is made the dependent variable (`y`).
    b.  Other indicators in `PE` (e.g., `PE2`, `PE3`, `PE4`) are made the independent variables (`X`).
    c.  A linear regression model (`sm.OLS`) is created to predict `PE1` from `PE2`, `PE3`, and `PE4`.
    d.  The `R-squared` from this regression model is taken. This value shows how much variance in `PE1` can be explained by other indicators in the same construct.
    e.  **VIF** is calculated with the classic formula: `1 / (1 - R_squared)`.
3.  VIF values generally considered good are below 5, or sometimes below 3 for stricter criteria.

## 8. Hypothesis Testing (Bootstrapping)

Bootstrapping is a resampling procedure to test the statistical significance of path coefficients.

**Code Location:** Executed by `Plspm` and results are processed in `run_analysis`.

```python
# Inside run_analysis
# 1. Run PLS-PM with bootstrapping
plspm_calc = Plspm(..., bootstrap=True, bootstrap_iterations=5000, ...)
boot_results = plspm_calc.bootstrap()
boot_paths = boot_results.paths().reset_index()

# 2. Calculate T-Statistic
if 't stat.' not in boot_paths.columns:
     boot_paths['t stat.'] = (boot_paths['Original Sample (O)'] / boot_paths['Standard Deviation']).fillna(0)

# 3. Calculate P-Values
boot_paths['P-Values'] = 2 * (1 - t.cdf(boot_paths['t stat.'].abs(), df=499))

# 4. Determine Decision
boot_paths['Decision'] = np.where((boot_paths['P-Values'] < 0.05) & (boot_paths['t stat.'].abs() > 1.96), "ACCEPTED", "REJECTED")
```

**How it Works:**
1.  **Bootstrapping**: The `plspm` library runs the PLS algorithm `bootstrap_iterations` times (e.g., 5000 times). Each time, it takes a random sample from the original data *with replacement*. From each sample, path coefficients are calculated and stored.
2.  **Results**: After 5000 iterations, we get a distribution for each path coefficient.
    -   `Original Sample (O)`: Path coefficient calculated from original data (no resampling).
    -   `Standard Deviation`: This is the standard deviation of the 5000 path coefficients generated from bootstrap. This serves as the **Standard Error**.
3.  **`t stat.` (T-Statistic)**: Calculated with the formula: `Original Coefficient / Standard Error`. This value shows how far the original coefficient is from zero, in units of standard error.
4.  **`P-Values`**: Calculated from the T-Statistic. This is the probability of getting a result as extreme as the original coefficient if the null hypothesis (stating coefficient=0) were true. The code uses `scipy.stats.t.cdf` to calculate this. `df=499` is the *degrees of freedom*.
5.  **Decision**: The hypothesis is accepted if `P-Values < 0.05` and (redundantly but for clarity) `|T-Statistic| > 1.96` (critical value for 5% significance level).

---

