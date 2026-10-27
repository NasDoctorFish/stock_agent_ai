# Price familiarity & relative valuation ablation

Universe: 56 current config tickers (survivor bias), prices 2010->2026, first sample at day 756.
P/B not computed: no book-equity columns in fundamentals_q_full. EPS lag 45d (no filing dates).

## A. Price-only ablation (full sample)
127 OOS dates 2016-03-08 -> 2026-09-16
head alpha=100; spread = top-minus-bottom quintile, equal weight, 21d; cost = one-way bps on traded notional
```
                      IC    ICIR  ICIR_ann       t     hit   yrs>0     dIC  dIC_lo  dIC_hi  gross_mo    turn   net10   net25  sharpe_net25
set                                                                                                                                       
base              0.0088  0.0353    0.1222  0.3976  0.5276  0.7273  0.0000     NaN     NaN    0.0043  0.6165  0.0031  0.0012        0.0405
familiarity       0.0112  0.0466    0.1615  0.5254  0.4803  0.6364  0.0024 -0.0093  0.0139   -0.0006  0.7317 -0.0021 -0.0043       -0.1972
familiarity_only  0.0054  0.0211    0.0731  0.2378  0.4409  0.5455 -0.0034 -0.0244  0.0180   -0.0057  0.6839 -0.0070 -0.0091       -0.3340

mean IC by year
       base  familiarity  familiarity_only
date                                      
2016 -0.027       -0.043            -0.070
2017  0.026        0.047             0.064
2018  0.037        0.049             0.054
2019  0.012       -0.000            -0.004
2020  0.010        0.023             0.045
2021 -0.079       -0.064            -0.066
2022  0.091        0.058             0.040
2023 -0.023       -0.001            -0.020
2024  0.006        0.008             0.011
2025  0.033        0.016            -0.034
2026  0.003        0.027             0.030

mean IC by alpha
                    10      100     1000
base              0.0088  0.0088  0.0098
familiarity       0.0138  0.0112  0.0139
familiarity_only  0.0002  0.0054  0.0164
```
## B. Valuation ablation (val-eligible dates/rows only; exploratory, short sample)
14 OOS dates 2025-08-14 -> 2026-09-16
head alpha=100; spread = top-minus-bottom quintile, equal weight, 21d; cost = one-way bps on traded notional
```
                    IC    ICIR  ICIR_ann       t     hit  yrs>0     dIC  dIC_lo  dIC_hi  gross_mo    turn   net10   net25  sharpe_net25
set                                                                                                                                    
base            0.0303  0.1365    0.4727  0.5106  0.5714    1.0  0.0000     NaN     NaN    0.0210  0.5909  0.0198  0.0180        0.6193
valuation       0.0593  0.2982    1.0331  1.1159  0.6429    1.0  0.0290 -0.0436  0.0972    0.0197  0.6402  0.0184  0.0165        0.5682
combined        0.0646  0.3338    1.1563  1.2490  0.5714    1.0  0.0343 -0.0338  0.0984    0.0261  0.7879  0.0245  0.0222        0.9176
familiarity     0.0465  0.2052    0.7107  0.7676  0.6429    1.0  0.0162 -0.0172  0.0486    0.0239  0.7619  0.0224  0.0201        0.7151
valuation_only  0.0085  0.0371    0.1286  0.1389  0.5000    0.5 -0.0218 -0.1688  0.1345   -0.0097  0.8355 -0.0114 -0.0139       -0.5316

mean IC by year
       base  valuation  combined  familiarity  valuation_only
date                                                         
2025  0.079      0.066     0.064        0.082          -0.102
2026  0.003      0.055     0.065        0.027           0.070

mean IC by alpha
                  10      100     1000
base            0.0285  0.0303  0.0448
valuation       0.0446  0.0593  0.0565
combined        0.0759  0.0646  0.0432
familiarity     0.0686  0.0465  0.0329
valuation_only -0.0071  0.0085  0.0100
```
## C. Incremental signal beyond mom_12_1 + rev_1m (per-date cross-sectional, full panel)
```
             IC_alone  t_alone  IC_partial  t_partial      n
z_63           0.0005   0.0270      0.0001     0.0091  165.0
z_126          0.0212   1.2121      0.0167     1.0441  165.0
z_252          0.0305   1.5708      0.0114     0.6770  165.0
pctl_252       0.0311   1.5855      0.0171     1.0115  165.0
pctl_756       0.0312   1.5728      0.0107     0.6671  165.0
range_252      0.0322   1.6260      0.0149     0.8439  165.0
dd_126         0.0271   1.3178      0.0246     1.2643  165.0
ma_dist_200    0.0345   1.7273      0.0183     1.0350  165.0
ma_dist_500    0.0355   1.7383      0.0041     0.2436  165.0
```
valuation features (val-eligible rows)
```
            IC_alone  t_alone  IC_partial  t_partial     n
ey            0.0756   0.9888      0.1030     1.7147  14.0
pe_inv_rel    0.0119   0.3864     -0.0109    -0.2732  14.0
ey_pctl       0.0130   0.3707     -0.0132    -0.3173  14.0
pe           -0.0714  -1.6468     -0.0873    -2.2744  14.0
```
## D. Mean ridge coefficients (alpha=100)
```
             familiarity  combined
dd_126             0.031     0.031
dd_252            -0.003    -0.004
ey                   NaN     0.000
ey_pctl              NaN    -0.000
ma_dist_200        0.011     0.011
ma_dist_500        0.010     0.010
mom_12_1           0.022     0.022
mom_3m            -0.017    -0.017
mom_6_1           -0.009    -0.009
pctl_252           0.006     0.006
pctl_756           0.014     0.014
pe                   NaN    -0.002
pe_inv_rel           NaN    -0.000
range_252          0.013     0.013
rev_1m             0.002     0.002
vol_60            -0.010    -0.010
z_126             -0.002    -0.002
z_252              0.005     0.005
z_63              -0.032    -0.032
```
## E. Placebo (new features shuffled across stocks)
dIC combined-base = +0.0005 (t 0.06); expect ~0

## Verdict
- Familiarity vs base: dIC +0.0024, 95% CI [-0.0093, +0.0139] -> no statistically significant incremental power (CI includes 0).
- Check section C signs: a negative partial IC for price percentile/z-score means high-vs-history prices predicted lower returns; a positive one means the opposite. Low price was NOT assumed to mean undervalued.
- Caveats: survivor universe of 56 large caps (58 in price table = 56 + SPY,QQQ benchmarks); valuation window ~1.5y; overlapping nothing (H=21 sampling) but n is small.