# BLS Employment Revisions: Visual Analysis Portfolio

**Analyzing how CES estimates evolve from preliminary release through subsequent revisions**

This project reconstructs revision histories for employment series by accessing FRED's vintage data snapshots. The analysis covers 2012-2025, tracking how initial estimates (T) are adjusted in the first (T+1) and second (T+2) monthly revisions before the annual benchmark.

---

## 1. Total Nonfarm Employment (PAYNSA)

### Chart 1-2: Revision Patterns Over Time
**01_paynsa_revisions_with_trend.png** | **02_paynsa_cumulative_revisions.png**

Monthly revisions typically range ±50-60k jobs. The 2020-2021 period shows significantly larger revisions, likely reflecting sampling challenges during pandemic-related establishment closures and reopenings. Cumulative net revisions reveal a pattern of underestimation from 2021-2023, followed by recent downward adjustments.

### Chart 3-4: Distribution Analysis
**03_paynsa_distribution.png** | **04_paynsa_distribution_no_outliers.png**

The full distribution has heavy tails due to extreme values during the pandemic. The trimmed distribution (excluding top 2%) shows approximate normality with a mean of +8k, suggesting a modest positive bias in preliminary estimates. This is consistent with the lag between survey collection and initial publication.

### Chart 5-6: Annual Patterns and Revision Timing
**05_paynsa_annual_average.png** | **06_paynsa_1m_vs_2m.png**

Average annual revisions highlight periods of systematic bias—2012 and 2022 show large positive revisions (underestimation), while 2024-2025 show negative revisions (overestimation). The comparison of first-month (48k) versus second-month (45k) revision magnitudes shows roughly equivalent contributions, indicating that estimate refinement continues through both revision cycles rather than concentrating in the first month.

### Chart 7: Cross-Sector Revision Timing
**07_sector_1m_vs_2m.png**

Aggregating across all level 2 sectors, first-month revisions (0.14%) slightly exceed second-month revisions (0.12%). This gradual convergence differs from what might be expected if most respondent updates arrived by the first revision deadline.

---

## 2. Industry-Level Patterns

### Chart 8-9: Sectoral Bias and Volatility
**08_sector_net_revisions.png** | **09_sector_magnitude.png**

Information (+0.10%) and Construction (+0.02%) show consistent upward revisions, while Mining shows downward revisions (-0.10%). These patterns may reflect response rate differences—industries with smaller establishments or higher turnover tend to have lower initial response rates and larger subsequent adjustments.

Revision magnitude varies considerably across sectors. Mining and Information show 0.30-0.35% average absolute revisions, compared to Manufacturing at 0.05%. This range likely reflects differences in both sample composition and employment volatility.

### Chart 10: Employment Level vs. Revision Volatility  
**10_size_vs_volatility.png**

Larger sectors generally show lower percentage volatility in revisions, following an approximate power law. Leisure & Hospitality is a notable exception—despite its size, it maintains high volatility (0.43%), consistent with known measurement challenges in high-turnover industries.

### Chart 11-12: Inter-Sector Correlations and Temporal Patterns
**11_correlation_heatmap.png** | **12_sector_month_heatmap.png**

Revision correlations suggest common survey response timing patterns and economic linkages. Construction and Financial Activities show positive correlation (0.64), while some sector pairs show negative correlations (Leisure & Hospitality vs. Government: -0.40). The quarterly heatmap indicates these relationships weakened during Q2-Q3 2020, when different sectors faced distinct measurement challenges.

### Chart 13: Monthly Seasonality
**13_monthly_seasonality.png**

December shows substantially elevated revision magnitudes (1.0% vs. baseline 0.10-0.15% for other months), likely related to seasonal adjustment complexity during the holiday season. This 5-6x increase affects most sectors and suggests heightened measurement difficulty for this reference month.

### Chart 14-15: Period Comparisons and Directional Consistency
**14_covid_impact.png** | **15_upward_downward_bias.png**

Comparing pre-COVID (2015-2019), COVID (2020-2021), and post-COVID (2022+) periods shows the pandemic's impact on data collection and estimation. Revision magnitude and volatility approximately tripled during 2020-2021, with partial recovery since.

Most sectors show roughly balanced upward and downward revision frequencies (45-55%), indicating minimal systematic directional bias. Private Education is an exception with persistent upward revisions.

---

## 3. Manufacturing Sub-Industries

### Chart 16-18: Detailed Manufacturing Breakdowns
**16_manufacturing_net_revisions.png** | **17_manufacturing_magnitude.png** | **18_manufacturing_directional_bias.png**

Within Manufacturing, durable goods industries (shown in blue in Chart 17) generally show higher revision volatility than non-durables (orange). Transportation Equipment shows upward revision bias (+0.06%), while Apparel shows downward bias (-0.10%). 

Apparel (0.5%) and Petroleum Products (0.4%) have the highest absolute revision magnitudes among manufacturing sub-industries. These industries have notably different characteristics—Apparel has many small establishments, while Petroleum has high earnings volatility—suggesting multiple factors contribute to revision magnitude.

The directional bias chart shows most manufacturing sub-industries cluster around 50/50 positive/negative split, with Transportation Equipment as a notable exception (62% positive revisions). This relative balance contrasts with the broader sector-level patterns, suggesting aggregation may smooth out some sub-industry biases.

---

## Technical Implementation

**Data Source**: FRED vintage snapshots accessed via API, reconstructing the information set available at each publication date (T, T+1, T+2 releases)

**Coverage**: 
- Time range: 2012-2025 (160+ months)
- Level 2 sectors: 11 major industries + Total Nonfarm
- Manufacturing detail: 22 sub-industries at level 4

**Methodology**: Matched observation dates across vintages to calculate revision magnitudes. Analysis excludes annual benchmark revisions, focusing only on within-year monthly adjustments.

**Code**: Python pipeline handles BLS industry hierarchy, FRED API rate limits (120/min), and vintage-specific queries with caching for efficiency (~15-30 minutes runtime for full dataset).

**Outputs**: 18 PNG charts plus `sector_summary_statistics.csv` with comprehensive metrics (mean, std dev, absolute revisions) for all sectors covering both 1-month and 2-month revisions.

---

**Repository**: Complete code and datasets available  
**Contact**: [Your information]
