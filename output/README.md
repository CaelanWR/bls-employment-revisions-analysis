# BLS Employment Revisions: Visual Analysis Portfolio

**Reconstructing how employment estimates evolve over time using FRED vintage data**

This project analyzes 13+ years of BLS employment revisions across 20+ sectors and 40+ manufacturing sub-industries. The visualizations below demonstrate multi-scale analysis from headline aggregates to granular industry patterns.

---

## 1. Total Nonfarm Employment: The Headline Story

### Chart 1-2: Time Series & Cumulative Drift
**01_paynsa_revisions_with_trend.png** | **02_paynsa_cumulative_revisions.png**

Monthly revisions show COVID-era volatility spikes (±600k jobs) versus typical ±50-60k range. Cumulative tracking reveals systematic over-estimation in recent years, with T+1 revisions climbing to +3M jobs by 2023 before recent corrections.

### Chart 3-4: Statistical Distribution
**03_paynsa_revision_distribution.png** | **03b_paynsa_distribution_no_outliers.png**

Full distribution shows fat tails with extreme COVID outliers. Excluding top 2% reveals near-normal distribution (mean: 8k jobs) with slight positive skew, indicating modest systematic underestimation.

### Chart 5-6: Time Patterns & Revision Horizons
**05_paynsa_absolute_by_month.png** | **04b_paynsa_1m_vs_2m_levels.png**

December shows 5x higher revision magnitude than other months (1.0% vs 0.17% average). First revisions (48k) and second revisions (45k) contribute nearly equally—unlike most time series where initial adjustments dominate.

### Chart 7: Cross-Sector Revision Timing
**04c_sectors_1m_vs_2m_comparison.png**

Across all sectors, first and second revisions are remarkably balanced (0.14% vs 0.12%), suggesting BLS gradually converges to final estimates rather than making one large correction.

---

## 2. Cross-Sector Analysis: Industry Heterogeneity

### Chart 8-9: Directional Bias & Magnitude
**08_sector_net_revisions.png** | 09_sector_magnitude.png**

Information sector shows strong upward bias (+0.10%), while Mining consistently revised downward (-0.10%). Magnitude varies 10x: Mining/Information average 0.3-0.35% revisions versus Manufacturing at just 0.05%.

### Chart 10: The Size-Volatility Relationship
**07_size_vs_volatility_with_curve.png**

Power law emerges: larger sectors have more predictable revisions. Mining (small, 1.05% volatility) and Leisure & Hospitality (large outlier, 0.43% volatility) defy the trend, while Trade/Transportation (largest, 0.20% volatility) follows expected pattern.

### Chart 11-12: Co-Movement Patterns
**09_sector_correlations.png** | **10_sector_heatmap_timeseries.png**

Strong positive correlations: Construction ↔ Finance (0.64), Leisure ↔ Other Services (0.80). Notable negative correlations: Leisure ↔ Government (-0.40), Mining ↔ Information (-0.57). Quarterly heatmap shows COVID disrupted these relationships in Q2-Q3 2020.

### Chart 13: Seasonal Patterns
**11_industry_total_contributions_by_month.png**

December dominates revision magnitude (6x average) across nearly all sectors. Retail and Leisure show elevated revisions November-January (holiday hiring complexity), while Manufacturing remains stable year-round.

### Chart 14-15: COVID Era & Directional Consistency
**12_covid_impact.png** | **15_directional_bias.png**

COVID period (2020-2021) saw 2.4x increase in revision magnitude and 3x increase in volatility versus pre/post periods. Private Education shows strongest upward bias (positive revisions in 58% of months), while only Mining and Construction lean negative.

---

## 3. Manufacturing Deep Dive: Sub-Industry Patterns

### Chart 16: Net Revision by Sub-Industry
**16_manufacturing_net_revisions_durable_split.png**

Transportation Equipment leads upward revisions (+0.06%), while Apparel shows strongest downward bias (-0.10%). Durable goods (blue) cluster near zero, while non-durables (orange) show more directional variance.

### Chart 17: Volatility by Product Type
**17_manufacturing_magnitude_durable_split.png**

Apparel manufacturing has highest forecast uncertainty (0.5% average absolute revision), followed by Petroleum (0.4%). Durable goods average 30% higher volatility than non-durables. Food and Computer manufacturing are most predictable at 0.1%.

### Chart 18: Directional Bias Patterns
**18_manufacturing_directional_bias.png**

Transportation Equipment shows persistent underestimation (positive revisions in 62% of months). Most sub-industries cluster tightly around 50/50, indicating minimal systematic bias—unlike the broader sector-level patterns observed earlier.

---

## Technical Highlights

**Data Engineering**: Reconstructed revision history using FRED's vintage snapshot API, tracking how each monthly estimate evolved through T, T+1, and T+2 releases.

**Scale**: 3,500+ revision records across 60+ series spanning 2012-2025. Automated pipeline handles hierarchical BLS industry codes, rate limiting, and intelligent caching.

**Analysis Depth**: Three-tier investigation from macro aggregate (Total Nonfarm) → sector patterns (11 industries) → granular breakdowns (20+ manufacturing sub-industries).

**Key Findings**: COVID increased revision volatility 2-3x; December revisions 5x normal magnitude; size-volatility power law holds except for high-turnover sectors; first and second revisions contribute equally (unusual pattern); durable goods 30% more volatile than non-durables.

---

**Repository**: Complete code, datasets, and reproduction instructions available at [GitHub link]  
**Tech Stack**: Python, pandas, matplotlib, FRED API, BLS public data  
**Contact**: [Your details]
