"""
BLS Revision Trend Analyzer
Clean, informative visualizations with UNIFORM STYLING
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# UNIFIED STYLING CONFIGURATION
# =============================================================================

# Color Palette - Two-color scheme: Blue & Orange
COLORS = {
    'primary': '#1A476F',      # Dark blue (main)
    'secondary': '#FF6600',    # Orange (accent)
    'primary_light': '#2A5F8F',   # Lighter blue
    'primary_dark': '#0D2338',    # Darker blue
    'secondary_light': '#FF8533', # Lighter orange
    'secondary_dark': '#CC5200',  # Darker orange
    'neutral': '#6B7C8F',      # Blue-gray neutral
}

# Font sizes - Larger and more readable
FONT_SIZES = {
    'title': 18,
    'subtitle': 15,
    'axis_label': 15,
    'tick_label': 13,
    'legend': 13,
    'annotation': 12,
}

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': FONT_SIZES['tick_label'],
    'axes.labelsize': FONT_SIZES['axis_label'],
    'axes.titlesize': FONT_SIZES['title'],
    'xtick.labelsize': FONT_SIZES['tick_label'],
    'ytick.labelsize': FONT_SIZES['tick_label'],
    'legend.fontsize': FONT_SIZES['legend'],
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
    'grid.alpha': 0.3,
})

# Sequential colormap for heatmaps
CMAP_DIVERGING = 'RdBu_r'
CMAP_SEQUENTIAL = 'Blues'

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = "bls_revisions_level2_with_paynsa.csv"
LEISURE_DATA_FILE = "bls_revisions_level4_70_Leisure_and_hospitality.csv"
# =============================================================================


def load_data(filepath):
    """Load and prepare revision data"""
    df = pd.read_csv(filepath)
    df['obs_date'] = pd.to_datetime(df['obs_date'])
    return df


# =============================================================================
# PAYNSA ANALYSIS
# =============================================================================

def chart_1_paynsa_revisions_with_trend(df):
    """Chart 1: Total Nonfarm revisions with rolling trend"""
    paynsa = df[df['series_id'] == 'PAYNSA'].sort_values('obs_date').copy()
    
    if paynsa.empty:
        print("⚠ PAYNSA data not found")
        return
    
    paynsa['rolling_abs'] = paynsa['revision_1month'].abs().rolling(window=12, min_periods=6).mean()
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.bar(paynsa['obs_date'], paynsa['revision_1month'], 
           width=20, alpha=0.6, label='Monthly Revision', color=COLORS['primary'])
    
    ax2 = ax.twinx()
    ax2.plot(paynsa['obs_date'], paynsa['rolling_abs'], 
             linewidth=4, label='12-Month Avg Magnitude', color=COLORS['secondary'])
    ax2.set_ylabel('Average Absolute Revision (thousands)', 
                   fontsize=FONT_SIZES['axis_label'], fontweight='bold', color=COLORS['secondary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['secondary'], labelsize=FONT_SIZES['tick_label'])
    ax2.spines['top'].set_visible(False)
    
    ax.axhline(y=0, color='black', linewidth=2)
    ax.set_title('Total Nonfarm Employment: Revisions Over Time with Trend\nAre revisions getting larger or smaller?', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.set_ylabel('Revision (thousands of jobs)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold', color=COLORS['primary'])
    ax.tick_params(axis='y', labelcolor=COLORS['primary'], labelsize=FONT_SIZES['tick_label'])
    ax.legend(loc='upper left', frameon=True, fontsize=FONT_SIZES['legend'])
    ax2.legend(loc='upper right', frameon=True, fontsize=FONT_SIZES['legend'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('01_paynsa_revisions_with_trend.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 1: PAYNSA revisions with trend")
    plt.close()

# Alternative: Separate companion chart showing percentage view
def chart_1b_paynsa_revisions_percentage(df):
    """Chart 1B: PAYNSA revisions as percentage of total employment"""
    paynsa = df[df['series_id'] == 'PAYNSA'].sort_values('obs_date').copy()
    
    if paynsa.empty:
        print("⚠ PAYNSA data not found")
        return
    
    # Calculate percentage
    paynsa['revision_pct'] = (paynsa['revision_1month'] / paynsa['estimate_t']) * 100
    paynsa['rolling_abs_pct'] = paynsa['revision_pct'].abs().rolling(window=12, min_periods=6).mean()
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.bar(paynsa['obs_date'], paynsa['revision_pct'], 
           width=20, alpha=0.6, label='Monthly Revision', color=COLORS['primary'])
    
    ax2 = ax.twinx()
    ax2.plot(paynsa['obs_date'], paynsa['rolling_abs_pct'], 
             linewidth=4, label='12-Month Avg Magnitude', color=COLORS['secondary'])
    ax2.set_ylabel('Average Absolute Revision (%)', 
                   fontsize=FONT_SIZES['axis_label'], fontweight='bold', color=COLORS['secondary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['secondary'], labelsize=FONT_SIZES['tick_label'])
    ax2.spines['top'].set_visible(False)
    
    ax.axhline(y=0, color='black', linewidth=2)
    ax.set_title('Total Nonfarm Employment: Revisions as % of Total Employment\nRevisions in context of the ~150M employed', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.set_ylabel('Revision (% of total employment)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold', color=COLORS['primary'])
    ax.tick_params(axis='y', labelcolor=COLORS['primary'], labelsize=FONT_SIZES['tick_label'])
    ax.legend(loc='upper left', frameon=True, fontsize=FONT_SIZES['legend'])
    ax2.legend(loc='upper right', frameon=True, fontsize=FONT_SIZES['legend'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('01b_paynsa_revisions_percentage.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 1B: PAYNSA revisions percentage")
    plt.close()

def chart_2_paynsa_distribution(df):
    """Chart 2: Distribution of PAYNSA revisions"""
    paynsa = df[df['series_id'] == 'PAYNSA'].sort_values('obs_date').copy()
    
    if paynsa.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.hist(paynsa['revision_1month'].dropna(), bins=30, 
            color=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(x=0, color=COLORS['primary_dark'], linestyle='--', linewidth=3, 
               label='No Revision')
    ax.axvline(x=paynsa['revision_1month'].mean(), color=COLORS['secondary'], 
               linestyle='--', linewidth=3, 
               label=f"Mean: {paynsa['revision_1month'].mean():.0f}k")
    
    ax.set_title('Distribution of Total Nonfarm Revisions\nAre revisions symmetric or biased?', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.set_xlabel('Revision (thousands of jobs)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.legend(frameon=True, fontsize=FONT_SIZES['legend'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('02_paynsa_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 2: PAYNSA distribution")
    plt.close()


def chart_3_paynsa_distribution_no_outliers(df):
    """Chart 3: Distribution without top 2% outliers"""
    paynsa = df[df['series_id'] == 'PAYNSA'].sort_values('obs_date').copy()
    
    if paynsa.empty:
        return
    
    abs_revisions = paynsa['revision_1month'].abs()
    threshold = abs_revisions.quantile(0.98)
    paynsa_filtered = paynsa[abs_revisions <= threshold].copy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.hist(paynsa_filtered['revision_1month'].dropna(), bins=30, 
            color=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(x=0, color=COLORS['primary_dark'], linestyle='--', linewidth=3, 
               label='No Revision')
    ax.axvline(x=paynsa_filtered['revision_1month'].mean(), color=COLORS['secondary'], 
               linestyle='--', linewidth=3, 
               label=f"Mean: {paynsa_filtered['revision_1month'].mean():.0f}k")
    
    ax.set_title('Distribution of Total Nonfarm Revisions (Excluding Top 2% Outliers)\nTypical revision patterns without extreme values', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.set_xlabel('Revision (thousands of jobs)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.legend(frameon=True, fontsize=FONT_SIZES['legend'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('03_paynsa_distribution_no_outliers.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 3: PAYNSA distribution without outliers")
    plt.close()


def chart_4_paynsa_annual_average(df):
    """Chart 4: Average annual revisions for PAYNSA"""
    paynsa = df[df['series_id'] == 'PAYNSA'].sort_values('obs_date').copy()
    
    if paynsa.empty:
        return
    
    paynsa['year'] = paynsa['obs_date'].dt.year
    yearly_stats = paynsa.groupby('year').agg({
        'revision_1month': ['mean', lambda x: x.abs().mean()]
    }).reset_index()
    yearly_stats.columns = ['year', 'mean_revision', 'avg_abs_revision']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = [COLORS['secondary'] if x < 0 else COLORS['primary'] for x in yearly_stats['mean_revision']]
    
    ax.bar(yearly_stats['year'], yearly_stats['mean_revision'], 
           width=0.7, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=2)
    
    ax.set_title('Total Nonfarm: Average Annual Revisions\nWhich years had the biggest forecast misses?', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_ylabel('Average Revision (thousands)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('04_paynsa_annual_average.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 4: PAYNSA annual averages")
    plt.close()


def chart_4b_paynsa_1m_vs_2m(df, col1='revision_1month', col2='revision_2month',
                              col1_pct='revision_1month_pct', col2_pct='revision_2month_pct'):
    """Chart 4B: PAYNSA - 1-month vs 2-month revisions"""
    paynsa = df[df['series_id'] == 'PAYNSA'].sort_values('obs_date').copy()
    if paynsa.empty:
        print("⚠ PAYNSA data not found")
        return

    if col1 in paynsa.columns and col2 in paynsa.columns:
        vals = {
            '1-month': paynsa[col1].abs().mean(),
            '2-month': paynsa[col2].abs().mean()
        }
        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.bar(list(vals.keys()), list(vals.values()),
                      color=[COLORS['primary'], COLORS['secondary']], 
                      alpha=0.85, edgecolor='black', linewidth=2)
        ax.set_ylabel('Average Absolute Revision (jobs)', 
                      fontsize=FONT_SIZES['axis_label'], fontweight='bold')
        ax.set_title('PAYNSA: 1-Month vs 2-Month Revisions (Levels)', 
                     fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h, f'{h:,.0f}', 
                    ha='center', va='bottom', fontsize=FONT_SIZES['annotation'], fontweight='bold')

        plt.tight_layout()
        plt.savefig('04b_paynsa_1m_vs_2m_levels.png', dpi=300, bbox_inches='tight')
        print("✓ Chart 4B (levels): PAYNSA 1m vs 2m")
        plt.close()


def chart_5_sector_net_revisions(df):
    """Chart 5: Net average revision by sector"""
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    
    sector_stats = industry_df.groupby('industry_name')['revision_1month_pct'].mean().reset_index()
    sector_stats.columns = ['industry', 'mean_revision']
    sector_stats = sector_stats.sort_values('mean_revision')
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    colors = [COLORS['secondary'] if x < 0 else COLORS['primary'] for x in sector_stats['mean_revision']]
    
    ax.barh(sector_stats['industry'], sector_stats['mean_revision'],
            color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axvline(x=0, color='black', linewidth=2.5, alpha=0.7)
    
    ax.set_xlabel('Average Net Revision (%)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Average Revision by Sector\nWhich sectors are consistently revised up or down?', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('05_sector_net_revisions.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 5: Sector net revisions")
    plt.close()


def chart_6_sector_revision_magnitude(df):
    """Chart 6: Average absolute revision by sector"""
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    
    sector_stats = industry_df.groupby('industry_name')['revision_1month_pct'].apply(
        lambda x: x.abs().mean()
    ).reset_index()
    sector_stats.columns = ['industry', 'avg_abs_revision']
    sector_stats = sector_stats.sort_values('avg_abs_revision', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    ax.barh(sector_stats['industry'], sector_stats['avg_abs_revision'],
            color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Average Absolute Revision (%)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Revision Magnitude by Sector\nWhich sectors have the biggest forecast errors?', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('06_sector_magnitude.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 6: Sector revision magnitude")
    plt.close()


def chart_6b_sector_1m_vs_2m(df, col1_pct='revision_1month_pct', col2_pct='revision_2month_pct',
                             filename='06b_sector_1m_vs_2m.png'):
    """Chart 6B: Sectors - average absolute % revision, 1m vs 2m"""
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    if industry_df.empty or col1_pct not in industry_df.columns or col2_pct not in industry_df.columns:
        return

    per_industry = industry_df.groupby('industry_name').agg(
        avg_abs_1m=(col1_pct, lambda x: x.abs().mean()),
        avg_abs_2m=(col2_pct, lambda x: x.abs().mean())
    ).reset_index()

    overall_1m = per_industry['avg_abs_1m'].mean()
    overall_2m = per_industry['avg_abs_2m'].mean()

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(['1-month', '2-month'], [overall_1m, overall_2m],
                  color=[COLORS['primary'], COLORS['secondary']], 
                  alpha=0.85, edgecolor='black', linewidth=2)

    ax.set_ylabel('Average Absolute Revision (%)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Sectors: 1-Month vs 2-Month Revisions (Average Across Industries)', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2., h, f'{h:.2f}%', 
                ha='center', va='bottom', fontsize=FONT_SIZES['annotation'], fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Chart 6B: Sector 1m vs 2m")
    plt.close()


def chart_7_size_vs_volatility_with_curve(df):
    """Chart 7: Sector size vs volatility with fitted curve"""
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    
    sector_stats = industry_df.groupby('industry_name').agg({
        'estimate_t': 'mean',
        'revision_1month_pct': 'std'
    }).reset_index()
    
    sector_stats.columns = ['industry', 'avg_employment', 'std_revision']
    
    def power_law(x, a, b):
        return a * np.power(x, b)
    
    valid_data = sector_stats.dropna()
    
    try:
        popt, _ = curve_fit(power_law, valid_data['avg_employment'], valid_data['std_revision'])
        x_curve = np.linspace(valid_data['avg_employment'].min(), valid_data['avg_employment'].max(), 100)
        y_curve = power_law(x_curve, *popt)
    except:
        x_curve = None
        y_curve = None
    
    fig, ax = plt.subplots(figsize=(13, 9))
    
    ax.scatter(sector_stats['avg_employment'], 
               sector_stats['std_revision'],
               s=200, alpha=0.7, color=COLORS['secondary'], 
               edgecolors='black', linewidth=2)
    
    if x_curve is not None:
        ax.plot(x_curve, y_curve, '--', linewidth=3, alpha=0.7, 
                color=COLORS['primary'], label='Power Law Fit')
        ax.legend(loc='upper right', fontsize=FONT_SIZES['legend'], frameon=True)
    
    for _, row in sector_stats.iterrows():
        ax.annotate(row['industry'], 
                    (row['avg_employment'], row['std_revision']),
                    fontsize=FONT_SIZES['annotation'], xytext=(5, 5), 
                    textcoords='offset points')
    
    ax.set_title('Sector Size vs Revision Volatility\nDo larger sectors have more predictable revisions?', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.set_xlabel('Average Employment (thousands)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_ylabel('Revision Volatility (Std Dev %)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('07_size_vs_volatility_with_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 7: Size vs volatility with curve")
    plt.close()


def chart_8_correlation_heatmap(df):
    """Chart 8: Correlation heatmap with larger fonts"""
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    
    pivot = industry_df.pivot_table(
        values='revision_1month_pct',
        index='obs_date',
        columns='industry_name',
        aggfunc='first'
    )
    
    corr_matrix = pivot.corr()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap=CMAP_DIVERGING, center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax, annot_kws={'size': FONT_SIZES['annotation'], 'weight': 'bold'})
    
    ax.set_title('Revision Direction Correlation Across Sectors\nDo certain sectors get revised together?',
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick_label'])
    
    plt.tight_layout()
    plt.savefig('08_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 8: Correlation heatmap")
    plt.close()


def chart_9_sector_month_heatmap_clipped(df):
    """Chart 9: Heatmap with outliers clipped for readability"""
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    
    pivot = industry_df.pivot_table(
        values='revision_1month_pct',
        index='industry_name',
        columns=pd.Grouper(key='obs_date', freq='Q'),
        aggfunc='mean'
    )
    
    pivot.columns = [f"Q{col.quarter} {col.year}" for col in pivot.columns]
    
    vmin = np.nanpercentile(pivot.values.flatten(), 5)
    vmax = np.nanpercentile(pivot.values.flatten(), 95)
    
    fig, ax = plt.subplots(figsize=(22, 9))
    
    sns.heatmap(pivot, cmap=CMAP_DIVERGING, center=0, 
                vmin=vmin, vmax=vmax,
                cbar_kws={"label": "Revision (%, clipped at 5th/95th percentile)"},
                ax=ax, linewidths=0)
    
    ax.set_title('Sector Revisions Over Time (Quarterly Averages)\nOutliers clipped for readability',
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.set_xlabel('Quarter', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_ylabel('Sector', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.tick_params(axis='x', labelsize=FONT_SIZES['annotation'], rotation=45)
    ax.tick_params(axis='y', labelsize=FONT_SIZES['tick_label'])
    
    plt.tight_layout()
    plt.savefig('09_sector_month_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 9: Sector-month heatmap")
    plt.close()


def chart_10_monthly_seasonality(df):
    """Chart 10: Are certain months more error-prone?"""
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    industry_df['month'] = industry_df['obs_date'].dt.month
    
    monthly_stats = industry_df.groupby('month')['revision_1month_pct'].apply(
        lambda x: x.abs().mean()
    ).reset_index()
    monthly_stats.columns = ['month', 'avg_abs_revision']
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = [COLORS['secondary'] if x > monthly_stats['avg_abs_revision'].mean() 
              else COLORS['neutral'] for x in monthly_stats['avg_abs_revision']]
    bars = ax.bar(range(12), monthly_stats['avg_abs_revision'], 
                  color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    ax.axhline(y=monthly_stats['avg_abs_revision'].mean(), 
               color=COLORS['primary'], linestyle='--', linewidth=3, alpha=0.7,
               label=f"Average: {monthly_stats['avg_abs_revision'].mean():.2f}%")
    
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=FONT_SIZES['tick_label'], fontweight='bold')
    ax.set_ylabel('Average Absolute Revision (%)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Revision Seasonality: Which Months Are Hardest to Forecast?\nAverage across all sectors',
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.legend(fontsize=FONT_SIZES['legend'], frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('11_monthly_seasonality.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 11: Monthly seasonality")
    plt.close()


def chart_11_industry_total_contributions_by_month(df, revision_col='revision_1month', top_n=10,
                                                   filename='11_industry_total_contributions_by_month.png'):
    """Chart 11: Which industries drive the revisions each month?"""
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    if industry_df.empty:
        return

    industry_df['month'] = industry_df['obs_date'].dt.month
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    grp = (
        industry_df
        .groupby(['month', 'industry_name'])[revision_col]
        .mean()
        .reset_index(name='avg_revision_jobs')
    )

    overall = (
        grp.groupby('industry_name')['avg_revision_jobs']
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
    )
    top_inds = overall.head(top_n).index

    grp['industry_plot'] = grp['industry_name'].where(grp['industry_name'].isin(top_inds), 'Other')

    pivot = (
        grp.groupby(['month','industry_plot'])['avg_revision_jobs']
        .sum()
        .reset_index()
        .pivot_table(index='month', columns='industry_plot', values='avg_revision_jobs', fill_value=0)
    )

    pivot = pivot.reindex(range(1,13), fill_value=0)
    pivot.index = month_names

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use two-color scheme with variations
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = [COLORS['primary'], COLORS['primary_light'], COLORS['secondary_light'], 
                   COLORS['secondary'], COLORS['primary_dark'], COLORS['neutral']]
    n_cols = len(pivot.columns)
    
    # Create enough distinct colors by cycling through our palette
    plot_colors = [colors_list[i % len(colors_list)] for i in range(n_cols)]
    
    pivot.plot(
        kind='bar',
        stacked=True,
        figsize=(14,8),
        color=plot_colors,
        edgecolor='black',
        linewidth=1,
        alpha=0.9,
        ax=ax
    )

    ax.set_ylabel('Average Revision (Jobs)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_xlabel('Month', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Industry Contributions to Monthly Jobs Revisions\n(Average Level, Top Industries)',
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)

    ax.legend(title='Industry', bbox_to_anchor=(1.05,1), loc='upper left', 
              fontsize=FONT_SIZES['annotation'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=FONT_SIZES['tick_label'], rotation=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Chart 11 (total jobs contributions)")
    plt.close()


def chart_12_covid_impact_comparison(df):
    """Chart 12: Pre-COVID vs COVID vs Post-COVID revision patterns"""
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    
    industry_df['period'] = 'Pre-COVID (2015-2019)'
    industry_df.loc[industry_df['obs_date'].between('2020-01-01', '2021-12-31'), 'period'] = 'COVID Era (2020-2021)'
    industry_df.loc[industry_df['obs_date'] >= '2022-01-01', 'period'] = 'Post-COVID (2022+)'
    
    period_stats = industry_df.groupby('period').agg({
        'revision_1month_pct': [lambda x: x.abs().mean(), 'std']
    }).reset_index()
    period_stats.columns = ['period', 'avg_abs_revision', 'std_revision']
    
    period_order = ['Pre-COVID (2015-2019)', 'COVID Era (2020-2021)', 'Post-COVID (2022+)']
    period_stats['period'] = pd.Categorical(period_stats['period'], categories=period_order, ordered=True)
    period_stats = period_stats.sort_values('period')
    
    fig, ax = plt.subplots(figsize=(13, 8))
    
    x = np.arange(len(period_stats))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, period_stats['avg_abs_revision'], width,
                   label='Average Magnitude', color=COLORS['primary'], 
                   alpha=0.85, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, period_stats['std_revision'], width,
                   label='Volatility (Std Dev)', color=COLORS['secondary'], 
                   alpha=0.85, edgecolor='black', linewidth=2)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', 
                fontsize=FONT_SIZES['annotation'], fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', 
                fontsize=FONT_SIZES['annotation'], fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(period_stats['period'], fontsize=FONT_SIZES['tick_label'], fontweight='bold')
    ax.set_ylabel('Revision Magnitude (%)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('How COVID Changed Revision Patterns\nComparing forecast accuracy across eras',
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.legend(fontsize=FONT_SIZES['legend'], frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('12_covid_impact.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 12: COVID impact comparison")
    plt.close()


def chart_13_upward_downward_bias(df):
    """Chart 13: Upward vs downward revision frequency by sector"""
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    
    sector_bias = industry_df.groupby('industry_name').apply(
        lambda x: pd.Series({
            'pct_upward': (x['revision_1month_pct'] > 0).mean() * 100,
            'pct_downward': (x['revision_1month_pct'] < 0).mean() * 100
        })
    ).reset_index()
    
    sector_bias['net_bias'] = sector_bias['pct_upward'] - 50
    sector_bias = sector_bias.sort_values('net_bias')
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    colors = [COLORS['secondary'] if x < 0 else COLORS['primary'] for x in sector_bias['net_bias']]
    
    ax.barh(sector_bias['industry_name'], sector_bias['pct_upward'] - 50,
            color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.axvline(x=0, color='black', linewidth=3, alpha=0.8)
    ax.set_xlabel('← More Downward Revisions    |    More Upward Revisions →', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Revision Direction Bias by Sector\nDoes BLS tend to over- or under-estimate initially?',
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-30, 30)
    
    plt.tight_layout()
    plt.savefig('13_upward_downward_bias.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 13: Upward vs downward bias")
    plt.close()


def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    paynsa = df[df['series_id'] == 'PAYNSA']
    if not paynsa.empty:
        print("\nTotal Nonfarm (PAYNSA):")
        print(f"  • Average revision: {paynsa['revision_1month'].abs().mean():.0f}k jobs")
        print(f"  • Std deviation: {paynsa['revision_1month'].std():.0f}k jobs")
        print(f"  • Revised up: {(paynsa['revision_1month'] > 0).mean()*100:.0f}% of months")
        print(f"  • Largest revision: {paynsa['revision_1month'].abs().max():.0f}k jobs")
    
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    
    volatility = industry_df.groupby('industry_name')['revision_1month_pct'].std().sort_values(ascending=False)
    print("\nMost Volatile Sectors:")
    for idx, (sector, vol) in enumerate(volatility.head(3).items(), 1):
        print(f"  {idx}. {sector}: {vol:.2f}% std dev")
    
    avg_revisions = industry_df.groupby('industry_name')['revision_1month_pct'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
    print("\nLargest Average Revisions:")
    for idx, (sector, avg) in enumerate(avg_revisions.head(3).items(), 1):
        print(f"  {idx}. {sector}: {avg:.2f}%")
    
    print("="*60)


# =============================================================================
# LEISURE & HOSPITALITY DEEP DIVE
# =============================================================================

def chart_16_lh_net_revisions(df_leisure):
    """Chart 16: Net average revision by L&H sub-industry (same as Chart 5 but for L&H)"""
    if df_leisure.empty:
        print("⚠ No L&H data available")
        return
    
    industry_stats = df_leisure.groupby('industry_name')['revision_1month_pct'].mean().reset_index()
    industry_stats.columns = ['industry', 'mean_revision']
    industry_stats = industry_stats.sort_values('mean_revision')
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    colors = [COLORS['secondary'] if x < 0 else COLORS['primary'] for x in industry_stats['mean_revision']]
    
    ax.barh(industry_stats['industry'], industry_stats['mean_revision'],
            color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axvline(x=0, color='black', linewidth=2.5, alpha=0.7)
    
    ax.set_xlabel('Average Net Revision (%)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Leisure & Hospitality: Average Revision by Sub-Industry\nWhich L&H sub-sectors are consistently revised up or down?', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('16_lh_net_revisions.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 16: L&H net revisions by sub-industry")
    plt.close()


def chart_17_lh_revision_magnitude(df_leisure):
    """Chart 17: Average absolute revision by L&H sub-industry (same as Chart 6 but for L&H)"""
    if df_leisure.empty:
        return
    
    industry_stats = df_leisure.groupby('industry_name')['revision_1month_pct'].apply(
        lambda x: x.abs().mean()
    ).reset_index()
    industry_stats.columns = ['industry', 'avg_abs_revision']
    industry_stats = industry_stats.sort_values('avg_abs_revision', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    ax.barh(industry_stats['industry'], industry_stats['avg_abs_revision'],
            color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Average Absolute Revision (%)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Leisure & Hospitality: Revision Magnitude by Sub-Industry\nWhich L&H sub-sectors have the biggest forecast errors?', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('17_lh_magnitude.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 17: L&H revision magnitude by sub-industry")
    plt.close()


def chart_18_lh_directional_bias(df_leisure):
    """Chart 18: L&H upward vs downward revision frequency (same as Chart 13 but for L&H)"""
    if df_leisure.empty:
        return
    
    industry_bias = df_leisure.groupby('industry_name').apply(
        lambda x: pd.Series({
            'pct_upward': (x['revision_1month_pct'] > 0).mean() * 100,
            'pct_downward': (x['revision_1month_pct'] < 0).mean() * 100
        })
    ).reset_index()
    
    industry_bias['net_bias'] = industry_bias['pct_upward'] - 50
    industry_bias = industry_bias.sort_values('net_bias')
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    colors = [COLORS['secondary'] if x < 0 else COLORS['primary'] for x in industry_bias['net_bias']]
    
    ax.barh(industry_bias['industry_name'], industry_bias['pct_upward'] - 50,
            color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.axvline(x=0, color='black', linewidth=3, alpha=0.8)
    ax.set_xlabel('← More Downward Revisions    |    More Upward Revisions →', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Leisure & Hospitality: Revision Direction Bias by Sub-Industry\nDoes BLS tend to over- or under-estimate L&H sub-sectors initially?',
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-30, 30)
    
    plt.tight_layout()
    plt.savefig('18_lh_directional_bias.png', dpi=300, bbox_inches='tight')
    print("✓ Chart 18: L&H directional bias by sub-industry")
    plt.close()


def print_leisure_summary(df_leisure):
    """Print focused summary for L&H sector"""
    if df_leisure.empty:
        print("⚠ No L&H data available")
        return
    
    print("\n" + "="*60)
    print("LEISURE & HOSPITALITY SUB-INDUSTRY FINDINGS")
    print("="*60)
    
    # Overall L&H stats - using df_leisure only
    lh_avg_revision = df_leisure['revision_1month_pct'].abs().mean()
    lh_std = df_leisure['revision_1month_pct'].std()
    lh_upward_pct = (df_leisure['revision_1month_pct'] > 0).mean() * 100
    
    print(f"\nOverall L&H Sub-Industries:")
    print(f"  • Average revision: {lh_avg_revision:.2f}%")
    print(f"  • Std deviation: {lh_std:.2f}%")
    print(f"  • Revised upward: {lh_upward_pct:.0f}% of months")
    
    # Most volatile sub-industries
    worst_industries = df_leisure.groupby('industry_name')['revision_1month_pct'].apply(
        lambda x: x.abs().mean()
    ).nlargest(3)
    
    print(f"\nMost Volatile L&H Sub-Industries:")
    for idx, (industry, avg) in enumerate(worst_industries.items(), 1):
        print(f"  {idx}. {industry}: {avg:.2f}%")
    
    # COVID impact - using df_leisure only
    covid_data = df_leisure[df_leisure['obs_date'].between('2020-03-01', '2021-06-30')]
    other_data = df_leisure[~df_leisure['obs_date'].between('2020-03-01', '2021-06-30')]
    
    if not covid_data.empty and not other_data.empty:
        covid_avg = covid_data['revision_1month_pct'].abs().mean()
        other_avg = other_data['revision_1month_pct'].abs().mean()
        print(f"\nCOVID Impact:")
        print(f"  • COVID-era revisions: {covid_avg:.2f}% (avg)")
        print(f"  • Normal-era revisions: {other_avg:.2f}% (avg)")
        if other_avg > 0:
            print(f"  • COVID multiplier: {covid_avg/other_avg:.1f}x worse")
    
    print("="*60)


def main():
    """Print summary statistics"""
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    paynsa = df[df['series_id'] == 'PAYNSA']
    if not paynsa.empty:
        print("\nTotal Nonfarm (PAYNSA):")
        print(f"  • Average revision: {paynsa['revision_1month'].abs().mean():.0f}k jobs")
        print(f"  • Std deviation: {paynsa['revision_1month'].std():.0f}k jobs")
        print(f"  • Revised up: {(paynsa['revision_1month'] > 0).mean()*100:.0f}% of months")
        print(f"  • Largest revision: {paynsa['revision_1month'].abs().max():.0f}k jobs")
    
    industry_df = df[(df['display_level'] == 2) & (df['industry_name'].notna())].copy()
    
    volatility = industry_df.groupby('industry_name')['revision_1month_pct'].std().sort_values(ascending=False)
    print("\nMost Volatile Sectors:")
    for idx, (sector, vol) in enumerate(volatility.head(3).items(), 1):
        print(f"  {idx}. {sector}: {vol:.2f}% std dev")
    
    avg_revisions = industry_df.groupby('industry_name')['revision_1month_pct'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
    print("\nLargest Average Revisions:")
    for idx, (sector, avg) in enumerate(avg_revisions.head(3).items(), 1):
        print(f"  {idx}. {sector}: {avg:.2f}%")
    
    print("="*60)


def main():
    """Generate comprehensive visualizations"""
    print("\nBLS Revision Trend Analyzer - UNIFORM STYLING")
    print("="*60)
    
    if not Path(DATA_FILE).exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        return
    
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df):,} records\n")
    
    print("\n📊 PAYNSA ANALYSIS")
    chart_1_paynsa_revisions_with_trend(df)
    chart_1b_paynsa_revisions_percentage(df)
    chart_2_paynsa_distribution(df)
    chart_3_paynsa_distribution_no_outliers(df)
    chart_4_paynsa_annual_average(df)
    chart_4b_paynsa_1m_vs_2m(df)
    
    print("\n📊 SECTOR COMPARISON")
    chart_5_sector_net_revisions(df)
    chart_6_sector_revision_magnitude(df)
    chart_6b_sector_1m_vs_2m(df)
    chart_7_size_vs_volatility_with_curve(df)
    chart_8_correlation_heatmap(df)
    
    print("\n📊 GRANULAR DATA PATTERNS")
    chart_9_sector_month_heatmap_clipped(df)
    chart_13_upward_downward_bias(df)
    
    print_summary(df)
    
    # Leisure & Hospitality Deep Dive
    if Path(LEISURE_DATA_FILE).exists():
        print("\n" + "="*60)
        print("📊 LEISURE & HOSPITALITY SUB-INDUSTRY ANALYSIS")
        print("="*60)
        
        df_leisure = load_data(LEISURE_DATA_FILE)
        
        chart_16_lh_net_revisions(df_leisure)
        chart_17_lh_revision_magnitude(df_leisure)
        chart_18_lh_directional_bias(df_leisure)
        
        print_leisure_summary(df_leisure)
    else:
        print(f"\n⚠ Leisure data file not found: {LEISURE_DATA_FILE}")
        print("Skipping L&H deep dive")
    
    print("\n✓ All charts generated with uniform styling!")


if __name__ == "__main__":
    main()