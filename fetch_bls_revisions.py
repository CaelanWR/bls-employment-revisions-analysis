"""
BLS Revision Analyzer
Analyzes how employment estimates change as BLS releases subsequent revisions
Uses FRED API to pull different data vintages for revision analysis
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import hashlib

# =============================================================================
# CONFIGURATION
# =============================================================================

# Replace with your FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
API_KEY = "your_api_key_here"

if API_KEY == "your_api_key_here":
    raise ValueError("Please replace 'your_api_key_here' with your actual FRED API key")

# Analysis mode toggles
RUN_LEVEL_2_ANALYSIS = True      # Broad industries + total nonfarm
RUN_DETAILED_ANALYSIS = True      # Detailed industries within a supersector

# Detailed analysis parameters
DETAILED_SUPERSECTOR_CODE = "30"  # "30"=Manufacturing, "42"=Retail, etc.
DETAILED_LEVEL = 4                # Display level to analyze (typically 3 or 4)

# Date range for analysis
START_DATE = "2012-01-01"
END_DATE = "2025-07-31"

# =============================================================================


class OptimizedBLSAnalyzer:
    """
    Handles FRED API interactions with intelligent caching and rate limiting
    
    Key features:
    - Respects FRED's 120 requests/minute limit with adaptive delays
    - Caches responses to prevent duplicate API calls for the same data
    - Tracks failed requests to avoid retrying known errors
    
    Note: Cache is primarily for deduplication within a series, not across
    series, since each series has unique data. Low cache hit rates (10-20%)
    are expected when processing multiple series.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.request_count = 0
        self.start_time = time.time()
        self.response_times = []
        self.vintage_cache = {}
        self.failed_requests = set()
    
    def _calculate_adaptive_delay(self):
        """Dynamically adjust delay based on API response times"""
        base_delay = 0.5
        
        # Use recent response times to calculate optimal delay
        if len(self.response_times) >= 5:
            recent_times = self.response_times[-10:]
            avg_response_time = sum(recent_times) / len(recent_times)
            adaptive_delay = max(base_delay, avg_response_time * 1.5)
            return min(adaptive_delay, 2.0)
        
        return base_delay
    
    def _get_cache_key(self, series_id: str, realtime_date: str, obs_start: str, obs_end: str):
        """Generate unique identifier for caching API responses"""
        key_data = f"{series_id}|{realtime_date}|{obs_start}|{obs_end}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _api_call_with_retry(self, endpoint: str, params: dict, max_retries=3):
        """Execute API call with retry logic and cache management"""
        params.update({'api_key': self.api_key, 'file_type': 'json'})
        
        # Check cache before making API call
        if endpoint == 'series/observations':
            cache_key = self._get_cache_key(
                params.get('series_id', ''),
                params.get('realtime_start', ''),
                params.get('observation_start', ''),
                params.get('observation_end', '')
            )
            if cache_key in self.vintage_cache:
                return self.vintage_cache[cache_key]
            if cache_key in self.failed_requests:
                return None
        
        # Execute request with exponential backoff on failures
        for attempt in range(max_retries):
            try:
                self.request_count += 1
                time.sleep(self._calculate_adaptive_delay())
                
                # Progress indicator every 50 requests
                if self.request_count % 50 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.request_count / (elapsed / 60)
                    print(f"Progress: {self.request_count} calls, {rate:.0f}/min, {len(self.vintage_cache)} cached")
                
                start_request = time.time()
                response = requests.get(
                    f"{self.base_url}/{endpoint}",
                    params=params,
                    timeout=30
                )
                request_time = time.time() - start_request
                self.response_times.append(request_time)
                
                # Keep only recent response times for adaptive delay calculation
                if len(self.response_times) > 20:
                    self.response_times = self.response_times[-20:]
                
                if response.status_code == 200:
                    result = response.json()
                    if endpoint == 'series/observations':
                        self.vintage_cache[cache_key] = result
                    return result
                elif response.status_code == 429:
                    # Rate limited - wait with exponential backoff
                    wait_time = min(20, 5 * (2 ** attempt))
                    print(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 400:
                    # Bad request - mark as failed to avoid retrying
                    if endpoint == 'series/observations':
                        self.failed_requests.add(cache_key)
                    return None
                else:
                    return None
                    
            except Exception:
                if attempt == max_retries - 1:
                    if endpoint == 'series/observations':
                        self.failed_requests.add(cache_key)
                    return None
                else:
                    time.sleep(2 * (attempt + 1))
                    continue
        
        return None
    
    def get_data_as_of_date(self, series_id: str, realtime_date: str, obs_start: str, obs_end: str):
        """Retrieve vintage data snapshot as it existed on a specific date"""
        response = self._api_call_with_retry('series/observations', {
            'series_id': series_id,
            'observation_start': obs_start,
            'observation_end': obs_end,
            'realtime_start': realtime_date,
            'realtime_end': realtime_date
        })
        
        if not response or 'observations' not in response:
            return {}
        
        # Parse observations into dictionary
        data = {}
        for obs in response['observations']:
            if obs['value'] != '.':
                try:
                    data[obs['date']] = float(obs['value'])
                except (ValueError, TypeError):
                    continue
        return data
    
    def _calculate_vintage_dates_optimized(self, obs_months):
        """
        Calculate when each data point was released and subsequently revised
        
        BLS releases employment data approximately 1 week into the following month,
        then revises it in the next two monthly releases (t+1 and t+2 revisions).
        """
        vintage_requests = {}
        
        for obs_date in obs_months:
            # Initial release is ~6 days into following month
            release_month = obs_date + relativedelta(months=1)
            initial_release = release_month + timedelta(days=6)
            first_revision = initial_release + relativedelta(months=1)
            second_revision = initial_release + relativedelta(months=2)
            
            vintage_dates = [
                initial_release.strftime('%Y-%m-%d'),
                first_revision.strftime('%Y-%m-%d'),
                second_revision.strftime('%Y-%m-%d')
            ]
            
            # Track which observation dates are needed for each vintage
            for vintage_date in vintage_dates:
                if vintage_date not in vintage_requests:
                    vintage_requests[vintage_date] = set()
                vintage_requests[vintage_date].add(obs_date)
        
        return vintage_requests
    
    def analyze_series_revisions_optimized(self, series_id: str, start_date: str, end_date: str):
        """Build complete revision history for a single employment series"""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        obs_months = pd.date_range(start=start_dt, end=end_dt, freq='MS')
        
        # Use consistent extended date range across all chunks for better caching
        global_start = (obs_months[0] - relativedelta(months=3)).strftime('%Y-%m-%d')
        global_end = (obs_months[-1] + relativedelta(months=4)).strftime('%Y-%m-%d')
        
        # Calculate all vintage dates upfront
        vintage_requests = self._calculate_vintage_dates_optimized(obs_months)
        
        # Fetch all vintage snapshots using the same date range (maximizes cache hits)
        vintage_data = {}
        for vintage_date in sorted(vintage_requests.keys()):
            data = self.get_data_as_of_date(
                series_id, vintage_date, global_start, global_end
            )
            vintage_data[vintage_date] = data
        
        # Process all revisions
        revisions = self._process_chunk_revisions(series_id, obs_months, vintage_data)
        
        return pd.DataFrame(revisions) if revisions else None
    
    def _process_chunk_revisions(self, series_id: str, obs_months, vintage_data):
        """
        Extract initial estimates and revisions from vintage snapshots.
        
        Calculates both incremental revisions (t to t1, t1 to t2) and cumulative
        revisions (t to t2) to show how estimates evolve over time.
        """
        revisions = []

        for obs_date in obs_months:
            obs_date_str = obs_date.strftime('%Y-%m-%d')

            # Calculate vintage dates: initial release ~6 days after month, then monthly revisions
            release_month = obs_date + relativedelta(months=1)
            initial_release = release_month + timedelta(days=6)
            first_revision = initial_release + relativedelta(months=1)
            second_revision = initial_release + relativedelta(months=2)

            initial_date_str = initial_release.strftime('%Y-%m-%d')
            first_rev_date_str = first_revision.strftime('%Y-%m-%d')
            second_rev_date_str = second_revision.strftime('%Y-%m-%d')

            # Pull the three estimates as of those vintages
            estimates = {}
            vintages = {}

            if initial_date_str in vintage_data and obs_date_str in vintage_data[initial_date_str]:
                estimates['t'] = vintage_data[initial_date_str][obs_date_str]
                vintages['t'] = initial_release

            if first_rev_date_str in vintage_data and obs_date_str in vintage_data[first_rev_date_str]:
                estimates['t1'] = vintage_data[first_rev_date_str][obs_date_str]
                vintages['t1'] = first_revision

            if second_rev_date_str in vintage_data and obs_date_str in vintage_data[second_rev_date_str]:
                estimates['t2'] = vintage_data[second_rev_date_str][obs_date_str]
                vintages['t2'] = second_revision

            # Build record with estimates and vintages
            if 't' in estimates:
                est_t  = estimates.get('t')
                est_t1 = estimates.get('t1')
                est_t2 = estimates.get('t2')

                rec = {
                    'series_id': series_id,
                    'obs_date': obs_date,
                    'estimate_t': est_t,
                    'estimate_t1': est_t1,
                    'estimate_t2': est_t2,
                    'vintage_t':  vintages.get('t'),
                    'vintage_t1': vintages.get('t1'),
                    'vintage_t2': vintages.get('t2'),
                }

                # First revision: t to t1 (incremental change after first month)
                if est_t is not None and est_t1 is not None:
                    rec['revision_1month'] = est_t1 - est_t
                    rec['revision_1month_pct'] = ((rec['revision_1month'] / est_t) * 100) if est_t else None

                # Second revision: t1 to t2 (incremental change after second month)
                if est_t1 is not None and est_t2 is not None:
                    rec['revision_2month'] = est_t2 - est_t1
                    rec['revision_2month_pct'] = ((rec['revision_2month'] / est_t1) * 100) if est_t1 else None

                # Also store as explicit incremental revision
                if est_t1 is not None and est_t2 is not None:
                    rec['revision_t1_to_t2'] = est_t2 - est_t1

                # Cumulative revision from initial to final (t to t2)
                if est_t is not None and est_t2 is not None:
                    rec['rev2_cum'] = est_t2 - est_t
                    rec['rev2_cum_pct'] = ((rec['rev2_cum'] / est_t) * 100) if est_t else None
                else:
                    rec['rev2_cum'] = None
                    rec['rev2_cum_pct'] = None

                # Incremental second-step revision (same as revision_2month, explicit alias)
                if est_t1 is not None and est_t2 is not None:
                    rev2_incr = est_t2 - est_t1
                    rec['rev2_incr'] = rev2_incr
                    rec['rev2_incr_pct'] = ((rev2_incr / est_t1) * 100) if est_t1 else None
                else:
                    rec['rev2_incr'] = None
                    rec['rev2_incr_pct'] = None

                revisions.append(rec)

        return revisions

    def build_dataset(self, series_list, start_date, end_date):
        """Compile revision data across multiple employment series"""
        print(f"\nAnalyzing {len(series_list)} series from {start_date} to {end_date}")
        
        all_data = []
        successful = 0
        
        for i, series_id in enumerate(series_list):
            print(f"{i+1}/{len(series_list)}: {series_id}")
            series_data = self.analyze_series_revisions_optimized(series_id, start_date, end_date)
            
            if series_data is not None and not series_data.empty:
                all_data.append(series_data)
                successful += 1
            
            # Brief pause between series to avoid rate limiting
            if i < len(series_list) - 1:
                time.sleep(1)
        
        if all_data:
            final_dataset = pd.concat(all_data, ignore_index=True)
            elapsed = time.time() - self.start_time
            print(f"\nCompleted: {len(final_dataset):,} records from {successful}/{len(series_list)} series in {elapsed:.0f}s")
            return final_dataset
        else:
            print("No data collected")
            return pd.DataFrame()


class BLSExtractor:
    """
    Downloads and parses BLS industry classification files
    
    Provides mapping between industry codes, NAICS codes, and series IDs
    for navigating the BLS employment statistics hierarchy
    """
    
    def __init__(self):
        self.base_url = "https://download.bls.gov/pub/time.series/ce/"

    def download_bls_file(self, filename: str):
        """Fetch BLS reference file from public data repository"""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        urls = [
            f"{self.base_url}{filename}",
            f"https://download.bls.gov/pub/time.series/ce/{filename}"
        ]
        
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 200:
                    return response.text
            except Exception:
                continue
        
        raise Exception("Could not download BLS industry file")

    def parse_industry_file(self, content: str):
        """Parse tab-delimited BLS industry classification format"""
        lines = content.strip().split('\n')
        industries = []
        
        for line in lines[1:]:  # Skip header row
            parts = line.split('\t')
            if len(parts) >= 5:
                try:
                    industries.append({
                        'industry_code': str(parts[0]).strip().zfill(8),
                        'naics_code': str(parts[1]).strip(),
                        'industry_name': str(parts[3]).strip(),
                        'display_level': int(parts[4]) if parts[4].isdigit() else 0
                    })
                except Exception:
                    continue
        
        return pd.DataFrame(industries)

    def get_supersector_mapping(self):
        """Map BLS supersector codes to industry names"""
        return {
            '00': 'Total nonfarm', '05': 'Total private', '06': 'Goods-producing',
            '07': 'Service-providing', '08': 'Private service-providing',
            '10': 'Mining and logging', '20': 'Construction', '30': 'Manufacturing',
            '31': 'Durable goods', '32': 'Nondurable goods',
            '40': 'Trade, transportation, and utilities', '41': 'Wholesale trade',
            '42': 'Retail trade', '43': 'Transportation and warehousing',
            '44': 'Utilities', '50': 'Information', '55': 'Financial activities',
            '60': 'Professional and business services', '65': 'Education and health services',
            '70': 'Leisure and hospitality', '80': 'Other services', '90': 'Government'
        }
    
    def get_supersector_hierarchy(self):
        """
        Map parent supersectors to their children
        
        Some supersectors like Manufacturing (30) aggregate multiple child
        supersectors (31, 32). When analyzing "30", we need to include both children.
        """
        return {
            '30': ['31', '32'],  # Manufacturing -> Durable, Nondurable
            '40': ['41', '42', '43', '44'],  # Trade/Transport/Utilities -> children
        }
    
    def get_effective_supersector_codes(self, supersector_code: str):
        """
        Get all supersector codes to search for a given code
        
        Returns the code itself plus any children if it's a parent aggregate.
        This handles cases like Manufacturing (30) which contains Durable (31)
        and Nondurable (32) goods.
        """
        hierarchy = self.get_supersector_hierarchy()
        
        if supersector_code in hierarchy:
            # Parent code with children - return all children
            return hierarchy[supersector_code]
        else:
            # Leaf code - return as-is
            return [supersector_code]

    def get_universe(self):
        """Build complete industry hierarchy with series IDs and metadata"""
        content = self.download_bls_file('ce.industry')
        df = self.parse_industry_file(content)
        supersector_map = self.get_supersector_mapping()
        
        # Add supersector information to each industry
        df['supersector_code'] = df['industry_code'].str[:2]
        df['supersector_name'] = df['supersector_code'].map(supersector_map)
        
        # Generate seasonally adjusted (SA) and not seasonally adjusted (NSA) series IDs
        df['sa_series_id'] = 'CES' + df['industry_code'] + '01'
        df['nsa_series_id'] = 'CEU' + df['industry_code'] + '01'
        
        return df.sort_values(['supersector_code', 'display_level', 'industry_code']).reset_index(drop=True)


def analyze_supersector_detailed(api_key, supersector_code, detail_level, start_date, end_date):
    """
    Analyze all detailed industries at specified level within a supersector
    
    Automatically handles parent supersectors by including all child supersectors.
    For example, requesting Manufacturing (30) will analyze both Durable (31)
    and Nondurable (32) goods.
    """
    extractor = BLSExtractor()
    bls_universe = extractor.get_universe()
    
    # Get all supersector codes to search (handles parent-child relationships)
    effective_codes = extractor.get_effective_supersector_codes(supersector_code)
    supersector_map = extractor.get_supersector_mapping()
    
    # Show available levels for this supersector
    matching_industries = bls_universe[bls_universe['supersector_code'].isin(effective_codes)]
    if not matching_industries.empty:
        available_levels = matching_industries['display_level'].value_counts().sort_index()
        print(f"\nAvailable levels for supersector {supersector_code} (searching codes: {effective_codes}):")
        for level, count in available_levels.items():
            print(f"  Level {level}: {count} industries")
        print(f"\nUsing level {detail_level} for analysis")
    
    # Filter to requested detail level
    supersector_detailed = bls_universe[
        (bls_universe['supersector_code'].isin(effective_codes)) & 
        (bls_universe['display_level'] == detail_level)
    ].copy()
    
    if supersector_detailed.empty:
        print(f"\nNo level {detail_level} industries found")
        print(f"Try a different DETAILED_LEVEL value (typically 3 or 4)")
        return None
    
    # Use the requested supersector name for the filename
    supersector_name = supersector_map.get(supersector_code, f"Supersector_{supersector_code}")
    detailed_series = supersector_detailed['nsa_series_id'].tolist()
    
    print(f"\n{'='*60}")
    print(f"Running: {supersector_name} (code {supersector_code}, level {detail_level})")
    print(f"Found {len(detailed_series)} series across codes {effective_codes}")
    print(f"{'='*60}")
    
    # Run analysis
    analyzer = OptimizedBLSAnalyzer(api_key)
    revision_dataset = analyzer.build_dataset(detailed_series, start_date, end_date)
    
    if not revision_dataset.empty:
        # Merge industry names and metadata
        revision_dataset = revision_dataset.merge(
            supersector_detailed[['nsa_series_id', 'industry_name', 'supersector_name']],
            left_on='series_id',
            right_on='nsa_series_id',
            how='left'
        )
        
        # Save to CSV
        safe_name = supersector_name.replace(' ', '_').replace(',', '').replace('/', '_')
        filename = f"bls_revisions_level{detail_level}_{supersector_code}_{safe_name}.csv"
        revision_dataset.to_csv(filename, index=False)
        print(f"Saved: {filename}")
        
        return revision_dataset
    else:
        print("No data collected")
        return None


def main():
    """Execute BLS revision analysis based on configuration settings"""
    print("BLS Revision Analysis")
    print(f"Date range: {START_DATE} to {END_DATE}\n")
    
    extractor = BLSExtractor()
    bls_universe = extractor.get_universe()
    
    if bls_universe.empty:
        print("Could not load BLS universe")
        return {}
    
    results = {}
    
    # Analyze broad industries (level 2) plus total nonfarm
    if RUN_LEVEL_2_ANALYSIS:
        print(f"\n{'='*60}")
        print("Running: Level 2 + PAYNSA")
        print(f"{'='*60}")
        
        level_2 = bls_universe[bls_universe['display_level'] == 2].copy()
        nsa_series = level_2['nsa_series_id'].tolist()
        nsa_series.append('PAYNSA')  # Add total nonfarm NSA series
        
        analyzer = OptimizedBLSAnalyzer(API_KEY)
        revision_dataset = analyzer.build_dataset(nsa_series, START_DATE, END_DATE)
        
        if not revision_dataset.empty:
            # Merge industry metadata
            revision_dataset = revision_dataset.merge(
                level_2[['nsa_series_id', 'industry_name', 'supersector_name', 'display_level']],
                left_on='series_id',
                right_on='nsa_series_id',
                how='left'
            )
            
            # Add metadata for PAYNSA (not in BLS universe file)
            revision_dataset.loc[revision_dataset['series_id'] == 'PAYNSA', 'industry_name'] = 'Total Nonfarm (PAYNSA)'
            revision_dataset.loc[revision_dataset['series_id'] == 'PAYNSA', 'supersector_name'] = 'Total nonfarm'
            revision_dataset.loc[revision_dataset['series_id'] == 'PAYNSA', 'display_level'] = 0
            
            filename = "bls_revisions_level2_with_paynsa.csv"
            revision_dataset.to_csv(filename, index=False)
            print(f"Saved: {filename}")
            
            results['level_2'] = revision_dataset
        else:
            print("No data collected")
    
    # Analyze detailed industries within a supersector
    if RUN_DETAILED_ANALYSIS:
        data = analyze_supersector_detailed(
            API_KEY, DETAILED_SUPERSECTOR_CODE, DETAILED_LEVEL,
            START_DATE, END_DATE
        )
        if data is not None:
            results['detailed'] = data
    
    return results


if __name__ == "__main__":
    results = main()
    
    print(f"\n{'='*60}")
    print("Analysis complete")
    for key, data in results.items():
        print(f"{key}: {len(data):,} records")
    print(f"{'='*60}")
