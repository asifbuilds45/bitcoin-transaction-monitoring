"""
Offline GeoIP Enrichment Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

NOTE: GeoIP is an offline network-layer IP enrichment component and is NOT
the anomaly detection algorithm.
"""

import os
import logging
import ipaddress
import pandas as pd
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Try importing geoip2 safely
try:
    import geoip2.database
    import geoip2.errors
    GEOIP2_AVAILABLE = True
except ImportError:
    GEOIP2_AVAILABLE = False


class GeoIPEnricher:
    """
    Offline GeoIP and ASN enrichment engine using local MaxMind GeoLite2 databases.
    Operates strictly locally with zero outbound network calls.
    """

# Standard ISO-3166-1 alpha-2 country code mappings for clean display
ISO_COUNTRY_NAMES = {
    "IN": "India", "US": "United States", "GB": "United Kingdom", "CA": "Canada",
    "AU": "Australia", "SG": "Singapore", "DE": "Germany", "FR": "France",
    "JP": "Japan", "CN": "China", "RU": "Russia", "BR": "Brazil",
    "NL": "Netherlands", "CH": "Switzerland", "KR": "South Korea", "SE": "Sweden",
    "NO": "Norway", "FI": "Finland", "ES": "Spain", "IT": "Italy",
    "ZA": "South Africa", "MX": "Mexico", "ID": "Indonesia", "TR": "Turkey",
    "SA": "Saudi Arabia", "AE": "United Arab Emirates", "IL": "Israel",
    "PL": "Poland", "UA": "Ukraine", "RO": "Romania", "NZ": "New Zealand",
    "IE": "Ireland", "AT": "Austria", "BE": "Belgium", "CZ": "Czech Republic",
    "DK": "Denmark", "PT": "Portugal", "GR": "Greece", "HU": "Hungary",
    "TH": "Thailand", "VN": "Vietnam", "MY": "Malaysia", "PH": "Philippines",
    "HK": "Hong Kong", "TW": "Taiwan", "AR": "Argentina", "CL": "Chile",
    "CO": "Colombia", "EG": "Egypt", "NG": "Nigeria", "KE": "Kenya"
}


def _find_default_db_path(filename: str) -> str:
    """Locate GeoLite2 database file in data/geoip/ or data/ with robust absolute root lookup."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    p1 = os.path.join(base_dir, "data", "geoip", filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(base_dir, "data", filename)
    if os.path.exists(p2):
        return p2
    # Fallback to current working directory relative paths
    p3 = os.path.join("data", "geoip", filename)
    if os.path.exists(p3):
        return p3
    p4 = os.path.join("data", filename)
    if os.path.exists(p4):
        return p4
    return p1


class GeoIPEnricher:
    """
    Offline GeoIP and ASN enrichment engine using local MaxMind GeoLite2 databases.
    Operates strictly locally with zero outbound network calls.
    """

    def __init__(
        self,
        country_db_path: Optional[str] = None,
        asn_db_path: Optional[str] = None
    ):
        self.country_db_path = country_db_path or _find_default_db_path("GeoLite2-Country.mmdb")
        self.asn_db_path = asn_db_path or _find_default_db_path("GeoLite2-ASN.mmdb")
        self.country_reader = None
        self.asn_reader = None
        self.databases_loaded = False
        
        self._init_readers()

    def _init_readers(self) -> None:
        """Initialize local MMDB readers if available."""
        if not GEOIP2_AVAILABLE:
            logger.warning("geoip2 library not installed — using synthetic dataset enrichment.")
            return

        has_country = os.path.exists(self.country_db_path)
        has_asn = os.path.exists(self.asn_db_path)

        if has_country:
            try:
                self.country_reader = geoip2.database.Reader(self.country_db_path)
            except Exception as e:
                logger.warning(f"Failed to open GeoIP Country DB at {self.country_db_path}: {e}")

        if has_asn:
            try:
                self.asn_reader = geoip2.database.Reader(self.asn_db_path)
            except Exception as e:
                logger.warning(f"Failed to open GeoIP ASN DB at {self.asn_db_path}: {e}")

        self.databases_loaded = (self.country_reader is not None or self.asn_reader is not None)

    def is_private_or_synthetic(self, ip_str: str) -> bool:
        """Check if an IP is private, loopback, link-local, multicast, or synthetic."""
        try:
            ip_obj = ipaddress.ip_address(ip_str.strip())
            return (
                ip_obj.is_private or
                ip_obj.is_loopback or
                ip_obj.is_link_local or
                ip_obj.is_multicast or
                ip_obj.is_reserved
            )
        except ValueError:
            return True

    def get_status(self) -> Dict[str, Any]:
        """Return status dictionary of local GeoLite2 databases."""
        country_loaded = self.country_reader is not None
        asn_loaded = self.asn_reader is not None
        is_offline = country_loaded or asn_loaded
        return {
            "country_db_loaded": country_loaded,
            "asn_db_loaded": asn_loaded,
            "country_status": "Loaded" if country_loaded else "Not available",
            "asn_status": "Loaded" if asn_loaded else "Not available",
            "mode": "Offline" if is_offline else "Fallback",
            "country_db_path": self.country_db_path,
            "asn_db_path": self.asn_db_path
        }

    def lookup_ip(self, ip_str: str, fallback_country: str = "Unknown", fallback_asn: str = "Unknown") -> Dict[str, Any]:
        """
        Perform local offline lookup for a single IP address.
        
        Public IPs are queried against local MaxMind GeoLite2 databases.
        Private/synthetic IPs preserve dataset telemetry fallback when present,
        or explicitly return 'Unknown' when no telemetry exists.
        Never fabricates location or network entity information.
        """
        clean_ip = str(ip_str).strip() if ip_str is not None else ""

        # Normalize fallback values
        has_fb_c = bool(fallback_country and str(fallback_country).strip().upper() not in ["NAN", "NONE", "UNKNOWN", ""])
        has_fb_a = bool(fallback_asn and str(fallback_asn).strip().upper() not in ["NAN", "NONE", "UNKNOWN", ""])

        fb_c_raw = str(fallback_country).strip() if has_fb_c else "Unknown"
        fb_a_raw = str(fallback_asn).strip() if has_fb_a else "Unknown"
        fb_c_name = ISO_COUNTRY_NAMES.get(fb_c_raw.upper(), fb_c_raw) if has_fb_c else "Unknown"
        fb_c_code = fb_c_raw.upper() if (has_fb_c and len(fb_c_raw) == 2) else ("Unknown" if fb_c_raw == "Unknown" else fb_c_raw)

        # 1. Missing IP handling
        if not clean_ip:
            return {
                "country": fb_c_name if has_fb_c else "Unknown",
                "country_code": fb_c_code if has_fb_c else "Unknown",
                "asn": fb_a_raw if has_fb_a else "Unknown",
                "asn_org": "Unknown",
                "lookup_status": "synthetic_fallback" if (has_fb_c or has_fb_a) else "missing_ip"
            }

        # 2. Validate IP syntax and private status
        try:
            ip_obj = ipaddress.ip_address(clean_ip)
            is_priv = (
                ip_obj.is_private or
                ip_obj.is_loopback or
                ip_obj.is_link_local or
                ip_obj.is_multicast or
                ip_obj.is_reserved
            )
        except ValueError:
            return {
                "country": fb_c_name if has_fb_c else "Unknown",
                "country_code": fb_c_code if has_fb_c else "Unknown",
                "asn": fb_a_raw if has_fb_a else "Unknown",
                "asn_org": "Unknown",
                "lookup_status": "synthetic_fallback" if (has_fb_c or has_fb_a) else "invalid_ip"
            }

        # 3. Private / synthetic IP handling
        if is_priv:
            if has_fb_c or has_fb_a:
                return {
                    "country": fb_c_name,
                    "country_code": fb_c_code,
                    "asn": fb_a_raw,
                    "asn_org": "Unknown",
                    "lookup_status": "synthetic_fallback"
                }
            return {
                "country": "Unknown",
                "country_code": "Unknown",
                "asn": "Unknown",
                "asn_org": "Unknown",
                "lookup_status": "private_ip"
            }

        # 4. If MMDB databases are not loaded, fallback cleanly
        if not self.databases_loaded:
            return {
                "country": fb_c_name,
                "country_code": fb_c_code,
                "asn": fb_a_raw,
                "asn_org": "Unknown",
                "lookup_status": "synthetic_fallback" if (has_fb_c or has_fb_a) else "not_found"
            }

        result = {
            "country": "Unknown",
            "country_code": "Unknown",
            "asn": "Unknown",
            "asn_org": "Unknown",
            "lookup_status": "not_found"
        }

        # Lookup country in GeoLite2 Country MMDB
        if self.country_reader:
            try:
                resp = self.country_reader.country(clean_ip)
                if resp.country and resp.country.name:
                    result["country"] = resp.country.name
                    result["country_code"] = resp.country.iso_code or "Unknown"
            except (geoip2.errors.AddressNotFoundError, ValueError):
                pass
            except Exception as e:
                logger.debug(f"Country lookup error for {clean_ip}: {e}")

        # Lookup ASN in GeoLite2 ASN MMDB
        if self.asn_reader:
            try:
                resp_asn = self.asn_reader.asn(clean_ip)
                if resp_asn.autonomous_system_number:
                    result["asn"] = f"AS{resp_asn.autonomous_system_number}"
                    result["asn_org"] = resp_asn.autonomous_system_organization or "Unknown"
            except (geoip2.errors.AddressNotFoundError, ValueError):
                pass
            except Exception as e:
                logger.debug(f"ASN lookup error for {clean_ip}: {e}")

        # Fallback to dataset metadata if MMDB didn't resolve specific attributes
        if result["country"] == "Unknown" and has_fb_c:
            result["country"] = fb_c_name
            result["country_code"] = fb_c_code

        if result["asn"] == "Unknown" and has_fb_a:
            result["asn"] = fb_a_raw

        # Set status appropriately
        if result["country"] != "Unknown" or result["asn"] != "Unknown":
            if (self.country_reader and result["country"] not in ["Unknown", fb_c_name]) or \
               (self.asn_reader and result["asn"] not in ["Unknown", fb_a_raw]):
                result["lookup_status"] = "geoip_resolved"
            else:
                result["lookup_status"] = "synthetic_fallback" if (has_fb_c or has_fb_a) else "geoip_resolved"

        return result

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich dataframe with GeoIP and ASN metadata without modifying input df.
        
        Adds columns:
        - geoip_src_country
        - geoip_dst_country
        - geoip_src_country_code
        - geoip_dst_country_code
        - geoip_src_asn
        - geoip_dst_asn
        - geoip_src_org
        - geoip_dst_org
        - geoip_lookup_status
        """
        df_enriched = df.copy(deep=True)

        has_src_c = 'src_country' in df_enriched.columns
        has_src_a = 'src_asn' in df_enriched.columns
        has_dst_c = 'dst_country' in df_enriched.columns
        has_dst_a = 'dst_asn' in df_enriched.columns

        unique_src_ips = df_enriched['src_ip'].dropna().unique() if 'src_ip' in df_enriched.columns else []
        unique_dst_ips = df_enriched['dst_ip'].dropna().unique() if 'dst_ip' in df_enriched.columns else []

        # Fast lookup mapping using first matching row values
        src_fb_map = {}
        if len(unique_src_ips) > 0 and (has_src_c or has_src_a):
            subset_cols = ['src_ip']
            if has_src_c: subset_cols.append('src_country')
            if has_src_a: subset_cols.append('src_asn')
            src_first = df_enriched[subset_cols].drop_duplicates(subset=['src_ip']).set_index('src_ip')
            for ip in unique_src_ips:
                fb_c = str(src_first.loc[ip, 'src_country']) if has_src_c and pd.notna(src_first.loc[ip, 'src_country']) else "Unknown"
                fb_a = str(src_first.loc[ip, 'src_asn']) if has_src_a and pd.notna(src_first.loc[ip, 'src_asn']) else "Unknown"
                src_fb_map[ip] = (fb_c, fb_a)

        dst_fb_map = {}
        if len(unique_dst_ips) > 0 and (has_dst_c or has_dst_a):
            subset_cols = ['dst_ip']
            if has_dst_c: subset_cols.append('dst_country')
            if has_dst_a: subset_cols.append('dst_asn')
            dst_first = df_enriched[subset_cols].drop_duplicates(subset=['dst_ip']).set_index('dst_ip')
            for ip in unique_dst_ips:
                fb_c = str(dst_first.loc[ip, 'dst_country']) if has_dst_c and pd.notna(dst_first.loc[ip, 'dst_country']) else "Unknown"
                fb_a = str(dst_first.loc[ip, 'dst_asn']) if has_dst_a and pd.notna(dst_first.loc[ip, 'dst_asn']) else "Unknown"
                dst_fb_map[ip] = (fb_c, fb_a)

        src_lookup_map = {}
        for ip in unique_src_ips:
            fb_c, fb_a = src_fb_map.get(ip, ("Unknown", "Unknown"))
            src_lookup_map[ip] = self.lookup_ip(ip, fallback_country=fb_c, fallback_asn=fb_a)

        dst_lookup_map = {}
        for ip in unique_dst_ips:
            fb_c, fb_a = dst_fb_map.get(ip, ("Unknown", "Unknown"))
            dst_lookup_map[ip] = self.lookup_ip(ip, fallback_country=fb_c, fallback_asn=fb_a)

        # Vectorized mapping
        if 'src_ip' in df_enriched.columns:
            df_enriched['geoip_src_country'] = df_enriched['src_ip'].map(lambda ip: src_lookup_map.get(ip, {}).get('country', 'Unknown'))
            df_enriched['geoip_src_country_code'] = df_enriched['src_ip'].map(lambda ip: src_lookup_map.get(ip, {}).get('country_code', 'Unknown'))
            df_enriched['geoip_src_asn'] = df_enriched['src_ip'].map(lambda ip: src_lookup_map.get(ip, {}).get('asn', 'Unknown'))
            df_enriched['geoip_src_org'] = df_enriched['src_ip'].map(lambda ip: src_lookup_map.get(ip, {}).get('asn_org', 'Unknown'))
            df_enriched['geoip_lookup_status'] = df_enriched['src_ip'].map(lambda ip: src_lookup_map.get(ip, {}).get('lookup_status', 'synthetic_fallback'))
        else:
            df_enriched['geoip_src_country'] = 'Unknown'
            df_enriched['geoip_src_country_code'] = 'Unknown'
            df_enriched['geoip_src_asn'] = 'Unknown'
            df_enriched['geoip_src_org'] = 'Unknown'
            df_enriched['geoip_lookup_status'] = 'synthetic_fallback'

        if 'dst_ip' in df_enriched.columns:
            df_enriched['geoip_dst_country'] = df_enriched['dst_ip'].map(lambda ip: dst_lookup_map.get(ip, {}).get('country', 'Unknown'))
            df_enriched['geoip_dst_country_code'] = df_enriched['dst_ip'].map(lambda ip: dst_lookup_map.get(ip, {}).get('country_code', 'Unknown'))
            df_enriched['geoip_dst_asn'] = df_enriched['dst_ip'].map(lambda ip: dst_lookup_map.get(ip, {}).get('asn', 'Unknown'))
            df_enriched['geoip_dst_org'] = df_enriched['dst_ip'].map(lambda ip: dst_lookup_map.get(ip, {}).get('asn_org', 'Unknown'))
        else:
            df_enriched['geoip_dst_country'] = 'Unknown'
            df_enriched['geoip_dst_country_code'] = 'Unknown'
            df_enriched['geoip_dst_asn'] = 'Unknown'
            df_enriched['geoip_dst_org'] = 'Unknown'

        # Ensure standard column names are populated from offline GeoIP resolution
        if 'src_country' not in df_enriched.columns or (df_enriched['src_country'].isin(['UNKNOWN', 'Unknown'])).all():
            df_enriched['src_country'] = df_enriched['geoip_src_country']
        else:
            df_enriched['src_country'] = df_enriched['geoip_src_country'].where(
                df_enriched['geoip_src_country'] != 'Unknown',
                df_enriched['src_country']
            )

        if 'dst_country' not in df_enriched.columns or (df_enriched['dst_country'].isin(['UNKNOWN', 'Unknown'])).all():
            df_enriched['dst_country'] = df_enriched['geoip_dst_country']
        else:
            df_enriched['dst_country'] = df_enriched['geoip_dst_country'].where(
                df_enriched['geoip_dst_country'] != 'Unknown',
                df_enriched['dst_country']
            )

        if 'src_asn' not in df_enriched.columns or (df_enriched['src_asn'].isin(['UNKNOWN', 'Unknown'])).all():
            df_enriched['src_asn'] = df_enriched['geoip_src_asn']
        else:
            df_enriched['src_asn'] = df_enriched['geoip_src_asn'].where(
                df_enriched['geoip_src_asn'] != 'Unknown',
                df_enriched['src_asn']
            )

        if 'dst_asn' not in df_enriched.columns or (df_enriched['dst_asn'].isin(['UNKNOWN', 'Unknown'])).all():
            df_enriched['dst_asn'] = df_enriched['geoip_dst_asn']
        else:
            df_enriched['dst_asn'] = df_enriched['geoip_dst_asn'].where(
                df_enriched['geoip_dst_asn'] != 'Unknown',
                df_enriched['dst_asn']
            )

        if 'country_count' not in df_enriched.columns:
            if 'source_wallet' in df_enriched.columns and 'src_country' in df_enriched.columns:
                c_map = df_enriched.groupby('source_wallet')['src_country'].nunique().to_dict()
                df_enriched['country_count'] = df_enriched['source_wallet'].map(c_map).fillna(1).astype(int)
            else:
                df_enriched['country_count'] = 1

        if 'asn_count' not in df_enriched.columns:
            if 'source_wallet' in df_enriched.columns and 'src_asn' in df_enriched.columns:
                a_map = df_enriched.groupby('source_wallet')['src_asn'].nunique().to_dict()
                df_enriched['asn_count'] = df_enriched['source_wallet'].map(a_map).fillna(1).astype(int)
            else:
                df_enriched['asn_count'] = 1

        return df_enriched

    def close(self) -> None:
        """Close database handles cleanly."""
        if self.country_reader:
            try:
                self.country_reader.close()
            except Exception:
                pass
            self.country_reader = None
        if self.asn_reader:
            try:
                self.asn_reader.close()
            except Exception:
                pass
            self.asn_reader = None


_STATUS_CACHE: Dict[Tuple[Optional[str], Optional[str]], Dict[str, Any]] = {}

def get_geoip_status(
    country_db_path: Optional[str] = None,
    asn_db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Query local GeoIP database availability without running full enrichment (cached)."""
    cache_key = (country_db_path, asn_db_path)
    if cache_key in _STATUS_CACHE:
        return _STATUS_CACHE[cache_key]
    
    enricher = GeoIPEnricher(country_db_path=country_db_path, asn_db_path=asn_db_path)
    try:
        status = enricher.get_status()
        _STATUS_CACHE[cache_key] = status
        return status
    finally:
        enricher.close()



def enrich_transactions_with_geoip(
    df: pd.DataFrame,
    country_db_path: Optional[str] = None,
    asn_db_path: Optional[str] = None
) -> pd.DataFrame:
    """Convenience wrapper for offline GeoIP enrichment."""
    enricher = GeoIPEnricher(country_db_path=country_db_path, asn_db_path=asn_db_path)
    try:
        return enricher.enrich_dataframe(df)
    finally:
        enricher.close()
