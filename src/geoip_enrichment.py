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

def _find_default_db_path(filename: str) -> str:
    """Locate GeoLite2 database file in data/geoip/ or data/."""
    p1 = os.path.join("data", "geoip", filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join("data", filename)
    if os.path.exists(p2):
        return p2
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
        
        Private/synthetic/invalid IPs strictly return 'Unknown' for Country and ASN.
        Never fabricates location or network entity information.
        """
        clean_ip = str(ip_str).strip() if ip_str is not None else ""

        # Validate IP syntax and private status
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
                "country": "Unknown",
                "country_code": "Unknown",
                "asn": "Unknown",
                "asn_org": "Unknown",
                "lookup_status": "invalid_ip"
            }

        # Private / synthetic IP handling (strictly return Unknown)
        if is_priv:
            return {
                "country": "Unknown",
                "country_code": "Unknown",
                "asn": "Unknown",
                "asn_org": "Unknown",
                "lookup_status": "private_ip"
            }

        # If MMDB databases are not loaded, fallback cleanly to existing dataset attributes
        if not self.databases_loaded:
            fb_c = fallback_country if fallback_country and str(fallback_country).strip().upper() not in ["NAN", "NONE", ""] else "Unknown"
            fb_a = fallback_asn if fallback_asn and str(fallback_asn).strip().upper() not in ["NAN", "NONE", ""] else "Unknown"
            return {
                "country": fb_c,
                "country_code": fb_c,
                "asn": fb_a,
                "asn_org": "Unknown",
                "lookup_status": "synthetic_fallback"
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

        if result["country"] != "Unknown" or result["asn"] != "Unknown":
            result["lookup_status"] = "geoip_resolved"

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

        # Optimization: Cache IP lookups across unique IP addresses
        has_src_c = 'src_country' in df_enriched.columns
        has_src_a = 'src_asn' in df_enriched.columns
        has_dst_c = 'dst_country' in df_enriched.columns
        has_dst_a = 'dst_asn' in df_enriched.columns

        unique_src_ips = df_enriched['src_ip'].dropna().unique() if 'src_ip' in df_enriched.columns else []
        unique_dst_ips = df_enriched['dst_ip'].dropna().unique() if 'dst_ip' in df_enriched.columns else []

        src_lookup_map = {}
        for ip in unique_src_ips:
            fb_c = "Unknown"
            fb_a = "Unknown"
            if has_src_c:
                row_match = df_enriched[df_enriched['src_ip'] == ip]
                if not row_match.empty:
                    fb_c = str(row_match['src_country'].iloc[0])
            if has_src_a:
                row_match = df_enriched[df_enriched['src_ip'] == ip]
                if not row_match.empty:
                    fb_a = str(row_match['src_asn'].iloc[0])
            src_lookup_map[ip] = self.lookup_ip(ip, fallback_country=fb_c, fallback_asn=fb_a)

        dst_lookup_map = {}
        for ip in unique_dst_ips:
            fb_c = "Unknown"
            fb_a = "Unknown"
            if has_dst_c:
                row_match = df_enriched[df_enriched['dst_ip'] == ip]
                if not row_match.empty:
                    fb_c = str(row_match['dst_country'].iloc[0])
            if has_dst_a:
                row_match = df_enriched[df_enriched['dst_ip'] == ip]
                if not row_match.empty:
                    fb_a = str(row_match['dst_asn'].iloc[0])
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


def get_geoip_status(
    country_db_path: Optional[str] = None,
    asn_db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Query local GeoIP database availability without running full enrichment."""
    enricher = GeoIPEnricher(country_db_path=country_db_path, asn_db_path=asn_db_path)
    try:
        return enricher.get_status()
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
