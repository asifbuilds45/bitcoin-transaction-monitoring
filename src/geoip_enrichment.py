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

    def __init__(
        self,
        country_db_path: str = os.path.join("data", "GeoLite2-Country.mmdb"),
        asn_db_path: str = os.path.join("data", "GeoLite2-ASN.mmdb")
    ):
        self.country_db_path = country_db_path
        self.asn_db_path = asn_db_path
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
        if not self.databases_loaded:
            print("GeoIP database not available — using synthetic dataset enrichment.")

    def is_private_or_synthetic(self, ip_str: str) -> bool:
        """Check if an IP is private, loopback, link-local, or synthetic."""
        try:
            ip_obj = ipaddress.ip_address(ip_str.strip())
            return ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
        except ValueError:
            return True

    def lookup_ip(self, ip_str: str, fallback_country: str = "UNKNOWN", fallback_asn: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Perform local offline lookup for a single IP address.
        
        Args:
            ip_str: Target IP address string.
            fallback_country: Country value from dataset to use if lookup fails/private.
            fallback_asn: ASN value from dataset to use if lookup fails/private.
            
        Returns:
            Dict containing country, country_code, asn, org, and lookup status.
        """
        result = {
            "country": fallback_country,
            "country_code": fallback_country,
            "asn": fallback_asn,
            "asn_org": "N/A",
            "lookup_status": "synthetic_fallback"
        }

        if not self.databases_loaded or self.is_private_or_synthetic(ip_str):
            # Synthetic 10.x.x.x or missing DB -> use synthetic fallback
            return result

        resolved_country = False
        resolved_asn = False

        # Lookup country
        if self.country_reader:
            try:
                resp = self.country_reader.country(ip_str)
                if resp.country and resp.country.name:
                    result["country"] = resp.country.name
                    result["country_code"] = resp.country.iso_code or fallback_country
                    resolved_country = True
            except (geoip2.errors.AddressNotFoundError, ValueError):
                pass
            except Exception as e:
                logger.debug(f"Country lookup error for {ip_str}: {e}")

        # Lookup ASN
        if self.asn_reader:
            try:
                resp_asn = self.asn_reader.asn(ip_str)
                if resp_asn.autonomous_system_number:
                    result["asn"] = f"AS{resp_asn.autonomous_system_number}"
                    result["asn_org"] = resp_asn.autonomous_system_organization or "Unknown Org"
                    resolved_asn = True
            except (geoip2.errors.AddressNotFoundError, ValueError):
                pass
            except Exception as e:
                logger.debug(f"ASN lookup error for {ip_str}: {e}")

        if resolved_country or resolved_asn:
            result["lookup_status"] = "geoip_resolved"
        else:
            result["lookup_status"] = "synthetic_fallback"

        return result

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich dataframe with GeoIP and ASN metadata without modifying input df.
        
        Adds columns:
        - geoip_src_country
        - geoip_dst_country
        - geoip_src_asn
        - geoip_dst_asn
        - geoip_lookup_status
        """
        df_enriched = df.copy(deep=True)

        # Optimization: Cache IP lookups across repeated IP addresses
        unique_src_ips = df_enriched[['src_ip', 'src_country', 'src_asn']].drop_duplicates()
        unique_dst_ips = df_enriched[['dst_ip', 'dst_country', 'dst_asn']].drop_duplicates()

        src_lookup_map = {}
        for _, row in unique_src_ips.iterrows():
            src_lookup_map[row['src_ip']] = self.lookup_ip(
                row['src_ip'],
                fallback_country=str(row['src_country']),
                fallback_asn=str(row['src_asn'])
            )

        dst_lookup_map = {}
        for _, row in unique_dst_ips.iterrows():
            dst_lookup_map[row['dst_ip']] = self.lookup_ip(
                row['dst_ip'],
                fallback_country=str(row['dst_country']),
                fallback_asn=str(row['dst_asn'])
            )

        # Vectorized mapping
        df_enriched['geoip_src_country'] = df_enriched['src_ip'].map(lambda ip: src_lookup_map.get(ip, {}).get('country', 'UNKNOWN'))
        df_enriched['geoip_dst_country'] = df_enriched['dst_ip'].map(lambda ip: dst_lookup_map.get(ip, {}).get('country', 'UNKNOWN'))
        df_enriched['geoip_src_asn'] = df_enriched['src_ip'].map(lambda ip: src_lookup_map.get(ip, {}).get('asn', 'UNKNOWN'))
        df_enriched['geoip_dst_asn'] = df_enriched['dst_ip'].map(lambda ip: dst_lookup_map.get(ip, {}).get('asn', 'UNKNOWN'))
        df_enriched['geoip_lookup_status'] = df_enriched['src_ip'].map(lambda ip: src_lookup_map.get(ip, {}).get('lookup_status', 'synthetic_fallback'))

        return df_enriched

    def close(self) -> None:
        """Close database handles cleanly."""
        if self.country_reader:
            self.country_reader.close()
        if self.asn_reader:
            self.asn_reader.close()


def enrich_transactions_with_geoip(
    df: pd.DataFrame,
    country_db_path: str = os.path.join("data", "GeoLite2-Country.mmdb"),
    asn_db_path: str = os.path.join("data", "GeoLite2-ASN.mmdb")
) -> pd.DataFrame:
    """Convenience wrapper for offline GeoIP enrichment."""
    enricher = GeoIPEnricher(country_db_path=country_db_path, asn_db_path=asn_db_path)
    try:
        return enricher.enrich_dataframe(df)
    finally:
        enricher.close()
