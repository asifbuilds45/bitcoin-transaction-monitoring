"""
PostgreSQL & Database Persistence Manager
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Provides scalable enterprise database persistence for Network Telemetry, Blockchain Transactions,
Correlations, Communities, Anomaly Results, and Investigation Alerts.
Supports local PostgreSQL with QueuePool connection pooling, automatic index creation,
and graceful fallback to local SQLite when operating in minimal environments.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from sqlalchemy import create_engine, text, inspect, Index
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# Load local environment if .env.local exists
def _load_env_file():
    env_file = Path(".env.local")
    if not env_file.exists():
        env_file = Path(__file__).resolve().parent.parent.parent / ".env.local"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() not in os.environ:
                            os.environ[k.strip()] = v.strip()
        except Exception as e:
            logger.debug(f"Could not parse .env.local: {e}")

_load_env_file()

DEFAULT_POSTGRES_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/bitcoin_monitoring"
)
DEFAULT_SQLITE_URI = f"sqlite:///{os.path.abspath(os.path.join('data', 'bitcoin_monitoring.db'))}"


class DatabaseManager:
    """
    Manages connections, schema initialization, indexing, and persistence
    for the Bitcoin Monitoring analysis pipeline.
    """

    def __init__(self, db_uri: Optional[str] = None):
        self.db_uri = db_uri or os.getenv("DATABASE_URL", DEFAULT_POSTGRES_URI)
        self.engine: Optional[Engine] = None
        self.db_type = "UNKNOWN"
        self.is_connected = False
        self.connection_status_msg = ""
        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize database engine with connection pooling. Try PostgreSQL first, fall back to SQLite."""
        # 1. Try PostgreSQL if configured
        if self.db_uri.startswith("postgresql"):
            try:
                pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
                max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
                pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))

                engine = create_engine(
                    self.db_uri,
                    poolclass=QueuePool,
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                    pool_timeout=pool_timeout,
                    pool_recycle=1800,
                    pool_pre_ping=True,
                    connect_args={"connect_timeout": 5}
                )
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self.engine = engine
                self.db_type = "PostgreSQL"
                self.is_connected = True
                self.connection_status_msg = f"Connected to local PostgreSQL database ({self.db_uri.split('@')[-1] if '@' in self.db_uri else 'PostgreSQL'})."
                logger.info(self.connection_status_msg)
                self._ensure_indexes()
                return
            except Exception as e:
                msg = f"PostgreSQL unavailable at {self.db_uri}. Using local SQLite fallback database. ({e})"
                logger.warning(msg)

        # 2. Fallback to local SQLite DB file
        try:
            os.makedirs("data", exist_ok=True)
            self.engine = create_engine(
                DEFAULT_SQLITE_URI,
                connect_args={"check_same_thread": False}
            )
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.db_type = "SQLite (Fallback)"
            self.is_connected = True
            self.connection_status_msg = "PostgreSQL not running — active on local SQLite database."
            logger.info(self.connection_status_msg)
        except Exception as e:
            self.is_connected = False
            self.connection_status_msg = f"Failed to connect to any database: {e}"
            logger.error(self.connection_status_msg)

    def _ensure_indexes(self) -> None:
        """Create indexes on known analysis tables for fast forensic queries."""
        if not self.is_connected or self.engine is None or self.db_type != "PostgreSQL":
            return
        
        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_tx_txid ON transactions (txid);",
            "CREATE INDEX IF NOT EXISTS idx_tx_src_ip ON transactions (src_ip);",
            "CREATE INDEX IF NOT EXISTS idx_tx_src_wallet ON transactions (source_wallet);",
            "CREATE INDEX IF NOT EXISTS idx_tx_dst_wallet ON transactions (destination_wallet);",
            "CREATE INDEX IF NOT EXISTS idx_tx_risk_level ON transactions (risk_level);",
            "CREATE INDEX IF NOT EXISTS idx_tx_is_anomaly ON transactions (is_anomaly);",
            "CREATE INDEX IF NOT EXISTS idx_alerts_txid ON alerts (txid);",
            "CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts (priority);",
        ]
        try:
            with self.engine.connect() as conn:
                for q in index_queries:
                    try:
                        conn.execute(text(q))
                        conn.commit()
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Index creation notice: {e}")

    def save_dataframe_table(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> bool:
        """
        Persist dataframe into database table with clean serialization.
        """
        if not self.is_connected or self.engine is None or df is None or df.empty:
            return False

        try:
            # Flatten non-primitive column types (lists/dicts) for SQL compatibility
            df_sql = df.copy(deep=True)
            for col in df_sql.columns:
                if df_sql[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    df_sql[col] = df_sql[col].astype(str)

            df_sql.to_sql(table_name, con=self.engine, if_exists=if_exists, index=False)
            
            # If we created or replaced transactions or alerts on PostgreSQL, re-ensure indexes
            if self.db_type == "PostgreSQL" and table_name in ("transactions", "alerts", "telemetry"):
                self._ensure_indexes()
                
            return True
        except Exception as e:
            logger.error(f"Error saving to table {table_name}: {e}")
            return False

    def query_table(self, query_str: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute a SELECT query and return results as a DataFrame."""
        if not self.is_connected or self.engine is None:
            return pd.DataFrame()
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(query_str), conn, params=params or {})
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return pd.DataFrame()

    def get_table_summary(self) -> Dict[str, Any]:
        """Retrieve table listing and row counts from the database."""
        if not self.is_connected or self.engine is None:
            return {"connected": False, "message": self.connection_status_msg}

        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            summary = {}
            with self.engine.connect() as conn:
                for t in tables:
                    try:
                        cnt = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                        summary[t] = int(cnt)
                    except Exception:
                        summary[t] = 0

            return {
                "connected": True,
                "db_type": self.db_type,
                "status_message": self.connection_status_msg,
                "tables": summary
            }
        except Exception as e:
            return {
                "connected": False,
                "db_type": self.db_type,
                "status_message": f"Error listing tables: {e}",
                "tables": {}
            }

    def health_check(self) -> Dict[str, Any]:
        """Check live database health and connection status."""
        if not self.is_connected or self.engine is None:
            return {"status": "unhealthy", "db_type": self.db_type, "message": self.connection_status_msg}
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "healthy", "db_type": self.db_type, "message": self.connection_status_msg}
        except Exception as e:
            return {"status": "unhealthy", "db_type": self.db_type, "error": str(e)}
