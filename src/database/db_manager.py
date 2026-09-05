"""
PostgreSQL & Database Persistence Manager
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Provides database persistence for Network Telemetry, Blockchain Transactions,
Correlations, Communities, Anomaly Results, and Investigation Alerts.
Supports local PostgreSQL database with automatic fallback to local SQLite.
"""

import os
import logging
import pandas as pd
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Default PostgreSQL Connection String
DEFAULT_POSTGRES_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/bitcoin_monitoring")
DEFAULT_SQLITE_URI = f"sqlite:///{os.path.abspath(os.path.join('data', 'bitcoin_monitoring.db'))}"


class DatabaseManager:
    """
    Manages connections and persistence for analysis pipeline tables.
    """

    def __init__(self, db_uri: str = DEFAULT_POSTGRES_URI):
        self.db_uri = db_uri
        self.engine: Optional[Engine] = None
        self.db_type = "UNKNOWN"
        self.is_connected = False
        self.connection_status_msg = ""
        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize database engine. Try PostgreSQL first, fall back to SQLite."""
        # 1. Try PostgreSQL if configured
        if self.db_uri.startswith("postgresql"):
            try:
                engine = create_engine(self.db_uri, connect_args={"connect_timeout": 3})
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self.engine = engine
                self.db_type = "PostgreSQL"
                self.is_connected = True
                self.connection_status_msg = "Connected to local PostgreSQL database."
                logger.info(self.connection_status_msg)
                return
            except Exception as e:
                msg = f"PostgreSQL unavailable at {self.db_uri}. Using local SQLite fallback database. ({e})"
                logger.warning(msg)

        # 2. Fallback to local SQLite DB file
        try:
            os.makedirs("data", exist_ok=True)
            self.engine = create_engine(DEFAULT_SQLITE_URI)
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

    def save_dataframe_table(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> bool:
        """
        Persist dataframe into database table.
        """
        if not self.is_connected or self.engine is None or df.empty:
            return False

        try:
            # Flatten non-primitive column types (lists/dicts) for SQL compatibility
            df_sql = df.copy(deep=True)
            for col in df_sql.columns:
                if df_sql[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    df_sql[col] = df_sql[col].astype(str)

            df_sql.to_sql(table_name, con=self.engine, if_exists=if_exists, index=False)
            return True
        except Exception as e:
            logger.error(f"Error saving to table {table_name}: {e}")
            return False

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
                    cnt = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                    summary[t] = int(cnt)

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
