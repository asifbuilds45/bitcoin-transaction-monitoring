"""
SQLite to PostgreSQL Data Migration Script
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Transfers all tables and historical forensic data from local SQLite database
(data/bitcoin_monitoring.db) into the production local PostgreSQL database (bitcoin_monitoring).
"""

import os
import sys
import logging
from pathlib import Path
from sqlalchemy import create_engine, text, inspect
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.database.db_manager import DatabaseManager, DEFAULT_SQLITE_URI, DEFAULT_POSTGRES_URI


def migrate_sqlite_to_postgres():
    """Migrate all tables from SQLite to PostgreSQL."""
    sqlite_path = ROOT_DIR / "data" / "bitcoin_monitoring.db"
    
    if not sqlite_path.exists():
        logger.info(f"No existing SQLite database found at {sqlite_path}. Nothing to migrate.")
        return True

    logger.info(f"Found SQLite database at {sqlite_path}. Inspecting tables...")
    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")
    
    postgres_db = DatabaseManager(DEFAULT_POSTGRES_URI)
    if not postgres_db.is_connected or postgres_db.db_type != "PostgreSQL":
        logger.error(f"Cannot migrate: PostgreSQL is not connected! Status: {postgres_db.connection_status_msg}")
        return False

    inspector = inspect(sqlite_engine)
    tables = inspector.get_table_names()
    logger.info(f"Tables found in SQLite: {tables}")

    migrated_counts = {}
    for table_name in tables:
        try:
            df = pd.read_sql_table(table_name, sqlite_engine)
            if df.empty:
                logger.info(f"Skipping empty table: {table_name}")
                continue
            
            success = postgres_db.save_dataframe_table(df, table_name, if_exists="replace")
            if success:
                migrated_counts[table_name] = len(df)
                logger.info(f"Successfully migrated table '{table_name}': {len(df)} rows.")
            else:
                logger.error(f"Failed to migrate table '{table_name}'.")
        except Exception as e:
            logger.error(f"Error migrating table '{table_name}': {e}")

    logger.info("=" * 60)
    logger.info(f"Migration completed! Summary: {migrated_counts}")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = migrate_sqlite_to_postgres()
    sys.exit(0 if success else 1)
