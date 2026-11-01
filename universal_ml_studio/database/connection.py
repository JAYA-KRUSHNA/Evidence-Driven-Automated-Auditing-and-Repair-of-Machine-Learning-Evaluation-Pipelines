"""
SQLite connection manager.

Provides:
    get_engine()   → singleton SQLAlchemy Engine  (WAL mode, pool_pre_ping)
    get_session()  → new Session bound to the engine
    init_db()      → create all tables from ORM models
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from ..config import DATABASE_URL

_engine = None
_SessionFactory = None


def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL journal mode and foreign keys for every connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine():
    """Return the singleton SQLAlchemy Engine (creates it on first call)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )
        event.listen(_engine, "connect", _set_sqlite_pragma)
    return _engine


def get_session() -> Session:
    """Return a new Session.  Caller is responsible for closing it."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory()


def init_db():
    """Create all tables that don't exist yet."""
    from .models import Base  # local import to avoid circular deps
    Base.metadata.create_all(bind=get_engine())
