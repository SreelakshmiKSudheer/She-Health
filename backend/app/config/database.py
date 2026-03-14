# Compatibility shim — all DB access goes through app.db.database.MongoDB
from app.db.database import MongoDB

def get_database():
    """Return the active Motor database from the canonical MongoDB singleton."""
    return MongoDB.db
