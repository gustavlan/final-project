from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
from typing import Any

db: SQLAlchemy = SQLAlchemy()
# Alias the declarative base so mypy can resolve it as a class.
Model: Any = db.Model  # type: ignore[attr-defined]
csrf = CSRFProtect()
