from extensions import Model, db


class Index(Model):
    """Market indices such as S&P 500 or Dow Jones."""

    __tablename__ = "indices"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    symbol = db.Column(db.String(10), nullable=False, unique=True)


class HistoricalPrice(Model):
    id = db.Column(db.Integer, primary_key=True)
    index_id = db.Column(db.Integer, db.ForeignKey("indices.id"), nullable=False)
    date = db.Column(db.Date, nullable=False)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Integer)
    index = db.relationship("Index", backref=db.backref("prices", lazy=True))


class MacroData(Model):
    id = db.Column(db.Integer, primary_key=True)
    indicator = db.Column(db.String(50), nullable=False)
    date = db.Column(db.Date, nullable=False)
    value = db.Column(db.Float)


class Strategy(Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    parameters = db.Column(db.Text)  # Store as JSON string


class BacktestResult(Model):
    id = db.Column(db.Integer, primary_key=True)
    strategy_id = db.Column(db.Integer, db.ForeignKey("strategy.id"), nullable=False)
    index_id = db.Column(db.Integer, db.ForeignKey("indices.id"), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    returns = db.Column(db.Float)
    alpha = db.Column(db.Float)
    strategy = db.relationship("Strategy", backref=db.backref("backtests", lazy=True))
    index = db.relationship("Index", backref=db.backref("backtests", lazy=True))
