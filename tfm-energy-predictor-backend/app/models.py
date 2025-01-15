from app.database import db


class HistoricalData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    energy_type = db.Column(db.String(50), nullable=False)
    value = db.Column(db.Float, nullable=False)
    prediction_type = db.Column(db.String(50), nullable=False)
    updated_timestamp = db.Column(db.DateTime, nullable=False)
