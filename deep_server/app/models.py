from app import db

class Record(db.Model):
    __tablename__ = 'records'
    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(db.DateTime)
    model_id = db.Column(db.Integer)
    category = db.Column(db.Integer)
    img_path = db.Column(db.String(255))

    def __repr__(self):
        return '<Record %r>' % self.ID

    def add(self):
        db.session.add(self)
        db.session.commit()