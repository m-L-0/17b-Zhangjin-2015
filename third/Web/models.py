# encoding:utf-8

from exts import db


class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    telephone = db.Column(db.String(11),nullable=False)
    username = db.Column(db.String(50),nullable=False)
    password = db.Column(db.String(100),nullable=False)
    imgs = db.relationship('Image',backref='user')
    questions = db.relationship('Question', backref='user')


class Image(db.Model):
    __tablename__ ='image'
    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    img = db.Column(db.LargeBinary,nullable=True)
    user_id = db.Column(db.Integer,db.ForeignKey('user.id'))


class Question(db.Model):
    __table__name = 'question'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String,nullable=True)
    content = db.Column(db.Text,nullable=True)
    create_time = db.Column(db.String,nullable=True)
    author = db.Column(db.Integer, db.ForeignKey('user.id'))