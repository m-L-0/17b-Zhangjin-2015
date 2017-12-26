#encoding:utf-8
import os

DEBUG = True

SECRET_KEY = os.urandom(24)

# HOSTNAME = '127.0.0.1'
# PORT     = '3306'
# DATABASE = 'search_world'
# USERNAME = 'root'
# PASSWORD = 'root'
# 'mysql+mysqldb://{}:{}@{}/{}?charset=utf8'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)

basedir = os.path.abspath(os.path.dirname(__file__))
DB_URI = 'sqlite:///'+os.path.join(basedir, 'data.sqlite')
SQLALCHEMY_DATABASE_URI = DB_URI
SQLALCHEMY_COMMIT_ON_TEARDOWN = True
SQLALCHEMY_TRACK_MODIFICATIONS = False