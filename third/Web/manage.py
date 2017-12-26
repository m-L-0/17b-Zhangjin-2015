from flask_script import Manager
from flask_migrate import Migrate,MigrateCommand
from flask_ import app
from exts import db
manager = Manager(app)
from models import User

# 使用Migrate绑定app和db

migrate = Migrate(app,db)
manager.add_command('db',MigrateCommand)

if __name__ == '__main__':
    manager.run()