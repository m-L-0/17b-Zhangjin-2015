from flask import Flask, render_template, request, redirect, url_for, session
import config
from models import User, Question, Image
from exts import db
import time

app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)
log = False

@app.route('/index')
def index():
    context = {
        'questionss': Question.query.all()
    }
    return render_template('index.html',**context)


@app.route('/login/',methods=['GET','POST'])
def login():
    global log
    if request.method == 'GET':
        return render_template('login.html')
    else:
        telephone = request.form.get('telephone')
        password = request.form.get('password')
        user = User.query.filter(User.telephone==telephone,User.password==password).first()
        if user:
            session['user_id'] = user.id
            session['username'] = user.username
            session.permanent = True
            log = True
            return redirect(url_for('index'))
        else:
            return u'手机号码或者密码错误，请重新登陆'


@app.route('/logout/')
def logout():
    global log
    #session.pop('user_id')
    #del session['user_id']
    log = False
    session.clear()
    return redirect(url_for('index'))


@app.route('/regist/',methods=['GET','POST'])
def regist():
    if request.method == 'GET':
        return render_template('regist.html')
    else:
        telephone = request.form.get('telephone')
        username = request.form.get('username')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter(User.telephone == telephone).first()
        if user:
            return u'改手机号已被注册，请更换手机号'
        else:
            if password1 != password2:
                return u'两次密码不相等，请核对后重写'
            else:
                user = User(telephone=telephone,username=username,password=password1)
                db.session.add(user)
                db.session.commit()
                return redirect(url_for('login'))


@app.route('/question/',methods=['GET','POST'])
def question():
    if not log:
        return redirect(url_for('login'))
    if request.method == 'GET':
        return render_template('question.html')
    else:
        create_time = time.strftime('%y-%m-%d %H:%M:%S', time.localtime(time.time()))
        title = request.form.get('title')
        content = request.form.get('content')
        author = session['username']
        questions = Question(title=title, content=content, author=author, create_time=create_time)
        db.session.add(questions)
        db.session.commit()
        return redirect(url_for('index'))


@app.route('/make_img',methods=['GET','POST'])
def make_img():
    if request.method == 'GET':
        return render_template('make_img.html')
    else:
        img = session['file']
        print(img)
        user = session['username']
        makes = Image(img=img,user_id=user)
        db.session.add(makes)
        db.session.commit()
        content = {
            'imgs': session[img][-1]
        }
        return render_template('make_img.html',**content)


@app.context_processor
def my_context_processor():
    user_id = session.get('user_id')
    if user_id:
        user = User.query.filter(User.id == user_id).first()
        if user:

            return {'user': user}

    return {}


if __name__ == '__main__':
    app.run()
