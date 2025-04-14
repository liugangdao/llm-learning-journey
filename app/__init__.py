from flask import Flask, render_template

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-unique-secret-key'
    from app.routes import quiz
    from app.routes import quiz_admin
    app.register_blueprint(quiz.bp)
    app.register_blueprint(quiz_admin.bp)
    @app.route("/")
    def index():
        return render_template('index.html')

    return app
