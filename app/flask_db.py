"""Flask application setup and runtime entrypoint.

Quick rerun script
export FLASK_APP=flask_db.py > name of flask file
flask db init > run when need to reinialise db
flask db migrate -m "initial schema" > create migration script
flask db upgrade > apply migration
flask run > start the server
"""

from pathlib import Path

from flask import Flask 

try:
    from .api_routes import register_routes
    from .db_models import db, migrate
except ImportError:  
    from api_routes import register_routes
    from db_models import db, migrate


def create_app(database_url: str = "sqlite:///data/learner_recommender.db") -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__)

    if database_url.startswith("sqlite:///") and not database_url.startswith("sqlite:////"):
        db_file = Path(app.root_path) / database_url.removeprefix("sqlite:///")
        db_file.parent.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:////{db_file}"

    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    migrate.init_app(app, db)
    register_routes(app)

    return app


app = create_app()


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
