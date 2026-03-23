.PHONY: reset-db db-upgrade db-current db-downgrade run-api

reset-db:
	bash scripts/reset_db.sh

db-upgrade:
	FLASK_APP=app/flask_db.py ./.venv/bin/python -m flask db upgrade

db-current:
	FLASK_APP=app/flask_db.py ./.venv/bin/python -m flask db current

db-downgrade:
	FLASK_APP=app/flask_db.py ./.venv/bin/python -m flask db downgrade

run-api:
	FLASK_APP=app/flask_db.py ./.venv/bin/python -m flask run --debug
