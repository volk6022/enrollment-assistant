#!/bin/sh
set -eu

python - <<'PY'
import os, time
from sqlalchemy import create_engine, text

url = os.environ.get('DATABASE_URL')
if not url:
    raise SystemExit('DATABASE_URL is not set')

for i in range(30):
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        print('Database is ready')
        break
    except Exception as e:
        print(f'Waiting for database... ({i+1}/30) {e}')
        time.sleep(2)
else:
    raise SystemExit('Database is not ready after waiting')
PY

alembic upgrade head
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
