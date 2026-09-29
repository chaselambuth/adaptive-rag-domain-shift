import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from app.web import create_app

# Local development entry point. Docker uses Gunicorn with the same factory.
if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=int(os.environ.get('PORT', '5000')), debug=False)
