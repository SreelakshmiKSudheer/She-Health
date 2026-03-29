from app.main import app\nprint('total routes', len(app.routes))\nfor r in app.routes: print(r.name, r.path)
