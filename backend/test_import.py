import sys
import importlib.util

# Load the module directly without going through __init__.py
spec = importlib.util.spec_from_file_location("cycle_router", "app/routes/cycle_router.py")
cr = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(cr)
    print("Module loaded successfully")
    print("Has router:", hasattr(cr, 'router'))
    if hasattr(cr, 'router'):
        print("Router type:", type(cr.router))
        print("Router:", cr.router)
except Exception as e:
    import traceback
    print("Error loading module:")
    traceback.print_exc()

