# Hugging Face Spaces entrypoint wrapper
# Spaces expects a file named 'app.py'. We already have 'streamlit_app.py'
# so import it here to keep a single codebase.

from streamlit_app import *
