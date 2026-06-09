@echo off
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest %*
