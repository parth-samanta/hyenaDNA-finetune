@echo off
setlocal
cd /d "%~dp0"
call conda run -n hyenadna-pooling python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
call conda run -n hyenadna-pooling python -m pip install --no-deps --no-build-isolation -e .
if errorlevel 1 exit /b 1
call conda run -n hyenadna-pooling python -m pip check
if errorlevel 1 exit /b 1
echo Ready. Run: conda activate hyenadna-pooling
echo Then: spyder --new-instance src\run_study.py
