@echo off
echo ====================================================
echo Starting Full Concrete Intelligence Training Pipeline
echo ====================================================
echo.

echo [1/5] Generating Augmented Crack Dataset...
python process_defect_data.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] process_defect_data.py failed!
    exit /b %ERRORLEVEL%
)
echo.

echo [2/5] Training Crack Intelligence Models...
python train_models.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] train_models.py failed!
    exit /b %ERRORLEVEL%
)
echo.

echo [3/5] Generating Defect Volume Dataset...
python generate_defect_dataset.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] generate_defect_dataset.py failed!
    exit /b %ERRORLEVEL%
)
echo.

echo [4/5] Training Defect Volume Models...
python train_defect_models.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] train_defect_models.py failed!
    exit /b %ERRORLEVEL%
)
echo.

echo [5/5] Recompiling RAG Search Index...
python run_pipeline.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] run_pipeline.py failed!
    exit /b %ERRORLEVEL%
)
echo.

echo ====================================================
echo Pipeline complete! All models trained and saved.
echo You can now run: streamlit run app.py
echo ====================================================
pause
