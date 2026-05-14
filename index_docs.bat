@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   本地知识库 - 增量索引
echo ========================================
echo.

set PYTHON=python312\python.exe
set INDEX_SCRIPT=index_docs.py
set DOCS_DIR=md-source

if not exist "%PYTHON%" (
    echo [错误] 未找到 %PYTHON%
    pause
    exit /b 1
)

if not exist "%INDEX_SCRIPT%" (
    echo [错误] 未找到 %INDEX_SCRIPT%
    pause
    exit /b 1
)

if not exist "%DOCS_DIR%" (
    echo [错误] 未找到文档目录 %DOCS_DIR%
    pause
    exit /b 1
)

echo Python:   %PYTHON%
echo 脚  本:   %INDEX_SCRIPT%
echo 文档目录: %DOCS_DIR%
echo.

"%PYTHON%" "%INDEX_SCRIPT%" "%DOCS_DIR%"

echo.
echo ========================================
echo   索引入库完成！
echo ========================================
pause
