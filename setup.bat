@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo ========================================
echo RVC Realtime for XPU - Environment Setup
echo ========================================
echo.

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv が見つかりません。
    echo https://docs.astral.sh/uv/
    echo.
    pause
    exit /b 1
)

echo [1/5] Python 3.12.13 を確認しています...
uv python install 3.12.13
if errorlevel 1 goto :error

echo.
echo [2/5] Python 3.12.13 の仮想環境を作成しています...
if exist ".venv" (
    echo .venv は既に存在します。
) else (
    uv venv .venv --python 3.12.13
    if errorlevel 1 goto :error
)

echo.
echo [3/5] PyTorch XPU をインストールしています...
uv pip install --python ".venv\Scripts\python.exe" ^
    torch==2.7.1+xpu torchaudio==2.7.1+xpu ^
    --index-url https://download.pytorch.org/whl/xpu ^
    --extra-index-url https://pypi.org/simple ^
    --index-strategy unsafe-best-match
if errorlevel 1 goto :error

echo.
echo [4/5] RVC の依存パッケージをインストールしています...
uv pip install --python ".venv\Scripts\python.exe" ^
    -r "requirments_xpu_py312.txt"
if errorlevel 1 goto :error

echo.
echo [5/5] 必要なモデルをダウンロードしています...

if not exist "assets\hubert_base\pytorch_model.bin" (
    echo HuBERTモデルをダウンロードしています...
    uv pip install --python ".venv\Scripts\python.exe" huggingface_hub
    if errorlevel 1 goto :error

    ".venv\Scripts\hf.exe" download ^
        lj1995/VoiceConversionWebUI ^
        hubert_base/pytorch_model.bin ^
        --revision main ^
        --local-dir assets
    if errorlevel 1 goto :error
) else (
    echo HuBERTモデルは既に存在します。スキップします。
)

if not exist "assets\rmvpe\rmvpe.pt" (
    echo RMVPEモデルをダウンロードしています...

    if not exist ".venv\Scripts\hf.exe" (
        uv pip install --python ".venv\Scripts\python.exe" huggingface_hub
        if errorlevel 1 goto :error
    )

    if not exist "assets\rmvpe" mkdir "assets\rmvpe"

    ".venv\Scripts\hf.exe" download ^
        lj1995/VoiceConversionWebUI ^
        rmvpe.pt ^
        --revision main ^
        --local-dir assets\rmvpe
    if errorlevel 1 goto :error
) else (
    echo RMVPEモデルは既に存在します。スキップします。
)

echo.
echo ========================================
echo 環境構築が完了しました。
echo ========================================
echo.
pause
exit /b 0

:error
echo.
echo ========================================
echo 環境構築に失敗しました。
echo ========================================
echo.
pause
exit /b 1