@echo off
setlocal

:: DigiKam Batch Queue Manager Custom Script wrapper
:: Usage in DigiKam: path\to\wb_wrapper.bat "%INPUT%" "%OUTPUT%"

set INPUT=%~1
set OUTPUT=%~2

:: Execute the AI White Balance tool.
:: It will generate an .xmp file in the same directory as the INPUT file.
:: You can change --method to ASH, AWB, or DWB
"%~dp0wb_ai.exe" "%INPUT%" --method DWB

if %errorlevel% neq 0 (
    echo Error processing %INPUT%
    exit /b %errorlevel%
)

:: DigiKam requires the script to create the %OUTPUT% file to mark the step as successful.
:: Since we process in-place (creating a sidecar), we just copy the original file to the output path.
copy "%INPUT%" "%OUTPUT%" /Y

endlocal
exit /b 0
