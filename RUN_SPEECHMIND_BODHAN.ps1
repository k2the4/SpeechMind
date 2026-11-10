# SpeechMind optional Bodhan hosted-provider launcher.
# Audio and transcript text are sent to Bodhan AI; API usage is metered.
# The key is requested securely and is not written to disk.

$ErrorActionPreference = "Stop"
$secureKey = Read-Host "Enter Bodhan API key (input is hidden)" -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    $env:BODHAN_API_KEY = $plainKey
    $env:SPEECHMIND_ASR_PROVIDER = "bodhan"
    $env:SPEECHMIND_TRANSLATOR_PROVIDER = "bodhan"

    $python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        $python = "python"
    }

    Write-Host "Starting SpeechMind with automatic code-mixed ASR and English translation."
    Write-Host "Bodhan's transcription endpoint currently accepts up to 30 seconds per request."
    & $python -m streamlit run (Join-Path $PSScriptRoot "app.py")
    if ($LASTEXITCODE -ne 0) {
        throw "SpeechMind exited with code $LASTEXITCODE"
    }
}
finally {
    $env:BODHAN_API_KEY = $null
    $env:SPEECHMIND_ASR_PROVIDER = $null
    $env:SPEECHMIND_TRANSLATOR_PROVIDER = $null
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    $plainKey = $null
}
