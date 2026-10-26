$ErrorActionPreference = "Stop"

$python = "D:\Projects\BioSNN-MemoryTerminal\.venv\Scripts\python.exe"
$script = "D:\Projects\BioSNN-Plug-neuron\scripts\axo_real_memory_50k.py"
$out = "D:\Projects\BioSNN-Plug-neuron\sweep_results\axo_real_50k\real_memory.json"
$logDir = "D:\Projects\_scratch_axo"
$log = Join-Path $logDir "real_50k_trigger.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log([string]$message) {
    Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format o), $message)
}

Write-Log "check start"
$output = & $python $script --data-root "D:/AxoData" --threshold 50000 --queries 100 --full-scan-queries 10 --require-semantic --out $out 2>&1
$code = $LASTEXITCODE
$output | Out-File -FilePath $log -Append -Encoding utf8
Write-Log ("python exit={0}" -f $code)
exit $code
