# Publica el chatbot por un Cloudflare quick tunnel para una demo remota.
#
#   cd rag_pipeline
#   powershell -ExecutionPolicy Bypass -File scripts\demo_remota.ps1
#
# Levanta Streamlit en una ventana aparte, espera a que responda, abre el tunel
# y copia la URL https://....trycloudflare.com al portapapeles. El enlace NO
# lleva contrasena: cualquiera que lo tenga entra. Vive mientras esta ventana
# siga abierta; Ctrl+C cierra el tunel y Streamlit. Al reiniciar, la URL cambia.
#
# Mientras corre, Windows no se suspende ni apaga la pantalla (sin cambiar la
# configuracion de energia: el efecto termina al cerrar el script).

param([int]$Puerto = 8501)

$ErrorActionPreference = "Continue"
$app    = Split-Path $PSScriptRoot -Parent                   # rag_pipeline
$python = Join-Path (Split-Path $app -Parent) ".venv\Scripts\python.exe"

function Write-Paso($texto) { Write-Host "`n== $texto" -ForegroundColor Cyan }
function Stop-Demo($texto)  { Write-Host "ERROR: $texto" -ForegroundColor Red; exit 1 }

# --- cloudflared ------------------------------------------------------------
$cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cf) {
    $cf = Get-ChildItem "C:\Program Files*\cloudflared\cloudflared.exe" -ErrorAction SilentlyContinue |
          Select-Object -First 1 -ExpandProperty FullName
}
if (-not $cf) { Stop-Demo "falta cloudflared: winget install --id Cloudflare.cloudflared" }
if (-not (Test-Path $python)) { Stop-Demo "no encuentro el entorno virtual en $python" }

# --- Ollama -----------------------------------------------------------------
Write-Paso "Ollama"
try {
    Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 | Out-Null
    Write-Host "Ollama ya esta corriendo."
} catch {
    Write-Host "Ollama no responde; lo arranco."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Minimized
    $ok = $false
    foreach ($i in 1..30) {
        Start-Sleep 1
        try { Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null; $ok = $true; break } catch {}
    }
    if (-not $ok) { Stop-Demo "Ollama no arranco en 30 s" }
}

# --- Streamlit --------------------------------------------------------------
# Un segundo Streamlit chocaria con el candado de indices/qdrant.
if (Get-NetTCPConnection -LocalPort $Puerto -State Listen -ErrorAction SilentlyContinue) {
    Stop-Demo "el puerto $Puerto ya esta en uso (otro Streamlit abierto?). Cierralo primero."
}

Write-Paso "Streamlit"
# El modelo se queda en la GPU durante toda la reunion. Ollama recibe el valor
# como texto y exige unidad: "-1" a secas responde 400 ("missing unit").
$env:KEEP_ALIVE = "24h"
$st = Start-Process $python -WorkingDirectory $app -PassThru -ArgumentList @(
    "-m", "streamlit", "run", "app.py",
    "--server.headless", "true", "--server.port", "$Puerto",
    "--browser.gatherUsageStats", "false")

$ok = $false
foreach ($i in 1..120) {
    Start-Sleep 1
    if ($st.HasExited) { Stop-Demo "Streamlit se cerro; revisa su ventana" }
    try {
        if ((Invoke-WebRequest "http://localhost:$Puerto/_stcore/health" -UseBasicParsing -TimeoutSec 2).Content -eq "ok") { $ok = $true; break }
    } catch {}
}
if (-not $ok) { Stop-Demo "Streamlit no respondio en 2 minutos" }
Write-Host "Streamlit listo en http://localhost:$Puerto"
Write-Host "Abrelo una vez en tu navegador para que cargue los modelos (~30 s) antes de compartir el enlace." -ForegroundColor Yellow

# --- que Windows no se duerma mientras corre el script ----------------------
Add-Type -Namespace Demo -Name Energia -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
[Demo.Energia]::SetThreadExecutionState([uint32]"0x80000003") | Out-Null

# --- tunel ------------------------------------------------------------------
Write-Paso "Tunel de Cloudflare (Ctrl+C para terminar)"
try {
    & $cf tunnel --no-autoupdate --url "http://localhost:$Puerto" 2>&1 | ForEach-Object {
        $linea = "$_"
        if ($linea -match "https://[a-z0-9-]+\.trycloudflare\.com") {
            $url = $Matches[0]
            Set-Clipboard $url
            Write-Host ""
            Write-Host "  ENLACE PARA COMPARTIR: $url" -ForegroundColor Green
            Write-Host "  (copiado al portapapeles)" -ForegroundColor Green
            Write-Host ""
        } elseif ($linea -match " ERR ") {
            Write-Host $linea -ForegroundColor DarkYellow
        }
    }
} finally {
    [Demo.Energia]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
    if (-not $st.HasExited) { Stop-Process -Id $st.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "Tunel y Streamlit cerrados."
}
