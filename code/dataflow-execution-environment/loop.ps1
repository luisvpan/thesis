# Uso: ./loop.ps1 -ModeArg [mode] -MaxIterations [iterations] (-Push)
# Ejemplos:
#   ./loop.ps1 (modo build, tareas sin límite, sin push automático)
#   ./loop.ps1 -MaxIterations 20 (modo build, 20 tareas máximo, sin push automático)
#   ./loop.ps1 -ModeArg plan (modo plan, tareas sin límite, sin push automático)
#   ./loop.ps1 -Push (modo plan, tareas sin límite, push automático)
#   ./loop.ps1 -ModeArg plan -MaxIterations 5 -Push (modo plan, 5 tareas máximo, push automático)

param (
    [string]$ModeArg = "build",
    [int]$MaxIterations = 0,
    [switch]$Push
)

# Configuración de archivos y modo
$Mode = if ($ModeArg -eq "plan") { "plan" } else { "build" }
$PromptFile = "PROMPT_$Mode.md"
$Iteration = 0
$CurrentBranch = git branch --show-current

# Verificar existencia del archivo .env
if (-not (Test-Path ".env")) {
    Write-Error "❌ Error: El archivo .env no existe en la raíz."
    exit 1
}

# Cargar y validar variables críticas
$envContent = Get-Content ".env" -Raw
$requiredVars = @("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "OLLAMA_HOST")
$missingVars = @()

foreach ($var in $requiredVars) {
    if ($envContent -notmatch "(?m)^$var=.") {
        $missingVars += $var
    }
}

# Reportar errores si faltan variables
if ($missingVars.Count -gt 0) {
    Write-Host "❌ Error: Faltan variables en el .env:" -ForegroundColor Red
    $missingVars | ForEach-Object { Write-Host "   - $_" -ForegroundColor Yellow }
    Write-Host "Ejemplo de formato: GIT_AUTHOR_NAME=Tu Nombre" -ForegroundColor Gray
    exit 1
}

Write-Host "✅ .env validado correctamente." -ForegroundColor Green

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "Modo:   $Mode"
Write-Host "Prompt: $PromptFile"
Write-Host "Branch: $CurrentBranch"
Write-Host "Push:   $Push"
if ($MaxIterations -gt 0) { Write-Host "Max:    $MaxIterations iteraciones" }
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if (-not (Test-Path $PromptFile)) {
    Write-Error "Error: $PromptFile no encontrado."
    exit 1
}

try {
    while ($true) {
        if ($MaxIterations -gt 0 -and $Iteration -ge $MaxIterations) {
            Write-Host "🏁 Límite alcanzado: $MaxIterations" -ForegroundColor Yellow
            break
        }

        Write-Host "🚀 Iniciando iteración $($Iteration + 1)..." -ForegroundColor Magenta

        # Ejecutamos a Ralph pasando el Prompt
        # Importante: Montamos la carpeta .git y pasamos config de usuario
        docker run --rm --name "ralph-agent" `
          -v "${PWD}/../../:/thesis" `
          --env-file .env `
          --add-host=host.docker.internal:host-gateway `
          opencode-ralph `
          run "$(Get-Content $PromptFile -Raw)"

        if ($Push) {
            Write-Host "Sincronizando con repositorio remoto..." -ForegroundColor Gray
            # git push origin "$CurrentBranch"
        }

        $Iteration++
        Write-Host "=== ITERACIÓN $Iteration COMPLETADA ===" -ForegroundColor Green
    }
}
finally {
    Write-Host "`n🛑 Interrupción detectada. Limpiando recursos..." -ForegroundColor Yellow
    # Forzamos la detención si quedó algo vivo
    docker stop "ralph-agent" 2>$null
    Write-Host "✅ Cleanup completado. ¡Hasta la próxima, Ralph!" -ForegroundColor Green
}
