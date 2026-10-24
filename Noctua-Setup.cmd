@echo off
rem Noctua (AstroForge) - installazione con doppio clic. Tutto il lavoro lo fa lo script PowerShell qui sotto.
rem Cartella: %LOCALAPPDATA%\AstroForge (o la variabile ASTROFORGE_DIR). Si puo' rilanciare: aggiorna e ripara.
set "AF_SELF=%~f0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = Get-Content -Raw -LiteralPath $env:AF_SELF; $i = $s.IndexOf('#' + 'PS1START'); Invoke-Expression $s.Substring($i)"
echo.
pause
exit /b
#PS1START
$ErrorActionPreference = "Stop"
$Repo = if ($env:ASTROFORGE_REPO) { $env:ASTROFORGE_REPO } else { "https://github.com/nicoladfurnari/noctua.git" }
$Dir = if ($env:ASTROFORGE_DIR) { $env:ASTROFORGE_DIR } else { Join-Path $env:LOCALAPPDATA "AstroForge" }
function Say($m) { Write-Host "== $m" -ForegroundColor Cyan }
function RefreshPath {
  $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}
function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function WingetInstall($id, $what) {
  if (-not (Have "winget")) { throw "$what manca e winget non c'e': installa $what a mano e rilancia." }
  Say "installo $what (winget $id)"
  winget install -e --id $id --scope user --accept-package-agreements --accept-source-agreements --silent
  if ($LASTEXITCODE -ne 0) { winget install -e --id $id --accept-package-agreements --accept-source-agreements --silent }
  RefreshPath
}

try {
  Say "Noctua: installazione in $Dir"

  # 1. Python (3.12, 3.13 o 3.11: quello che c'e' gia') e Git
  $py = $null
  foreach ($v in @("3.12", "3.13", "3.11")) {
    if (-not $py -and (Have "py")) { try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); Say "uso Python $v gia' installato" } } catch {} }
  }
  if (-not $py -and (Have "python")) {
    try { $ver = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)
      if ($ver -in @("3.11", "3.12", "3.13")) { $py = @("python"); Say "uso Python $ver gia' installato" } } catch {}
  }
  if (-not $py) {
    WingetInstall "Python.Python.3.12" "Python 3.12"
    if (Have "py") { $py = @("py", "-3.12") } else { $py = @("python") }
  }
  if (-not (Have "git")) { WingetInstall "Git.Git" "Git" }
  if (-not (Have "git")) { throw "Git non trovato dopo l'installazione: chiudi e rilancia questo file." }

  # 2. il programma (repository pubblico nicoladfurnari/noctua; ASTROFORGE_REPO per un altro)
  if (Test-Path (Join-Path $Dir ".git")) {
    Say "Noctua c'e' gia': aggiorno"
    git -C $Dir pull --ff-only
  } else {
    Say "scarico Noctua da GitHub"
    git clone $Repo $Dir
    if ($LASTEXITCODE -ne 0) { throw "download non riuscito: controlla la rete" }
  }

  # 3. ambiente Python e pacchetti (CuPy solo con una GPU NVIDIA)
  $venvPy = Join-Path $Dir ".venv\Scripts\python.exe"
  if (-not (Test-Path $venvPy)) {
    Say "creo l'ambiente Python"
    & $py[0] $py[1..9] -m venv (Join-Path $Dir ".venv")
  }
  & $venvPy -m pip install --upgrade --disable-pip-version-check pip | Out-Null
  Say "installo i pacchetti (qualche minuto la prima volta)"
  Push-Location $Dir
  & $venvPy -m astroforge.updater --deps --config
  $ok = $LASTEXITCODE -eq 0
  & $venvPy -c "import astroforge.plugins as p; print('plugin:', len(p.REGISTRY))"
  Pop-Location
  if (-not $ok) { throw "installazione dei pacchetti non riuscita (vedi sopra); rilancia il file per riprovare" }

  # 4. icona sul desktop
  if (-not $env:ASTROFORGE_NO_SHORTCUT) {
    $desk = [Environment]::GetFolderPath("Desktop")
    $old = Join-Path $desk "AstroForge.lnk"          # nome di prima del cambio in Noctua
    if (Test-Path $old) { Remove-Item $old -Force }
    $lnk = Join-Path $desk "Noctua.lnk"
    $sh = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
    $sh.TargetPath = Join-Path $Dir "start-gui.cmd"
    $sh.WorkingDirectory = $Dir
    $sh.Description = "Noctua: si aggiorna da solo a ogni avvio"
    $ico = Join-Path $Dir "astroforge\gui\static\icons\noctua.ico"
    if (Test-Path $ico) { $sh.IconLocation = "$ico,0" }
    $sh.Save()
    Say "icona 'Noctua' sul desktop"
  }
  Write-Host ""
  Write-Host "Fatto. Avvia Noctua con l'icona sul desktop: a ogni avvio controlla gli aggiornamenti." -ForegroundColor Green
  Write-Host "Programmi esterni gratuiti (StarNet2, GraXpert, DeepSNR, Cosmic Clarity, ASTAP): mettili in C:\Programs"
  Write-Host "oppure indica dove sono dalla GUI (Programmi esterni)."
} catch {
  Write-Host ""
  Write-Host "ERRORE: $($_.Exception.Message)" -ForegroundColor Red
}
