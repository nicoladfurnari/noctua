@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Noctua non e' installato in questa cartella: lancia Noctua-Setup.cmd
  pause
  exit /b 1
)
rem controlla e scarica gli aggiornamenti (senza rete o con modifiche locali parte con la versione che c'e')
rem (una riga sola: l'aggiornamento puo' riscrivere questo file mentre gira, e cmd lo legge riga per riga)
.venv\Scripts\python.exe -m astroforge.updater & .venv\Scripts\python.exe -m astroforge gui --open %*
