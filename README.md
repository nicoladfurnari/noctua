# Noctua

Software di astrofotografia in Python, a plugin, ancora in fase di sviluppo. Obiettivo: dalla cartella dei grezzi
(light, dark, flat, bias) all'immagine finale, con automazioni e preset, senza dipendere da PixInsight.
(Il pacchetto Python e i comandi si chiamano `astroforge`, es. `python -m astroforge gui`.)

Stato attuale: **flusso completo dai grezzi all'immagine finale**, con GUI a wizard, GPU (anche più schede),
drizzle 2x/3x, camere a colori e mono con filtri (SHO/HOO/LRGB), local normalization, flat per sessione,
dark optimization, rimozione satelliti, correzione distorsione, mosaici, comet stacking, dual-band OSC,
plate solving + SPCC. Stato delle funzioni e limiti in [ROADMAP.md](ROADMAP.md).

**Versione beta** (`0.6.0-beta.1`): funziona dai grezzi all'immagine finale ed è provata sui dati dell'autore
(QHY168C, QHY163C, QHY163M); con altre camere e programmi di acquisizione possono esserci sorprese. Le
segnalazioni vanno nelle [Issues](https://github.com/nicoladfurnari/noctua/issues) con l'intestazione FITS di un
frame e il log del lavoro (pagina Elaborazione).

© 2026 Nicola Furnari. Tutti i diritti riservati: il codice è pubblicato per installare e provare il programma;
non è concesso ridistribuirlo o riusarlo in altri progetti senza accordo con l'autore.

## Avvio rapido

Doppio clic su `start-gui.cmd`, poi apri <http://127.0.0.1:8765> nel browser.
Il wizard guida in 7 passi: sorgenti → frame → impostazioni → elaborazione → risultati → post (anteprima dal vivo) → export.

Da riga di comando:

```bash
# colore (OSC), drizzle 2x, post automatica
.venv\Scripts\python.exe -m astroforge stack "D:\Astro\QHY168C\M8 - Lagoon Nebula" -c "D:\Astro\QHY168C\#Bias" "D:\Astro\QHY168C\#Dark" "D:\Astro\QHY168C\#Flat" -w work\m8 -d 2 --post nebula
# mono con filtri, combinazione SHO
.venv\Scripts\python.exe -m astroforge stack "D:\Astro\QHY163M\NGC2244 - Nebulosa Rosetta" -c "...\Bias" "...\Dark" -w work\rosetta --filters SHO --post sho
# OSC con filtro dual-band -> Ha, OIII e HOO; local normalization per gradienti che cambiano (Luna)
.venv\Scripts\python.exe -m astroforge stack "...\NGC7000" -c "...\Bias" "...\Flat" --dualband --local-norm
# mosaico: una cartella per pannello
.venv\Scripts\python.exe -m astroforge stack "...\Pannello1" "...\Pannello2" -c "...\Bias" --mosaic
# cometa: nucleo nel primo e nell'ultimo frame (pixel del frame di riferimento)
.venv\Scripts\python.exe -m astroforge stack "...\C2026" -c "...\Bias" --comet 812 604 870 633
```

**Master di calibrazione**: nella GUI, scheda Calibrazione, "+ File per creare i master" (bias/dark/flat grezzi) o
"+ Master già pronti" (PixInsight, NINA, AstroForge). "Crea i master…" li crea subito dai grezzi con i parametri
(rigetto e sigma, Superbias, mappa dei difetti del sensore, un flat per notte, formato XISF/FITS, libreria) e un
rapporto di qualità; "Usa questi master" li mette tra i pronti. Da riga di comando:
`python -m astroforge masters "D:\Astro\QHY163M\#MASTERS" -o "D:\Astro\QHY163M\AstroForge masters" --superbias`.
Una cartella unica con tutto va bene: i bias si separano per esposizione (si usa il più corto) e i flat per notte.

Altre opzioni di `stack`: `--debayer rcd|bilinear|ea` (RCD predefinito),
`--rejection auto|winsorized|linear_fit|esd|sigma|percentile`, `--weighting psf|standard|nebula|galaxy|cluster|<espressione>`,
`--approval 70`, `--exposure-tolerance 200`, `--dark-optimize auto|always|never`, `--satellites`, `--no-distortion`.
Flat per notte, libreria master e dark optimization "auto" (solo se manca il dark con la stessa posa) sono attivi di default.
Come WBPP (impostazioni JSON e GUI): `grouping_keywords` (es. `["TELESCOP", "NIGHT"]`, gruppi integrati a parte),
`calibration_keywords` (es. `["ROTATOR"]`), `group_settings` per gruppo, `all_groups`, `pedestal` (ADU o `"auto"`),
`large_scale_rejection` (`{"high": 2, "growth": 2}`), `plate_solve` (WCS scritto nello stack), `local_norm_reference`
(`"integration"` dei frame migliori) e il piano con la stima dei tempi (`POST /api/plan`, scheda "Piano" nella GUI).
**Profili automatici** (scelti da soli in base ai dati con `-p auto`, `--post auto` e nella GUI):
`rgb_starless` (camera a colori), `sho_starless` (mono Ha/OIII/SII), `hoo_starless` (bicolore), `mono`, `galaxy`
(oggetto riconosciuto come galassia dal nome o da OBJECT). Seguono il metodo starless: gradiente, colore e denoise
lineari, StarNet subito, nebulosa lavorata da sola (stretch, contrasto, MultiscaleLinearTransform 5 livelli), stelle
a parte (stretch, trasformazione morfologica ripetuta che lascia il nucleo brillante, curva sulle ali), unione e
infine `look.match`, tarato sulle immagini finali di riferimento.
Misurare il look di un'immagine finita e farne un profilo: `python -m astroforge look finale.jpg --save "Come M8"`
(`--base sho_starless` per partire da un altro profilo). Dare un nome proprio a un profilo:
`python -m astroforge rename rgb_starless "Le mie nebulose"` (il nome vale ovunque, il file del profilo non cambia;
i nomi stanno in `~/.astroforge/preset_labels.json`). Nella GUI: pulsanti **Rinomina** e **Da immagine…**.
I master XISF di PixInsight si aprono direttamente (`run master.xisf -p auto`).

Altri preset di post: `nebula`, `nebula_pro` (gradiente all-sky + SPCC + deconvoluzione con PSF misurata + DeepSNR),
`video_trifida` (il workflow del video "elaborazione pixinsight workflow 2026"), `gaeeb` (il workflow
PixInsight di Papotti/Airola passo per passo), `galaxy`, `sho`, ... Ogni passo di un preset può avere una maschera:
`{"plugin": "contrast.local", "mask": {"type": "range", "lower": 0.25, "upper": 1, "fuzziness": 0.1}}`
(tipi: `range`, `color`, `luminance`, `channel`, `stars`, `lum`, `layer`, `paint`; `invert`, `blur`). Con
`"target": "stars"` un passo lavora sul livello stelle lasciato da StarNet/Dark Star. Metadati FITS in uscita:
`python -m astroforge run img.fits -p gaeeb --meta OBSERVER=Nicola --meta COPYRIGHT="All rights reserved"`.
GUI in rete: `python -m astroforge gui --host 0.0.0.0` stampa l'indirizzo con il token d'accesso.

GPU: usate automaticamente se c'è CuPy (`cupy-cuda12x`). `ASTROFORGE_GPU=0` per disattivarle,
`ASTROFORGE_GPUS=0,1` per scegliere quali schede usare.

## Installazione

**Con doppio clic** (Windows 10/11): scarica [`Noctua-Setup.cmd`](https://github.com/nicoladfurnari/noctua/raw/main/Noctua-Setup.cmd)
e aprilo (se SmartScreen chiede, "Ulteriori informazioni → Esegui comunque"). Installa da solo quello che manca
(Python 3.12 e Git con winget), scarica Noctua in `%LOCALAPPDATA%\AstroForge`, crea l'ambiente Python con tutti i
pacchetti (CuPy per le GPU solo se c'è una scheda NVIDIA), scrive `config.json` con i programmi esterni trovati in
`C:\Programs` e mette l'icona **Noctua** sul desktop. Rilanciarlo ripara un'installazione. Niente viene installato
fuori da quella cartella (a parte Python e Git); per toglierlo basta cancellarla e l'icona.

Tempo medio di installazione, con una connessione normale: **5 minuti** senza GPU (circa 300 MB di pacchetti);
**10-15 minuti** con una scheda NVIDIA, perché scarica anche CuPy e le librerie CUDA (circa 2 GB); più un paio di
minuti se mancano Python e Git. Gli avvii successivi sono immediati.

Programmi esterni gratuiti che Noctua sa usare se li trova (`C:\Programs` o indicati dalla GUI, pagina
Programmi esterni): [StarNet2](https://www.starnetastro.com/), [GraXpert](https://www.graxpert.com/),
[ASTAP](https://www.hnsky.org/astap.htm) con il catalogo D50 (plate solving e SPCC),
[Cosmic Clarity](https://www.setiastro.com/cosmic-clarity), DeepSNR.

**Aggiornamenti**: a ogni avvio (`start-gui.cmd`, l'icona) Noctua controlla GitHub e, se c'è una versione nuova,
la scarica e reinstalla i pacchetti solo se sono cambiati; poi apre la GUI. Senza rete parte con la
versione che ha; con modifiche locali non salvate o su un altro ramo non tocca niente. Per spegnerlo:
variabile `ASTROFORGE_NO_UPDATE=1` o un file vuoto `no-auto-update` nella cartella. A mano:
`.venv\Scripts\python.exe -m astroforge.updater`. La versione in uso è in basso a sinistra nella GUI.

A mano (sviluppo):

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m astroforge.updater --deps --config
```

Programmi esterni opzionali: copia `config.example.json` in `config.json` e metti i percorsi giusti.
Se un programma manca, il suo plugin viene saltato con un messaggio.

## Uso

```bash
.venv\Scripts\python.exe -m astroforge run "D:\Astro\...\m8.fit" -p nebula -o out
.venv\Scripts\python.exe -m astroforge plugins     # elenco plugin e parametri
.venv\Scripts\python.exe -m astroforge presets     # elenco preset
```

Escono `<nome>_<preset>.fits` (32 bit), `.tif` (16 bit), `.jpg` e un'anteprima.

## Plugin

| Categoria | Plugin |
|---|---|
| geometria | `crop.auto`, `crop.manual`, `geometry.rotate`, `geometry.integer_resample` (binning/ingrandimento intero), `ext.cc_superres` |
| cosmetica | `cosmetic.hot_pixels`, `cosmetic.chroma_noise`, `cosmetic.heal` (clone stamp), `ext.cc_satellite` |
| fondo cielo | `background.auto` (sceglie DBE/GraXpert o superficie semplice dall'area della nebulosa), `background.allsky` (riferimento DSS2 via CDS), `background.dbe`, `background.poly`, `ext.graxpert_background` |
| colore | `color.spcc` (ASTAP + Gaia DR3, curve sensori/filtri), `color.neutralize_background`, `color.white_balance_stars`, `color.scnr`, `color.selective`, `color.linear_fit`, `color.pcc` (APASS), `color.gradient_fit`, `color.fix_magenta`, `color.nb_normalization`, `color.nb_rgb` (NBRGBCombination: Ha/OIII/SII in una foto a colori, con cursori), `color.lrgb` (LRGBCombination), `color.channel_combine` (ChannelCombination da file mono) |
| rumore | `noise.tgv`, `noise.mmt`, `noise.first_layer_subtract`, `noise.wavelet`, `ext.graxpert_denoise`, `ext.deepsnr` |
| nitidezza | `sharpen.mlt` (MultiscaleLinearTransform, bias per livello), `sharpen.deconvolution`, `sharpen.wavelet`, `ext.cosmic_clarity`, `ext.rc_astro`, `sharpen.unsharp`, `sharpen.larson_sekanina` (getti delle comete) |
| stretch | `stretch.auto`, `stretch.linked_mtf`, `stretch.histogram`, `stretch.ghs`, `stretch.arcsinh`, `stretch.statistical`, `stretch.stf` (AutoSTF con canali collegati o separati, Boost), `stretch.masked` (MaskedStretch), `stretch.adaptive` (AdaptiveStretch) |
| stelle | `stars.morph` (MorphologicalTransformation ripetuta, nucleo protetto), `stars.starnet`, `stars.darkstar`, `stars.reduce`, `stars.halo_reduce`, `stars.color_boost`, `stars.remove_basic`, `stars.recombine`, `stars.stretch`, `stars.size` |
| contrasto | `contrast.curves` (RGB/R/G/B/L*/a*/b*/C/S/H), `tools.hdr_composition`, `contrast.saturation`, `contrast.local`, `contrast.dark_structure`, `contrast.hdr` |
| look | `look.match` (cielo, fondo debole, nebulosa, saturazione, alte luci verso un look di riferimento) |
| strumenti | `tools.pixelmath`, `tools.convolution` (gaussiana/box/Moffat), `tools.invert`, `tools.binarize`, `tools.grayscale`, `tools.channel_extract` (R/G/B/L/Lab/HSV) |
| analisi | `analysis.stats`, `analysis.plate_solve`, `annotate.objects` (Messier/NGC/IC, griglia, bussola), `annotate.text` |

Un preset è un JSON con la lista dei passi:

```json
{"name": "mio", "steps": [{"plugin": "crop.auto"}, {"plugin": "stretch.auto", "params": {"target": 0.2}, "preview": true}]}
```

## Scrivere un plugin

```python
from astroforge.plugins.base import Param, Plugin, register

@register
class MioPlugin(Plugin):
    id = "contrast.mio"
    name = "Il mio plugin"
    category = "contrast"
    stage = "nonlinear"          # linear | nonlinear | any
    params = {"amount": Param(0.5, 0.0, 1.0, "forza")}

    def run(self, img, ctx, amount):
        return img               # float32 (H, W, C), valori 0..1
```

e importalo in `astroforge/plugins/__init__.py`.

## Crediti dei dati

Catalogo deep-sky da [OpenNGC](https://github.com/mattiaverga/OpenNGC) di Mattia Verga, licenza CC-BY-SA-4.0 (vedi `astroforge/data/ATTRIBUTION.txt`); stelle brillanti dal Yale Bright Star Catalogue; Sharpless da VizieR VII/20. Curve spettrali di sensori e filtri: approssimazioni dai dati pubblicati, non dati dei costruttori.
