# ROADMAP — AstroForge "da zero"

Obiettivo: dalla cartella dei file singoli (light, dark, flat, bias, dark-flat) all'immagine finale, con un
**wizard** guidato e una **GUI** curata, più una modalità esperto con editor di pipeline.
Spunti: PixInsight (WBPP, SubframeSelector, LocalNormalization, DrizzleIntegration, MGC, SPCC),
DeepSkyStacker (semplicità, scelta automatica dei master, report), Siril 1.4 (Bayer drizzle Hubble-style,
registrazione con distorsione, mosaici, SPCC con Gaia offline, color management), Seti Astro Suite
(blink, rimozione satelliti, statistical stretch, Halo-B-Gon, Blemish Blaster), APP (normalizzazione,
mosaici, light pollution removal), workflow PixInsight GAEEB (Papotti/Airola, preset `gaeeb`), video
"elaborazione pixinsight workflow 2026" (Celestial Vibrations, preset `video_trifida`).

Legenda: ✅ fatto e coperto da test (sintetici) · 🧪 fatto, da provare su dati veri / hardware · ⬜ da fare.
Tutto il lavoro del 21/10 è verificato solo su dati sintetici: va confermato su M8 / Rosetta / M31.

## 1. Ingestione e wizard
- ✅ Scansione cartella (ricorsiva), header FITS con fallback su nome file/cartella.
- ✅ Raggruppamento per camera/gain/offset/temperatura impostata/esposizione/binning/filtro; notte da DATE-OBS.
- ✅ **Tolleranza di esposizione** (WBPP): light di pose diverse in un'unica integrazione, ognuna col suo dark.
- 🧪 **Libreria master** (`~/.astroforge/masters` + cartelle `#MASTERS` in sola lettura): salva e riusa
  bias/dark/flat quando una sessione non li ha, punteggio di compatibilità, scadenza, master difettosi scartati.
- ✅ Multi-filtro mono anche con più esposizioni per filtro; 🧪 OSC dual-band → HOO.

## 2. Calibrazione
- ✅ Master bias/dark/flat/dark-flat; darkflat per esposizione dei flat; cache legata a bias/darkflat.
- ✅ Flat per sessione; ✅ flat normalizzati per sito CFA (colori invariati) sulle camere a colori.
- 🧪 Dark optimization (rumore minimo + amp glow), di default solo se manca un dark con la stessa posa.
- ✅ Riduzione del banding (righe/colonne), opzionale. ✅ Pedestal, avvisi su dark/flat non adatti.
- ✅ Cosmetic correction automatica.

## 3. Debayer
- ✅ Bilineare, edge-aware e **RCD** (predefinito). ✅ Bayer drizzle. ✅ OSC senza BAYERPAT riconosciuta.

## 4. Analisi e selezione frame
- ✅ FWHM/eccentricità da fit Moffat, rumore MRS, SNR, peso PSF signal.
- ✅ **Formule del SubframeSelector** (standard / nebulosa / galassia / ammasso / espressione) e soglia di
  approvazione ("Weight > 70").
- ✅ GUI: grafici per frame (FWHM, eccentricità, SNR, peso), blink con zoom 1:1–8:1 e trascinamento.

## 5. Registrazione
- ✅ Stelle + triangoli + affine, Lanczos con clamping. 🧪 Distorsione (polinomio/TPS), anche nel drizzle.
- 🧪 Plate solving ASTAP; 🧪 mosaici registrati sulle stelle o per plate solving (con RA/DEC negli header),
  livellamento del fondo, feathering, da CLI e da GUI.

## 6. Normalizzazione e integrazione
- ✅ Normalizzazione globale; 🧪 local normalization (anche nel drizzle).
- ✅ Reiezione sigma/winsorized/linear fit/ESD/percentile/mediana, crescita, mappe; 🧪 stessi stimatori su GPU.
- ✅ Scie di satelliti anche nell'1x. 🧪 Comet stacking. ✅ Memoria da RAM libera; frame registrati in float16 (opzione).

## 7. Drizzle
- ✅ 1x/2x/3x a impronta esatta, kernel square/gaussian/lanczos3, Bayer drizzle, CPU multi-thread, multi-GPU.

## 8. Post-produzione
- ✅ ~55 plugin, preset JSON (`nebula`, `nebula_pro`, `gaeeb`, `galaxy`, `sho`...), CLI.
- ✅ **Maschere per singolo passo** (range, colore, luminanza, stelle, layer; inversione, sfocatura).
- 🧪 SPCC con curve di sensori (Sony, Panasonic, mono+RGB) e filtri (UV/IR, L-Pro, L-eNhance, L-eXtreme,
  narrowband) combinabili, riferimento galassia spirale media, grafico. Curve approssimate.
- ✅ DBE (campioni automatici o cliccati sull'anteprima), polinomio, GraXpert AI.
- ✅ Deconvoluzione RL-TV con PSF Moffat misurata; TGV denoise, MMT, sottrazione primo layer, wavelet, DeepSNR.
- ✅ Stretch: auto, linked MTF (HT Stretch Linked RGB), histogram transformation, GHS, arcsinh (protezione luci).
- ✅ Curve per canale (RGB, R, G, B, L*, a*, b*, C, S, H), colore selettivo, SCNR completo, linear fit, HDR composition.
- ✅ Stelle: StarNet2, Dark Star, riduzione, aloni, colore, ricombinazione; StarStretch, dimensione, correzione
  stelle magenta, qualsiasi passo applicabile al solo livello stelle (`"target": "stars"`). Clone stamp / heal, rotazione, crop.
- ✅ Annotazione oggetti (OpenNGC, Messier/NGC/IC/Sharpless, stelle brillanti, griglia, bussola, scala).
- ✅ Color management: sRGB / Display P3 / Adobe RGB con profilo ICC incorporato in TIFF/JPEG/PNG.
- ✅ Esterni: StarNet2, DeepSNR, GraXpert, Cosmic Clarity (sharpen, denoise, Dark Star, super-res, satelliti).
  RC-Astro CLI con la sintassi documentata (non provato, serve licenza).
- 🧪 **Gradiente con riferimento all-sky** (idea MGC/MARS): survey DSS2 dal servizio CDS hips2fits sul WCS
  dell'immagine; conserva la nebulosità estesa (test: nebulosa 106% vs 33% con DBE automatico).

## 9. GUI e wizard
- ✅ Wizard a 7 passi con tutte le opzioni nuove; modalità mosaico; anteprima dal vivo fedele (parametri scalati,
  StarNet in cache); confronto prima/dopo; coda lavori con annulla e storico.
- ✅ Protezione in rete: con `--host` non locale serve un token (stampato all'avvio), config dei programmi solo
  da localhost, file solo nelle cartelle della sessione. In locale nulla cambia.
- ✅ Editor delle maschere per passo (range/colore/luminanza/canale/stelle/pennello) con sovrapposizione rossa e
  Ctrl+K; editor pipeline trascinabile con selettore dei plugin; punto di simmetria GHS col clic; passi sul
  livello stelle; header FITS (autore, copyright...) nell'export; tabelle frame per filtro/pannello.

## 10. Prestazioni
- ✅ CPU: calibrazione in parallelo, drizzle multi-thread, RCD a bande, mediane NaN veloci.
- 🧪 GPU: tutti i metodi di reiezione con codice condiviso numpy/CuPy, bande da memoria libera per scheda.
- 🧪 GPU per deconvoluzione, TGV e wavelet (codice unico numpy/CuPy, mai eseguito su GPU vera).
  CPU: deconvoluzione 16,9 → 5,9 s, wavelet 4,4 → 2,4 s su 24 MP.

## 11. Dal video "workflow 2026" (21/10)
- ✅ PCC con APASS DR9 (VizieR II/336), linear fit automatico dei gradienti di colore (HLP Astro Image Premiere),
  GHS completo (SP/LP/HP, inverso, lineare), unsharp mask, Narrowband Normalization (ricostruita), testo
  sull'immagine, maschera da canale, maschera a pennello (Dynamic Paint Brush), metadati FITS.
- ⬜ BlurX/StarX/NoiseX restano a pagamento (RC-Astro CLI pronto, serve licenza).
