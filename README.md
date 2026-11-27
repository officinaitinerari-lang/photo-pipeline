# photo-pipeline

Pipeline fotografica automatica a costo zero per NAS UGREEN NASync DXP2800 (Intel N100).
Metti una foto in `/data/input` e ottieni:

- **Etsy**: 5 formati full-bleed + 2 Gallery Poster a 300 DPI, divisi in zip sotto i 20 MB, 3 mockup e `etsy_listing.txt` (EN + IT).
- **Microstock**: JPEG q95 sRGB con IPTC/XMP (titolo, descrizione, 35–45 keyword) caricato via SFTP (Adobe Stock) e/o FTPS (Shutterstock), solo se l'AI lo giudica commercialmente sicuro.

```
photo-pipeline/
├── .github/workflows/ci-cd.yml   ruff + pytest + push immagine su ghcr.io
├── Dockerfile                    multi-stage: export ONNX Real-ESRGAN + runtime python:3.11-slim
├── docker-compose.yml            limiti DXP2800 (2.5 CPU, 2560 MB, cpu_shares 512)
├── .env.example
├── requirements.txt / requirements-dev.txt / pyproject.toml
├── scripts/export_upscaler_onnx.py
├── src/
│   ├── main.py             CLI: run | process | upload-pending | retry-failed | status | healthcheck
│   ├── config.py           impostazioni da variabili d'ambiente
│   ├── database.py         SQLite: coda, SHA-256, cache AI, stato upload
│   ├── watcher.py          watchdog + debounce (dimensione stabile 5 s)
│   ├── image_loader.py     JPEG/TIFF/PNG/NEF/DNG, ICC→sRGB, EXIF/GPS, contesto dal percorso, pellicola
│   ├── ai_analyzer.py      Gemini 2.5 Flash, JSON Pydantic, rate limit, backoff, controllo legale
│   ├── upscaler.py         ONNX Runtime CPU a tile / Lanczos / modalità pellicola
│   ├── smart_crop.py       saliency spectral residual → Sobel/Canny + volti Haar
│   ├── print_generator.py  formati full-bleed + Gallery Poster
│   ├── mockup_generator.py 3 immagini vetrina procedurali
│   ├── packaging.py        zip divisi sotto il limite Etsy
│   ├── metadata_writer.py  IPTC/XMP via ExifTool
│   ├── ftp_uploader.py     FTPS / SFTP, DRY_RUN, verifica dimensione
│   ├── notifier.py         Telegram (opzionale)
│   ├── typography.py, logging_setup.py
│   └── pipeline.py         orchestrazione e gestione errori
└── tests/                  smart crop, zip, contesto, AI post-processing, DB, upscaler tiling
```

## 1. Installazione sul NAS (UGOS Pro)

1. **Cartella dati**: in UGOS crea una cartella condivisa, es. `PhotoPipeline` (percorso tipico `/volume1/PhotoPipeline`). Le sottocartelle (`input`, `output`, `processed`, `failed`, `state`) vengono create al primo avvio.
2. **Immagine Docker**: fai il push del repository su GitHub. Il workflow pubblica `ghcr.io/<utente>/photo-pipeline:latest`. Se il pacchetto GHCR è privato, sul NAS esegui `docker login ghcr.io` con un token che abbia il permesso `read:packages`, oppure rendi pubblico il pacchetto.
3. **Configurazione**: copia `docker-compose.yml` e `.env.example` in una cartella del NAS (es. `/volume1/docker/photo-pipeline/`), rinomina `.env.example` in `.env` e compila almeno `GEMINI_API_KEY`, `GHCR_OWNER`, `PUID` e `PGID`. Per conoscere UID/GID: SSH sul NAS → `id`.
4. Controlla il volume in `docker-compose.yml` (`/volume1/PhotoPipeline:/data`).
5. **Avvio**: in UGOS → Docker → Progetto → Crea, incolla il compose. Da SSH:
   ```bash
   cd /volume1/docker/photo-pipeline && docker compose up -d && docker compose logs -f
   ```
6. **Aggiornamenti**: `docker compose pull && docker compose up -d`.

In alternativa, per compilare sul NAS (lento: il primo stage scarica PyTorch CPU per esportare il modello): decommenta `build: .` nel compose. Con `--build-arg BUILD_UPSCALER_MODEL=false` salti il modello e usi solo Lanczos.

## 2. Chiavi e credenziali

**Gemini (gratuita)**: vai su <https://aistudio.google.com/apikey>, accedi con un account Google, clicca *Create API key* e incollala in `GEMINI_API_KEY`. I limiti del piano gratuito (richieste al minuto e al giorno) li decide Google e cambiano nel tempo. Quando la quota giornaliera finisce, la foto non va in errore: viene rimandata di `GEMINI_QUOTA_COOLDOWN_MINUTES`. Attenzione: sul piano gratuito Google può usare i contenuti inviati per migliorare i propri prodotti. Viene inviata solo un'anteprima di 1024 px.

**Adobe Stock (SFTP)**: Contributor Portal → Upload → icona SFTP → *Generate password*. Imposta `ADOBE_ENABLED=true`, `ADOBE_USERNAME` (l'ID numerico mostrato) e `ADOBE_PASSWORD`. Host: `sftp.contributor.adobestock.com:22`. Al primo collegamento la chiave del server viene salvata in `/data/state/known_hosts`; se poi cambia, l'upload viene rifiutato.

**Shutterstock (FTPS)**: dal portale contributor recupera le credenziali FTPS (host `ftps.shutterstock.com`, porta 21, TLS esplicito). Imposta `SHUTTERSTOCK_ENABLED=true`, `SHUTTERSTOCK_USERNAME` e `SHUTTERSTOCK_PASSWORD`.

Verifica sempre host e procedure sui portali delle agenzie: possono cambiare.

**Telegram (opzionale)**: crea un bot con @BotFather (`TELEGRAM_BOT_TOKEN`), scrivigli un messaggio e ricava `chat.id` da `https://api.telegram.org/bot<TOKEN>/getUpdates`.

## 3. Primo test sicuro (`DRY_RUN=true`)

`DRY_RUN=true` è il valore predefinito: nessun file lascia il NAS e gli upload sono solo simulati nei log.

1. Copia 2–3 foto in `input/`, ad esempio in una sottocartella che descrive il luogo: `input/Giappone_Kyoto_Inverno/tempio.jpg`. Il nome della cartella e del file diventa contesto per l'AI. Per le scansioni di pellicola usa parole come `35mm`, `film`, `kodak` nel percorso.
2. Segui i log: `docker compose logs -f` (con `LOG_FORMAT=text` per una lettura più comoda).
3. Controlla:
   - `output/etsy/<slug>/`: zip, `mockups/`, `etsy_listing.txt`
   - `output/microstock_pending/`: JPEG con metadati (controlla con `exiftool -XMP-dc:all -IPTC:all file.jpg`)
   - `output/microstock_review_editorial/`: foto bloccate (persone, targhe, loghi, opere protette); il motivo è nel nome del file
   - `processed/`: originali completati; `failed/`: originali + `.error.log`
4. Quando sei soddisfatto: imposta `DRY_RUN=false`, `docker compose up -d`, poi invia i file già pronti:
   ```bash
   docker exec photo-pipeline python -m src.main upload-pending
   ```

Comandi utili:
```bash
docker exec photo-pipeline python -m src.main status        # conteggio lavori per stato
docker exec photo-pipeline python -m src.main retry-failed  # rimette in coda i falliti
docker exec photo-pipeline python -m src.main process /data/input/foto.jpg
```

## Comportamento e affidabilità

- **Una foto alla volta**, con memoria liberata dopo ogni file. L'upscaling ONNX usa tile da 512 px e scrive direttamente alla risoluzione finale, senza immagini intermedie a scala 4×.
- **Idempotenza**: lo SHA-256 identifica la foto. Un duplicato già completato finisce in `processed/_duplicates/`. Dopo un riavvio i lavori interrotti tornano in coda, l'analisi AI salvata nel DB non viene ripetuta e un upload già riuscito non viene ripetuto.
- **Errori**: gli errori temporanei vengono ritentati (`MAX_JOB_ATTEMPTS`, con attesa crescente). I file illeggibili vanno subito in `failed/`. Gli upload falliti vengono ritentati ogni ora.
- **Microstock**: si esporta l'originale sviluppato, non la versione ingrandita (le agenzie rifiutano le immagini ingrandite). Sotto i 4 MP viene ingrandito con Lanczos. Il GPS viene rimosso per default.
- **Upscaling**: solo se il lato lungo è sotto 6000 px, con obiettivo 7200 px. Le pellicole usano solo Lanczos a piccoli passi più un leggero micro-contrasto, mai il modello AI.
- **Etsy**: ogni stampa JPEG resta sotto `MAX_PRINT_FILE_MB` e ogni zip sotto 19 MiB (meno di 20 MB). Etsy accetta al massimo 5 file per inserzione: se gli zip sarebbero di più, la pipeline rifà le stampe con qualità JPEG un po' più bassa (mai sotto `JPEG_QUALITY_MIN`) e, come ultima risorsa, lascia fuori le Gallery Poster.
- Il formato 2:3 è 4800×7200 px: 300 DPI nativi fino a 16×24 in (40×60 cm); 24×36 in risulta a circa 200 DPI, come indicato nell'infografica.

## Sviluppo

```bash
pip install -r requirements-dev.txt
ruff check . && pytest -q
```

## Bozze Etsy automatiche (facoltativo)

Con `ETSY_ENABLED=true` la pipeline crea su Etsy **solo bozze** (titolo, descrizione e tag nella lingua `ETSY_LANGUAGE`, prezzo `ETSY_PRICE`, tipo download, 3 mockup e gli zip). La pubblicazione resta manuale: la tassa di inserzione si paga quando premi "Pubblica".

1. Registra un'app personale su https://www.etsy.com/developers e annota *keystring* e *shared secret*.
2. Registra come callback lo stesso indirizzo https che metti in `ETSY_REDIRECT_URI`.
3. Imposta `ETSY_API_KEY`, `ETSY_SHARED_SECRET`, `ETSY_REDIRECT_URI`, `ETSY_ENABLED=true` e `DRY_RUN=false`, poi riavvia.
4. La pipeline scrive il link di autorizzazione in `state/etsy_auth_url.txt`. Aprilo, premi "Consenti" e salva l'indirizzo completo della pagina a cui vieni rimandato in `state/etsy_auth_code.txt`. Entro pochi minuti il token viene salvato in `state/etsy_token.json`.
5. Ogni `ETSY_SYNC_MINUTES` la pipeline crea le bozze mancanti. In ogni cartella `output/etsy/<slug>/` il file `etsy_draft.json` registra la bozza; un file con `{"status": "skip"}` esclude la cartella.
