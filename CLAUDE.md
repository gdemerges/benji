# Benji

macOS real-time transcription app. Pipeline: mic → VAD → STT → subtitle overlay.

## Rules

- Python 3.12, **PySide6** (binding Qt officiel, LGPL — PyQt6 était GPL, incompatible avec une distribution binaire fermée ; `PySide6-Essentials` seul, jamais `Addons`). No mypy, no type stubs. **Deux moteurs locaux selon l'OS** (décision du 2026-09-18, cf. `benji/stt/CLAUDE.md`) : Parakeet/MLX sur Apple Silicon, `FasterWhisperBackend` (CTranslate2) ailleurs — détecté par sonde (`parakeet_mlx` absent), jamais par `platform.system()`. Windows/Linux restent un chantier en cours (raccourci global, packaging, onboarding cross-OS non finis) ; seul le moteur STT est câblé à ce stade.
- Dependency management: **uv** (`pyproject.toml` is source of truth). `uv sync` to install, `uv run benji` to launch.
- Tunable config lives in `benji/config.py` (dataclasses), not config files. A few operational/secret settings are read from env vars instead: `BENJI_LAUNCH_MODE`, `BENJI_LOG_LEVEL`, `BENJI_VIBRANCY`, `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` (diarization), `ANTHROPIC_API_KEY` (cloud summary), `BENJI_SENTRY_DSN` + `BENJI_ENV` (crash reporting, inactif sans DSN), `BENJI_SUPPORT_EMAIL` (destinataire du « Signaler un problème », défaut = adresse perso).
- **Confidentialité — règle non négociable** : Benji transcrit des réunions. Rien de ce qui sort de la machine (log fichier, rapport de bug, événement Sentry) ne doit contenir de texte transcrit, de glossaire, de chemin d'historique ni de jeton. Les transcriptions sont logguées en **DEBUG** uniquement ; `benji/monitoring.py` et `benji/report.py` scrubbent le reste. Des tests verrouillent ces trois canaux — ne les contourne pas.
- **Écouter et garder ne sont pas le même geste** : Benji transcrit dès le lancement, mais rien n'est écrit sur disque avant l'accord de l'utilisateur (`benji/recording.py`, bandeau du Live ou tray). Ce qui a déjà été dit est **versé** au moment de l'accord — c'est ce qui distingue ce modèle d'un bouton « démarrer ». `STTConfig.confirm_before_saving = False` rend le comportement d'avant.
- Three inter-thread queues: `audio_queue` → `transcribe_queue` → `display_queue`. Never block the Qt thread. `display_queue` est une `NotifyingQueue` (`benji/queues.py`) : le producteur **réveille** le bus d'affichage au lieu d'être sondé 62 fois par seconde. Et **une passe partielle = un message**, jamais un par mot — la file est bornée et ses `put` sont bloquants, publier mot à mot faisait attendre le thread STT sur le tick de Qt. Avec `AudioConfig.system_audio`, un `AudioMixer` s'intercale en amont d'`audio_queue` (micro + audio système) — cf. `benji/audio/CLAUDE.md`.
- **Le backend (`backend/`) est dégelé partiellement** depuis le 2026-09-18 : il sert l'offre payante du modèle freemium (achat unique local sur Mac reste le défaut ; Windows/Linux choisissent entre un moteur local plus léger et ce backend, cf. bandeau du 1er lancement dans `benji/ui/onboarding_window.py`). Utilisable en développement/test avec des clés Stripe **sandbox**. Restent explicitement gelés, décision à reprendre avant d'y toucher : passage Stripe en **live** (action sur le dashboard, pas du code), `/v1/history`, migration Postgres, clients mobiles — cf. le bandeau dans `backend/README.md`.
- **Données utilisateur** : historique, résumés, `meetings.json` et identifiants vivent sous `~/Library/Application Support/Benji` (`benji/paths.py`), pas dans `~/.cache` — qui ne garde que les poids de modèles. Ne jamais résoudre ces chemins à l'import : `tests/conftest.py` isole `HOME`, un chemin figé irait déplacer les vraies données de l'utilisateur.
- **Une réunion est une entité de premier ordre** (`benji/meetings.py`) : chaque entrée d'historique porte son identifiant, et l'export / le résumé / l'effacement s'y cantonnent. Les entrées d'avant cette notion sont regroupées sous `meetings.LEGACY_ID`.
- **Direction visuelle** : une seule couleur saturée (le rouge d'enregistrement, jamais pour une action), trois voix typographiques (SF Pro pour l'app, New York pour les paroles transcrites, SF Mono pour le temps), et la ligne de temps du transcript comme élément signature. Toute couleur vit dans `benji/ui/style.py` — cf. `benji/ui/CLAUDE.md`.
- **Parakeet TDT est le moteur, Whisper est le rattrapage.** Parakeet ne paie que l'audio reçu là où Whisper encode toujours une fenêtre paddée de 30 s — 58 ms contre ~680 ms mesurés. Mais Parakeet détecte la langue sans pouvoir la forcer, et dérive vers l'anglais sur les segments difficiles. La passe finale est donc **hybride** (`STTConfig.final_engine="hybrid"`) : Parakeet décode, on relit le texte, et Whisper ne reprend que les segments qui ont dérivé — ses poids ne sont chargés que ce jour-là. Le contexte glissant reste parti avec `initial_prompt` ; le **glossaire est revenu à un autre étage** (relecture du texte final, `benji/stt/lexicon.py`) et non comme prompt au moteur. Détails dans `benji/stt/CLAUDE.md`.
- `STTConfig.language` defaults to `"fr"`. Keep French in mind when touching STT logic.
- **Premier lancement** : `benji/onboarding.py` (logique pure) + `benji/ui/onboarding_window.py` — permission micro, puis téléchargement des poids avec progression. Marqueur dans les données utilisateur : supprimer le dossier rejoue l'assistant. `app.run()` l'exécute après `_create_qapp` et avant le splash ; fermer l'assistant **quitte** l'app plutôt que de démarrer sans micro ni modèle.
- **Raccourci global** (`benji/hotkeys/`, un module par OS, `build_hotkeys()` choisit) : macOS via Carbon `RegisterEventHotKey`, qui n'exige **aucune autorisation** — contrairement au moniteur global Cocoa, qui échoue silencieusement sans « Surveillance de la saisie ». Windows via `RegisterHotKey` + interception `WM_HOTKEY` sur la boucle Qt. Linux via `XGrabKey` (X11 seulement — Wayland interdit la capture globale par design, sans solution unifiée). C'est le seul chemin praticable pour couper le micro pendant une visio en plein écran. Toute défaillance dégrade en silence : un raccourci absent est une gêne, une app qui ne démarre pas est une panne. **Windows et Linux non validés sur machine réelle** (aucun poste disponible pour ce projet) — même statut que Carbon à sa création.
- macOS: accessory policy must be set before `QApplication()` — see `benji/main.py:9`.
- Run: `uv run benji`. Tests: `uv run pytest`.

## Vault Obsidian

Le suivi de ce projet est documenté dans : `/Users/guillaumedemerges/Documents/Life/wiki/projects/Benji`

Notes du vault : `Benji.md` (fiche principale, état d'avancement) + `Benji-Architecture.md`, `Benji-Backend-Cloud.md`, `Benji-Distribution.md`.

**Règle** : à la fin d'une session de travail significative (feature terminée, architecture changée, checklist publication avancée), mets à jour la note concernée avec les changements. Garde le format existant (frontmatter, sections, checkboxes cochées/décochées).

## Modules

- [benji/](benji/CLAUDE.md) — core package, entry point, config, history, stats
- [benji/audio/](benji/audio/CLAUDE.md) — mic capture + Silero VAD
- [benji/stt/](benji/stt/CLAUDE.md) — Whisper transcription, diarization, post-processing
- [benji/ui/](benji/ui/CLAUDE.md) — PySide6 overlay, tray, history window, live summary
