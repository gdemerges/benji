# benji/ — Core package

`main.py` est un point d'entrée mince : logging, Sentry, politique « accessory » macOS (avant tout import Qt), puis délègue à `BenjiApplication`.

## Observabilité

`logging_config.py` — deux handlers : stderr + un `RotatingFileHandler` (2 Mo × 3) vers `log_file_path()` (`~/Library/Logs/Benji/benji.log`). Lancée depuis le Finder, l'app n'a **pas de terminal** : le fichier est le seul canal de diagnostic. Les transcriptions ne sont logguées qu'en DEBUG (le fichier est joint aux rapports de bug).

`monitoring.py` — Sentry, **inactif sans `BENJI_SENTRY_DSN`**. L'essentiel du module est le scrubbing, pas l'init : `include_local_variables=False` (une exception dans `stt/transcriber.py` a le texte de la réunion dans ses locales), breadcrumbs à INFO (donc pas les logs DEBUG), et `_scrub_event` en dernier rempart (jetons, chemin du home).

`report.py` — rendu pur (sans Qt) du `mailto:` de signalement : version, OS, config moteur, `SessionStats`. Aucune donnée personnelle par construction. Câblé au tray (« Signaler un problème… »), qui révèle le log à côté — un `mailto:` ne peut pas porter de pièce jointe.

`app.py` est le **composition root** : la classe `BenjiApplication` porte l'état et découpe le démarrage/arrêt en phases (`_build_configs`, `_build_account`, `_build_pipeline`, `_create_qapp`, `_load_transcriber`, `_start_stt`, `_build_display`, `_build_windows`, `_build_tray_and_shortcuts`, `shutdown`). Les cinq configs sont regroupées dans `AppConfigs` (injectable → les phases non-Qt sont testables sans lancer l'app, cf. `tests/test_app_composition.py`).

`config.py` contient tous les paramètres sous forme de dataclasses (`AudioConfig`, `VADConfig`, `STTConfig`, `UIConfig`, `LLMConfig`).

`onboarding.py` — premier lancement, **sans Qt** : état de l'autorisation micro (AVFoundation, `UNKNOWN` plutôt qu'une affirmation fausse si l'API manque), présence des poids dans le cache HF, et `ModelDownloader` dont la progression est **observée sur le disque** (l'API tqdm de `huggingface_hub` n'est pas stable d'une version à l'autre ; compter les octets marche partout et compte aussi ce qui était déjà là). Le marqueur vit dans les données utilisateur, pas dans QSettings.

`hotkeys/` — raccourci **global**, un module par OS derrière `build_hotkeys()` : `macos.GlobalHotkeys` (Carbon — voir la note du CLAUDE.md racine sur le choix de Carbon plutôt que `NSEvent.addGlobalMonitor`), `windows.WindowsHotkeys` (`RegisterHotKey` + `WM_HOTKEY` intercepté sur la boucle Qt), `linux.LinuxHotkeys` (`XGrabKey`, X11 seulement — échoue silencieusement sous Wayland ; le fil X11 **rend l'action au thread Qt**, elle touche le tray). `parsing.py` est pur et porte toute la logique testable : `_tokenize()` partagé (« Ctrl+Alt+Cmd+B » → modificateurs/touche), puis `parse_shortcut()`/`parse_shortcut_windows()`/`parse_shortcut_x11()` encodent dans les codes de chaque OS et refusent tous une touche sans modificateur — la réserver la retirerait de toutes les autres apps. Les modules Windows/Linux ne sont importés que sur leur OS. Windows/Linux non validés sur machine réelle.

`search.py` — recherche plein-texte **pure** : insensible aux accents et à la casse, tous les termes présents dans n'importe quel ordre, le locuteur compte comme texte cherchable. À l'échelle d'une réunion, la proximité entre deux mots ne veut rien dire.

`paths.py` — emplacement des données utilisateur (`~/Library/Application Support/Benji`) vs cache re-téléchargeable (`~/.cache/benji`, modèles seulement), avec migration au premier accès. Les chemins sont résolus **à l'appel**, jamais à l'import.

`recording.py` — **portillon de conservation**, pur. Le direct s'affiche toujours ; l'écriture disque attend l'accord, et l'attente (bornée) est versée à ce moment-là. Ne logue que des comptes, jamais du texte.

`stats.py` — `SessionStats` compte les métriques de la session (segments, latences, drops), en mémoire. `save()` persiste un instantané dans `log_dir()/sessions.jsonl` (borné à 20, comme la rotation des logs) — appelé au shutdown propre de l'app, après les joins des threads. Sans ça, une session qui se termine mal ne laissait aucune trace passé le process : le rapport de bug suivant ne parlait que d'une session vierge. Diagnostic anonyme, comme le reste (cf. `report.py`).

`queues.py` — `NotifyingQueue` : le producteur réveille son consommateur. Sonder `display_queue` plus lentement aurait retardé le premier mot d'un énoncé, la sonder vite réveillait le CPU en permanence.

`meetings.py` — registre des réunions (id, titre, début, fin, **noms de locuteurs**, **moments marqués**) dans `meetings.json` (0600, écriture atomique). La réunion courante est un état **du process** (plusieurs `TranscriptionHistory` écrivent le même fichier) et n'est ouverte que par la première transcription : `current_meeting_id()` lit sans créer.

`llm/mlx_runner.py` — **fil unique d'inférence mlx-lm**. Un stream MLX n'appartient qu'au thread qui l'a créé, et `mlx_lm` crée le sien *à l'import* : le premier des quatre consommateurs (correcteur sur le thread STT, titreur, résumé en direct, `SummaryWorker` de Qt) confisquait le moteur, les autres levaient « There is no Stream(gpu, N) in current thread ». Tout passe désormais par ce fil — chargement des poids compris, MLX liant un tableau au thread qui l'évalue en premier — et il **s'approprie** le `generation_stream` du module au cas où `mlx_lm` aurait été importé ailleurs avant lui. Piège : `mlx_lm.generate` est la *fonction*, pas le module ; la réattribution vise `sys.modules["mlx_lm.generate"]`.

`llm/titler.py` — nomme la réunion en cours à partir de ses premières phrases, via le modèle local déjà chargé. Trois règles : **on n'écrase jamais un titre choisi** (la condition est que le titre soit *exactement* celui par défaut), on attend d'avoir de quoi juger (300 caractères), et on ne nomme qu'une fois — un titre qui bouge en cours de réunion serait pire que l'horodatage. Le titre est du contenu de réunion : il n'est pas loggué.

`history.py` + `stats.py` reçoivent chaque utterance finale pour persister l'historique et les métriques de session. `add()` est sur le chemin chaud : la troncature est amortie par un compteur en mémoire, jamais une relecture du fichier par segment. Chaque entrée porte son `meeting`. `group_by_meeting()` rend tout le fichier indexé par réunion **en une lecture** — la fenêtre Réunions relisait le fichier une fois par réunion pour compter les échanges.

`export.py` — rendu pur (sans Qt) des entrées d'historique vers `txt` / `md` / `srt`, avec renommage optionnel des locuteurs (`speaker_names`). Le SRT dérive les bornes de temps des horodatages (fin d'un segment = début du suivant, durée estimée pour le dernier). Câblé aux boutons Copier/Exporter de `history_window`.

`account.py` — compte Benji côté app : `Session` (login/register/refresh auto) + persistance des jetons dans `credentials.json` (0600) sous les données utilisateur. L'abonnement suit le **compte**, pas le poste → mêmes identifiants = même plan sur toute plateforme. Au démarrage, `main.py` injecte l'access token dans `LLMConfig.backend_token`.

`billing.py` — client Stripe côté app : appelle le backend authentifié (`/v1/billing/checkout`, `/v1/billing/portal`) et ouvre l'URL renvoyée dans le navigateur. Aucune clé Stripe sur le poste. Câblé au menu tray (token fourni par la `Session`).
