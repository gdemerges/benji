# benji/stt/

- `backend.py` — **deux moteurs, et un arbitrage entre eux**, parce que les deux
  types de passe n'ont pas le même cahier des charges.

  | | passes partielles | passe finale |
  |---|---|---|
  | moteur | Parakeet TDT | **hybride** : Parakeet, relayé par Whisper |
  | ce qu'on optimise | la latence | la justesse de ce qui reste |
  | coût mesuré | ~50 ms | ~150 ms, ~800 ms sur les segments repris |
  | langue | détection auto | **`fr` garanti** sur les segments qui dérivent |

  **Pourquoi un hybride.** Parakeet fait de la détection automatique sur 25
  langues et **n'offre aucun levier pour la forcer** — confirmé par la fiche
  NVIDIA, et `parakeet-mlx` n'a pas une seule occurrence de « language » dans son
  code. Sur des segments difficiles il bascule en anglais : au milieu d'une
  réunion française on récupère « Enwiten ne se content plus de relier the
  utility devient also the chef d'orchestre ». Whisper accepte `language="fr"`
  et rend la dérive impossible, mais coûte ~5× plus cher sur *tous* les segments
  — pour se prémunir d'un accident qui n'en concerne qu'une poignée.

  `HybridFinalBackend` (défaut) tranche **après coup** : décoder avec Parakeet,
  puis relire le texte produit (`language.py`) et ne relancer Whisper que si les
  mots-outils trahissent l'autre langue, ou si Parakeet n'a rien rendu. Corollaire
  heureux : sur une réunion française propre, les poids de Whisper ne sont même
  **jamais chargés** (`WhisperBackend` est paresseux, `eager_warmup = False`).

  Prix payé : la passe finale ne streame plus mot à mot, puisqu'il faut avoir lu
  tout le texte pour décider s'il est recevable. Sans effet visible — les mots
  sont déjà à l'écran, posés par les partielles, et le final les remplace en bloc.

  `STTConfig.final_engine="whisper"` repasse au tout-Whisper (la garantie
  maximale, si l'hybride décevait en réunion) ; `"parakeet"` réutilise le moteur
  des partielles sans aucune garantie de langue.

  **Whisper tourne sur un thread dédié au backend**, créé une fois pour toutes :
  chargé depuis le thread STT, il deviendrait inutilisable si le superviseur
  relançait ce thread après un incident (même piège MLX que ci-dessous).

  Ce compromis **ne se voit pas sur de l'audio de synthèse**, trop propre pour
  faire dériver la détection. Il faut de la vraie parole pour l'observer : s'en
  souvenir avant de conclure quoi que ce soit d'un banc d'essai TTS.

  **Windows/Linux → `FasterWhisperBackend`** (CTranslate2), détecté par
  `_parakeet_available()` (sonde `parakeet_mlx`, pas l'OS — un Mac Intel sans
  mlx suit le même chemin) plutôt que branché sur `IS_MACOS`. Contrairement à
  Parakeet, faster-whisper accepte `language=` à la demande : pas de relais
  hybride, un seul moteur sert aux deux passes, dimensionné différemment.
  Mesuré sur CPU (audio français `say`, `beam_size=1`, mêmes tampons que le
  banc Parakeet) :

  | clip | Parakeet (MLX) | fw-tiny (CPU) | fw-small (CPU) | fw-medium (CPU) |
  |---|---|---|---|---|
  | 1,2 s | 58 ms | 116 ms | 948 ms | 2 524 ms |
  | 14,7 s | 222 ms | 242 ms | 1 237 ms | 3 780 ms |

  `tiny` (partielles) reste dans l'ordre de grandeur de Parakeet ; `small`
  (finale, `STTConfig.final_model_size` par défaut hors Mac) coûte ~1 s mais
  n'est payé qu'une fois par segment, sans arbitrage à attendre. `medium` est
  trop lent même en final (mesuré). CUDA détecté automatiquement
  (`ctranslate2.get_cuda_device_count()`) mais **non validé** — pas de GPU
  NVIDIA disponible pour la mesure ; les benchmarks connus de la lib
  suggèrent ×10-20, de quoi rapprocher small/medium du niveau Parakeet.
  Chargé **au constructeur** (pas paresseusement comme `WhisperBackend`) :
  sur ce chemin c'est le moteur de tous les segments, le charger tard ne
  ferait que déplacer le coût sur le premier segment réel, en pleine réunion.

  **Piège MLX — le modèle appartient au thread qui l'a chargé.** MLX charge paresseusement et lie les tableaux au **stream du thread qui les évalue en premier**. `ParakeetBackend.__init__` appelle donc `mx.eval(model.parameters())` pour matérialiser les poids sur place ; sans ça la liaison n'aurait lieu qu'au premier décodage réel et l'inférence depuis le thread STT lèverait « There is no Stream(gpu, N) in current thread ». `warmup()` ne peut pas jouer ce rôle : il préchauffe sur du silence, dont Parakeet ne décode aucun token. Et le chargement doit se faire sur un thread qui **vit aussi longtemps que l'app** — d'où `_load_transcriber` sur le thread principal dans `app.py`. Chargé depuis un thread éphémère, Parakeet devient **définitivement** inutilisable, sans réparation possible par `mx.new_stream()`.

  **Alimenté en mémoire** : l'API publique de `parakeet-mlx` ne transcrit que des chemins de fichiers ; on passe par `get_logmel()` + `generate()`. Écrire les tampons d'une réunion dans un fichier temporaire serait une régression de confidentialité — et ça évite la dépendance à ffmpeg. Ne pas « simplifier » vers `model.transcribe(path)`.

  **Sous-mots** : Parakeet rend des morceaux de mots (`" c"`, `"ô"`, `"té"`), recollés par `group_tokens_into_words()` — **par phrase** (`words_from_result`), sinon la fin d'une phrase se colle au début de la suivante (« Apple.Ça »).

- `transcriber.py` — consomme `transcribe_queue`, stream les mots vers `display_queue`. Chaque passe partielle re-décode le tampon **entier** via Parakeet (~123 ms pour 8 s) et fige le préfixe sur lequel deux passes successives s'accordent (LocalAgreement-2) ; le figeage est **monotone**, un mot affiché n'est jamais repris. La passe finale passe par le moteur hybride : c'est ce texte-là qui est conservé, puis relu par le glossaire (`lexicon.py`). Si `llm_correction` est activé, le texte brut est affiché immédiatement puis corrigé sur un thread dédié (`STT-corrector`) : chaque final porte un `seq` que l'overlay utilise pour remplacer la bonne ligne (et ignorer une correction dont le segment n'est plus à l'écran). Un segment peut rendre **plusieurs** finals — un par tour de parole, cf. `diarization.py`.
- `diarization.py` — labellisation de locuteurs (activée par défaut). Le tagger ne dépend que de l'audio : il tourne **en parallèle** de la passe finale (pool à un worker, `_await_spans` avec timeout) au lieu d'ajouter son coût au délai d'affichage. Un tagger bloqué rend un segment sans locuteur, jamais un thread STT figé. Backend `pyannote` (embeddings, N locuteurs, modèle HF **gated** → accepter les conditions sur hf.co/pyannote/embedding) avec fallback automatique sur `pitch` (F0, A/B) si indisponible. Le label voyage comme champ `speaker` dans le message `final_text` (jamais collé au texte) — la couleur par locuteur vient de `benji.ui.style.speaker_color`.

  **Un segment VAD n'est pas un tour de parole.** Avec `silence_duration_ms = 600`,
  deux personnes qui s'enchaînent sans pause franche tiennent dans le même
  tampon : une étiquette unique fondait les deux voix en une phrase, d'une seule
  couleur. On étiquette donc **par fenêtres glissantes** (`label_windows`, 1,5 s
  / pas 0,75 s) et on recoupe les mots — qui sont horodatés — aux frontières
  (`split_by_speaker`, **pure**). Le segment rend alors *N* `final_text`, un par
  tour, chacun avec son locuteur, son post-traitement et son entrée d'historique.

  Trois garde-fous, parce que le clustering se trompe : les fenêtres se
  recouvrent, donc la frontière est floue à un demi-pas près → on la **recale sur
  le silence** le plus proche (`_snap_to_pauses` : on se relaie dans un blanc),
  mais seulement s'il y a un vrai blanc, sinon bouger serait décider au hasard ;
  un tour plus court que `diarization_min_turn_words` est **refondu** dans son
  voisin, sinon une fenêtre isolée qui change d'avis ferait du confetti ; une
  fenêtre sans étiquette (trop courte, inférence en échec) est **ignorée** plutôt
  que d'ouvrir un tour anonyme au milieu de la phrase. `diarization_hop_s = 0`
  revient à une étiquette par segment.

  Coût : un embedding par fenêtre au lieu d'un par segment, toujours sur le pool
  de diarisation pendant que le décodage tourne — absorbé tant qu'il reste sous
  la durée du final. Une voix unique ne paie rien : un seul tour, le chemin d'avant.

  **La parole vraiment simultanée reste hors de portée** : sur un signal mélangé,
  le moteur STT lui-même n'entrelace qu'une transcription. Aucun ré-étiquetage a
  posteriori ne le rattrape ; il faudrait de la séparation de sources. Ne pas
  espérer que la découpe en tours règle ce cas-là.
- `language.py` — **pur**. Détection de dérive de langue par mots-outils (pas par
  vocabulaire : une réunion française truffée d'anglicismes métier garde ses
  « le », « du », « sur »). Conservateur par construction — il faut au moins deux
  marqueurs de l'autre langue *et* plus que la langue attendue, sur au moins
  quatre mots. Un faux positif coûte une passe Whisper inutile, un faux négatif
  laisse une phrase dans la mauvaise langue dans l'historique.
- `lexicon.py` — **pur**. Le glossaire utilisateur, revenu **à un autre étage**.
  Le retrait d'`initial_prompt` avec Whisper avait fait disparaître tout levier
  sur les noms propres — le point faible n°1 d'une transcription de réunion. Ici
  on ne souffle rien au modèle, on **relit sa sortie** : un terme du glossaire
  remplace une suite de mots qui *sonne* comme lui (clé phonétique approchée
  orientée français, comparée sur la fenêtre **concaténée** pour rattraper un
  terme découpé — « data dogue » → « Datadog »). Appliqué **seulement au final**.
  Le fichier (`glossary.txt`, données utilisateur, 0600, éditable dans les
  Préférences) liste des clients et des projets : il n'est **jamais loggué** et
  ne part dans aucun rapport — un test le verrouille.
- `postprocessing.py` — nettoyage grammaire/ponctuation appliqué après la passe finale. En `fr`, **typographie française** (`french_typography`, pure, idempotente) : espace fine insécable avant `; ! ?`, insécable avant `:` et dans « » — insécables pour qu'un retour à la ligne de l'overlay ne laisse jamais un « ? » seul. `14:30` et `?!` épargnés. Filtre d'hallucinations en **deux familles** : les crédits de sous-titrage (jamais prononcés en réunion) valent n'importe où ; les formules de fin (« merci à tous », « à la prochaine »…) seulement si elles sont **tout** le texte — les chercher en sous-chaîne jetait les phrases d'ouverture et de clôture de réunion. Répétition dégénérée : 6 fois le même mot (« non non non non » est humain).
- **Contexte après coupure forcée** : un message de `transcribe_queue` peut porter `context_s` (cf. `benji/audio/CLAUDE.md`). `drop_context_words` (pure) retire les mots dont le milieu tombe dans ce contexte ; les horodatages restent relatifs au tampon complet, même base de temps que les étendues de diarisation.

`VADConfig.partial_growth_factor` est à 0 (cadence fixe) : le frein progressif protégeait du coût de Whisper et ne faisait plus que rendre le direct poussif.
