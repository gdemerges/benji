# Benji — Contrat d'API (backend ↔ clients)

> Document normatif. Statut : **proposition v1**. Fige l'interface entre le
> backend et les clients (macOS, iOS, Android, web). Tant que ce contrat n'est
> pas figé, on n'écrit pas le backend. Voir le cadrage dans
> [`cloud-architecture.md`](./cloud-architecture.md).

## 0. Principes

- **Versionnement** : tout est préfixé `/v1`. Une rupture incompatible → `/v2`.
- **Transports** :
  - **REST** (JSON) — auth, compte/quotas, historique.
  - **SSE** — streaming des tokens de résumé (sens serveur→client uniquement).
  - **WebSocket** — transcription temps réel (audio montant + events descendants).
- **Encodage** : JSON UTF-8 pour le contrôle ; audio en **frames binaires** sur
  le WebSocket (pas de base64 — surcoût évité).
- **Base URL** : `https://api.benji.app` (exemple). WebSocket : `wss://`.
- **Auth** : jeton **Bearer** (JWT court + refresh). Voir §1.
- **Alignement client** : les events de transcription (§3.3) reprennent le
  vocabulaire interne actuel du `display_queue` (`segment_start`, `word`,
  `final_text`, `vad_status`) pour que le client macOS existant se branche sans
  réécriture de l'UI.

## 1. Authentification

### 1.1 `POST /v1/auth/login`

```jsonc
// Requête
{ "email": "user@example.com", "password": "..." }

// 200
{
  "access_token": "<JWT>",          // courte durée (~15 min)
  "refresh_token": "<opaque>",      // longue durée
  "expires_in": 900,                 // secondes
  "token_type": "Bearer"
}
```

### 1.2 `POST /v1/auth/refresh`

```jsonc
// Requête
{ "refresh_token": "<opaque>" }
// 200 : même forme que /login
```

**Rotation** : chaque refresh réussi **révoque** le `refresh_token` présenté et en
émet un nouveau — le client doit donc persister le refresh renvoyé à chaque appel.
Rejouer un refresh déjà tourné est traité comme un vol : toute la famille de
jetons de l'utilisateur est révoquée et l'appel renvoie `401 unauthenticated`
(reconnexion requise).

> Les endpoints `/v1/auth/*` sont **rate-limités** (par IP) : un excès de
> tentatives renvoie `429 rate_limited`.

### 1.3 Usage

- REST/SSE : header `Authorization: Bearer <access_token>`.
- WebSocket : le jeton voyage dans le **premier message `start`** (§3.2), car les
  clients navigateurs ne peuvent pas fixer de header sur un WS. Les clients
  natifs peuvent *aussi* l'envoyer en header `Authorization` — le backend
  accepte les deux, le message `start` faisant foi.

## 2. Compte & quotas

### 2.1 `GET /v1/me`

```jsonc
// 200
{
  "user_id": "usr_…",
  "plan": "free | pro",
  "entitlements": {
    "cloud_stt": true,                 // droit à la transcription cloud
    "cloud_summary": true              // droit au résumé cloud
  },
  "quota": {
    "stt_seconds_used": 4210,          // période courante
    "stt_seconds_limit": 36000,        // null = illimité
    "period_end": "2026-07-01T00:00:00Z"
  }
}
```

Le client lit ça au démarrage pour activer/désactiver les modes cloud et
afficher la conso. Le **métering fait foi côté serveur** (le client n'est pas de
confiance).

## 3. Transcription temps réel — WebSocket `/v1/transcribe`

### 3.1 Cycle de vie

```
client                              serveur
  │ — connect wss://…/v1/transcribe ─►│
  │ ── {start} (JSON) ───────────────►│   (auth + config)
  │ ◄───────────── {ready} (JSON) ────│
  │ ══ frames audio binaires ════════►│   (flux continu)
  │ ◄── {vad}/{segment_start}/{word}/ │   (events descendants)
  │      {final}/{error} (JSON) ──────│
  │ ── {stop} (JSON) ────────────────►│
  │ ◄───────────── {closed} (JSON) ───│
  │ — close ──────────────────────────│
```

### 3.2 Messages montants (client → serveur)

**`start`** (premier message, obligatoire) :

```jsonc
{
  "type": "start",
  "token": "<access_token>",          // si pas en header Authorization
  "audio": {
    "encoding": "pcm_s16le",          // v1 : PCM 16-bit signé little-endian
    "sample_rate": 16000,             // Hz : 8000/16000/24000/44100/48000, sinon bad_request
    "channels": 1
  },
  "language": "fr",                   // défaut "fr" ; null = détection automatique
  "diarization": true,                // labels de locuteurs côté serveur
  "glossary": ["Demergès", "Anthropic", "MLX"]  // biais lexical, optionnel
}
```

**Frames audio** : messages WebSocket **binaires**, chacun un bloc PCM brut
(`pcm_s16le`, 16 kHz, mono). Taille indicative 20–100 ms par frame. Pas de JSON,
pas de base64.

**`stop`** : `{ "type": "stop" }` — fin de session, le serveur flpush le segment
en cours puis renvoie `closed`.

### 3.3 Events descendants (serveur → client) — JSON

Tous portent `type`. Le vocabulaire reflète le `display_queue` interne.

```jsonc
{ "type": "ready" }                                   // session ouverte, prête

{ "type": "vad_status", "speaking": true }            // parole détectée on/off

{ "type": "segment_start" }                           // début d'un nouveau segment

{ "type": "word", "text": "bonjour" }                 // partiel incrémental

{ "type": "final_text",                               // segment finalisé
  "text": "Bonjour le monde",
  "speaker": "A",                                     // optionnel (diarisation)
  "drop": false }                                     // true = annuler le partiel

{ "type": "error", "code": "quota_exceeded",          // voir §6
  "message": "Quota STT atteint." }

{ "type": "closed", "stt_seconds": 142 }              // conso de la session
```

Règles :
- `speaker` est un **champ structuré**, jamais collé au texte (la couleur vient
  de `benji.ui.style.speaker_color` côté client).
- `drop: true` sur `final_text` annule l'overlay partiel (hallucination/silence).
- Le client doit tolérer l'absence de `speaker` (diarisation off ou indispo).
- Un segment peut rendre **plusieurs** `final_text`, un par tour de parole,
  quand deux voix s'enchaînent sans pause (découpe par mot, cf.
  `backend/app/stt/turns.py`) — même comportement que le mode local.

## 4. Résumé — `POST /v1/summary` (SSE)

Streaming des tokens, équivalent réseau du `CloudSummaryProvider` actuel.

```jsonc
// Requête — Authorization: Bearer …
{
  "entries": [                          // mêmes dicts que l'historique local
    { "timestamp": "…", "text": "…", "speaker": "A" }
  ],
  "model": "haiku"                      // alias logique ; le serveur mappe le modèle
}
```

Réponse `Content-Type: text/event-stream` :

```
event: token
data: {"text": "Voici "}

event: token
data: {"text": "un résumé."}

event: done
data: {"summary_id": "sum_…"}

event: error
data: {"code": "upstream_error", "message": "…"}
```

> Le client ne choisit pas le modèle Anthropic exact (`claude-haiku-4-5`) — il
> envoie un **alias logique** (`haiku`/`sonnet`) que le backend résout. La clé
> Anthropic ne quitte jamais le serveur.

## 5. Historique — `GET /v1/history`

```jsonc
// GET /v1/history?limit=50&before=<cursor>
// 200
{
  "items": [
    { "id": "utt_…", "timestamp": "…", "text": "…", "speaker": "A" }
  ],
  "next_cursor": "…"        // null si fin
}
```

(Synchronisation multi-appareils ; le client desktop local peut continuer à
gérer son historique en local en mode 100 % local.)

## 6. Modèle d'erreur

REST : code HTTP + corps JSON normalisé.

```jsonc
{ "error": { "code": "quota_exceeded", "message": "Quota STT atteint." } }
```

WebSocket/SSE : event `error` avec le même `{code, message}`.

| `code`             | HTTP | Sens |
|--------------------|------|------|
| `unauthenticated`  | 401  | jeton absent/invalide/expiré |
| `forbidden`        | 403  | plan sans le droit (`entitlements`) |
| `quota_exceeded`   | 429  | quota STT/usage dépassé |
| `rate_limited`     | 429  | trop de requêtes |
| `bad_request`      | 400  | payload invalide |
| `upstream_error`   | 502  | échec provider (STT/Claude) |
| `internal`         | 500  | erreur serveur |

Codes de fermeture WebSocket : `1000` normal, `4401` non authentifié, `4403`
droit manquant, `4429` quota dépassé.

## 7. Métering & facturation

- Le serveur compte les **secondes d'audio STT** par session (`closed.stt_seconds`)
  et agrège dans `quota`. C'est le poste facturable dimensionnant.
- Le résumé est compté mais négligeable (coût ~0,008 $/résumé en Haiku).
- Dépassement de quota → `quota_exceeded` (429 / event / close 4429). Vérifié à
  l'ouverture **et en cours de session** : le segment en cours est finalisé, la
  conso comptée, puis la session fermée.
- La conso est comptée **quelle que soit l'issue** (arrêt, déconnexion, panne du
  provider). Une panne du provider en cours de session → `upstream_error` puis
  close `1011`.

### Stripe (abonnement Pro)

- `POST /v1/billing/checkout` (auth) → `{ "checkout_url" }`. Crée une Checkout
  Session Stripe en mode `subscription` (Price `STRIPE_PRICE_ID`), reliée à
  l'utilisateur via `client_reference_id`. Sans `STRIPE_SECRET_KEY` → repli stub.
- `POST /v1/billing/portal` (auth) → `{ "portal_url" }`. Billing Portal Stripe
  pour gérer/résilier (requiert un client Stripe lié, sinon `400`).
- `POST /v1/billing/webhook` → bascule de plan (`pro`/`free`) sur les events
  `checkout.session.completed`, `customer.subscription.*`. Signature `Stripe-
  Signature` vérifiée si `STRIPE_WEBHOOK_SECRET` défini, sinon `401`.
- Config : `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`, `STRIPE_WEBHOOK_SECRET`,
  `BILLING_SUCCESS_URL`, `BILLING_CANCEL_URL`, `BILLING_PORTAL_RETURN_URL`.

## 8. Décisions ouvertes

- **Encodage audio v1** : PCM `pcm_s16le`/16 kHz retenu pour la simplicité.
  Opus en option ultérieure (réduit la bande passante mobile/4G) — déclaré via
  `start.audio.encoding`.
- **Fournisseur STT** côté serveur (Deepgram en tête) et hébergement EU/RGPD
  (l'audio est personnel).
- **Diarisation** : déléguée au provider cloud, ou conservée côté serveur via le
  pipeline pyannote existant.
- **Auth** : email/mot de passe en v1 ; OAuth/Sign in with Apple à prévoir pour
  l'app iOS.
```
