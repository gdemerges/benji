"""Panneau Préférences : réglages transcription + affichage, persistés.

Modal, sur le thread Qt. Chaque changement validé est écrit dans `UserSettings`
(QSettings) et appliqué à la config vivante. Les réglages d'affichage sont
poussés à chaud sur l'overlay via `on_live_change` ; ceux de transcription ne
prennent effet qu'au prochain lancement — un bandeau le signale.

Le style suit la fenêtre principale (`benji.ui.style`) : un aplat, des sections
séparées par un filet plutôt qu'encadrées, titrées en casse normale — les cadres
de groupe à titre en capitales étaient le look Qt par défaut, celui d'un panneau
de réglages générique. Bouton d'enregistrement en aplat d'encre. Se recharge au
changement de thème système.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFontComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from benji.settings import UserSettings
from benji.stt import lexicon
from benji.ui.style import FONT_UI, current_theme, install_theme_listener

# (code langue ISO, libellé). "" = détection automatique.
_LANGUAGES = [
    ("", "Détection automatique"),
    ("fr", "Français"),
    ("en", "English"),
    ("es", "Español"),
    ("de", "Deutsch"),
    ("it", "Italiano"),
    ("pt", "Português"),
    ("nl", "Nederlands"),
]


# (secondes, libellé). 0 = désactivé.
_SUMMARY_INTERVALS = [
    (0, "Désactivé"),
    (300, "Toutes les 5 min"),
    (600, "Toutes les 10 min"),
    (900, "Toutes les 15 min"),
]

# (valeur config, libellé). "remote" = backend Benji, débloqué par l'abonnement
# Pro. Le mode "cloud" (clé Anthropic sur le poste) est réservé au dev : il
# n'apparaît que s'il est déjà actif dans la config.
_PROVIDERS = [
    ("local", "Local — sur ce Mac"),
    ("remote", "Cloud Benji — abonnement Pro"),
]

# Le STT et le résumé n'offrent pas les mêmes choix : le résumé a un mode
# « cloud » de développement que la transcription n'a pas.
_STT_PROVIDERS = [
    ("parakeet", "Local — sur ce Mac"),
    ("remote", "Cloud Benji — abonnement Pro"),
]


# Famille système macOS (privée, préfixée « . ») : QFontComboBox ne sait pas
# l'afficher et retombe sur une entrée arbitraire de la liste. On la représente
# par « Helvetica Neue », la police native la plus proche réellement listée.
_SYSTEM_FONT_DISPLAY = "Helvetica Neue"


def _query_input_devices() -> list[dict]:
    """Énumère les périphériques d'entrée. Isolé pour être remplaçable en test
    et pour qu'un échec de PortAudio n'empêche pas d'ouvrir les Préférences."""
    try:
        import sounddevice as sd

        return list(sd.query_devices())
    except Exception:
        return []


def rgba_hex(c) -> str:
    return f"#{c.red():02x}{c.green():02x}{c.blue():02x}"


def _resolve_font(family: str) -> QFont:
    """QFont affichable dans le combo pour une famille de config donnée."""
    if not family or family.startswith("."):
        return QFont(_SYSTEM_FONT_DISPLAY)
    return QFont(family)


class PreferencesDialog(QDialog):
    def __init__(
        self,
        stt_config,
        ui_config,
        settings: UserSettings,
        on_live_change: Callable[[object], None] | None = None,
        llm_config=None,
        audio_config=None,
        device_lister=None,
        parent=None,
    ):
        """on_live_change: reçoit une UIConfig mise à jour pour application à chaud
        (typiquement `overlay.apply_config`). llm_config: LLMConfig — si fournie,
        expose le choix des moteurs (transcription/résumé local vs cloud Benji).
        audio_config: AudioConfig — si fournie, expose la capture de l'audio
        système. device_lister: callable() -> list[dict] au format
        `sounddevice.query_devices()` ; injectable pour tester sans matériel."""
        super().__init__(parent)
        self._stt = stt_config
        self._ui = ui_config
        self._llm = llm_config
        self._audio = audio_config
        self._device_lister = device_lister or _query_input_devices
        self._settings = settings
        self._on_live_change = on_live_change
        self.setWindowTitle("Préférences Benji")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._build_ui()
        install_theme_listener(self._apply_theme)
        self._apply_theme()
        # Après le thème : c'est la feuille de style qui fixe le corps des
        # étiquettes. Mesurées avant, elles l'étaient dans la fonte par défaut,
        # plus petite, et la plus longue se faisait rogner (« ésumé en direct »).
        self._align_forms()
        self._fit_to_screen()

    def _build_ui(self) -> None:
        # Les sections défilent, les boutons restent : sur un écran de portable,
        # la fenêtre dépassait le bas de l'écran et « Enregistrer » avec elle —
        # sans défilement, la fin des réglages était tout simplement inatteignable.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._content = QWidget()
        self._content.setObjectName("prefs_content")
        layout = QVBoxLayout(self._content)
        layout.setContentsMargins(28, 12, 28, 12)
        layout.setSpacing(6)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("prefs_scroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll, 1)

        # === Transcription (redémarrage requis) ===
        self._stt_box = QGroupBox("Transcription")
        stt_form = QFormLayout(self._stt_box)
        # Première section : pas de filet au-dessus, rien à séparer.
        self._stt_box.setObjectName("first_section")
        self._forms = [stt_form]
        stt_form.setContentsMargins(16, 18, 16, 16)
        stt_form.setSpacing(12)
        stt_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._language = QComboBox()
        for code, label in _LANGUAGES:
            self._language.addItem(label, code)
        self._select_data(self._language, self._stt.language or "")
        stt_form.addRow("Langue", self._language)


        self._diarization = QCheckBox("Identifier les locuteurs")
        self._diarization.setChecked(bool(self._stt.diarization))
        stt_form.addRow("Diarisation", self._diarization)

        self._confirm_saving = QCheckBox("Demander avant de conserver une réunion")
        self._confirm_saving.setChecked(bool(self._stt.confirm_before_saving))
        self._confirm_saving.setToolTip(
            "Benji transcrit dès le lancement, mais n'écrit rien sur disque tant "
            "que vous ne l'avez pas accordé. Ce qui a déjà été dit est conservé "
            "au moment de l'accord, pas perdu."
        )
        stt_form.addRow("Conservation", self._confirm_saving)

        self._summary = QComboBox()
        for secs, label in _SUMMARY_INTERVALS:
            self._summary.addItem(label, secs)
        self._select_data(self._summary, self._stt.live_summary_interval_s)
        stt_form.addRow("Résumé en direct", self._summary)

        self._hint = QLabel("Ces réglages prennent effet au prochain démarrage.")
        stt_form.addRow(self._hint)
        layout.addWidget(self._stt_box)

        # === Glossaire (redémarrage requis) ===
        # Le point faible d'une transcription de réunion n'est pas la grammaire,
        # ce sont les noms propres. Le moteur n'accepte aucun prompt : on relit
        # donc sa sortie (cf. benji/stt/lexicon.py).
        self._glossary_box = QGroupBox("Glossaire")
        glossary_layout = QVBoxLayout(self._glossary_box)
        glossary_layout.setContentsMargins(16, 18, 16, 16)
        glossary_layout.setSpacing(8)

        self._glossary = QPlainTextEdit()
        self._glossary.setPlaceholderText("Kubernetes\nCrédit Agricole\nDatadog")
        self._glossary.setPlainText(lexicon.read_raw())
        self._glossary.setFixedHeight(96)
        glossary_layout.addWidget(self._glossary)

        self._hint_glossary = QLabel(
            "Un nom propre ou un terme maison par ligne. Benji remplacera ce qui "
            "leur ressemble à l'oreille dans le texte final. Ce fichier ne quitte "
            "jamais votre Mac. Effet au prochain démarrage."
        )
        self._hint_glossary.setWordWrap(True)
        glossary_layout.addWidget(self._hint_glossary)
        layout.addWidget(self._glossary_box)

        # === Moteurs local / cloud Benji (redémarrage requis) ===
        self._engine_box = None
        self._hint_engines = None
        if self._llm is not None:
            self._engine_box = QGroupBox("Moteurs")
            engine_form = QFormLayout(self._engine_box)
            self._forms.append(engine_form)
            engine_form.setContentsMargins(16, 18, 16, 16)
            engine_form.setSpacing(12)
            engine_form.setLabelAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )

            self._stt_provider = self._provider_combo(
                self._stt.stt_provider, choices=_STT_PROVIDERS
            )
            engine_form.addRow("Transcription", self._stt_provider)

            self._summary_provider = self._provider_combo(self._llm.summary_provider)
            engine_form.addRow("Résumé", self._summary_provider)

            self._hint_engines = QLabel(
                "Le cloud Benji nécessite un compte avec abonnement Pro. "
                "Effet au prochain démarrage."
            )
            engine_form.addRow(self._hint_engines)
            layout.addWidget(self._engine_box)

        # === Audio système (redémarrage requis) ===
        # Sans ça Benji ne transcrit que le micro — donc, en visio, seulement
        # l'utilisateur. macOS n'expose aucune API publique pour capter la
        # sortie audio : il faut un pilote de boucle, d'où l'assistant ci-dessous.
        self._audio_box = None
        self._hint_audio = None
        if self._audio is not None:
            self._audio_box = QGroupBox("Audio des visios")
            audio_form = QFormLayout(self._audio_box)
            self._forms.append(audio_form)
            audio_form.setContentsMargins(16, 18, 16, 16)
            audio_form.setSpacing(12)
            audio_form.setLabelAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )

            self._system_audio = QCheckBox("Capter le son des visios")
            self._system_audio.setChecked(bool(self._audio.system_audio))
            audio_form.addRow("Réunions", self._system_audio)

            self._system_device = QComboBox()
            audio_form.addRow("Périphérique", self._system_device)

            self._hint_audio = QLabel()
            self._hint_audio.setWordWrap(True)
            audio_form.addRow(self._hint_audio)

            self._refresh_devices()
            self._system_audio.toggled.connect(self._on_system_audio_toggled)
            self._on_system_audio_toggled(self._system_audio.isChecked())
            layout.addWidget(self._audio_box)

        # === Affichage (application immédiate) ===
        self._ui_box = QGroupBox("Sous-titres")
        ui_form = QFormLayout(self._ui_box)
        self._forms.append(ui_form)
        ui_form.setContentsMargins(16, 18, 16, 16)
        ui_form.setSpacing(12)
        ui_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._font = QFontComboBox()
        self._font.setCurrentFont(_resolve_font(self._ui.font_family))
        ui_form.addRow("Police", self._font)

        self._font_size = QSpinBox()
        self._font_size.setRange(10, 96)
        self._font_size.setSuffix(" px")
        self._font_size.setValue(int(self._ui.font_size))
        ui_form.addRow("Taille", self._font_size)

        self._opacity = QSpinBox()
        self._opacity.setRange(0, 255)
        self._opacity.setValue(int(self._ui.bg_opacity))
        ui_form.addRow("Opacité du fond", self._opacity)

        self._duration = QSpinBox()
        self._duration.setRange(1, 60)
        self._duration.setSuffix(" s")
        self._duration.setValue(round(int(self._ui.display_duration_ms) / 1000))
        ui_form.addRow("Durée d'affichage", self._duration)

        layout.addWidget(self._ui_box)
        layout.addStretch(1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self._save_btn = self._buttons.button(QDialogButtonBox.StandardButton.Save)
        self._cancel_btn = self._buttons.button(QDialogButtonBox.StandardButton.Cancel)
        self._save_btn.setText("Enregistrer")
        self._cancel_btn.setText("Annuler")
        self._save_btn.setObjectName("accent_btn")
        self._cancel_btn.setObjectName("ghost_btn")
        self._buttons.accepted.connect(self._save)
        self._buttons.rejected.connect(self.reject)

        self._footer = QWidget()
        self._footer.setObjectName("prefs_footer")
        footer_layout = QVBoxLayout(self._footer)
        footer_layout.setContentsMargins(28, 12, 28, 16)
        footer_layout.addWidget(self._buttons)
        outer.addWidget(self._footer)

    def _fit_to_screen(self) -> None:
        """Assez haute pour tout montrer si l'écran le permet, jamais plus haute
        que lui : au-delà, c'est la zone défilante qui prend le relais."""
        wanted = self._content.sizeHint().height() + self._footer.sizeHint().height()
        screen = self.screen() or QGuiApplication.primaryScreen()
        limit = int(screen.availableGeometry().height() * 0.85) if screen else wanted
        # Assez large pour le contenu *et* la barre de défilement : plus étroite,
        # la zone rognait la colonne des étiquettes (« ésumé en direct »).
        bar = self._scroll.verticalScrollBar().sizeHint().width()
        width = max(self.minimumWidth(), self._content.minimumSizeHint().width() + bar)
        self.setMinimumWidth(width)
        self.resize(width, min(wanted, limit))

    def _align_forms(self) -> None:
        """Une seule colonne d'étiquettes pour toutes les sections.

        Chaque `QFormLayout` dimensionne ses étiquettes pour lui seul : d'une
        section à l'autre, les champs partaient de deux abscisses différentes.
        Et sur macOS le formulaire entier est **centré** par défaut quand ses
        champs ne s'étirent pas : un formulaire plus étroit glissait à droite.
        """
        labels = []
        for form in self._forms:
            form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if item is not None and isinstance(item.widget(), QLabel):
                    item.widget().ensurePolished()
                    labels.append(item.widget())
        if not labels:
            return
        width = max(label.sizeHint().width() for label in labels)
        for label in labels:
            label.setFixedWidth(width)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def _refresh_devices(self) -> None:
        """Peuple la liste des boucles détectées et rédige le message d'aide.

        Trois états possibles, chacun avec une consigne actionnable : aucun
        pilote (installer), seulement un périphérique appartenant à une app
        (avertir qu'il ne capte que cette app), pilote fiable (rappeler le
        routage, qui est l'étape que tout le monde oublie).
        """
        from benji.audio.loopback import find_loopback_devices

        try:
            devices = find_loopback_devices(list(self._device_lister()))
        except Exception:
            devices = []

        self._system_device.clear()
        self._system_device.addItem("Détection automatique", None)
        for device in devices:
            label = device.name if device.is_reliable else f"{device.name} (cette app seulement)"
            self._system_device.addItem(label, device.name)
        self._select_data(self._system_device, self._audio.system_audio_device)

        reliable = [d for d in devices if d.is_reliable]
        if reliable:
            self._hint_audio.setText(
                f"Détecté : {reliable[0].name}. Dans Réglages Son, choisissez un "
                "périphérique multi-sortie combinant vos enceintes et cette boucle — "
                "sinon vous n'entendrez plus rien. Effet au prochain démarrage."
            )
        elif devices:
            self._hint_audio.setText(
                "Seul un périphérique appartenant à une application a été trouvé : "
                "il ne captera que celle-ci. Installez BlackHole (gratuit) pour "
                "capter tout le son du Mac."
            )
        else:
            self._hint_audio.setText(
                "Aucun pilote de boucle détecté. macOS ne permet pas de capter le son "
                "sortant sans en installer un : installez BlackHole (gratuit, "
                "existential.audio/blackhole), puis créez un périphérique multi-sortie "
                "dans Configuration audio et MIDI."
            )

    def _on_system_audio_toggled(self, checked: bool) -> None:
        self._system_device.setEnabled(checked)

    def _apply_theme(self) -> None:
        t = current_theme()
        label = t.label
        sec = t.secondary_label
        # Le bouton d'enregistrement est un aplat d'encre : dans Benji le rouge
        # ne dit qu'une chose, « on enregistre » — pas « valider ».
        on_ink = rgba_hex(t.sheet if not t.is_dark else t.paper)
        field_bg = t.card
        field_border = t.spine

        def rgba(c):
            return f"rgba({c.red()},{c.green()},{c.blue()},{c.alpha()})"

        self.setStyleSheet(f"""
            QDialog {{ background-color: {rgba(t.sheet)}; }}
            QScrollArea#prefs_scroll, QWidget#prefs_content {{ background: transparent; }}
            QWidget#prefs_footer {{
                background: transparent;
                border-top: 1px solid {rgba(t.spine)};
            }}
            QGroupBox {{
                font-family: {FONT_UI};
                font-size: 13px;
                font-weight: 600;
                color: {rgba(label)};
                border: none;
                border-top: 1px solid {rgba(t.spine)};
                margin-top: 14px;
                padding-top: 22px;
                background: transparent;
            }}
            QGroupBox#first_section {{ border-top: none; margin-top: 0px; }}
            QGroupBox::title {{
                subcontrol-origin: padding;
                subcontrol-position: top left;
                left: 0px;
                top: 8px;
            }}
            QLabel {{
                font-family: {FONT_UI};
                font-size: 13px;
                color: {rgba(sec)};
                background: transparent;
            }}
            QComboBox, QFontComboBox, QSpinBox {{
                font-family: {FONT_UI};
                font-size: 13px;
                color: {rgba(label)};
                background-color: {rgba(field_bg)};
                border: 1px solid {rgba(field_border)};
                border-radius: 7px;
                padding: 4px 8px;
                min-height: 22px;
            }}
            QComboBox:hover, QFontComboBox:hover, QSpinBox:hover {{
                border-color: {rgba(t.ink_alpha(30))};
            }}
            QComboBox::drop-down, QFontComboBox::drop-down {{ border: none; width: 18px; }}
            QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; border: none; }}
            QCheckBox {{
                font-family: {FONT_UI};
                font-size: 13px;
                color: {rgba(label)};
                spacing: 8px;
                background: transparent;
            }}
            QPushButton#accent_btn {{
                font-family: {FONT_UI};
                font-size: 13px;
                font-weight: 600;
                color: {on_ink};
                background-color: {rgba(t.ink_alpha(92) if not t.is_dark else t.ink)};
                border: none;
                padding: 7px 18px;
                border-radius: 7px;
            }}
            QPushButton#accent_btn:hover {{ background-color: {rgba(t.ink)}; }}
            QPushButton#ghost_btn {{
                font-family: {FONT_UI};
                font-size: 13px;
                font-weight: 500;
                color: {rgba(sec)};
                background: transparent;
                border: none;
                padding: 7px 14px;
                border-radius: 7px;
            }}
            QPushButton#ghost_btn:hover {{
                color: {rgba(label)};
                background-color: {rgba(t.ink_alpha(7))};
            }}
        """)
        # Le bandeau d'info en couleur secondaire (le QSS QLabel ci-dessus cible
        # aussi ce label, on le repasse en tertiaire ici pour le distinguer).
        hint_qss = (
            f"font-family: {FONT_UI}; font-size: 12px; color: {rgba(t.ink_faint)}; "
            "background: transparent;"
        )
        self._hint.setStyleSheet(hint_qss)
        self._hint_glossary.setStyleSheet(hint_qss)
        self._glossary.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 12px; color: {rgba(label)}; "
            f"background-color: {rgba(field_bg)}; border: 1px solid {rgba(field_border)}; "
            "border-radius: 7px; padding: 6px;"
        )
        if self._hint_engines is not None:
            self._hint_engines.setStyleSheet(hint_qss)
        if self._hint_audio is not None:
            self._hint_audio.setStyleSheet(hint_qss)

    @staticmethod
    def _select_data(combo: QComboBox, data) -> None:
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    @staticmethod
    def _provider_combo(current: str, choices=None) -> QComboBox:
        """Combo de choix de moteur ; une valeur hors liste (ex. « cloud » en dev)
        est ajoutée telle quelle pour ne pas être écrasée silencieusement."""
        combo = QComboBox()
        for value, label in (choices or _PROVIDERS):
            combo.addItem(label, value)
        if combo.findData(current) < 0:
            combo.addItem(current, current)
        PreferencesDialog._select_data(combo, current)
        return combo

    def _save(self) -> None:
        s = self._settings

        # --- Transcription : persister + mettre à jour la config (effet au reboot) ---
        language = self._language.currentData() or None
        diarization = self._diarization.isChecked()
        summary_interval = self._summary.currentData()

        s.set_value("language", language)
        s.set_value("diarization", diarization)
        s.set_value("live_summary_interval_s", summary_interval)
        confirm_saving = self._confirm_saving.isChecked()
        s.set_value("confirm_before_saving", confirm_saving)
        self._stt.confirm_before_saving = confirm_saving
        # Le glossaire n'est pas une préférence QSettings : c'est un fichier de
        # données utilisateur, écrit en 0600 comme l'historique.
        lexicon.save_glossary(self._glossary.toPlainText())
        self._stt.language = language
        self._stt.diarization = diarization
        self._stt.live_summary_interval_s = summary_interval

        # --- Moteurs : persister + mettre à jour la config (effet au reboot) ---
        if self._llm is not None:
            stt_provider = self._stt_provider.currentData()
            summary_provider = self._summary_provider.currentData()
            s.set_value("stt_provider", stt_provider)
            s.set_value("summary_provider", summary_provider)
            self._stt.stt_provider = stt_provider
            self._llm.summary_provider = summary_provider

        # --- Audio système : persister + mettre à jour la config (effet au reboot) ---
        if self._audio is not None:
            system_audio = self._system_audio.isChecked()
            system_device = self._system_device.currentData()
            s.set_value("system_audio", system_audio)
            s.set_value("system_audio_device", system_device)
            self._audio.system_audio = system_audio
            self._audio.system_audio_device = system_device

        # --- Affichage : persister + appliquer à chaud ---
        font_family = self._font.currentFont().family()
        font_size = self._font_size.value()
        bg_opacity = self._opacity.value()
        display_ms = self._duration.value() * 1000

        s.set_value("font_family", font_family)
        s.set_value("font_size", font_size)
        s.set_value("bg_opacity", bg_opacity)
        s.set_value("display_duration_ms", display_ms)
        self._ui.font_family = font_family
        self._ui.font_size = font_size
        self._ui.bg_opacity = bg_opacity
        self._ui.display_duration_ms = display_ms

        if self._on_live_change is not None:
            # Copie figée pour que l'overlay reçoive un snapshot cohérent.
            self._on_live_change(replace(self._ui))

        self.accept()
