"""Assistant de premier lancement : ce que Benji fait, le micro, les modèles.

Trois écrans, dans l'ordre où les questions se posent réellement : ce que fait
l'app et où vont les données, l'autorisation du micro, puis le téléchargement
des poids. Chaque étape est **bloquante mais explicable** — c'est tout l'écart
avec l'état précédent, où l'app demandait le micro sans contexte et téléchargeait
plus d'un gigaoctet derrière une fenêtre figée.

La logique (état de la permission, présence des poids, avancement) vit dans
`benji/onboarding.py`, sans Qt. Ici il n'y a que la mise en scène et le
marshalling vers le thread Qt : les rappels d'AVFoundation et du téléchargement
arrivent de threads système, jamais de celui de l'interface.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from benji import onboarding
from benji.ui.style import (
    FONT_DISPLAY,
    FONT_UI,
    current_theme,
    install_theme_listener,
    panel_background_qss,
    primary_button_qss,
    secondary_button_qss,
)

log = logging.getLogger(__name__)


def _rgba(color) -> str:
    return f"rgba({color.red()},{color.green()},{color.blue()},{color.alpha()})"


class OnboardingWindow(QDialog):
    """Renvoie `Accepted` quand l'utilisateur est allé au bout.

    `session` (cf. `benji.account.Session`) est injectée par `app.py`, qui
    l'a déjà construite avant l'assistant : sans elle, l'option payante reste
    affichée mais ne peut rien connecter (utile en test).
    """

    _mic_result = Signal(bool)
    _download_progress = Signal(float, str)
    _download_done = Signal(str)  # "" = succès

    def __init__(self, parent=None, session=None):
        super().__init__(parent)
        self.setWindowTitle("Bienvenue dans Benji")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setModal(True)
        self.setFixedSize(640, 540)

        self._downloader: onboarding.ModelDownloader | None = None
        self._mic_state = onboarding.microphone_status()
        self._session = session

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_welcome())
        self.pages.addWidget(self._build_offer())
        self.pages.addWidget(self._build_microphone())
        self.pages.addWidget(self._build_models())

        self.back_btn = QPushButton("Retour")
        self.back_btn.clicked.connect(self._back)
        self.next_btn = QPushButton("Commencer")
        self.next_btn.setDefault(True)
        self.next_btn.clicked.connect(self._next)

        footer = QHBoxLayout()
        footer.addWidget(self.back_btn)
        footer.addStretch(1)
        footer.addWidget(self.next_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(34, 30, 34, 24)
        root.setSpacing(18)
        root.addWidget(self.pages, 1)
        root.addLayout(footer)

        self._mic_result.connect(self._on_mic_result)
        self._download_progress.connect(self._on_download_progress)
        self._download_done.connect(self._on_download_done)

        install_theme_listener(self._apply_theme)
        self._apply_theme()
        self._refresh_microphone()
        self._refresh_models()
        self._refresh_nav()

    # --- écrans ---

    def _build_welcome(self) -> QWidget:
        page = QWidget()
        self.welcome_title = QLabel("Benji")
        self.welcome_lede = QLabel(
            "Des sous-titres de vos réunions, écrits sur votre Mac."
        )
        self.welcome_lede.setWordWrap(True)
        # Ce qui distingue Benji tient en une phrase, et c'est celle qui décide
        # de son usage en réunion : rien ne sort de la machine.
        self.welcome_body = QLabel(
            "Rien ne quitte cet ordinateur : ni votre voix, ni le texte. Chaque "
            "réunion se relit, s'exporte ou s'efface quand vous le voulez.\n\n"
            "Deux réglages, une seule fois : le micro, puis le moteur de "
            "transcription."
        )
        self.welcome_body.setWordWrap(True)

        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        layout.addWidget(self.welcome_title)
        layout.addWidget(self.welcome_lede)
        layout.addSpacing(4)
        layout.addWidget(self.welcome_body)
        layout.addSpacing(10)
        layout.addWidget(self._build_specimen())
        layout.addStretch(1)
        return page

    def _build_specimen(self) -> QWidget:
        """Un extrait de réunion, composé avec les vrais composants du direct.

        Plutôt que de décrire Benji en puces, on le montre : la feuille, la
        ligne de temps, les noms en colonne, la ligne qui s'écrit. C'est la
        page que l'utilisateur verra dans une minute.
        """
        from datetime import datetime

        from benji.ui.widgets.chat_item import ChatItem
        from benji.ui.widgets.partial_bubble import PartialBubble
        from benji.ui.widgets.sheet import Sheet

        sheet = Sheet(margins=(10, 12, 16, 10))
        at = datetime.now().replace(hour=14, minute=32)
        self._specimen_items = [
            ChatItem("On part sur le 21 pour la mise en production ?",
                     ts=at, speaker="A", name="Marie"),
            ChatItem("Le 21, avec un gel du code le 17 au soir.",
                     ts=at, speaker="B", name="Karim", show_ts=False),
        ]
        for item in self._specimen_items:
            sheet.body.addWidget(item)
        self._specimen_partial = PartialBubble()
        self._specimen_partial.set_text("Je préviens les clients pilotes")
        # Un spécimen ne s'anime pas : l'onde danserait sans que personne parle.
        self._specimen_partial.wave.set_active(False)
        sheet.body.addWidget(self._specimen_partial)
        self._specimen = sheet
        return sheet

    def _build_offer(self) -> QWidget:
        """Gratuit (local) et/ou payant (cloud), pas exclusifs l'un de l'autre.

        Cocher le payant ouvre la connexion tout de suite — c'est le seul
        moment où le choix a un sens : décider après coup, dans les
        Préférences, reviendrait à ne jamais savoir qu'un abonnement existe.
        `_persist_offer_choice()` (appelé à la fin de l'assistant) écrit
        `stt_provider` dans QSettings ; comme le reste des Moteurs, ça ne
        prend effet qu'au prochain lancement.
        """
        page = QWidget()
        self.offer_title = QLabel("Comment transcrire vos réunions ?")
        self.offer_body = QLabel(
            "Gratuit, en local : le moteur tourne sur cet ordinateur, rien ne "
            "sort de la machine.\n\n"
            "Payant, via le cloud Benji : une meilleure qualité, quel que "
            "soit l'ordinateur. Les deux peuvent être actifs à la fois."
        )
        self.offer_body.setWordWrap(True)

        self.offer_free = QCheckBox("Gratuit — transcription locale")
        self.offer_free.setChecked(True)
        self.offer_cloud = QCheckBox("Payant — cloud Benji, abonnement")
        self.offer_cloud.setChecked(
            self._session is not None and self._session.is_authenticated
        )
        self.offer_cloud.toggled.connect(self._on_offer_cloud_toggled)

        self.offer_status = QLabel("")
        self.offer_status.setWordWrap(True)
        self.offer_warning = QLabel("")
        self.offer_warning.setWordWrap(True)
        self.offer_warning.hide()

        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        layout.addWidget(self.offer_title)
        layout.addWidget(self.offer_body)
        layout.addSpacing(6)
        layout.addWidget(self.offer_free)
        layout.addWidget(self.offer_cloud)
        layout.addWidget(self.offer_status)
        layout.addWidget(self.offer_warning)
        layout.addStretch(1)
        self._refresh_offer_status()
        return page

    def _on_offer_cloud_toggled(self, checked: bool) -> None:
        already_in = self._session is not None and self._session.is_authenticated
        if checked and self._session is not None and not already_in:
            from benji.ui.login_dialog import LoginDialog

            if not LoginDialog(self._session, parent=self).exec():
                self.offer_cloud.setChecked(False)
        self._refresh_offer_status()

    def _refresh_offer_status(self) -> None:
        if self._session is not None and self._session.is_authenticated:
            self.offer_status.setText(f"Connecté : {self._session.email}")
        else:
            self.offer_status.setText("")
        if self.offer_free.isChecked() or self.offer_cloud.isChecked():
            self.offer_warning.hide()

    def _persist_offer_choice(self) -> None:
        """`stt_provider`, comme le reste des Moteurs : effet au redémarrage.

        Priorité au cloud si les deux sont cochés — un abonné s'attend à la
        meilleure qualité par défaut ; le local reste sélectionnable dans les
        Préférences.
        """
        from benji.settings import UserSettings

        connected = self._session is not None and self._session.is_authenticated
        provider = "remote" if (self.offer_cloud.isChecked() and connected) else "parakeet"
        UserSettings().set_value("stt_provider", provider)

    def _build_microphone(self) -> QWidget:
        page = QWidget()
        self.mic_title = QLabel("Le micro")
        self.mic_body = QLabel(
            "Benji a besoin du microphone pour transcrire. macOS vous demandera "
            "votre accord — c'est le système qui décide, pas Benji.\n\n"
            "L'audio n'est jamais enregistré sur le disque : il traverse la "
            "mémoire, devient du texte, et disparaît."
        )
        self.mic_body.setWordWrap(True)
        self.mic_status = QLabel("")
        self.mic_status.setWordWrap(True)

        self.mic_btn = QPushButton("Autoriser le micro")
        self.mic_btn.clicked.connect(self._request_microphone)

        row = QHBoxLayout()
        row.addWidget(self.mic_btn)
        row.addStretch(1)

        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        layout.addWidget(self.mic_title)
        layout.addWidget(self.mic_body)
        layout.addSpacing(6)
        layout.addLayout(row)
        layout.addWidget(self.mic_status)
        layout.addStretch(1)
        return page

    def _build_models(self) -> QWidget:
        page = QWidget()
        self.models_title = QLabel("Le moteur de transcription")
        self.models_body = QLabel("")
        self.models_body.setWordWrap(True)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)

        self.download_btn = QPushButton("Télécharger")
        self.download_btn.clicked.connect(self._start_download)

        row = QHBoxLayout()
        row.addWidget(self.download_btn)
        row.addStretch(1)

        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        layout.addWidget(self.models_title)
        layout.addWidget(self.models_body)
        layout.addSpacing(6)
        layout.addLayout(row)
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_label)
        layout.addStretch(1)
        return page

    # --- micro ---

    def _request_microphone(self) -> None:
        if self._mic_state == onboarding.DENIED:
            # macOS ne redemande jamais après un refus : le seul chemin de
            # réparation passe par les Réglages Système.
            QDesktopServices.openUrl(QUrl(onboarding.open_privacy_settings()))
            return
        if not onboarding.request_microphone_access(self._mic_result.emit):
            self._mic_state = onboarding.UNKNOWN
            self._refresh_microphone()

    def _on_mic_result(self, granted: bool) -> None:
        self._mic_state = onboarding.GRANTED if granted else onboarding.DENIED
        self._refresh_microphone()
        self._refresh_nav()

    def _refresh_microphone(self) -> None:
        state = self._mic_state
        if state == onboarding.GRANTED:
            self.mic_status.setText("Micro autorisé.")
            self.mic_btn.setEnabled(False)
        elif state == onboarding.DENIED:
            self.mic_status.setText(
                "Accès refusé. macOS ne redemandera pas : il faut cocher Benji "
                "dans Réglages Système › Confidentialité et sécurité › Microphone, "
                "puis relancer l'application."
            )
            self.mic_btn.setText("Ouvrir les Réglages Système")
            self.mic_btn.setEnabled(True)
        elif state == onboarding.UNKNOWN:
            self.mic_status.setText(
                "Impossible de connaître l'état de l'autorisation sur ce système. "
                "macOS vous la demandera au démarrage de la transcription."
            )
            self.mic_btn.setEnabled(False)
        else:
            self.mic_status.setText("")
            self.mic_btn.setEnabled(True)

    # --- modèles ---

    def _refresh_models(self) -> None:
        missing = onboarding.missing_models()
        if not missing:
            self.models_body.setText(
                "Le moteur est déjà sur votre Mac. Rien à télécharger."
            )
            self.download_btn.hide()
            self.progress_label.setText("")
            return
        total = sum(size for _repo, _label, size in missing)
        lines = "\n".join(
            f"• {label} — {onboarding.format_size(size)}"
            for _repo, label, size in missing
        )
        self.models_body.setText(
            f"{lines}\n\nSoit environ {onboarding.format_size(total)} à télécharger "
            "une seule fois. Les fichiers restent dans votre cache : ils ne seront "
            "pas retéléchargés au prochain lancement."
        )
        self.download_btn.show()

    def _start_download(self) -> None:
        if self._downloader is not None:
            return
        self.download_btn.setEnabled(False)
        self.download_btn.setText("Téléchargement…")
        self.progress.show()
        self.progress.setValue(0)
        self.progress_label.setText("Connexion…")
        self._downloader = onboarding.ModelDownloader(
            on_progress=lambda frac, text: self._download_progress.emit(frac, text),
            on_done=lambda err: self._download_done.emit(err or ""),
        )
        self._downloader.start()
        self._refresh_nav()

    def _on_download_progress(self, fraction: float, text: str) -> None:
        self.progress.setValue(int(fraction * 1000))
        self.progress_label.setText(text)

    def _on_download_done(self, error: str) -> None:
        self._downloader = None
        self.progress.hide()
        if error:
            # Un échec ne condamne pas le premier lancement : les poids seront
            # retéléchargés au besoin, et l'utilisateur peut être hors ligne.
            self.progress_label.setText(
                f"Téléchargement interrompu ({error}). Benji réessaiera au "
                "démarrage — vérifiez votre connexion."
            )
            self.download_btn.setEnabled(True)
            self.download_btn.setText("Réessayer")
        else:
            self.progress_label.setText("Moteur prêt.")
            self.download_btn.hide()
        self._refresh_nav()

    # --- navigation ---

    def _back(self) -> None:
        self.pages.setCurrentIndex(max(0, self.pages.currentIndex() - 1))
        self._refresh_nav()

    def _next(self) -> None:
        index = self.pages.currentIndex()
        if index == 1 and not self.offer_free.isChecked() and not self.offer_cloud.isChecked():
            # Aucune des deux offres n'est active : Benji ne transcrirait rien.
            self.offer_warning.setText(
                "Choisissez au moins une option pour pouvoir transcrire."
            )
            self.offer_warning.show()
            return
        if index >= self.pages.count() - 1:
            onboarding.mark_done(microphone=self._mic_state)
            self._persist_offer_choice()
            self.accept()
            return
        self.pages.setCurrentIndex(index + 1)
        self._refresh_nav()

    def _refresh_nav(self) -> None:
        index = self.pages.currentIndex()
        last = index == self.pages.count() - 1
        self.back_btn.setVisible(index > 0)
        # On ne bloque que **pendant** un téléchargement : quitter l'assistant à
        # ce moment-là laisserait un cache à moitié écrit. Terminer sans avoir
        # téléchargé reste permis — un utilisateur hors ligne doit pouvoir aller
        # au bout, l'app reprendra les poids au démarrage suivant.
        downloading = self._downloader is not None
        self.next_btn.setEnabled(not downloading)
        self.next_btn.setText("Terminer" if last else "Continuer")
        if index == 0:
            self.next_btn.setText("Commencer")

    def reject(self) -> None:
        """Fermer la fenêtre = renoncer, pas « passer ».

        L'appelant (`benji/app.py`) en conclut qu'il faut quitter : démarrer sans
        micro ni modèle donnerait une app qui ne transcrit rien et n'explique
        pas pourquoi.
        """
        if self._downloader is not None:
            self._downloader.cancel()
        super().reject()

    # --- thème ---

    def _apply_theme(self) -> None:
        t = current_theme()
        self.setStyleSheet(
            panel_background_qss(t, "#OnboardingWindow")
            + f"""
            QDialog {{ background-color: {_rgba(t.paper)}; }}
            QLabel {{ background: transparent; }}
            QProgressBar {{
                background-color: {_rgba(t.ink_alpha(8))};
                border: none; border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {_rgba(t.ink_alpha(70))}; border-radius: 3px;
            }}
            """
        )
        title_qss = (
            f"font-family: {FONT_DISPLAY}; font-size: 26px; font-weight: 600; "
            f"letter-spacing: -0.4px; color: {_rgba(t.ink)}; background: transparent;"
        )
        for label in (self.welcome_title, self.offer_title, self.mic_title, self.models_title):
            label.setStyleSheet(title_qss)
        body_qss = (
            f"font-family: {FONT_UI}; font-size: 13px; color: {_rgba(t.ink_muted)}; "
            "background: transparent;"
        )
        for label in (self.welcome_body, self.offer_body, self.mic_body, self.models_body):
            label.setStyleSheet(body_qss)
        checkbox_qss = (
            f"font-family: {FONT_UI}; font-size: 13px; color: {_rgba(t.ink)}; "
            "background: transparent;"
        )
        for box in (self.offer_free, self.offer_cloud):
            box.setStyleSheet(checkbox_qss)
        self.welcome_lede.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 15px; color: {_rgba(t.ink)}; "
            "background: transparent;"
        )
        # Pas `meta_qss` : il compose en mono, réservé au temps. Un message
        # d'état n'est pas un horodatage.
        status_qss = (
            f"font-family: {FONT_UI}; font-size: 12px; color: {_rgba(t.ink_faint)}; "
            "background: transparent;"
        )
        self.mic_status.setStyleSheet(status_qss)
        self.progress_label.setStyleSheet(status_qss)
        self.offer_status.setStyleSheet(status_qss)
        self.offer_warning.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 12px; color: {_rgba(t.record)}; "
            "background: transparent;"
        )
        self.next_btn.setStyleSheet(primary_button_qss(t))
        for btn in (self.back_btn, self.mic_btn, self.download_btn):
            btn.setStyleSheet(secondary_button_qss(t))
