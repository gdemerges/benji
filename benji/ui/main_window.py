"""Fenêtre principale : toolbar + onglets Live/Résumés (style macOS natif)."""

from __future__ import annotations

import logging
import platform
import uuid
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from benji.ui.account_controller import AccountController
from benji.ui.live_tab import LiveTab
from benji.ui.style import (
    apply_window_vibrancy,
    current_theme,
    install_theme_listener,
    panel_background_qss,
    primary_button_qss,
    secondary_button_qss,
)
from benji.ui.summaries_tab import SummariesTab
from benji.ui.widgets.icons import (
    mic_icon,
    mic_slash_icon,
    minimize_icon,
    person_icon,
    sliders_icon,
)
from benji.ui.widgets.segmented_control import SegmentedControl
from benji.ui.widgets.sheet import Sheet
from benji.ui.widgets.status_pill import StatusPill

log = logging.getLogger(__name__)

_SETTINGS_ORG = "benji"
_SETTINGS_APP = "benji"
_GEOM_KEY = "main_window/geometry"
_TAB_KEY = "main_window/tab_index"


class MainWindow(QMainWindow):
    def __init__(
        self,
        bus,
        history,
        session_start: datetime,
        summary_worker,
        on_minimize=None,
        on_open_preferences=None,
        on_toggle_pause=None,
        on_save_meeting=None,
        on_learn_term=None,
        session=None,
        backend_url: str = "",
        parent=None,
    ):
        """on_toggle_pause: callable() -> bool — bascule la pause micro côté app
        et retourne le nouvel état (True = en pause)."""
        super().__init__(parent)
        self.setWindowTitle("Benji")
        self._bus = bus
        self._history = history
        self._session_start = session_start
        self._worker = summary_worker
        self._on_minimize = on_minimize
        self._on_open_preferences = on_open_preferences
        self._on_toggle_pause = on_toggle_pause
        # Accord de conservation (cf. benji/recording.py) : None = pas de
        # portillon, tout est conservé d'office et le bandeau ne s'affiche pas.
        self._on_save_meeting = on_save_meeting
        # Apprentissage du glossaire depuis le transcript (moteur local seul).
        self._on_learn_term = on_learn_term
        self._paused = False
        # Contrôleur compte/facturation (login + abonnement Stripe) — présent
        # seulement si une session est fournie. Succès/erreurs via QMessageBox.
        self._account = None
        if session is not None:
            self._account = AccountController(session, backend_url, self._notify_account, parent=self)
            self._account.failed.connect(
                lambda msg: QMessageBox.warning(self, "Benji", f"Action impossible : {msg}")
            )
        self._pending_summary_id: str | None = None
        self._has_unread_summary = False
        self._vibrancy_applied = False
        self._vibrancy_active = False

        self._build_ui()
        self._wire_worker()
        self._restore_state()

        if platform.system() == "Darwin":
            self.setUnifiedTitleAndToolBarOnMac(True)

        install_theme_listener(self._apply_theme)
        self._apply_theme()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._vibrancy_applied:
            self._vibrancy_active = apply_window_vibrancy(self)
            self._vibrancy_applied = True
            if self._vibrancy_active:
                # Fond transparent pour laisser voir le flou natif derrière.
                self._apply_theme()

    def _build_ui(self) -> None:
        # === Toolbar ===
        tb = QToolBar("main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.status_pill = StatusPill(self._session_start, title_provider=self._meeting_title)
        tb.addWidget(self.status_pill)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        # Pause micro — coupe réellement la capture (indicateur macOS éteint).
        self.pause_btn = QPushButton()
        self.pause_btn.setObjectName("icon_btn")
        self.pause_btn.setToolTip("Suspendre le micro")
        self.pause_btn.clicked.connect(self._toggle_pause)
        if self._on_toggle_pause is not None:
            tb.addWidget(self.pause_btn)

        # Compte / abonnement (modèle distant plus puissant) — icône seule + menu.
        self.account_btn = QPushButton()
        self.account_btn.setObjectName("icon_btn")
        self.account_btn.setToolTip("Compte et abonnement")
        self.account_btn.clicked.connect(self._open_account_menu)
        if self._account is not None:
            tb.addWidget(self.account_btn)

        # Réglages (ouvre le panneau Préférences).
        self.settings_btn = QPushButton()
        self.settings_btn.setObjectName("icon_btn")
        self.settings_btn.setToolTip("Préférences")
        self.settings_btn.clicked.connect(self._open_preferences)
        if self._on_open_preferences is not None:
            tb.addWidget(self.settings_btn)

        self.summarize_btn = QPushButton("Résumer")
        self.summarize_btn.setObjectName("summarize_btn")
        self.summarize_btn.clicked.connect(self._request_summary)
        tb.addWidget(self.summarize_btn)

        self.minimize_btn = QPushButton("Réduire")
        self.minimize_btn.setObjectName("minimize_btn")
        self.minimize_btn.clicked.connect(self._minimize)
        tb.addWidget(self.minimize_btn)

        # === Central ===
        central = QWidget()
        central.setObjectName("central")
        v = QVBoxLayout(central)
        v.setContentsMargins(18, 2, 18, 18)
        v.setSpacing(6)

        self.stack = QStackedWidget()
        self.live_tab = LiveTab()
        self.summaries_tab = SummariesTab()
        self.stack.addWidget(self.live_tab)
        self.stack.addWidget(self.summaries_tab)

        # Le transcript se pose sur une feuille surélevée ; la toolbar et les
        # onglets restent sur le plan de travail, derrière. La feuille est bornée
        # en largeur : au-delà, elle mangerait tout le plan et le relief
        # disparaîtrait — c'est une page posée sur un bureau, pas un fond.
        # Onglets alignés à gauche *dans* la feuille : ils désignent une vue sur
        # la matière qu'elle porte. Posés à côté d'elle, ils flottaient sur le
        # plan de travail sans jamais tomber d'aplomb avec son bord.
        self.segmented = SegmentedControl(["Live", "Résumés"])
        seg_wrap = QWidget()
        seg_row = QHBoxLayout(seg_wrap)
        seg_row.setContentsMargins(18, 10, 18, 0)
        seg_row.addWidget(self.segmented)

        self.tab_rule = QFrame()
        self.tab_rule.setFrameShape(QFrame.Shape.HLine)
        self.tab_rule.setFixedHeight(1)

        self.sheet = Sheet(margins=(6, 6, 6, 6))
        self.sheet.setMaximumWidth(900)
        self.sheet.body.addWidget(seg_wrap)
        self.sheet.body.addWidget(self.tab_rule)
        self.sheet.body.addWidget(self.stack, 1)
        sheet_row = QHBoxLayout()
        sheet_row.setContentsMargins(0, 0, 0, 0)
        sheet_row.addStretch(1)
        sheet_row.addWidget(self.sheet, 20)
        sheet_row.addStretch(1)
        v.addLayout(sheet_row, 1)

        self.segmented.currentChanged.connect(self.stack.setCurrentIndex)
        self.segmented.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(central)

        # === Bus wiring ===
        self.live_tab.save_requested.connect(self._save_meeting)
        if self._on_learn_term is not None:
            self.live_tab.set_learn_handler(self._on_learn_term)
        self.live_tab.set_consent_pending(self._on_save_meeting is not None)

        self._bus.event.connect(self.live_tab.on_event)
        self._bus.event.connect(self._update_vad_indicator)
        self._bus.event.connect(self._maybe_refresh_summarize_enabled)

        self._refresh_summarize_enabled()

    def _save_meeting(self) -> None:
        """Accorde la conservation, et le dit — l'accord doit se voir confirmé."""
        if self._on_save_meeting is None:
            return
        versees = self._on_save_meeting()
        self.live_tab.set_consent_pending(False)
        log.info("Conservation accordée depuis la fenêtre (%s entrée(s))", versees)

    def _apply_theme(self) -> None:
        t = current_theme()
        # Vibrancy active : fond transparent pour laisser passer le flou natif.
        # Sinon : un aplat de papier — pas de dégradé, il salirait la couleur sous
        # la ligne de temps.
        # Le fond est peint sur le **widget central**, pas sur la QMainWindow :
        # une feuille de style posée sur la QMainWindow n'est jamais rendue (le
        # widget ne remplit pas son fond de lui-même), si bien que la couleur du
        # plan de travail n'apparaissait pas et macOS remplissait la fenêtre avec
        # son gris système — exactement le look générique qu'on cherche à quitter.
        # La bande de la toolbar est en dehors du widget central : sans fond
        # explicite, elle reste elle aussi non peinte et laisse voir la couleur
        # système au-dessus du plan de travail.
        if self._vibrancy_active:
            window_bg = "QMainWindow, #central { background: transparent; }"
            toolbar_bg = "background: transparent;"
        else:
            window_bg = (
                panel_background_qss(t, "QMainWindow")
                + panel_background_qss(t, "#central")
            )
            toolbar_bg = f"background-color: rgb({t.paper.red()},{t.paper.green()},{t.paper.blue()});"
        self.setStyleSheet(f"""
            {window_bg}
            QToolBar {{ {toolbar_bg} border: none; padding: 10px 14px; spacing: 6px; }}
        """)
        self.status_pill.apply_theme()
        self.segmented.apply_theme()
        self.tab_rule.setStyleSheet(
            f"background-color: rgba({t.spine.red()},{t.spine.green()},"
            f"{t.spine.blue()},{t.spine.alpha()}); border: none;"
        )
        self.sheet.update()
        self._apply_toolbar_button_styles()
        if hasattr(self.live_tab, "apply_theme"):
            self.live_tab.apply_theme()
        if hasattr(self.summaries_tab, "apply_theme"):
            self.summaries_tab.apply_theme()

    def _apply_toolbar_button_styles(self) -> None:
        t = current_theme()
        ink_hex = f"#{t.ink.red():02x}{t.ink.green():02x}{t.ink.blue():02x}"

        # Action principale : aplat d'encre (le rouge est réservé au direct).
        # Pas d'icône : désactivé, l'icône (encre sur encre) disparaissait mais
        # gardait sa place et décentrait le mot. Le mot suffit.
        self.summarize_btn.setStyleSheet(
            primary_button_qss(t).replace("QPushButton", "QPushButton#summarize_btn")
        )
        self.minimize_btn.setIcon(minimize_icon(ink_hex))
        self.minimize_btn.setStyleSheet(
            secondary_button_qss(t).replace("QPushButton", "QPushButton#minimize_btn")
        )

        # Boutons icône seule (compte, réglages, pause) : fantômes.
        hover = t.ink_alpha(7)
        icon_qss = f"""
            QPushButton#icon_btn {{
                background: transparent;
                border: none;
                padding: 7px;
                border-radius: 7px;
            }}
            QPushButton#icon_btn:hover {{
                background-color: rgba({hover.red()},{hover.green()},{hover.blue()},{hover.alpha()});
            }}
        """
        self.account_btn.setIcon(person_icon(ink_hex))
        self.account_btn.setStyleSheet(icon_qss)
        self.settings_btn.setIcon(sliders_icon(ink_hex))
        self.settings_btn.setStyleSheet(icon_qss)
        self.pause_btn.setStyleSheet(icon_qss)
        self._refresh_pause_icon()

    def _refresh_pause_icon(self) -> None:
        t = current_theme()
        if self._paused:
            # Micro barré en rouge : état inhabituel, bien visible.
            red = t.live_red
            self.pause_btn.setIcon(mic_slash_icon(f"#{red.red():02x}{red.green():02x}{red.blue():02x}"))
            self.pause_btn.setToolTip("Reprendre le micro")
        else:
            label = t.label
            self.pause_btn.setIcon(mic_icon(f"#{label.red():02x}{label.green():02x}{label.blue():02x}"))
            self.pause_btn.setToolTip("Suspendre le micro")

    def _toggle_pause(self) -> None:
        if self._on_toggle_pause is None:
            return
        self.set_paused(bool(self._on_toggle_pause()))

    def set_paused(self, paused: bool) -> None:
        """Reflète l'état pause (appelé après un toggle local ou depuis le tray)."""
        self._paused = paused
        self.status_pill.set_paused(paused)
        self._refresh_pause_icon()

    def _wire_worker(self) -> None:
        self._worker.started.connect(self._on_summary_started)
        self._worker.chunk.connect(self._on_summary_chunk)
        self._worker.finished.connect(self._on_summary_finished)
        self._worker.failed.connect(self._on_summary_failed)

    def _update_vad_indicator(self, item) -> None:
        if isinstance(item, dict) and item.get("type") == "vad_status":
            self.status_pill.set_speaking(bool(item.get("speaking")))

    def _maybe_refresh_summarize_enabled(self, item) -> None:
        if isinstance(item, dict) and item.get("type") == "final_text" and item.get("text"):
            self._refresh_summarize_enabled()

    def _meeting_title(self) -> str:
        """Titre de la réunion en cours, vide tant qu'aucune phrase n'a été dite."""
        from benji import meetings

        meeting_id = meetings.current_meeting_id()
        if meeting_id is None:
            return ""
        meeting = meetings.store().get(meeting_id)
        return meeting.title if meeting else ""

    def _current_entries(self) -> list:
        """Entrées de la réunion en cours (vide tant qu'aucune n'a été dite)."""
        from benji import meetings

        meeting_id = meetings.current_meeting_id()
        if meeting_id is None:
            return []
        return self._history.get_for_meeting(meeting_id)

    def _refresh_summarize_enabled(self) -> None:
        try:
            has_history = bool(self._current_entries())
        except Exception:
            has_history = False
        idle = self._pending_summary_id is None
        self.summarize_btn.setEnabled(has_history and idle)

    def _request_summary(self) -> None:
        entries = self._current_entries()
        if not entries:
            return
        sid = uuid.uuid4().hex
        self._pending_summary_id = sid
        self._refresh_summarize_enabled()
        self.summaries_tab.begin_pending(sid)
        self._refresh_tab_badge()
        self._worker.request(entries=entries, summary_id=sid)

    def _on_summary_started(self, sid: str) -> None:
        log.info("Summary started: %s", sid)

    def _on_summary_chunk(self, sid: str, chunk: str) -> None:
        self.summaries_tab.append_chunk(sid, chunk)

    def _on_summary_finished(self, sid: str, path: Path) -> None:
        self.summaries_tab.finalize_pending(sid, path)
        self._pending_summary_id = None
        self._has_unread_summary = (self.segmented.currentIndex() != 1)
        self._refresh_tab_badge()
        self._refresh_summarize_enabled()

    def _on_summary_failed(self, sid: str, err: str) -> None:
        self.summaries_tab.fail_pending(sid, err)
        self._pending_summary_id = None
        self._refresh_tab_badge()
        self._refresh_summarize_enabled()

    def _on_tab_changed(self, idx: int) -> None:
        if idx == 1:
            self._has_unread_summary = False
            self._refresh_tab_badge()

    def _refresh_tab_badge(self) -> None:
        has_badge = self._pending_summary_id is not None or self._has_unread_summary
        self.segmented.setBadge(1, has_badge)

    def _minimize(self) -> None:
        if self._on_minimize is not None:
            self._on_minimize()

    def _open_preferences(self) -> None:
        if self._on_open_preferences is not None:
            self._on_open_preferences()

    def _notify_account(self, title: str, msg: str) -> None:
        QMessageBox.information(self, f"Benji — {title}", msg)

    def _build_account_menu(self) -> QMenu | None:
        """Menu compte selon l'état de session (None si pas de compte câblé).

        L'abonnement Pro débloque le modèle de transcription/résumé distant, plus
        puissant (cf. STTConfig.stt_provider = "remote")."""
        if self._account is None:
            return None
        menu = QMenu(self)
        session = self._account.session
        if session.is_authenticated:
            email = menu.addAction(session.email or "Connecté")
            email.setEnabled(False)
            menu.addSeparator()
            menu.addAction("Passer Pro — modèle distant…", self._account.open_checkout)
            menu.addAction("Gérer l'abonnement…", self._account.open_portal)
            menu.addSeparator()
            menu.addAction("Se déconnecter", self._account.logout)
        else:
            menu.addAction(
                "Se connecter / créer un compte…",
                lambda: self._account.login(parent=self),
            )
        return menu

    def _open_account_menu(self) -> None:
        """Reconstruit le menu à l'ouverture et le pose sous le bouton compte."""
        menu = self._build_account_menu()
        if menu is None:
            return
        pos = self.account_btn.mapToGlobal(self.account_btn.rect().bottomLeft())
        menu.exec(pos)

    def _restore_state(self) -> None:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        geom = s.value(_GEOM_KEY)
        if geom is not None:
            try:
                self.restoreGeometry(geom)
            except Exception:
                self.resize(960, 640)
        else:
            self.resize(960, 640)
        tab = s.value(_TAB_KEY, 0, type=int)
        self.segmented.setCurrentIndex(tab)
        self.stack.setCurrentIndex(tab)

    def closeEvent(self, event) -> None:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        s.setValue(_GEOM_KEY, self.saveGeometry())
        s.setValue(_TAB_KEY, self.segmented.currentIndex())
        super().closeEvent(event)
