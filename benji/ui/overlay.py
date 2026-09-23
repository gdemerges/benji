import logging

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from benji.config import IS_MACOS, IS_WINDOWS, UIConfig
from benji.stt.postprocessing import join_words
from benji.ui.style import speaker_color
from benji.ui.widgets.waveform import WaveformDot

log = logging.getLogger(__name__)

_space_observer_cls = None

# Hauteur non textuelle de l'overlay : marges du layout, rangée de l'indicateur
# VAD et padding du label. Retranchée du plafond de la fenêtre pour savoir ce
# qui reste au texte.
_CHROME_HEIGHT = 60


def _space_observer_class():
    """Lazily build (once) an NSObject subclass that forwards NSWorkspace
    active-space-change notifications to a Python callback.

    Defined lazily so this module still imports on non-macOS platforms (where
    Foundation isn't available). An Objective-C notification target must be a
    real NSObject subclass — a plain Python object is silently never called,
    which is exactly the bug this replaces.
    """
    global _space_observer_cls
    if _space_observer_cls is not None:
        return _space_observer_cls
    import objc
    from Foundation import NSObject

    class _SpaceObserver(NSObject):
        def initWithCallback_(self, cb):
            self = objc.super(_SpaceObserver, self).init()
            if self is None:
                return None
            self._cb = cb
            return self

        def spaceDidChange_(self, _notification):
            try:
                self._cb()
            except Exception:
                log.exception("Space-change reassert failed")

    _space_observer_cls = _SpaceObserver
    return _space_observer_cls


class VADIndicator(WaveformDot):
    """Forme d'onde signature en haut de l'overlay : danse quand la voix est
    détectée, quasi invisible sinon (repos discret plutôt que rond vert)."""

    def __init__(self, parent=None):
        super().__init__(bar_width=2, gap=2, height=12, parent=parent)
        self.set_color(QColor(255, 255, 255, 170))
        self.is_speaking = False

    def set_speaking(self, speaking: bool):
        self.is_speaking = speaking
        self.set_color(QColor(255, 255, 255, 220 if speaking else 70))
        self.set_active(speaking)


class SubtitleOverlay(QWidget):
    new_text_signal = Signal(str)
    new_word_signal = Signal(dict)
    vad_status_signal = Signal(bool)

    def __init__(self, bus, config: UIConfig = None, on_click=None, interactive: bool = False):
        """bus: DisplayBus. on_click: callable() appelé sur mousePressEvent (mode .app).
        interactive: if True, the overlay accepts mouse clicks (needed when on_click
        is bound to bring back the main window). Default False = click-through.
        """
        super().__init__()
        self._bus = bus
        self._on_click = on_click
        self._interactive = interactive
        self.setWindowTitle("BenjiOverlay")
        self.config = config or UIConfig()
        self.current_text = []  # For streaming mode
        # Finals de l'énoncé courant, une entrée par tour de parole : un segment
        # VAD peut en tenir plusieurs (cf. stt/diarization.py). Les remplacer les
        # uns par les autres n'afficherait que le dernier locuteur.
        self._final_lines: list[dict] = []
        # Prénoms donnés aux locuteurs (étiquette → nom), poussés par la fenêtre
        # principale (cf. `set_speaker_name`). En visio, c'est ici qu'un « A »
        # sert le moins : on lit les sous-titres, pas le transcript.
        self._speaker_names: dict[str, str] = {}
        self._shutting_down = False  # Flag to prevent operations during shutdown
        self._current_screen = None  # Screen the overlay is currently anchored to

        # Window flags (cross-platform)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if IS_MACOS:
            self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)

        # VAD indicator
        self.vad_indicator = VADIndicator()

        # Label
        self.label = QLabel("")
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self._apply_label_style()

        # Ombre portée douce : donne au bloc sous-titres un rendu « carte
        # flottante » et le détache d'un fond clair. Les marges du layout
        # ci-dessous réservent la place pour que le flou ne soit pas rogné.
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setColor(QColor(0, 0, 0, 120))
        shadow.setOffset(0, 3)
        self.label.setGraphicsEffect(shadow)

        # Layout with VAD indicator
        indicator_layout = QHBoxLayout()
        indicator_layout.addStretch()
        indicator_layout.addWidget(self.vad_indicator)
        indicator_layout.setContentsMargins(8, 8, 20, 0)

        main_layout = QVBoxLayout()
        main_layout.addLayout(indicator_layout)
        main_layout.addWidget(self.label)
        main_layout.setContentsMargins(20, 0, 20, 18)
        main_layout.setSpacing(4)
        self.setLayout(main_layout)

        # Position
        self._position_window()

        # Opacity animation for fade
        self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self.fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        # Fondu terminé = plus rien à l'écran : la garde de fenêtre se désarme.
        self.fade_anim.finished.connect(self._disarm_window_guard)

        # Subscribe to display bus events
        bus.event.connect(self._dispatch_event)

        # Auto-hide timer
        self.hide_timer = QTimer()
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self._start_fade)

        # Signal connections
        self.new_text_signal.connect(self._update_text)
        self.new_word_signal.connect(self._update_word)
        self.vad_status_signal.connect(self._update_vad_status)

        self.show()
        self._make_click_through()

    @Slot(bool)
    def _update_vad_status(self, speaking: bool):
        """Update VAD indicator."""
        if self._shutting_down:
            return
        try:
            self.vad_indicator.set_speaking(speaking)
        except Exception:
            if not self._shutting_down:
                log.exception("Error in _update_vad_status")

    def closeEvent(self, event):
        """Handle window close event - cleanup before Qt destroys objects."""
        self.cleanup()
        event.accept()

    def _apply_label_style(self):
        """(Re)applique police + fond depuis self.config au label de sous-titres."""
        # DemiBold plutôt que Bold : aussi lisible sur fond sombre, moins criard.
        self.label.setFont(
            QFont(self.config.font_family, self.config.font_size, QFont.Weight.DemiBold)
        )
        self.label.setStyleSheet(f"""
            QLabel {{
                color: rgba(255, 255, 255, 242);
                padding: 14px 26px;
                background-color: rgba(0, 0, 0, {self.config.bg_opacity});
                border-radius: 14px;
            }}
        """)

    def apply_config(self, config: UIConfig):
        """Applique à chaud une nouvelle UIConfig (police, opacité, durée, position).

        Appelé depuis le panneau Préférences pour les réglages « live » — pas de
        redémarrage. `display_duration_ms`/`fade_duration_ms` sont lus au vol lors
        des prochains démarrages de timer, donc réassigner self.config suffit.
        """
        if self._shutting_down:
            return
        self.config = config
        self._apply_label_style()
        self._position_window()

    def _target_screen(self):
        """Screen to anchor the overlay on.

        Multi-monitor: follow the screen under the cursor (the user's active
        display) when `follow_active_screen` is set; otherwise the primary.
        Falls back to primary if the cursor screen can't be resolved.
        """
        if getattr(self.config, "follow_active_screen", True):
            from PySide6.QtGui import QCursor
            screen = QApplication.screenAt(QCursor.pos())
            if screen is not None:
                return screen
        return QApplication.primaryScreen()

    def _position_window(self):
        screen = self._target_screen()
        if not screen:
            return
        self._current_screen = screen
        geom = screen.availableGeometry()
        width = int(geom.width() * self.config.window_width_ratio)
        x = geom.x() + (geom.width() - width) // 2
        self.setFixedWidth(width)
        self.setMaximumHeight(int(geom.height() * 0.4))  # Max 40% of screen height
        self.adjustSize()
        y = geom.y() + geom.height() - self.height() - self.config.bottom_margin
        self.move(x, y)

    def _reposition(self):
        """Reanchor window to bottom margin after content height changes.

        Stays on the screen the current utterance started on (self._current_screen)
        so the overlay never jumps mid-sentence; screen re-selection happens on
        segment_start via _position_window.
        """
        screen = self._current_screen or self._target_screen()
        if not screen:
            return
        geom = screen.availableGeometry()
        self.adjustSize()
        y = geom.y() + geom.height() - self.height() - self.config.bottom_margin
        self.move(self.x(), y)

    def _make_click_through(self):
        if IS_MACOS:
            self._click_through_macos()
        elif IS_WINDOWS:
            self._click_through_windows()

    def _apply_macos_window_settings(self, verbose: bool = False):
        """(Re)apply window level + collection behavior + private sticky tag.

        Also forces activation policy to Accessory each time, because Qt can
        reset it to Regular on certain events.
        """
        try:
            import ctypes
            import ctypes.util

            from AppKit import NSApp, NSApplicationActivationPolicyAccessory

            # Re-force Accessory policy (Qt may reset to Regular)
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
            current_policy = int(NSApp.activationPolicy())

            cg_path = ctypes.util.find_library("CoreGraphics")
            cg = ctypes.CDLL(cg_path)
            cg.CGShieldingWindowLevel.restype = ctypes.c_int32
            max_level = cg.CGShieldingWindowLevel()

            # Private SkyLight/CGS API for sticky-across-fullscreen windows.
            # Tag 0x400 = Sticky. May be restricted on macOS 15+ (Sequoia/Tahoe).
            sticky_ok = False
            conn_id = 0
            try:
                cg.CGSMainConnectionID.restype = ctypes.c_uint32
                conn_id = cg.CGSMainConnectionID()
                cg.CGSSetWindowTags.argtypes = [
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_int,
                ]
                cg.CGSSetWindowTags.restype = ctypes.c_int
                sticky_ok = True
            except Exception as e:
                if verbose:
                    log.debug("CGS private API unavailable: %s", e)

            window_debug = []
            for ns_window in NSApp.windows():
                # Only style the overlay's own NSWindow — otherwise we'd float the
                # main app window above everything too, and break its mouse events.
                if ns_window.title() != "BenjiOverlay":
                    continue
                if not self._interactive:
                    ns_window.setIgnoresMouseEvents_(True)
                else:
                    ns_window.setIgnoresMouseEvents_(False)
                ns_window.setLevel_(max_level)
                ns_window.setCollectionBehavior_(
                    (1 << 0) | (1 << 4) | (1 << 6) | (1 << 8)
                )
                ns_window.setCanHide_(False)
                ns_window.setHidesOnDeactivate_(False)
                ns_window.setOpaque_(False)

                tag_result = None
                if sticky_ok:
                    try:
                        wid = int(ns_window.windowNumber())
                        if wid > 0:
                            tags = (ctypes.c_uint32 * 2)(0x00000400, 0x00000000)
                            tag_result = cg.CGSSetWindowTags(conn_id, wid, tags, 32)
                    except Exception as e:
                        tag_result = f"err:{e}"

                if verbose:
                    window_debug.append({
                        "wid": int(ns_window.windowNumber()),
                        "level": int(ns_window.level()),
                        "collection": int(ns_window.collectionBehavior()),
                        "visible": bool(ns_window.isVisible()),
                        "sticky_tag": tag_result,
                    })

            if verbose:
                log.debug("policy=%s (1=Accessory, 0=Regular)", current_policy)
                log.debug("max_level=%s, cgs_conn=%s", max_level, conn_id)
                for w in window_debug:
                    log.debug("window %s", w)
            return max_level
        except Exception as e:
            log.warning("macOS window settings failed: %s", e)
            return None

    def _click_through_macos(self):
        try:
            from AppKit import (
                NSApp,
                NSApplicationActivationPolicyAccessory,
            )

            # Make the process an "accessory" app: no dock icon, behaves like
            # a menu-bar agent, and — crucially — allowed to float over
            # other apps' native fullscreen Spaces.
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

            level = self._apply_macos_window_settings(verbose=True)

            # Re-assert settings when the active Space changes (entering/leaving
            # another app's fullscreen). macOS resets window level on Space change,
            # so this event is the primary trigger — driven by NSWorkspace, not a
            # busy poll.
            try:
                from AppKit import (
                    NSWorkspace,
                    NSWorkspaceActiveSpaceDidChangeNotification,
                )

                observer_cls = _space_observer_class()
                # Strong ref: PyObjC must not GC an observer the notification
                # center holds only weakly.
                self._space_observer = observer_cls.alloc().initWithCallback_(
                    self._apply_macos_window_settings
                )
                self._workspace_nc = NSWorkspace.sharedWorkspace().notificationCenter()
                self._workspace_nc.addObserver_selector_name_object_(
                    self._space_observer,
                    b"spaceDidChange:",
                    NSWorkspaceActiveSpaceDidChangeNotification,
                    None,
                )

                # Filet de sécurité si une notification est manquée (les tags
                # privés peuvent aussi se réinitialiser hors changement d'Espace).
                # **Armé seulement quand des sous-titres sont à l'écran** : c'est
                # le seul moment où le niveau de fenêtre protège quelque chose.
                # Tournant en permanence, il réveillait le CPU toutes les 2 s
                # pendant les heures où l'overlay est invisible.
                self._reassert_timer = QTimer(self)
                self._reassert_timer.setInterval(2000)
                self._reassert_timer.timeout.connect(self._apply_macos_window_settings)

                # Verbose diagnostic dump — diagnostic builds only (off by default;
                # Ctrl+Shift+D gives the same dump on demand).
                if getattr(self.config, "debug_macos_window", False):
                    self._debug_timer = QTimer(self)
                    self._debug_timer.timeout.connect(
                        lambda: self._apply_macos_window_settings(verbose=True)
                    )
                    self._debug_timer.start(5000)
            except Exception as e:
                log.warning("Space-change observer not installed: %s", e)

            log.info("Click-through enabled (macOS, level=%s, policy=Accessory)", level)
        except Exception as e:
            log.warning("macOS click-through failed: %s", e)

    def _click_through_windows(self):
        try:
            import win32con
            import win32gui
            hwnd = int(self.winId())
            # Add layered + transparent + tool window extended styles
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            ex_style |= (
                win32con.WS_EX_LAYERED
                | win32con.WS_EX_TRANSPARENT
                | win32con.WS_EX_TOOLWINDOW
                | win32con.WS_EX_TOPMOST
                | win32con.WS_EX_NOACTIVATE
            )
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)
            # Force topmost position
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE,
            )
            log.info("Click-through enabled (Windows)")
        except Exception as e:
            log.warning("Windows click-through failed: %s", e)

    @Slot(str)
    def _update_text(self, text: str):
        """Classic mode: replace all text at once."""
        if self._shutting_down:
            return
        try:
            self.fade_anim.stop()
            self.setWindowOpacity(1.0)
            self.label.setText(text)
            self._reposition()
            self._arm_window_guard()
            self.hide_timer.start(self.config.display_duration_ms)
        except Exception:
            if not self._shutting_down:
                log.exception("Error in _update_text")

    def set_speaker_name(self, label: str, name: str) -> None:
        """Nomme un locuteur (nom vide = retour à l'étiquette), à l'écran aussi.

        La couleur reste celle de l'**étiquette** : renommer ne doit pas faire
        changer quelqu'un de couleur en pleine réunion.
        """
        name = (name or "").strip()
        if name:
            self._speaker_names[label] = name
        else:
            self._speaker_names.pop(label, None)
        if self._final_lines and not self._shutting_down:
            self._render_final_lines()
            self._reposition()

    def speaker_name(self, label: str) -> str | None:
        """Le nom donné à une étiquette pendant cette réunion, s'il y en a un."""
        return self._speaker_names.get(label)

    def clear_speaker_names(self) -> None:
        """Nouvelle réunion : les étiquettes ne désignent plus les mêmes gens."""
        self._speaker_names.clear()

    @staticmethod
    def _lines_html(lines, names: dict[str, str] | None = None) -> str:
        """Compose les tours de parole, un par ligne.

        Le nom du locuteur est coloré et le corps échappé : une transcription
        contenant `<`, `&` ou `>` ne doit pas être interprétée comme du balisage.
        """
        from html import escape

        names = names or {}
        parts = []
        for line in lines:
            body = escape(line["text"])
            speaker = line.get("speaker")
            if speaker:
                # L'overlay est toujours sur fond noir : variante claire.
                c = speaker_color(speaker, on_dark=True)
                parts.append(
                    f'<span style="color:{c.name()};font-weight:bold;">'
                    f"{escape(names.get(speaker, speaker))}</span> {body}"
                )
            else:
                parts.append(body)
        return "<br>".join(parts)

    def _render_final_lines(self) -> None:
        """Affiche les tours de l'énoncé, en n'en gardant que ce qui tient.

        La fenêtre est plafonnée à 40 % de la hauteur d'écran et le label ne
        défile pas : au-delà, le texte était **coupé en silence** — mesuré à
        820 px voulus pour 320 disponibles sur six tours empilés, et la réplique
        de quelqu'un disparaissait sans que rien ne le signale. On retire donc
        les tours les plus anciens jusqu'à ce que l'énoncé tienne : le plus
        récent est celui qu'on est en train de lire, et la ligne complète reste
        dans le transcript et l'historique.
        """
        budget = self.maximumHeight() - _CHROME_HEIGHT
        # On rend une **copie** : `_final_lines` garde le texte entier, sinon une
        # correction LLM tardive s'appliquerait à un texte déjà rogné.
        lines = list(self._final_lines)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setText(self._lines_html(lines, self._speaker_names))

        while len(lines) > 1 and self.label.sizeHint().height() > budget:
            lines.pop(0)
            self.label.setText(self._lines_html(lines, self._speaker_names))

        # Un tour unique trop long ne peut pas être retiré : on montre sa fin —
        # la partie qu'on vient d'entendre — derrière une ellipse qui dit que le
        # début est ailleurs. Mieux qu'une coupe nette qui se fait passer pour
        # la fin de la phrase.
        if lines and self.label.sizeHint().height() > budget:
            words = lines[0]["text"].split()
            while len(words) > 4 and self.label.sizeHint().height() > budget:
                words = words[max(1, len(words) // 8):]
                lines[0] = {**lines[0], "text": "… " + " ".join(words)}
                self.label.setText(self._lines_html(lines, self._speaker_names))

    def _arm_window_guard(self) -> None:
        """Réassertion périodique du niveau de fenêtre, pendant l'affichage."""
        timer = getattr(self, "_reassert_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()

    def _disarm_window_guard(self) -> None:
        timer = getattr(self, "_reassert_timer", None)
        if timer is not None:
            timer.stop()

    def _show_partial(self) -> None:
        """Peint l'énoncé en cours et relance le fondu.

        Le minuteur n'est relancé que quand des mots arrivent, jamais sur
        `segment_start` : le texte précédent reste lisible pendant que le moteur
        décode le suivant.
        """
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setText(join_words(self.current_text))
        self._reposition()
        self.fade_anim.stop()
        self.setWindowOpacity(1.0)
        self._arm_window_guard()
        self.hide_timer.start(self.config.display_duration_ms)

    @Slot(dict)
    def _update_word(self, message: dict):
        """Streaming mode: add words progressively."""
        if self._shutting_down:
            return
        try:
            msg_type = message.get("type")

            if msg_type == "segment_start":
                # Reset internal buffer but keep label visible until first word arrives
                self.current_text = []
                # Un nouvel énoncé commence : une correction LLM tardive pour
                # le final précédent ne s'applique plus à ce qui est à l'écran.
                self._final_lines = []
                # Between utterances (faded out), re-evaluate which screen is
                # active so subtitles follow the user to another monitor.
                if self.windowOpacity() == 0.0 or not self.isVisible():
                    self._position_window()
            elif msg_type == "partial":
                # Instantané complet de l'énoncé en cours : une seule peinture.
                # Le moteur local publie une passe entière par message (cf.
                # stt/transcriber.py) — repeindre par mot coûtait un adjustSize
                # et un move de fenêtre à chaque mot, pour un état transitoire.
                self.current_text = [w["text"] for w in message.get("words", [])]
                self._show_partial()
            elif msg_type == "word":
                # Mot à mot : chemin du mode remote, dont les events viennent du
                # backend (cf. stt/remote.py) et arrivent réellement un par un.
                self.current_text.append(message["text"])
                self._show_partial()
            elif msg_type == "final_text":
                # Replace the streamed (raw) text with the post-processed/corrected
                # final version. If `drop` is set, the segment was a hallucination —
                # clear the overlay immediately instead of leaving garbage on screen.
                if message.get("drop"):
                    self.current_text = []
                    self._final_lines = []
                    self.label.setText("")
                    self.fade_anim.stop()
                    self.setWindowOpacity(0.0)
                    self._disarm_window_guard()
                    return
                seq = message.get("seq")
                text = message.get("text") or ""
                if message.get("corrected"):
                    # Async LLM correction: only replace if that segment is still
                    # on screen (no newer utterance has started since).
                    line = next((l for l in self._final_lines
                                 if seq is not None and l["seq"] == seq), None)
                    if line is None:
                        return
                    line["text"] = text
                else:
                    # Tour de parole suivant du même énoncé : on l'ajoute sous le
                    # précédent au lieu de l'écraser — sinon la réplique de l'un
                    # effacerait celle de l'autre en une fraction de seconde.
                    self.current_text = []
                    self._final_lines.append(
                        {"text": text, "speaker": message.get("speaker"), "seq": seq}
                    )
                self._render_final_lines()
                self._reposition()
                self.fade_anim.stop()
                self.setWindowOpacity(1.0)
                self._arm_window_guard()
                self.hide_timer.start(self.config.display_duration_ms)
        except Exception:
            if not self._shutting_down:
                log.exception("Error in _update_word")

    def _dispatch_event(self, item) -> None:
        if self._shutting_down:
            return
        try:
            if isinstance(item, dict):
                msg_type = item.get("type")
                if msg_type == "vad_status":
                    self.vad_status_signal.emit(item["speaking"])
                else:
                    self.new_word_signal.emit(item)
            elif isinstance(item, str):
                self.new_text_signal.emit(item)
        except Exception:
            if not self._shutting_down:
                log.exception("Error in _dispatch_event")

    def mousePressEvent(self, event):
        if self._on_click is not None:
            try:
                self._on_click()
            except Exception:
                log.exception("Overlay on_click handler raised")
        super().mousePressEvent(event)

    def _start_fade(self):
        if self._shutting_down:
            return
        try:
            self.fade_anim.setDuration(self.config.fade_duration_ms)
            self.fade_anim.setStartValue(1.0)
            self.fade_anim.setEndValue(0.0)
            self.fade_anim.start()
        except Exception:
            if not self._shutting_down:
                log.exception("Error in _start_fade")

    def cleanup(self):
        """Stop all timers and animations before shutdown."""
        self._shutting_down = True
        self.hide_timer.stop()
        self.fade_anim.stop()
        if hasattr(self, "_reassert_timer"):
            self._reassert_timer.stop()
        if hasattr(self, "_debug_timer"):
            self._debug_timer.stop()
        if hasattr(self, "_workspace_nc") and hasattr(self, "_space_observer"):
            try:
                self._workspace_nc.removeObserver_(self._space_observer)
            except Exception:
                pass
        # Disconnect signals to prevent any pending emissions
        try:
            self.new_text_signal.disconnect()
            self.new_word_signal.disconnect()
        except:
            pass
