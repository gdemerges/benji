"""Contrôleur compte + facturation Stripe, partagé entre le tray et la fenêtre.

Login : modal, sur le thread Qt. Facturation (checkout/portail) : réseau hors
thread Qt — la résolution du token peut rafraîchir → off-thread — avec erreurs
remontées par signal vers le thread principal (`failed`). Les notifications de
succès passent par un callback `notify(title, message)` fourni par l'appelant
(le tray affiche une bulle système ; la fenêtre peut afficher autre chose).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

log = logging.getLogger(__name__)


class AccountController(QObject):
    failed = Signal(str)

    def __init__(
        self,
        session,
        base_url: str,
        notify: Callable[[str, str], None],
        parent=None,
    ):
        super().__init__(parent)
        self._session = session
        self._base_url = base_url
        self._notify = notify

    @property
    def session(self):
        return self._session

    def login(self, parent=None) -> None:
        from benji.ui.login_dialog import LoginDialog
        if LoginDialog(self._session, parent=parent).exec():
            self._notify("Connecté", f"Compte : {self._session.email}")

    def logout(self) -> None:
        self._session.logout()
        self._notify("Déconnecté", "Tu peux te reconnecter à tout moment.")

    def open_checkout(self) -> None:
        from benji import billing
        self._run(lambda token: billing.open_checkout(self._base_url, token))

    def open_portal(self) -> None:
        from benji import billing
        self._run(lambda token: billing.open_portal(self._base_url, token))

    def _run(self, fn) -> None:
        def worker():
            try:
                token = self._session.access_token()
                if not token:
                    # Session gardée mais pas de jeton : backend injoignable.
                    if self._session.is_authenticated:
                        raise RuntimeError("Service Benji injoignable — réessaie dans un instant.")
                    raise RuntimeError("Session expirée — reconnecte-toi.")
                fn(token)
            except Exception as e:  # réseau, 401/402, backend down…
                log.warning("Facturation: %s", e)
                self.failed.emit(str(e))

        threading.Thread(target=worker, daemon=True, name="benji-billing").start()
