"""Dialogue de connexion / inscription au compte Benji.

Modal, sur le thread Qt. La requête réseau (login/register) est rapide (un POST)
et reste synchrone ; le bouton passe en « Connexion… » le temps de l'appel.

Un seul formulaire, deux modes : se connecter ou créer un compte. L'ancienne
boîte posait les deux boutons côte à côte au-dessus d'un `QFormLayout` natif —
on ne savait pas lequel était attendu, et le rouge d'erreur (#d9534f) était la
seule couleur de la fenêtre, codée en dur. Même feuille que le reste de l'app :
titres en SF Pro, champs en creux d'encre, action principale en aplat d'encre.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from benji.account import AuthError, Session
from benji.ui.style import (
    FONT_UI,
    Theme,
    current_theme,
    field_qss,
    primary_button_qss,
)

LOGIN = "login"
REGISTER = "register"

_COPY = {
    LOGIN: {
        "title": "Se connecter à Benji",
        "action": "Se connecter",
        "busy": "Connexion…",
        "switch_prompt": "Pas encore de compte ?",
        "switch": "Créer un compte",
    },
    REGISTER: {
        "title": "Créer votre compte",
        "action": "Créer le compte",
        "busy": "Création…",
        "switch_prompt": "Déjà un compte ?",
        "switch": "Se connecter",
    },
}

# Ce que le compte fait — et surtout ce qu'il ne fait pas. Benji transcrit des
# réunions : la première question d'un inscrit est « qu'est-ce qui part ? ».
_BODY = (
    "Le compte porte votre abonnement Pro, sur tous vos appareils. "
    "Vos transcriptions n'y sont pas stockées."
)


_WIDTH = 380
_MARGIN = 32


def _rgba(c) -> str:
    return f"rgba({c.red()},{c.green()},{c.blue()},{c.alpha()})"


class LoginDialog(QDialog):
    def __init__(self, session: Session, parent=None, mode: str = LOGIN):
        super().__init__(parent)
        self._session = session
        self._mode = mode if mode in _COPY else LOGIN
        self.setWindowTitle("Compte Benji")
        self.setModal(True)
        self.setFixedWidth(_WIDTH)
        self._build_ui()
        self._set_mode(self._mode)
        self._apply_theme()

    def _build_ui(self) -> None:
        self._title = QLabel()
        self._body = QLabel(_BODY)
        self._body.setWordWrap(True)
        # Largeur fixe, pas celle de la disposition : un libellé à retours dans
        # une fenêtre à largeur fixe voit sa hauteur mal calculée, et la
        # dernière ligne (« stockées. ») disparaissait.
        self._body.setFixedWidth(_WIDTH - 2 * _MARGIN)

        self._email_label = QLabel("Adresse e-mail")
        self._email = QLineEdit()
        self._email.setPlaceholderText("vous@exemple.com")
        self._password_label = QLabel("Mot de passe")
        self._password = QLineEdit()
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        self._confirm_label = QLabel("Confirmer le mot de passe")
        self._confirm = QLineEdit()
        self._confirm.setEchoMode(QLineEdit.EchoMode.Password)

        self._error = QLabel()
        self._error.setWordWrap(True)
        self._error.setFixedWidth(_WIDTH - 2 * _MARGIN)
        self._error.hide()

        self._submit = QPushButton()
        self._submit.setDefault(True)
        self._submit.setCursor(Qt.CursorShape.PointingHandCursor)
        self._submit.clicked.connect(self._do_submit)

        self._switch_prompt = QLabel()
        self._switch = QPushButton()
        self._switch.setObjectName("link_btn")
        self._switch.setCursor(Qt.CursorShape.PointingHandCursor)
        self._switch.setFlat(True)
        self._switch.clicked.connect(self._toggle_mode)

        switch_row = QWidget()
        row = QHBoxLayout(switch_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        row.addStretch(1)
        row.addWidget(self._switch_prompt)
        row.addWidget(self._switch)
        row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(_MARGIN, 28, _MARGIN, 22)
        layout.setSpacing(0)
        layout.addWidget(self._title)
        layout.addSpacing(6)
        layout.addWidget(self._body)
        layout.addSpacing(22)
        for label, field in (
            (self._email_label, self._email),
            (self._password_label, self._password),
            (self._confirm_label, self._confirm),
        ):
            layout.addWidget(label)
            layout.addSpacing(5)
            layout.addWidget(field)
            layout.addSpacing(14)
        layout.addWidget(self._error)
        layout.addSpacing(6)
        layout.addWidget(self._submit)
        layout.addSpacing(14)
        layout.addWidget(switch_row)

        self._email.returnPressed.connect(self._password.setFocus)
        self._password.returnPressed.connect(self._on_password_return)
        self._confirm.returnPressed.connect(self._do_submit)

    # --- modes ---

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        copy = _COPY[mode]
        self._title.setText(copy["title"])
        self._submit.setText(copy["action"])
        self._switch_prompt.setText(copy["switch_prompt"])
        self._switch.setText(copy["switch"])
        registering = mode == REGISTER
        self._confirm_label.setVisible(registering)
        self._confirm.setVisible(registering)
        self._password.setPlaceholderText("8 caractères ou plus" if registering else "")
        self._error.hide()
        self.adjustSize()

    def _toggle_mode(self) -> None:
        self._set_mode(REGISTER if self._mode == LOGIN else LOGIN)
        (self._email if not self._email.text() else self._password).setFocus()

    def _on_password_return(self) -> None:
        if self._mode == REGISTER:
            self._confirm.setFocus()
        else:
            self._do_submit()

    # --- envoi ---

    def _credentials(self) -> tuple[str, str] | None:
        email = self._email.text().strip()
        password = self._password.text()
        if not email or not password:
            self._show_error("Adresse e-mail et mot de passe requis.")
            return None
        if "@" not in email:
            self._show_error("Cette adresse e-mail ne semble pas complète.")
            return None
        if self._mode == REGISTER:
            if len(password) < 8:
                self._show_error("Choisissez un mot de passe d'au moins 8 caractères.")
                return None
            if password != self._confirm.text():
                self._show_error("Les deux mots de passe ne correspondent pas.")
                return None
        return email, password

    def _do_submit(self) -> None:
        action = self._session.register if self._mode == REGISTER else self._session.login
        creds = self._credentials()
        if creds is None:
            return
        self._error.hide()
        self._set_busy(True)
        try:
            action(*creds)
        except AuthError as e:
            self._show_error(str(e))
            return
        finally:
            self._set_busy(False)
        self.accept()

    def _set_busy(self, busy: bool) -> None:
        self._submit.setEnabled(not busy)
        self._switch.setEnabled(not busy)
        copy = _COPY[self._mode]
        self._submit.setText(copy["busy"] if busy else copy["action"])
        if busy:
            # Le POST est synchrone : sans ça, « Connexion… » ne se peint pas.
            QApplication.processEvents()

    def _show_error(self, msg: str) -> None:
        self._error.setText(msg)
        self._error.show()
        self.adjustSize()

    # --- thème ---

    def _apply_theme(self) -> None:
        t: Theme = current_theme()
        self.setStyleSheet(
            f"QDialog {{ background-color: {_rgba(t.sheet)}; }}"
            + field_qss(t).replace("padding: 6px 10px;", "padding: 8px 11px; font-size: 13px;")
            + f"""
            QPushButton#link_btn {{
                font-family: {FONT_UI}; font-size: 12px; font-weight: 600;
                color: {_rgba(t.ink)}; background: transparent; border: none;
                padding: 2px 2px;
            }}
            QPushButton#link_btn:hover {{ text-decoration: underline; }}
            QPushButton#link_btn:disabled {{ color: {_rgba(t.ink_faint)}; }}
            """
        )
        self._submit.setStyleSheet(
            primary_button_qss(t).replace("padding: 7px 16px;", "padding: 10px 16px;")
            .replace("font-size: 12px;", "font-size: 13px;")
        )
        self._title.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 19px; font-weight: 600; "
            f"color: {_rgba(t.ink)}; background: transparent;"
        )
        self._body.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 13px; color: {_rgba(t.ink_muted)}; "
            "background: transparent;"
        )
        field_label = (
            f"font-family: {FONT_UI}; font-size: 12px; font-weight: 500; "
            f"color: {_rgba(t.ink_muted)}; background: transparent;"
        )
        for label in (self._email_label, self._password_label, self._confirm_label):
            label.setStyleSheet(field_label)
        self._switch_prompt.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 12px; color: {_rgba(t.ink_faint)}; "
            "background: transparent;"
        )
        # Une erreur n'est pas le direct, mais elle doit se voir : c'est le seul
        # autre usage du rouge admis, comme l'action destructive — du texte,
        # jamais un aplat.
        self._error.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 12px; color: {_rgba(t.record)}; "
            "background: transparent; padding-bottom: 6px;"
        )
        # Les corps viennent de changer : la hauteur se recalcule avec eux.
        self.adjustSize()
