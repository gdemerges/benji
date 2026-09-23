"""Fenêtre de compte : un formulaire, deux modes, des erreurs avant le réseau."""

import pytest
from PySide6.QtWidgets import QApplication

from benji.account import AuthError
from benji.ui.login_dialog import LOGIN, REGISTER, LoginDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class FakeSession:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def login(self, email, password):
        self.calls.append(("login", email))
        if self.fail:
            raise AuthError(self.fail)

    def register(self, email, password):
        self.calls.append(("register", email))


def _fill(d, email="a@b.fr", pw="motdepasse", confirm=None):
    d._email.setText(email)
    d._password.setText(pw)
    d._confirm.setText(pw if confirm is None else confirm)


def test_connexion_par_defaut_sans_confirmation(qapp):
    d = LoginDialog(FakeSession())
    assert d._mode == LOGIN
    assert d._confirm.isHidden()


def test_basculer_vers_la_creation_montre_la_confirmation(qapp):
    d = LoginDialog(FakeSession())
    d._switch.click()
    assert d._mode == REGISTER
    assert not d._confirm.isHidden()
    assert d._submit.text() == "Créer le compte"


def test_connexion_appelle_login(qapp):
    s = FakeSession()
    d = LoginDialog(s)
    _fill(d)
    d._submit.click()
    assert s.calls == [("login", "a@b.fr")]


def test_creation_refuse_des_mots_de_passe_differents_sans_appel_reseau(qapp):
    s = FakeSession()
    d = LoginDialog(s, mode=REGISTER)
    _fill(d, confirm="autrechose")
    d._submit.click()
    assert s.calls == []
    assert not d._error.isHidden()


def test_creation_valide_appelle_register(qapp):
    s = FakeSession()
    d = LoginDialog(s, mode=REGISTER)
    _fill(d)
    d._submit.click()
    assert s.calls == [("register", "a@b.fr")]


def test_erreur_du_backend_affichee_et_bouton_rendu(qapp):
    d = LoginDialog(FakeSession(fail="Identifiants invalides"))
    _fill(d)
    d._submit.click()
    assert d._error.text() == "Identifiants invalides"
    assert d._submit.isEnabled() and d._submit.text() == "Se connecter"
