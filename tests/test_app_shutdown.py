from queue import Queue

from benji.app import BenjiApplication


def test_le_sentinelle_darret_ne_bloque_pas_sur_une_file_pleine():
    q = Queue(maxsize=1)
    q.put("x")
    BenjiApplication._put_sentinel(q)  # doit rendre la main au lieu de bloquer
    assert q.qsize() == 1
