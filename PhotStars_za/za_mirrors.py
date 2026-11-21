"""
Point astroquery's SIMBAD and VizieR queries at the South African mirrors
at IDIA (simbad.idia.ac.za and vizier.idia.ac.za) instead of the CDS servers
in Strasbourg.

Import this module BEFORE anything else uses astroquery's Simbad/Vizier:

    import za_mirrors  # noqa: F401

Every PhotStars_za module already does this. Set USE_MIRRORS = False to send
everything back to CDS.
"""
import inspect

from astroquery.simbad import conf as _simbad_conf, Simbad as _Simbad
from astroquery.vizier import conf as _vizier_conf, Vizier as _Vizier

USE_MIRRORS = True

SIMBAD_SERVER = "simbad.idia.ac.za" if USE_MIRRORS else "simbad.cds.unistra.fr"
VIZIER_SERVER = "vizier.idia.ac.za" if USE_MIRRORS else "vizier.cds.unistra.fr"


def apply():
    # 1. Config defaults (used by any Simbad() instance created from now on)
    #    astroquery only accepts servers from its built-in lists here, and
    #    the IDIA mirrors are not in those lists, so a refusal is expected and
    #    harmless: steps 2 and 3 set the servers directly.
    for _conf, _server in ((_simbad_conf, SIMBAD_SERVER), (_vizier_conf, VIZIER_SERVER)):
        try:
            _conf.server = _server
        except (TypeError, ValueError):
            pass

    # 2. The ready-made module-level objects (Simbad.query_object, Vizier.query_region)
    #    astroquery >= 0.4.8 (TAP): the public `server` setter refuses mirrors
    #    outside its built-in list, so set the underlying attribute, and make
    #    any new Simbad() instance pick up the same server.
    if SIMBAD_SERVER not in getattr(_simbad_conf, "servers_list", [SIMBAD_SERVER]):
        _simbad_conf.servers_list.append(SIMBAD_SERVER)
    _Simbad._server = SIMBAD_SERVER
    _Simbad._tap = None
    cls = type(_Simbad)
    if not getattr(cls, "_za_patched", False):
        _orig_init = cls.__init__

        def _init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self._server = SIMBAD_SERVER
            self._tap = None

        cls.__init__ = _init
        cls._za_patched = True
    _Simbad.SIMBAD_URL = f"https://{SIMBAD_SERVER}/simbad/sim-script"   # older astroquery
    type(_Simbad).SIMBAD_URL = _Simbad.SIMBAD_URL
    _Vizier.VIZIER_SERVER = VIZIER_SERVER

    # 3. New Vizier(...) instances: the server is a default argument that was
    #    frozen at import time, so the default itself has to be replaced.
    init = type(_Vizier).__init__
    params = inspect.signature(init).parameters
    if "vizier_server" in params:
        if params["vizier_server"].kind is inspect.Parameter.KEYWORD_ONLY:
            init.__kwdefaults__["vizier_server"] = VIZIER_SERVER
        else:
            with_defaults = [n for n, p in params.items()
                             if p.default is not inspect.Parameter.empty
                             and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
            d = list(init.__defaults__)
            d[with_defaults.index("vizier_server")] = VIZIER_SERVER
            init.__defaults__ = tuple(d)
    type(_Vizier).VIZIER_SERVER = VIZIER_SERVER


apply()
print(f"[za_mirrors] SIMBAD -> {SIMBAD_SERVER} | VizieR -> {VIZIER_SERVER}")
