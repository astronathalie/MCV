"""
Point astroquery's SIMBAD and VizieR queries at the Harvard (CfA) mirrors
instead of the CDS servers in Strasbourg.

Import this module BEFORE anything else uses astroquery's Simbad/Vizier:

    import harvard_mirrors  # noqa: F401

Every PhotStars_h module already does this. To go back to CDS once it is up
again, set USE_HARVARD = False below (or just use the original PhotStars).
"""
import inspect

from astroquery.simbad import conf as _simbad_conf, Simbad as _Simbad
from astroquery.vizier import conf as _vizier_conf, Vizier as _Vizier

USE_HARVARD = True

SIMBAD_SERVER = "simbad.harvard.edu" if USE_HARVARD else "simbad.cds.unistra.fr"
VIZIER_SERVER = "vizier.cfa.harvard.edu" if USE_HARVARD else "vizier.cds.unistra.fr"


def apply():
    # 1. Config defaults (used by any Simbad() instance created from now on)
    _simbad_conf.server = SIMBAD_SERVER
    _vizier_conf.server = VIZIER_SERVER

    # 2. The ready-made module-level objects (Simbad.query_object, Vizier.query_region)
    try:
        _Simbad.server = SIMBAD_SERVER          # astroquery >= 0.4.8 (TAP)
    except Exception:
        pass
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
print(f"[harvard_mirrors] SIMBAD -> {SIMBAD_SERVER} | VizieR -> {VIZIER_SERVER}")
