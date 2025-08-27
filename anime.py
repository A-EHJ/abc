import re
import json
import time
import functools
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from urllib.parse import urljoin, urlencode, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
import streamlit as st
from streamlit.components.v1 import html as st_html

# Opcional: persistencia en localStorage
try:
    from streamlit_js_eval import get_local_storage, set_local_storage
except Exception:
    get_local_storage = None
    set_local_storage = None

# ==========================
#   CONFIG
# ==========================
BASE_URL = "https://www3.animeflv.net"
LS_KEY_SEEN   = "aflv_nav_seen_v1"
LS_KEY_WATCH  = "aflv_nav_watch_v1"
LS_KEY_PREFS  = "aflv_nav_prefs_v1"

# ==========================
#   UTIL / HTTP
# ==========================
def HEADERS():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Referer": BASE_URL,
        "Connection": "keep-alive",
    }

def fetch(url: str, params: Optional[dict] = None) -> str:
    r = requests.get(url, params=params, headers=HEADERS(), timeout=20)
    r.raise_for_status()
    return r.text

@st.cache_data(ttl=60, show_spinner=False)
def fetch_cached(url: str, params: Optional[dict] = None) -> str:
    return fetch(url, params)

def abs_url(base: str, path: str) -> str:
    return urljoin(base, path or "")

def _qp_href(params: Dict[str, str]) -> str:
    """Devuelve ?query=... para navegación interna."""
    return "?" + urlencode(params, doseq=True)

# ==========================
#   ESTADO DE FILTROS (Directorio)
# ==========================
def _dir_defaults():
    return {
        "dir_genres": [],
        "dir_years": [],
        "dir_types": [],
        "dir_status": [],
        "dir_order": "default",
        "dir_page": 1,
        "dir_q": "",
        "dir_cols": 5,   # NUEVO: columnas de grilla
    }

def _ensure_dir_state():
    for k, v in _dir_defaults().items():
        st.session_state.setdefault(k, v)

# ==========================
#   MODELOS
# ==========================
@dataclass
class Card:
    title: str
    url: str
    image: Optional[str]

@dataclass
class Episode:
    number: int
    id: int
    url: Optional[str]

@dataclass
class AnimeDetail:
    title: str
    image: Optional[str]
    genres: List[str]
    description: str
    episodes: List[Episode]

@dataclass
class HomeEp:
    anime_title: str
    anime_slug: str
    anime_url: str
    episode_num: int
    episode_url: str
    image: Optional[str]

# ==========================
#   PARSEADORES
# ==========================
def extract_episode_servers(html: str):
    m = re.search(r'var\s+videos\s*=\s*(\{.*?\});', html, re.DOTALL)
    if not m:
        return []
    data = json.loads(m.group(1))
    servers = []
    for lang, items in data.items():
        for it in items:
            link = it.get("url") or it.get("code")
            servers.append({
                "lang": lang,
                "server": it.get("server"),
                "title": it.get("title"),
                "ads": bool(it.get("ads", 0)),
                "allow_mobile": it.get("allow_mobile", True),
                "link": link,
            })
    return servers

def parse_browse(html: str, base_url: str) -> List[Card]:
    soup = BeautifulSoup(html, "html.parser")
    cards: List[Card] = []
    seen = set()
    for a in soup.select('a[href^="/anime/"]'):
        href = a.get("href")
        if not href or href in seen:
            continue
        seen.add(href)
        title = a.get("title") or a.get_text(strip=True)
        img = None
        img_tag = a.find("img")
        if img_tag:
            img = img_tag.get("data-src") or img_tag.get("data-original") or img_tag.get("src")
        cards.append(Card(title=title or "", url=abs_url(base_url, href), image=abs_url(base_url, img) if img else None))
    return cards

def parse_anime_detail(html: str, page_url: str, base_url: str) -> AnimeDetail:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.select_one(".Ficha .Title") or soup.find("h1")).get_text(strip=True)
    img_tag = soup.select_one(".AnimeCover .Image img") or soup.select_one("meta[property='og:image']")
    image = abs_url(base_url, (img_tag.get("src") or img_tag.get("content"))) if img_tag else None
    genres = [a.get_text(strip=True) for a in soup.select(".Nvgnrs a")]
    desc_tag = soup.select_one(".Description") or soup.find("p")
    description = desc_tag.get_text(" ", strip=True) if desc_tag else ""
    episodes: List[Episode] = []
    m = re.search(r"var\s+episodes\s*=\s*(\[\[.*?\]\]);", html, re.S)
    slug = None
    ms = re.search(r'var\s+anime_info\s*=\s*\[\s*"[^\"]*",\s*"[^\"]*",\s*"([^\"]+)"', html)
    if ms:
        slug = ms.group(1)
    if m:
        try:
            data = json.loads(m.group(1))
            for num, eid in data:
                ep_url = abs_url(base_url, f"/ver/{slug}-{num}") if slug else None
                episodes.append(Episode(number=int(num), id=int(eid), url=ep_url))
        except Exception:
            episodes = []
    return AnimeDetail(title=title, image=image, genres=genres, description=description, episodes=episodes)

def parse_home_week(html: str, base_url: str, limit: int = 24) -> List[HomeEp]:
    soup = BeautifulSoup(html, "html.parser")
    items: List[HomeEp] = []
    seen = set()
    candidates = soup.select(".ListEpisodios a[href^='/ver/'], .ListCaps a[href^='/ver/']") or soup.select("a[href^='/ver/']")
    rx = re.compile(r"^/ver/([a-z0-9\-]+)-(\d+)$", re.I)
    for a in candidates:
        href = (a.get("href") or "").strip()
        m = rx.match(href)
        if not m or href in seen:
            continue
        seen.add(href)
        slug, num = m.group(1), int(m.group(2))
        anime_url = abs_url(base_url, f"/anime/{slug}")
        episode_url = abs_url(base_url, href)
        title = a.get("title") or a.get_text(strip=True) or slug.replace("-", " ").title()
        image = None
        parent = a.find_parent(["li", "article", "div"]) or a.parent
        if parent:
            imgt = parent.find("img")
            if imgt:
                image = imgt.get("data-src") or imgt.get("data-original") or imgt.get("src")
        if image:
            image = abs_url(base_url, image)
        items.append(HomeEp(title, slug, anime_url, num, episode_url, image))
        if len(items) >= limit:
            break
    return items

# ==========================
#   FILTROS (Directorio)
# ==========================
GENRE_SLUGS = [
    "accion","artes-marciales","aventuras","carreras","ciencia-ficcion","comedia","deportes","drama","ecchi",
    "escolar","espacial","fantasia","harem","historico","infantil","josei","juegos","magia","militar","misterio",
    "musica","parodia","policia","psicologico","recuerdos-de-la-vida","romance","samurai","seinen","shoujo","shounen",
    "sobrenatural","superpoderes","suspenso","terror","vampiros","yaoi","yuri","demencia",
]
TYPE_MAP   = {"tv": "TV", "movie": "Película", "ova": "OVA", "special": "Especial"}
STATUS_MAP = {"1": "En emisión", "2": "Finalizado", "3": "Próximamente"}
ORDER_MAP  = {"default": "Por Defecto", "updated": "Recientemente Actualizados", "added": "Recientemente Agregados", "name":"Nombre A-Z", "rating":"Calificación"}
YEARS = list(range(1990, 2026))

def _build_browse_url(base=BASE_URL):
    p: Dict[str, List[str] | str | int] = {}
    ss = st.session_state
    if ss["dir_q"].strip():            p["q"] = ss["dir_q"].strip()
    if ss["dir_genres"]:               p["genre[]"] = ss["dir_genres"]
    if ss["dir_years"]:                p["year[]"] = [str(y) for y in ss["dir_years"]]
    if ss["dir_types"]:                p["type[]"] = ss["dir_types"]
    if ss["dir_status"]:               p["status[]"] = ss["dir_status"]
    p["order"] = ss["dir_order"] if ss["dir_order"] != "default" else "default"
    if ss["dir_page"] and ss["dir_page"] != 1: p["page"] = ss["dir_page"]
    query = urlencode(p, doseq=True)
    return abs_url(base, "/browse") + (f"?{query}" if query else "")

def import_filters_from_url(url: str) -> Tuple[List[str], List[int], List[str], List[str], str, int]:
    try:
        u = urlparse(url); q = parse_qs(u.query)
        genres = [g for g in q.get("genre[]", [])]
        years = [int(y) for y in q.get("year[]", []) if y.isdigit()]
        types = [t for t in q.get("type[]", [])]
        statuses = [s for s in q.get("status[]", [])]
        order = q.get("order", ["default"])[0]
        page = int(q.get("page", [1])[0])
        return genres, years, types, statuses, order, page
    except Exception:
        return [], [], [], [], "default", 1

# ==========================
#   PREFERENCIAS / LOCALSTORAGE
# ==========================
def _ls_get(key: str, fallback_key: str) -> Dict:
    if f"cache_{key}" in st.session_state:
        return st.session_state[f"cache_{key}"]
    raw = "{}"
    if get_local_storage is not None:
        try:
            raw = get_local_storage(key, default="{}") or "{}"
        except Exception:
            raw = "{}"
    else:
        raw = st.session_state.get(f"_ls_fallback_{fallback_key}", "{}")
    try:
        data = json.loads(raw)
    except Exception:
        data = {}
    st.session_state[f"cache_{key}"] = data
    return data

def _ls_put(key: str, fallback_key: str, data: Dict):
    st.session_state[f"cache_{key}"] = data
    raw = json.dumps(data, ensure_ascii=False)
    if set_local_storage is not None:
        try: set_local_storage(key, raw)
        except Exception: pass
    else:
        st.session_state[f"_ls_fallback_{fallback_key}"] = raw

def _prefs_defaults():
    return {
        "default_player_h": 540,
        "prefer_no_ads": True,
        "host_priority":"streamtape.com,streamwish.to,ok.ru,mega.nz",
        "open_newtab_default": False,
        "dedup_mirrors": True,
    }

def _prefs_load() -> Dict:
    data = _ls_get(LS_KEY_PREFS, "prefs")
    for k, v in _prefs_defaults().items(): data.setdefault(k, v)
    return data

def _prefs_save(data: Dict): _ls_put(LS_KEY_PREFS, "prefs", data)

def _ls_seen_load() -> Dict:
    data = _ls_get(LS_KEY_SEEN, "seen")
    if "animes" not in data: data["animes"] = {}
    return data

def _ls_seen_save(data: Dict): _ls_put(LS_KEY_SEEN, "seen", data)

def _ls_watch_load() -> Dict[str, Dict]:
    data = _ls_get(LS_KEY_WATCH, "watch")
    if "animes" not in data: data["animes"] = {}
    return data

def _ls_watch_save(data: Dict[str, Dict]): _ls_put(LS_KEY_WATCH, "watch", data)

# ==========================
#   HISTORIAL (vistos) y FAVORITOS
# ==========================
def seen_add_episode(slug: str, title: str, anime_url: str, ep_num: int, ep_url: str, image: Optional[str] = None):
    data = _ls_seen_load()
    animes = data.setdefault("animes", {})
    a = animes.setdefault(slug, {"title": title, "url": anime_url, "image": image, "episodes": {}, "last_seen": 0})
    a["title"] = a.get("title") or title
    a["url"] = anime_url
    if image: a["image"] = image
    ts = int(time.time()); a["last_seen"] = ts
    a["episodes"][str(ep_num)] = {"url": ep_url, "ts": ts}
    _ls_seen_save(data)

def next_episode_for(slug: str) -> Optional[int]:
    data = _ls_seen_load().get("animes", {}).get(slug)
    if not data: return None
    vistos = sorted(int(k) for k in data.get("episodes", {}))
    return (vistos[-1] + 1) if vistos else None

def episode_seen(slug: str, num: int) -> bool:
    data = _ls_seen_load().get("animes", {}).get(slug, {})
    return str(num) in data.get("episodes", {})

def seen_delete_anime(slug: str):
    data = _ls_seen_load()
    data.get("animes", {}).pop(slug, None)
    _ls_seen_save(data); st.rerun()

def seen_clear_all():
    _ls_seen_save({"animes": {}}); st.rerun()

def watch_toggle(slug: str, title: str, url: str, image: Optional[str]):
    data = _ls_watch_load()
    if slug in data["animes"]: data["animes"].pop(slug, None)
    else: data["animes"][slug] = {"title": title, "url": url, "image": image, "ts": int(time.time())}
    _ls_watch_save(data)

def is_watchlisted(slug: str) -> bool: return slug in _ls_watch_load().get("animes", {})

# ==========================
#   RUTEO (query params)
# ==========================
def _restore_from_query_params():
    if st.session_state.get("_qp_restored"): return
    qp = st.query_params
    if qp:
        st.session_state["mode"] = qp.get("mode", st.session_state.get("mode", "home"))
        _ensure_dir_state()
        st.session_state["dir_q"]     = qp.get("q", st.session_state["dir_q"])
        st.session_state["dir_order"] = qp.get("order", st.session_state["dir_order"])
        st.session_state["dir_page"]  = int(qp.get("page", st.session_state["dir_page"]))
        st.session_state["dir_genres"] = [x for x in qp.get("genres","").split(",") if x] or st.session_state["dir_genres"]
        st.session_state["dir_years"]  = [int(x) for x in qp.get("years","").split(",") if x.isdigit()] or st.session_state["dir_years"]
        st.session_state["dir_types"]  = [x for x in qp.get("types","").split(",") if x] or st.session_state["dir_types"]
        st.session_state["dir_status"] = [x for x in qp.get("status","").split(",") if x] or st.session_state["dir_status"]
        st.session_state["anime_url"]  = qp.get("anime", st.session_state.get("anime_url"))
        st.session_state["episode_url"]= qp.get("ep", st.session_state.get("episode_url"))
    st.session_state["_qp_restored"] = True

def _sync_query_params():
    try:
        st.query_params.update({
            "mode":   st.session_state.get("mode","home"),
            "q":      st.session_state.get("dir_q",""),
            "order":  st.session_state.get("dir_order","default"),
            "page":   str(st.session_state.get("dir_page",1)),
            "genres": ",".join(st.session_state.get("dir_genres",[])),
            "years":  ",".join(map(str, st.session_state.get("dir_years",[]))),
            "types":  ",".join(st.session_state.get("dir_types",[])),
            "status": ",".join(st.session_state.get("dir_status",[])),
            "anime":  st.session_state.get("anime_url") or "",
            "ep":     st.session_state.get("episode_url") or "",
        })
    except Exception:
        pass

# ==========================
#   UI HELPERS
# ==========================
def _copy_button(label: str, text: str, key: str):
    st_html(
        f"""
        <button id="copybtn-{key}" style="padding:6px 10px;border-radius:8px;border:1px solid #ddd;cursor:pointer;">
            {label}
        </button>
        <script>
        (function(){{
            const b = document.getElementById('copybtn-{key}');
            if(!b) return;
            b.addEventListener('click', async () => {{
                try {{
                    await navigator.clipboard.writeText({json.dumps(text)});
                    const old = b.innerText; b.innerText = '¡Copiado!';
                    setTimeout(()=>{{ b.innerText = old; }}, 1200);
                }} catch(e) {{ console.log(e); }}
            }});
        }})();
        </script>
        """,
        height=40,
    )

def _image_link(href: str, img: str, width_px: int = 160, radius: int = 10):
    st.markdown(
        f"""<a href="{href}">
                <img src="{img}" style="width:{width_px}px;max-width:100%;border-radius:{radius}px;border:1px solid #eee"/>
            </a>""",
        unsafe_allow_html=True,
    )

# ==========================
#   ANIME – VISTAS
# ==========================
def _pick_best_server(servers: List[Dict]) -> Optional[str]:
    if not servers: return None
    prefs = _prefs_load(); candidates = servers[:]
    if prefs.get("prefer_no_ads", True):
        noads = [s for s in candidates if not s.get("ads")]; candidates = noads or candidates
    pri = [h.strip().lower() for h in prefs.get("host_priority","").split(",") if h.strip()]
    if pri:
        for host in pri:
            for s in candidates:
                try: net = urlparse(s.get("link","")).netloc.lower()
                except Exception: net = ""
                if host and host in net and s.get("link"): return s["link"]
    for s in candidates:
        if s.get("link"): return s["link"]
    return None

def _group_servers_by_host(servers: List[Dict]) -> List[Dict]:
    prefs = _prefs_load()
    if not prefs.get("dedup_mirrors", True):
        return [{"_host": urlparse(s.get("link","")).netloc, "_count": 1, **s} for s in servers]
    grouped: Dict[str, Dict] = {}
    for s in servers:
        host = urlparse(s.get("link","")).netloc
        if host not in grouped: grouped[host] = {"_host": host, "_count": 0, **s}
        grouped[host]["_count"] += 1
    return list(grouped.values())

def view_home():
    prefs = _prefs_load()
    st.subheader("Continuar viendo")
    seen = _ls_seen_load().get("animes", {})
    rows = []
    for slug, a in seen.items():
        nxt = next_episode_for(slug)
        if nxt:
            rows.append((a.get("title") or slug.replace("-"," ").title(), slug, nxt,
                         abs_url(BASE_URL, f"/ver/{slug}-{nxt}"), a.get("image")))
    if rows:
        cols = st.columns(4)
        for i, (title, slug, nxt, url, img) in enumerate(sorted(rows, key=lambda t: t[2])[:8]):
            with cols[i % 4]:
                if img: st.image(img, use_container_width=True)
                st.caption(f"{title} — Episodio {nxt}")
                st.button("Reanudar ▶", key=f"resume_{slug}",
                          on_click=lambda u=url: st.session_state.update({"mode":"episode","episode_url":u,"player_url":None}))
    else:
        st.info("Aún no hay episodios para continuar.")

    st.header("En emisión / últimos episodios (semana)")
    with st.spinner("Cargando episodios recientes…"):
        html = fetch_cached(abs_url(BASE_URL, "/"))
        items = parse_home_week(html, BASE_URL, limit=24)
    if not items:
        st.info("No pude detectar episodios recientes.")
        return
    cols = st.columns(4)
    for i, it in enumerate(items):
        with cols[i % 4]:
            if it.image:
                st.image(it.image, use_container_width=True)
            st.caption(f"{it.anime_title} — Episodio {it.episode_num}")
            c1, c2 = st.columns(2)
            if c1.button("Ver episodios", key=f"h_go_{i}"):
                st.session_state.update({"mode": "detail", "anime_url": it.anime_url}); st.rerun()
            if c2.button("Ver capítulo ▶", key=f"h_ep_{i}"):
                st.session_state.update({"mode": "episode", "episode_url": it.episode_url, "player_url": None}); st.rerun()

def view_browse():
    _ensure_dir_state()
    st.header("Directorio")
    st.text_input("Buscar por título", key="dir_q", placeholder="Escribe y presiona Enter")

    # Sidebar filtros y preferencias
    with st.sidebar:
        st.subheader("Filtros")
        st.multiselect("Géneros", GENRE_SLUGS, key="dir_genres")
        st.multiselect("Años", list(reversed(YEARS)), key="dir_years")
        st.multiselect("Tipo", list(TYPE_MAP.keys()), format_func=lambda x: TYPE_MAP[x], key="dir_types")
        st.multiselect("Estado", list(STATUS_MAP.keys()), format_func=lambda x: STATUS_MAP[x], key="dir_status")
        st.selectbox("Orden", list(ORDER_MAP.keys()),
                     index=list(ORDER_MAP.keys()).index(st.session_state["dir_order"]),
                     format_func=lambda x: ORDER_MAP[x], key="dir_order")
        st.number_input("Página", min_value=1, step=1, key="dir_page")

        st.slider("Columnas (grilla)", min_value=2, max_value=7, key="dir_cols")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Aplicar"):
                st.rerun()
        with c2:
            if st.button("Limpiar filtros"):
                for k, v in _dir_defaults().items():
                    st.session_state[k] = v
                st.rerun()

        with st.expander("Importar filtros desde URL /browse"):
            raw = st.text_input("Pega la URL de AnimeFLV")
            if st.button("Importar"):
                g,y,t,s,o,p = import_filters_from_url(raw)
                st.session_state.update({"dir_genres": g, "dir_years": y, "dir_types": t, "dir_status": s, "dir_order": o, "dir_page": p}); st.rerun()

        st.divider()
        st.subheader("Preferencias Visor")
        prefs = _prefs_load()
        new_h = st.number_input("Altura por defecto (px)", min_value=320, max_value=1200, value=int(prefs.get("default_player_h", 540)))
        prefer_no_ads = st.checkbox("Preferir servidores sin ads", value=bool(prefs.get("prefer_no_ads", True)))
        host_priority = st.text_input("Prioridad hosts (coma)", value=prefs.get("host_priority",""))
        open_newtab_default = st.checkbox("Abrir directo en nueva pestaña", value=bool(prefs.get("open_newtab_default", False)))
        dedup_mirrors = st.checkbox("Agrupar mirrors por host", value=bool(prefs.get("dedup_mirrors", True)))
        if st.button("Guardar preferencias"):
            prefs.update({"default_player_h": int(new_h), "prefer_no_ads": bool(prefer_no_ads), "host_priority": host_priority, "open_newtab_default": bool(open_newtab_default), "dedup_mirrors": bool(dedup_mirrors)})
            _prefs_save(prefs); st.success("Preferencias guardadas")

    # Traer y pintar
    url = _build_browse_url()
    with st.spinner("Cargando resultados…"):
        html = fetch_cached(url)
        cards = parse_browse(html, BASE_URL)

    if not cards:
        st.info("Sin resultados.")
    else:
        n = max(2, min(7, int(st.session_state.get("dir_cols", 5))))
        cols = st.columns(n)
        for i, c in enumerate(cards):
            with cols[i % n]:
                if c.image:
                    href = _qp_href({"mode":"detail","anime":c.url})
                    _image_link(href, c.image, width_px=160)
                # título clicable
                st.markdown(f"[{c.title or 'Abrir'}]({_qp_href({'mode':'detail','anime':c.url})})")

    # Paginación rápida debajo
    pc1, _, pc3 = st.columns([1, 2, 1])
    with pc1:
        if st.button("« Anterior", disabled=st.session_state["dir_page"] <= 1):
            st.session_state["dir_page"] = max(1, st.session_state["dir_page"] - 1); st.rerun()
    with pc3:
        if st.button("Siguiente »"):
            st.session_state["dir_page"] += 1; st.rerun()

    # URL final abajo (para copiar)
    st.markdown("**URL generada:**")
    st.code(url, language="text")

def view_episode(ep_url: str):
    prefs = _prefs_load()
    m = re.search(r"/ver/([a-z0-9\-]+)-(\d+)", ep_url, re.I)
    slug, ep_num = (m.group(1), int(m.group(2))) if m else (None, None)

    with st.spinner("Cargando episodio…"):
        html_page = fetch(ep_url)
        servers = extract_episode_servers(html_page)

    title_match = re.search(r"<h1[^>]*>(.*?)</h1>", html_page, re.S)
    title_text = BeautifulSoup(title_match.group(1), "html.parser").get_text(" ", strip=True) if title_match else ep_url
    m2 = re.search(r"(.+?)\s+Episodio\s+\d+", title_text, re.I)
    anime_title = m2.group(1).strip() if m2 else None

    # Marcar visto
    if slug and ep_num is not None:
        seen_add_episode(slug, anime_title or slug.replace("-", " ").title(), abs_url(BASE_URL, f"/anime/{slug}"), ep_num, ep_url, image=None)

    # Autoselección de servidor
    if servers and not st.session_state.get("player_url"):
        st.session_state["player_url"] = _pick_best_server(servers)

    # Nav prev/next + copiar link interno al visor
    prev_url = abs_url(BASE_URL, f"/ver/{slug}-{ep_num-1}") if slug and ep_num and ep_num>1 else None
    next_url = abs_url(BASE_URL, f"/ver/{slug}-{ep_num+1}") if slug and ep_num is not None else None
    prev_q = _qp_href({"mode":"episode","ep": prev_url}) if prev_url else ""
    next_q = _qp_href({"mode":"episode","ep": next_url}) if next_url else ""
    share_q = _qp_href({"mode":"episode","ep": ep_url})

    nav = st.columns([1,1,4,2])
    with nav[0]:
        st.button("⭠ Anterior", disabled=not prev_url, on_click=lambda u=prev_url: st.session_state.update({"episode_url":u,"player_url":None}) if u else None)
    with nav[1]:
        st.button("Siguiente ⭢", disabled=not next_url, on_click=lambda u=next_url: st.session_state.update({"episode_url":u,"player_url":None}) if u else None)
    with nav[3]:
        _copy_button("Compartir este capítulo", share_q, key=f"share_ep_{slug}_{ep_num}")

    st.subheader(title_text)

    # Reproductor
    st.write("### Reproductor")
    h = st.slider("Altura del reproductor", min_value=320, max_value=1200, step=20, value=st.session_state.get("player_h", int(prefs.get("default_player_h", 540))))
    st.session_state["player_h"] = h

    player_url = st.session_state.get("player_url")
    if player_url:
        st_html(f"""
            <div id="player-wrap" style="position:relative;">
              <iframe id="the-player" src="{player_url}" width="100%" height="{h}" frameborder="0"
                      allow="autoplay; fullscreen; picture-in-picture" allowfullscreen referrerpolicy="no-referrer"></iframe>
              <button onclick="(function(){{
                  const el = document.getElementById('the-player');
                  if (el && el.requestFullscreen) el.requestFullscreen();
                  else if (el && el.webkitRequestFullscreen) el.webkitRequestFullscreen();
                  else if (el && el.msRequestFullscreen) el.msRequestFullscreen();
              }})()" style="position:absolute;right:10px;top:10px;padding:6px 10px;border-radius:8px;border:0;background:#0ea5e9;color:white;cursor:pointer;">
                Pantalla completa (F)
              </button>
            </div>
            <div id="navkeys" data-prev="{prev_q}" data-next="{next_q}"></div>
            <script>
            document.addEventListener('keydown', function(e){{
              const n = document.getElementById('navkeys');
              if(!n) return;
              if(e.key==='ArrowRight') {{
                const u = n.dataset.next; if(u) window.location.search = u.substring(1);
              }} else if(e.key==='ArrowLeft') {{
                const u = n.dataset.prev; if(u) window.location.search = u.substring(1);
              }} else if(e.key==='f' || e.key==='F') {{
                const el = document.getElementById('the-player');
                if(!el) return;
                (el.requestFullscreen||el.webkitRequestFullscreen||el.msRequestFullscreen||function(){{}}).call(el);
              }}
            }});
            </script>
            """, height=h+60)
        st.markdown(f"<div style='text-align:right;margin-top:6px;'><a href='{player_url}' target='_blank' rel='noopener'>Abrir en pestaña nueva ↗</a></div>", unsafe_allow_html=True)
    else:
        st.info("Elige un servidor para reproducir.")

    # Lista de servidores
    st.markdown("### Servidores")
    if not servers:
        st.info("No se encontraron servidores para este episodio."); return
    grouped = _group_servers_by_host(servers)
    col_a, col_b = st.columns(2); cols = [col_a, col_b]
    for idx, s in enumerate(grouped):
        with cols[idx % 2]:
            badge = "🟡 con ads" if s.get("ads") else "🟢 sin ads"
            mirrors = f" · {s.get('_count',1)} mirror(s)" if s.get("_count",1)>1 else ""
            host = s.get("_host","")
            link = s.get("link","")
            st.markdown(f"**{s.get('title','')}** — `{s.get('server','')}` — `{host}` {mirrors} — {badge}  \n[Abrir enlace ↗]({link})")
            btns = st.columns([1,1,1,3])
            with btns[0]:
                st.button("Ver aquí", key=f"play_{idx}", on_click=lambda url=link: st.session_state.update({"player_url": url}), disabled=not link)
            with btns[1]:
                _copy_button("Copiar enlace", link, key=f"copy_{idx}")
            with btns[2]:
                if link and _prefs_load().get("open_newtab_default", False):
                    st.markdown(f"[Abrir pestaña]({link})")
            st.divider()

def view_anime(url: str):
    with st.spinner("Cargando ficha…"):
        html = fetch_cached(url); detail = parse_anime_detail(html, url, BASE_URL)

    mslug = re.search(r"/anime/([a-z0-9\-]+)", url); slug = mslug.group(1) if mslug else None
    watchlisted = is_watchlisted(slug) if slug else False

    # Barra superior
    btns = st.columns([1,2,2,2,3])
    with btns[0]:
        st.button("← Directorio", on_click=lambda: st.session_state.update({"mode": "browse", "anime_url": None}))
    with btns[1]:
        if slug:
            st.button("⭐ Favorito" if not watchlisted else "★ Quitar",
                      on_click=lambda s=slug, t=detail.title, u=url, img=detail.image: watch_toggle(s,t,u,img))
    with btns[2]:
        # Reproducir último
        if detail.episodes:
            last_ep = max(e.number for e in detail.episodes)
            st.button(f"▶ Último (Ep {last_ep})",
                      on_click=lambda u=abs_url(BASE_URL, f"/ver/{slug}-{last_ep}"): st.session_state.update({"mode":"episode","episode_url":u,"player_url":None}))
    with btns[3]:
        # Continuar (siguiente no visto)
        nxt = next_episode_for(slug) if slug else None
        if nxt:
            st.button(f"⏭ Continuar (Ep {nxt})",
                      on_click=lambda u=abs_url(BASE_URL, f"/ver/{slug}-{nxt}"): st.session_state.update({"mode":"episode","episode_url":u,"player_url":None}))
    with btns[4]:
        # Compartir link interno a esta ficha
        share_detail = _qp_href({"mode":"detail","anime":url})
        _copy_button("Compartir ficha", share_detail, key=f"share_detail_{slug}")

    cols = st.columns([1, 3])
    with cols[0]:
        if detail.image: st.image(detail.image, width=220, use_container_width=False)
    with cols[1]:
        st.subheader(detail.title)
        if detail.genres: st.write("Géneros: " + ", ".join(detail.genres))
        if detail.description: st.write(detail.description)

    st.markdown("### Lista de episodios")
    if not detail.episodes:
        st.info("No se detectaron episodios.")
    else:
        for ep in sorted(detail.episodes, key=lambda e: e.number, reverse=True):
            visto = episode_seen(slug, ep.number) if slug else False
            cols = st.columns([3,1,1])
            cols[0].write(f"Episodio {ep.number} {'✅' if visto else '🟦'}")
            if ep.url:
                if cols[1].button("Ver en visor", key=f"ep_{ep.number}"):
                    st.session_state["mode"] = "episode"
                    st.session_state["episode_url"] = ep.url
                    st.session_state["player_url"] = None
                    if slug:
                        seen_add_episode(slug, detail.title, abs_url(BASE_URL, f"/anime/{slug}"), ep.number, ep.url, image=detail.image)
                    st.rerun()
                cols[2].link_button("Directo ↗", ep.url)
            else:
                cols[1].write("-")

def view_seen():
    st.header("Vistos")
    data = _ls_seen_load(); animes: Dict[str, Dict] = data.get("animes", {})
    if not animes:
        st.info("Aún no hay animes/episodios vistos.")
    else:
        items = sorted(animes.items(), key=lambda kv: kv[1].get("last_seen", 0), reverse=True)
        for slug, a in items:
            title = a.get("title") or slug.replace("-", " ").title()
            image = a.get("image"); url = a.get("url") or abs_url(BASE_URL, f"/anime/{slug}")
            eps = a.get("episodes", {})
            with st.expander(f"{title} — {len(eps)} episodios vistos"):
                if image: st.image(image, width=220)
                st.markdown(f"[Ver ficha del anime]({_qp_href({'mode':'detail','anime':url})})")
                ep_items = sorted(((int(k), v) for k, v in eps.items()), key=lambda x: x[0], reverse=True)
                cols = st.columns(4)
                for i, (num, info) in enumerate(ep_items):
                    with cols[i % 4]:
                        st.write(f"Episodio {num}")
                        st.link_button("Abrir", _qp_href({"mode":"episode","ep":info.get('url', '#')}))
                st.button("Quitar de vistos", key=f"del_{slug}", on_click=lambda s=slug: seen_delete_anime(s))

    st.divider()
    st.subheader("Exportar / Importar historial")
    data_json = json.dumps(_ls_seen_load(), ensure_ascii=False, indent=2)
    st.download_button("Exportar vistos (JSON)", data=data_json, file_name="vistos_animeflv.json", mime="application/json")
    imp = st.text_area("Pega aquí un JSON para importar/combinar")
    c1, _ = st.columns([1,4])
    with c1:
        if st.button("Importar"):
            try:
                new = json.loads(imp)
                merged = _ls_seen_load(); merged.setdefault("animes", {})
                for slug, v in new.get("animes", {}).items():
                    merged["animes"].setdefault(slug, v)
                    merged["animes"][slug].setdefault("episodes", {})
                    merged["animes"][slug]["episodes"].update(v.get("episodes", {}))
                    merged["animes"][slug]["title"] = merged["animes"][slug].get("title") or v.get("title")
                    merged["animes"][slug]["url"] = merged["animes"][slug].get("url") or v.get("url")
                    if v.get("image"): merged["animes"][slug]["image"] = v.get("image")
                _ls_seen_save(merged); st.success("Importado/Combinado correctamente.")
            except Exception:
                st.error("JSON inválido")
    st.button("Vaciar historial", on_click=seen_clear_all)

def view_watchlist():
    st.header("Favoritos / Ver más tarde")
    data = _ls_watch_load().get("animes", {})
    if not data:
        st.info("No tienes animes en favoritos."); return
    items = sorted(data.items(), key=lambda kv: kv[1].get("ts", 0), reverse=True)
    n = 5
    cols = st.columns(n)
    for i, (slug, a) in enumerate(items):
        with cols[i % n]:
            if a.get("image"):
                _image_link(_qp_href({"mode":"detail","anime":a.get("url","")}), a["image"], width_px=160)
            st.caption(a.get("title", slug.replace("-", " ").title()))
            c1, c2 = st.columns(2)
            c1.button("Abrir", key=f"wl_open_{slug}", on_click=lambda u=a.get("url"): st.session_state.update({"mode":"detail","anime_url":u}))
            c2.button("Quitar", key=f"wl_rm_{slug}", on_click=lambda s=slug: (watch_toggle(s,"","",None), st.rerun()))

# ==========================
#   APP
# ==========================
def main():
    st.set_page_config(page_title="Anime Navigator", layout="wide")

    # Estado inicial
    if "mode" not in st.session_state:
        st.session_state.update({
            "mode": "home",
            "anime_url": None, "episode_url": None, "player_url": None, "player_h": 540,
        })
    _restore_from_query_params()

    # Navegación lateral
    st.sidebar.title("Anime Navigator")
    col_nav1, col_nav2, col_nav3, col_nav4 = st.sidebar.columns(4)
    if col_nav1.button("Inicio"):
        st.session_state["mode"] = "home";  st.rerun()
    if col_nav2.button("Directorio"):
        st.session_state["mode"] = "browse";st.rerun()
    if col_nav3.button("Vistos"):
        st.session_state["mode"] = "seen";  st.rerun()
    if col_nav4.button("Favoritos"):
        st.session_state["mode"] = "watch"; st.rerun()

    mode = st.session_state.get("mode", "home")
    if mode == "detail" and st.session_state.get("anime_url"):
        view_anime(st.session_state["anime_url"])
    elif mode == "episode" and st.session_state.get("episode_url"):
        view_episode(st.session_state["episode_url"])
    elif mode == "browse":
        view_browse()
    elif mode == "seen":
        view_seen()
    elif mode == "watch":
        view_watchlist()
    else:
        view_home()

    _sync_query_params()

if __name__ == "__main__":
    main()
