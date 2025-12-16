import re
import json
import time
import functools
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from urllib.parse import urljoin, urlencode

import requests
from bs4 import BeautifulSoup
import streamlit as st
from streamlit.components.v1 import html as st_html

# Persistencia opcional en localStorage. Si no está disponible, usa fallback en sesión.
try:
    from streamlit_js_eval import get_local_storage, set_local_storage  # pip install streamlit-js-eval
except Exception:  # pragma: no cover
    get_local_storage = None
    set_local_storage = None

BASE_URL = "https://www3.animeflv.net"
LS_KEY = "aflv_nav_seen_v1"   # clave en localStorage


# ---------------- Navegación interna (historial + atajos) ----------------
NAV_STATE_KEYS = ["mode", "anime_url", "episode_url", "player_url", "player_h"]


def _ensure_nav_stacks():
    st.session_state.setdefault("nav_history", [])  # pila para atrás
    st.session_state.setdefault("nav_future", [])   # pila para adelante


def _current_nav_state() -> Dict:
    return {k: st.session_state.get(k) for k in NAV_STATE_KEYS}


def _apply_nav_state(state: Dict):
    st.session_state.update({k: state.get(k) for k in NAV_STATE_KEYS})


def _navigate_to(mode: str, **kwargs):
    """Actualiza el modo guardando el estado anterior para un botón Atrás/Adelante."""
    _ensure_nav_stacks()
    st.session_state["nav_history"].append(_current_nav_state())
    st.session_state["nav_future"].clear()
    new_state = _current_nav_state()
    new_state.update(kwargs)
    new_state["mode"] = mode
    if mode != "episode":
        new_state["player_url"] = None
    _apply_nav_state(new_state)
    st.rerun()


def _nav_back():
    _ensure_nav_stacks()
    if not st.session_state["nav_history"]:
        return
    st.session_state["nav_future"].append(_current_nav_state())
    prev = st.session_state["nav_history"].pop()
    _apply_nav_state(prev)
    st.rerun()


def _nav_forward():
    _ensure_nav_stacks()
    if not st.session_state["nav_future"]:
        return
    st.session_state["nav_history"].append(_current_nav_state())
    nxt = st.session_state["nav_future"].pop()
    _apply_nav_state(nxt)
    st.rerun()


def _consume_nav_param():
    """Permite que la URL ?nav=back|forward active los controles de historial."""
    params = st.experimental_get_query_params()
    action = None
    if "nav" in params and params["nav"]:
        action = params["nav"][0]
        params.pop("nav", None)
        st.experimental_set_query_params(**params)
    if action == "back":
        _nav_back()
    elif action == "forward":
        _nav_forward()


def _inject_nav_shortcuts():
    """JS para interceptar botón atrás del navegador y atajos de teclado (←/→)."""
    if st.session_state.get("_nav_js_injected"):
        return
    st.session_state["_nav_js_injected"] = True
    st_html(
        """
        <script>
        (function() {
            const sendNav = (action) => {
                const url = new URL(window.location.href);
                url.searchParams.set('nav', action);
                window.location.replace(url.toString());
            };
            // Atajos de teclado
            document.addEventListener('keydown', (ev) => {
                if (ev.key === 'ArrowLeft') { sendNav('back'); }
                else if (ev.key === 'ArrowRight') { sendNav('forward'); }
            }, { passive: true });
            // Evita salir del sitio con el botón atrás del navegador
            window.history.pushState({}, '', window.location.href);
            window.addEventListener('popstate', () => sendNav('back'));
        })();
        </script>
        """,
        height=0,
    )


# ---------------- Estado de filtros (Directorio) ----------------
def _dir_defaults():
    return {
        "dir_genres": [],           # slugs: ["comedia", "fantasia", ...]
        "dir_years": [],            # [2008, 2009, ...]
        "dir_types": [],            # ["tv","movie","ova","special"]
        "dir_status": [],           # ["1","2","3"]
        "dir_order": "default",     # "default" | "updated" | "added" | "name" | "rating"
        "dir_page": 1,
        "dir_q": "",                # búsqueda por título
    }

def _ensure_dir_state():
    for k, v in _dir_defaults().items():
        st.session_state.setdefault(k, v)

def HEADERS():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Referer": BASE_URL,
        "Connection": "keep-alive",
    }

def fetch(url: str, params: Optional[dict] = None) -> str:
    r = requests.get(url, params=params, headers=HEADERS(), timeout=20)
    r.raise_for_status()
    return r.text

def abs_url(base: str, path: str) -> str:
    return urljoin(base, path or "")


# ---------------- Servidores del episodio ----------------
def extract_episode_servers(html: str):
    """
    Devuelve lista de dicts:
    {lang, server, title, ads, allow_mobile, link}
    Prioriza 'url' y si no existe usa 'code'.
    """
    m = re.search(r'var\s+videos\s*=\s*(\{.*?});', html, re.DOTALL)
    if not m:
        return []
    raw = m.group(1)
    data = json.loads(raw)
    servers = []
    for lang, items in data.items():
        for it in items:
            link = it.get("url") or it.get("code")
            servers.append(
                {
                    "lang": lang,
                    "server": it.get("server"),
                    "title": it.get("title"),
                    "ads": bool(it.get("ads", 0)),
                    "allow_mobile": it.get("allow_mobile", True),
                    "link": link,
                }
            )
    return servers


# ---------------- Modelos ----------------
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


# ---------------- Parseadores ----------------
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
            img = (
                img_tag.get("data-src")
                or img_tag.get("data-original")
                or img_tag.get("src")
            )
        cards.append(
            Card(
                title=title or "",
                url=abs_url(base_url, href),
                image=abs_url(base_url, img) if img else None,
            )
        )
    return cards

def parse_anime_detail(html: str, page_url: str, base_url: str) -> AnimeDetail:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.select_one(".Ficha .Title") or soup.find("h1")).get_text(strip=True)
    img_tag = soup.select_one(".AnimeCover .Image img") or soup.select_one(
        "meta[property='og:image']"
    )
    image = None
    if img_tag:
        src = img_tag.get("src") or img_tag.get("content")
        image = abs_url(base_url, src)
    genres = [a.get_text(strip=True) for a in soup.select(".Nvgnrs a")]
    desc_tag = soup.select_one(".Description") or soup.find("p")
    description = desc_tag.get_text(" ", strip=True) if desc_tag else ""
    episodes: List[Episode] = []
    m = re.search(r"var\s+episodes\s*=\s*(\[\[.*?]]);", html, re.S)
    slug = None
    ms = re.search(
        r'var\s+anime_info\s*=\s*\[\s*"[^\"]*",\s*"[^\"]*",\s*"([^\"]+)"', html
    )
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
    return AnimeDetail(
        title=title, image=image, genres=genres, description=description, episodes=episodes
    )


# --------- HOME: “de la semana / últimos episodios” ---------
def _clean_home_title(text: str) -> str:
    """Remueve prefijos/sufijos 'Episodio N' del texto recibido y normaliza espacios."""
    if not text:
        return text
    t = re.sub(r'^\s*Episodio\s*\d+\s*[-:–—]?\s*', '', text, flags=re.I)
    t = re.sub(r'\s*[-:–—]?\s*Episodio\s*\d+\s*$', '', t, flags=re.I)
    return ' '.join(t.split())

def parse_home_week(html: str, base_url: str, limit: int = 24) -> List[HomeEp]:
    soup = BeautifulSoup(html, "html.parser")
    items: List[HomeEp] = []
    seen = set()
    candidates = soup.select(".ListEpisodios a[href^='/ver/'], .ListCaps a[href^='/ver/']")
    if not candidates:
        candidates = soup.select("a[href^='/ver/']")
    rx = re.compile(r"^/ver/([a-z0-9\-]+)-(\d+)$", re.I)
    for a in candidates:
        href = a.get("href") or ""
        m = rx.match(href.strip())
        if not m or href in seen:
            continue
        seen.add(href)
        slug, num = m.group(1), int(m.group(2))
        anime_url = abs_url(base_url, f"/anime/{slug}")
        episode_url = abs_url(base_url, href)
        raw_title = a.get("title") or a.get_text(strip=True) or slug.replace("-", " ").title()
        title = _clean_home_title(raw_title) or slug.replace("-", " ").title()
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


# ---------- Filtros/constantes (Directorio) ----------
GENRE_SLUGS = [
    "accion", "artes-marciales", "aventuras", "carreras", "ciencia-ficcion",
    "comedia", "deportes", "drama", "ecchi", "escolar", "espacial", "fantasia",
    "harem", "historico", "infantil", "josei", "juegos", "magia", "militar", "misterio",
    "musica", "parodia", "policia", "psicologico", "recuerdos-de-la-vida",
    "romance", "samurai", "seinen", "shoujo", "shounen", "sobrenatural",
    "superpoderes", "suspenso", "terror", "vampiros", "yaoi", "yuri", "demencia",
]
TYPE_MAP   = {"tv": "TV", "movie": "Película", "ova": "OVA", "special": "Especial"}
STATUS_MAP = {"1": "En emisión", "2": "Finalizado", "3": "Próximamente"}
ORDER_MAP  = {
    "default": "Por Defecto",
    "updated": "Recientemente Actualizados",
    "added":  "Recientemente Agregados",
    "name":   "Nombre A-Z",     # (el sitio acepta 'name' para ordenar por título)
    "rating": "Calificación",
}
YEARS = list(range(1990, 2026))

def _build_browse_url(base=BASE_URL):
    """Construye la URL /browse con todos los filtros + q + page."""
    p: Dict[str, List[str] | str | int] = {}
    ss = st.session_state
    if ss["dir_q"].strip():
        p["q"] = ss["dir_q"].strip()
    if ss["dir_genres"]:
        p["genre[]"] = ss["dir_genres"]
    if ss["dir_years"]:
        p["year[]"] = [str(y) for y in ss["dir_years"]]
    if ss["dir_types"]:
        p["type[]"] = ss["dir_types"]
    if ss["dir_status"]:
        p["status[]"] = ss["dir_status"]
    if ss["dir_order"] and ss["dir_order"] != "default":
        p["order"] = ss["dir_order"]
    else:
        p["order"] = "default"
    if ss["dir_page"] and ss["dir_page"] != 1:
        p["page"] = ss["dir_page"]
    query = urlencode(p, doseq=True)
    return abs_url(base, "/browse") + (f"?{query}" if query else "")


# ---------------- Persistencia "Vistos" (localStorage + fallback) ----------------
def _ls_load() -> Dict:
    """Lee JSON desde localStorage (si hay lib) o desde la sesión."""
    if "seen_cache" in st.session_state:
        return st.session_state["seen_cache"]
    raw = "{}"
    if get_local_storage is not None:
        try:
            raw = get_local_storage(LS_KEY, default="{}") or "{}"
        except Exception:
            raw = "{}"
    else:
        raw = st.session_state.get("_ls_fallback", "{}")
    try:
        data = json.loads(raw)
    except Exception:
        data = {}
    if "animes" not in data:
        data["animes"] = {}
    # Normalizar episodios antiguos que no tengan bandera de confirmación
    for a in data.get("animes", {}).values():
        eps = a.get("episodes", {})
        for num, info in eps.items():
            if isinstance(info, dict):
                info.setdefault("confirmed", False)
    st.session_state["seen_cache"] = data
    return data

def _ls_save(data: Dict):
    st.session_state["seen_cache"] = data
    raw = json.dumps(data)
    if set_local_storage is not None:
        try:
            set_local_storage(LS_KEY, raw)
        except Exception:
            pass
    else:
        st.session_state["_ls_fallback"] = raw

def seen_add_episode(
    slug: str,
    title: str,
    anime_url: str,
    ep_num: int,
    ep_url: str,
    image: Optional[str] = None,
    confirmed: bool = False,
):
    data = _ls_load()
    animes = data.setdefault("animes", {})
    a = animes.setdefault(
        slug, {"title": title, "url": anime_url, "image": image, "episodes": {}, "last_seen": 0}
    )
    a["title"] = title or a.get("title")
    a["url"] = anime_url or a.get("url")
    if image:
        a["image"] = image
    ts = int(time.time())
    a["last_seen"] = ts
    ep_entry = a["episodes"].get(str(ep_num), {})
    ep_entry.update({"url": ep_url, "ts": ts})
    ep_entry["confirmed"] = confirmed or ep_entry.get("confirmed", False)
    a["episodes"][str(ep_num)] = ep_entry
    _ls_save(data)


def seen_confirm_episode(slug: str, ep_num: int):
    data = _ls_load()
    anime = data.get("animes", {}).get(slug)
    if not anime:
        return
    ep_entry = anime.get("episodes", {}).get(str(ep_num))
    if not ep_entry:
        return
    ep_entry["confirmed"] = True
    ep_entry["ts"] = int(time.time())
    anime["last_seen"] = max(anime.get("last_seen", 0), ep_entry["ts"])
    _ls_save(data)


def seen_remove_episode(slug: str, ep_num: int):
    data = _ls_load()
    anime = data.get("animes", {}).get(slug)
    if not anime:
        return
    episodes = anime.get("episodes", {})
    removed = episodes.pop(str(ep_num), None)
    if not removed:
        return
    remaining_ts = max((ep.get("ts", 0) for ep in episodes.values()), default=0)
    anime["last_seen"] = remaining_ts
    _ls_save(data)

def seen_delete_anime(slug: str):
    data = _ls_load()
    data.get("animes", {}).pop(slug, None)
    _ls_save(data)
    st.rerun()

def seen_clear_all():
    _ls_save({"animes": {}})
    st.rerun()


# ---------------- Utilidades de episodios (prev/next) ----------------
@functools.lru_cache(maxsize=256)
def get_anime_detail_cached(slug: str) -> AnimeDetail:
    """Descarga y cachea la ficha del anime para cálculos de navegación y metadatos."""
    html = fetch(abs_url(BASE_URL, f"/anime/{slug}"))
    return parse_anime_detail(html, abs_url(BASE_URL, f"/anime/{slug}"), BASE_URL)

def get_prev_next_urls(slug: str, current_num: int) -> Tuple[Optional[str], Optional[str], Optional[AnimeDetail]]:
    """Devuelve (prev_url, next_url, detail) para el episodio actual."""
    try:
        detail = get_anime_detail_cached(slug)
    except Exception:
        return None, None, None
    num_to_url = {e.number: e.url for e in detail.episodes if e.url}
    if not num_to_url:
        return None, None, detail
    nums = sorted(num_to_url.keys())
    if current_num not in num_to_url and nums:
        # Si el número no está (caso raro), ubicamos posición relativa
        nums.append(current_num)
        nums = sorted(set(nums))
    idx = nums.index(current_num)
    prev_url = num_to_url.get(nums[idx-1]) if idx > 0 else None
    next_url = num_to_url.get(nums[idx+1]) if idx < len(nums)-1 else None
    return prev_url, next_url, detail


# ---------------- Vistas ----------------
def view_home():
    st.header("En emisión / últimos episodios (semana)")
    with st.spinner("Cargando…"):
        html = fetch(abs_url(BASE_URL, "/"))
    items = parse_home_week(html, BASE_URL, limit=24)
    if not items:
        st.info("No pude detectar episodios recientes.")
        return
    cols = st.columns(4)
    for i, it in enumerate(items):
        with cols[i % 4]:
            if it.image:
                st.image(it.image, width=300)  # sin use_container_width
            st.caption(f"{it.anime_title} — Episodio {it.episode_num}")
            c1, c2 = st.columns(2)
            if c1.button("Ver episodios", key=f"h_go_{i}", help="Ir a la ficha del anime"):
                _navigate_to("detail", anime_url=it.anime_url)
            if c2.button("Ver capítulo ▸", key=f"h_ep_{i}", help="Abrir el episodio en el visor"):
                _navigate_to("episode", episode_url=it.episode_url, player_url=None)


def view_browse():
    _ensure_dir_state()

    # --- Búsqueda por título en la página ---
    st.header("Directorio")
    def _trigger_search():
        st.session_state["dir_page"] = 1
        st.rerun()

    st.text_input(
        "Buscar por título",
        key="dir_q",
        placeholder="Escribe para filtrar o pulsa el botón",
        on_change=_trigger_search,
        help="El filtrado se aplica automáticamente al escribir o con el botón Buscar.",
    )
    st.button("🔍 Buscar", on_click=_trigger_search)

    # --- Filtros en la IZQUIERDA (sidebar) ---
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

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Aplicar"):
                st.rerun()
        with c2:
            if st.button("Limpiar filtros"):
                for k, v in _dir_defaults().items():
                    st.session_state[k] = v
                st.rerun()

    # --- Traer y pintar ---
    url = _build_browse_url()
    with st.spinner("Buscando…"):
        html = fetch(url)
    cards = parse_browse(html, BASE_URL)

    if not cards:
        st.info("Sin resultados.")
    else:
        cols = st.columns(5)
        for i, c in enumerate(cards):
            with cols[i % 5]:
                if c.image:
                    st.image(c.image, width=160)
                st.caption(c.title or "")
                if st.button("Ver", key=f"ver_{i}", help="Abrir ficha del anime"):
                    _navigate_to("detail", anime_url=c.url)

    # --- Paginación segura (callbacks) ---
    def _prev_page():
        st.session_state["dir_page"] = max(1, int(st.session_state.get("dir_page", 1)) - 1)

    def _next_page():
        st.session_state["dir_page"] = int(st.session_state.get("dir_page", 1)) + 1

    pc1, pc2, pc3 = st.columns([1, 2, 1])
    with pc1:
        st.button("« Anterior", on_click=_prev_page, disabled=int(st.session_state.get("dir_page", 1)) <= 1, key="btn_prev")
    with pc3:
        st.button("Siguiente »", on_click=_next_page, key="btn_next")

    # URL final abajo (para copiar)
    st.markdown("**URL generada:**")
    st.code(url, language="text")


def view_episode(ep_url: str):
    # Marcar como visto + preparar navegación
    m = re.search(r"/ver/([a-z0-9\-]+)-(\d+)", ep_url, re.I)
    slug, ep_num = (m.group(1), int(m.group(2))) if m else (None, None)

    # Obtener info del anime + prev/next
    detail_for_seen: Optional[AnimeDetail] = None
    prev_url = next_url = None
    if slug and ep_num is not None:
        prev_url, next_url, detail_for_seen = get_prev_next_urls(slug, ep_num)

    with st.spinner("Cargando episodio…"):
        html_page = fetch(ep_url)
    servers = extract_episode_servers(html_page)

    # Título
    title_match = re.search(r"<h1[^>]*>(.*?)</h1>", html_page, re.S)
    title_text = BeautifulSoup(title_match.group(1), "html.parser").get_text(" ", strip=True) if title_match else ep_url

    # Marcar visto (usando metadatos reales si los tenemos)
    if slug and ep_num is not None:
        title_for_seen = (detail_for_seen.title if detail_for_seen else slug.replace("-", " ").title())
        image_for_seen = (detail_for_seen.image if detail_for_seen else None)
        seen_add_episode(
            slug,
            title_for_seen,
            abs_url(BASE_URL, f"/anime/{slug}"),
            ep_num,
            ep_url,
            image=image_for_seen,
            confirmed=False,
        )

    # autoselección: primer servidor con link
    if servers and not st.session_state.get("player_url"):
        for s in servers:
            if s.get("link"):
                st.session_state["player_url"] = s["link"]
                break

    # Controles superiores
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.button("◂ Anterior", disabled=not prev_url, on_click=lambda url=prev_url: _go_episode(url), help="Episodio anterior")
    target_anime_url = abs_url(BASE_URL, f"/anime/{slug}") if slug else st.session_state.get("anime_url")
    with c2:
        st.button(
            "Volver al anime",
            disabled=not target_anime_url,
            on_click=lambda: _navigate_to("detail", anime_url=target_anime_url),
        )
    with c3:
        st.button("Siguiente ▸", disabled=not next_url, on_click=lambda url=next_url: _go_episode(url), help="Próximo episodio")

    st.subheader(title_text)

    # Estado de visto + reconfirmación manual
    seen_info = _ls_load().get("animes", {}).get(slug or "", {}).get("episodes", {})
    current_seen = seen_info.get(str(ep_num), {}) if ep_num is not None else {}
    status = "✅ Confirmado" if current_seen.get("confirmed") else "👁️ Marcado (falta confirmar)"
    st.caption(status)
    if slug and ep_num is not None:
        st.button(
            "Confirmar que ya lo viste",
            disabled=current_seen.get("confirmed", False),
            on_click=lambda s=slug, n=ep_num: (seen_confirm_episode(s, n), st.rerun()),
            help="Úsalo para reafirmar que terminaste el episodio.",
        )

    # Reproductor
    st.write("### Reproductor")
    h = st.slider(
        "Altura del reproductor",
        min_value=320,
        max_value=900,
        step=20,
        value=st.session_state.get("player_h", 540),
        help="Ajusta la altura del iframe. Si ves 'Sandboxed embed is not allowed', prueba otro servidor o usa Abrir en pestaña nueva.",
    )
    st.session_state["player_h"] = h

    player_url = st.session_state.get("player_url")
    if player_url:
        st_html(
            f"""
            <div id="player-wrap" style="position:relative;">
              <iframe id="the-player" src="{player_url}" width="100%" height="{h}" frameborder="0"
                      allow="autoplay; fullscreen; picture-in-picture" allowfullscreen
                      referrerpolicy="no-referrer"></iframe>
              <button onclick="(function(){{
                  const el = document.getElementById('the-player');
                  if (el.requestFullscreen) el.requestFullscreen();
                  else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
                  else if (el.msRequestFullscreen) el.msRequestFullscreen();
              }})()"
                style="position:absolute;right:10px;top:10px;padding:6px 10px;border-radius:8px;border:0;background:#0ea5e9;color:white;cursor:pointer;">
                Pantalla completa
              </button>
            </div>
            """,
            height=h + 50,
        )
        st.markdown(
            f"<div style='text-align:right;margin-top:6px;'>"
            f"<a href='{player_url}' target='_blank' rel='noopener'>Abrir en pestaña nueva ↗</a>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.info(
            "Si el visor muestra mensajes como 'Sandboxed embed is not allowed' o la página del servidor se cae, abre el enlace directo o prueba otro servidor de la lista."
        )
    else:
        st.info("Elige un servidor para reproducir.")

    # Lista de servidores (2 columnas, compacta)
    st.markdown("### Servidores")
    if not servers:
        st.info("No se encontraron servidores para este episodio.")
    else:
        col_a, col_b = st.columns(2)
        cols = [col_a, col_b]
        for idx, s in enumerate(servers):
            with cols[idx % 2]:
                badge = "🟡 con ads" if s.get("ads") else "🟢 sin ads"
                st.markdown(
                    f"**{s.get('title','')}** (`{s.get('server','')}`) — {badge}  \n"
                    f"[Abrir enlace ↗]({s.get('link','')})",
                    unsafe_allow_html=False,
                )
                st.button(
                    "Ver aquí",
                    key=f"play_{idx}",
                    on_click=lambda url=s.get('link',''): st.session_state.update({"player_url": url}),
                    disabled=not s.get("link"),
                    help="Cargar este servidor en el visor"
                )
                st.divider()

    # Botón siguiente al final (cómodo para maratón)
    st.button("Siguiente ▸", disabled=not next_url, on_click=lambda url=next_url: _go_episode(url), key="next_bottom")

def _go_episode(url: Optional[str]):
    if not url:
        return
    _navigate_to("episode", episode_url=url, player_url=None)


def view_anime(url: str):
    with st.spinner("Cargando ficha…"):
        html = fetch(url)
    detail = parse_anime_detail(html, url, BASE_URL)

    seen_data = _ls_load()
    slug_match = re.search(r"/anime/([a-z0-9\-]+)$", url)
    slug = slug_match.group(1) if slug_match else None
    seen_eps = seen_data.get("animes", {}).get(slug or "", {}).get("episodes", {})

    st.button(
        "Volver al Directorio",
        on_click=lambda: _navigate_to("browse", anime_url=None),
    )
    cols = st.columns([1, 3])
    with cols[0]:
        if detail.image:
            st.image(detail.image, width=220)
    with cols[1]:
        st.subheader(detail.title)
        if detail.genres:
            st.write("Géneros: " + ", ".join(detail.genres))
        if detail.description:
            st.write(detail.description)

        # Si hay historial, ofrecer “Continuar”
        if slug and seen_eps:
            last_seen = max(int(k) for k in seen_eps.keys())
            # buscar siguiente válido en la ficha
            next_url = None
            for e in sorted(detail.episodes, key=lambda e: e.number):
                if e.number > last_seen:
                    next_url = e.url
                    break
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"Último visto: **Ep {last_seen}**")
            with c2:
                st.button("Continuar ▸", disabled=not next_url,
                          on_click=lambda url=next_url: _go_episode(url))

    st.markdown("### Lista de episodios")
    if not detail.episodes:
        st.info("No se detectaron episodios.")
    else:
        for ep in sorted(detail.episodes, key=lambda e: e.number, reverse=True):
            cols = st.columns([2, 1, 1, 1])
            cols[0].write(f"Episodio {ep.number}")
            ep_seen = seen_eps.get(str(ep.number), {}) if slug else {}
            if ep_seen:
                badge = "✅ Confirmado" if ep_seen.get("confirmed") else "👁️ Marcado"
                cols[0].caption(badge)
            if ep.url:
                if cols[1].button("Ver en visor", key=f"ep_{ep.number}"):
                    _navigate_to("episode", episode_url=ep.url, player_url=None)
                    # marcar como visto también aquí (por si no abre el visor)
                    m = re.search(r"/ver/([a-z0-9\-]+)-(\d+)", ep.url, re.I)
                    slug_for_seen = m.group(1) if m else slug
                    if slug_for_seen:
                        seen_add_episode(
                            slug_for_seen,
                            detail.title,
                            abs_url(BASE_URL, f"/anime/{slug_for_seen}"),
                            ep.number,
                            ep.url,
                            image=detail.image,
                            confirmed=False,
                        )
                    st.rerun()
                cols[2].link_button("Abrir directo ↗", ep.url)
                cols[3].button(
                    "Confirmar visto",
                    key=f"confirm_{ep.number}",
                    disabled=not slug or ep_seen.get("confirmed", False),
                    on_click=lambda s=slug, n=ep.number: (seen_confirm_episode(s, n), st.rerun()),
                    help="Pulsa para marcar manualmente que terminaste el episodio.",
                )
            else:
                cols[1].write("-")


def view_seen():
    st.header("Vistos")
    data = _ls_load()
    animes: Dict[str, Dict] = data.get("animes", {})
    if not animes:
        st.info("Aún no hay animes/episodios vistos.")
        return

    # ordenar por último visto desc
    items = sorted(animes.items(), key=lambda kv: kv[1].get("last_seen", 0), reverse=True)
    for slug, a in items:
        title = a.get("title") or slug.replace("-", " ").title()
        image = a.get("image")
        url = a.get("url") or abs_url(BASE_URL, f"/anime/{slug}")
        eps = a.get("episodes", {})
        last_seen = max((int(k) for k in eps.keys()), default=None)

        # Calcular “continuar”
        next_url = None
        try:
            detail = get_anime_detail_cached(slug)
            if last_seen is not None:
                for e in sorted(detail.episodes, key=lambda e: e.number):
                    if e.number > last_seen and e.url:
                        next_url = e.url
                        break
        except Exception:
            pass

        confirmed_count = sum(1 for _, info in eps.items() if info.get("confirmed"))
        pending_count = len(eps) - confirmed_count

        with st.expander(
            f"{title} — {len(eps)} episodios vistos — {confirmed_count} confirmados / {pending_count} pendientes"
        ):
            if image:
                st.image(image, width=220)
            st.markdown(f"[Ver ficha del anime]({url})")

            if last_seen is not None:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(f"Último visto: **Ep {last_seen}**")
                with c2:
                    first_url = eps.get(str(last_seen), {}).get("url")
                    if first_url:
                        st.link_button("Abrir último visto", first_url)
                with c3:
                    st.button("Continuar ▸", disabled=not next_url,
                              on_click=lambda url=next_url: _go_episode(url))

            # lista de episodios
            ep_items = sorted(
                ((int(k), v) for k, v in eps.items()), key=lambda x: x[0], reverse=True
            )

            def render_section(title: str, items: List[Tuple[int, Dict]], confirmed: bool):
                st.markdown(f"#### {title}")
                if not items:
                    st.caption("No hay episodios en esta sección.")
                    return

                cols = st.columns(4)
                for i, (num, info) in enumerate(items):
                    with cols[i % 4]:
                        st.write(f"Episodio {num}")
                        status = "✅ Confirmado" if confirmed else "⏳ Pendiente"
                        st.caption(status)
                        st.link_button("Abrir", info.get("url", "#"))
                        st.button(
                            "Reconfirmar",
                            key=f"reconfirm_{slug}_{num}",
                            disabled=info.get("confirmed", False),
                            on_click=lambda s=slug, n=num: (
                                seen_confirm_episode(s, n),
                                st.rerun(),
                            ),
                            help="Marca manualmente que terminaste este episodio.",
                        )

            pending_items = [(n, info) for n, info in ep_items if not info.get("confirmed")]
            confirmed_items = [(n, info) for n, info in ep_items if info.get("confirmed")]

            render_section("Pendientes", pending_items, confirmed=False)
            render_section("Confirmados", confirmed_items, confirmed=True)
            # borrar anime del historial
            st.button("Quitar de vistos", key=f"del_{slug}", on_click=lambda s=slug: seen_delete_anime(s))

    st.divider()
    st.button("Vaciar historial", on_click=seen_clear_all)


# ---------------- App ----------------
def main():
    st.set_page_config(page_title="AnimeFLV Navigator", layout="wide")
    _inject_nav_shortcuts()
    st.sidebar.title("AnimeFLV Navigator")

    # Estado inicial
    if "mode" not in st.session_state:
        st.session_state.update(
            {
                "mode": "home",           # Inicio
                "anime_url": None,
                "episode_url": None,
                "player_url": None,
                "player_h": 540,
            }
        )
    _ensure_nav_stacks()
    _consume_nav_param()

    # Navegación lateral
    st.sidebar.caption("Usa ←/→ o estos controles para moverte sin salir de la app")
    back_col, next_col = st.sidebar.columns(2)
    back_col.button("⬅️ Atrás", on_click=_nav_back, disabled=not st.session_state.get("nav_history"))
    next_col.button("Adelante ➡️", on_click=_nav_forward, disabled=not st.session_state.get("nav_future"))

    col_nav1, col_nav2, col_nav3 = st.sidebar.columns(3)
    if col_nav1.button("Inicio"):
        _navigate_to("home")
    if col_nav2.button("Directorio"):
        _navigate_to("browse")
    if col_nav3.button("Vistos"):
        _navigate_to("seen")

    mode = st.session_state.get("mode", "home")
    if mode == "detail" and st.session_state.get("anime_url"):
        view_anime(st.session_state["anime_url"])
    elif mode == "episode" and st.session_state.get("episode_url"):
        view_episode(st.session_state["episode_url"])
    elif mode == "browse":
        view_browse()
    elif mode == "seen":
        view_seen()
    else:
        view_home()


if __name__ == "__main__":
    main()
