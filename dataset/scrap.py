# LOOP:
# Detik → collect up to 250
#        ↓
# Kompas → collect up to 250
#        ↓
# CNN → collect up to 250
#        ↓
# Tribun → collect up to 250

import nest_asyncio
nest_asyncio.apply()

import os
import re
import time
import random
import asyncio
import urllib.parse
from datetime import date, timedelta
import pandas as pd
import requests
import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from tqdm.auto import tqdm

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# ==========================================
# GLOBAL CONFIG
# ==========================================
TARGET = 1000
PORTAL_CAP = 250
MAX_DATE = date(2026, 10, 3)
MIN_DATE = date(2025, 10, 3)
OUTPUT_COLUMNS = ["id", "source", "date", "title", "label", "url"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
]

# Titles containing these words are skipped (polls / quizzes, not real articles).
POLLING_WORDS = ("polling", "tebak", "vote", "voting", "pilih siapa")

DOMAINS = {
    "technology": {
        "label": "Technology",
        "output_file": "data/technology.csv",
        "keywords": ["software", "hardware", "it", "teknologi", "wifi", "internet", "robot", "ict"],
        "cnn_indeks": "teknologi/indeks/8",
        "cnn_section": "/teknologi/",
        "kompas_site": "tekno",
        "kompas_host": "tekno.kompas.com",
        "kompas_url_patterns": ["tekno.kompas.com/read/", "www.kompas.com/tekno/read/"],
        "detik_base": "https://inet.detik.com",
        "detik_host": "inet.detik.com",
        "detik_date_iso": True,
        "detik_channels": [
            "news", "telecommunication",
            "gadget", "tips-tricks",
            "cyber-life", "security",
            "review", "mobileapps", "inet-grafis",
        ],
        "tribun_channels": ["index-news/techno"],
        "tribun_marker": "techno",
    },
    "health": {
        "label": "Health",
        "output_file": "data/health.csv",
        "keywords": ["kesehatan", "sehat", "bugar", "obat", "pertolongan", "pencegahan", "vitamin"],
        "cnn_indeks": "health/indeks/62",
        "cnn_section": "/gaya-hidup/",
        "kompas_site": "health",
        "kompas_host": "health.kompas.com",
        "kompas_url_patterns": ["health.kompas.com/read/", "www.kompas.com/health/read/"],
        "detik_base": "https://health.detik.com",
        "detik_host": "health.detik.com",
        "detik_date_iso": False,
        "detik_channels": [
            "berita-detikhealth", "diet",
            "kebugaran", "true-story",
            "sexual-health", "wellness-beauty",
        ],
        "tribun_channels": ["index-news/kesehatan"],
        "tribun_marker": "kesehatan",
    },
    "education": {
        "label": "Education",
        "output_file": "data/education.csv",
        "keywords": ["kuliah", "materi", "pendidikan", "kelas", "pembelajaran", "guru", "dosen"],
        "cnn_indeks": "edukasi/indeks/559",
        "cnn_section": "/edukasi/",
        "kompas_site": "edukasi",
        "kompas_host": "www.kompas.com/edu",
        "kompas_url_patterns": ["www.kompas.com/edu/read/", "edukasi.kompas.com/read/"],
        "detik_base": "https://www.detik.com/edu",
        "detik_host": "www.detik.com",
        "detik_date_iso": False,
        "detik_channels": [
            "sekolah", "perguruan-tinggi",
            "beasiswa", "edutainment",
            "seleksi-masuk-pt", "detikpedia",
        ],
        "tribun_channels": ["index-news/pendidikan"],
        "tribun_marker": "pendidikan",
    },
    "sport": {
        "label": "Sport",
        "output_file": "data/sport.csv",
        "keywords": ["sepakbola", "bola", "soccer", "nbl", "basket", "volly", "asian games", "skor", "ht ft"],
        "cnn_indeks": "olahraga/indeks/7",
        "cnn_section": "/olahraga/",
        "kompas_site": "bola",
        "kompas_host": "bola.kompas.com",
        "kompas_url_patterns": ["bola.kompas.com/read/", "www.kompas.com/sports/read/"],
        "detik_base": "https://sport.detik.com",
        "detik_host": "sport.detik.com",
        "detik_date_iso": False,
        "detik_channels": [
            "sepakbola", "basket",
            "moto-gp", "f1",
            "raket", "sport-lain",
            "g-sport", "sportstyle",
        ],
        "tribun_channels": ["index-news/sport"],
        "tribun_marker": "sport",
    },
    "politics": {
        "label": "Politics",
        "output_file": "data/politic.csv",
        "keywords": ["politik", "parpol", "partai", "pemilu", "kpu", "pencoblosan", "pemungutan suara", "oposisi"],
        "cnn_indeks": "nasional/indeks/3",
        "cnn_section": "/nasional/",
        "kompas_site": "nasional",
        "kompas_host": "nasional.kompas.com",
        "kompas_url_patterns": [
            "nasional.kompas.com/read/",
            "www.kompas.com/nasional/read/",
            "sorotpolitik.kompas.com/read/",
        ],
        "detik_base": "https://news.detik.com",
        "detik_host": "news.detik.com",
        "detik_date_iso": False,
        "detik_channels": [
            "berita", "internasional",
            "bbc", "investigasi",
            "pemilu", "pilkada",
            "bangun-indonesia", "nyago-bumi-sriwijaya",
            "jabodetabek", "hukum",
        ],
        "tribun_channels": [
            "index-news/nasional",
            "index-news/dpd-ri",
            "index-news/mpr-ri",
            "index-news/kilas-kementerian",
        ],
        "tribun_marker": "nasional",
    },
}

PORTAL_ORDER = ["Detik", "Kompas", "CNN Indonesia", "Tribun News"]


# ==========================================
# SHARED HELPERS (unchanged)
# ==========================================
def keyword_matches(keyword, text):
    words = [re.escape(w) for w in keyword.lower().split()]
    pattern = r"\b" + r"[\s\-]+".join(words) + r"\b"
    return re.search(pattern, text.lower()) is not None


def title_has_keyword(title, keywords):
    return any(keyword_matches(k, title) for k in keywords)


def new_funnel():
    return {
        "candidates": 0, "date_rejected": 0, "keyword_rejected": 0,
        "duplicate": 0, "collected": 0,
    }


def print_funnel(portal_name, f):
    print(
        f"[FUNNEL] {portal_name} | candidates={f['candidates']} | "
        f"collected={f['collected']} | "
        f"reject(date={f['date_rejected']}, "
        f"keyword={f['keyword_rejected']}, "
        f"duplicate={f['duplicate']})",
        flush=True,
    )


def clean_title(title):
    if not title:
        return None
    title = re.sub(r"\s+", " ", title).strip()
    return title


def within_window(dstr):
    try:
        y, m, d = map(int, dstr.split("-"))
        return MIN_DATE <= date(y, m, d) <= MAX_DATE
    except Exception:
        return False


def record_article(articles, seen_urls, seen_titles, funnel, bar, source, url, title, dstr):
    """Save one accepted article.

    Every scraper runs the SAME final step once an article has passed all of
    its filters: remember the url/title so we never add it twice, append the
    row, bump the counters, and tick the progress bar. This helper is that
    shared step, so the four scrapers don't each repeat it.
    """
    seen_urls.add(url)
    seen_titles.add(title.lower())
    articles.append({
        "id": 0, "source": source, "date": dstr,
        "title": title, "label": None, "url": url,
    })
    funnel["collected"] += 1
    if bar is not None:
        bar.update(1)


def log_fetch(portal_name, url):
    """Line 1: the exact index URL we are about to request."""
    print(f"[INDEX] {portal_name} | {url}", flush=True)


def log_result(portal_name, scope, date_iso, page, raw_links, new, collected):
    """Line 2: the per-page result, in ONE shared format for every portal.

    `scope` is the channel name for Detik/Tribun, or "" for CNN/Kompas
    (which have no channel). A leading space is added only when scope is set,
    so the output stays clean either way.
    """
    scope_part = f"{scope} " if scope else ""
    print(
        f"[INDEX] {portal_name} | {scope_part}Date={date_iso} page={page} | "
        f"raw_links={raw_links} | new={new} | Collected={collected}/{TARGET}",
        flush=True,
    )


def new_portal_state():
    return {"collected": 0, "last_date": MAX_DATE, "exhausted": False}


_stop_remover = StopWordRemoverFactory().create_stop_word_remover()


def title_has_enough_words(title, min_words=5):
    """Filter only – does NOT change the saved title."""
    if not title:
        return False
    cleaned = _stop_remover.remove(title.lower())
    words = [w for w in re.findall(r"[a-zA-Z0-9]+", cleaned) if len(w) > 1]
    return len(words) >= min_words


# ==========================================
# DETIK  (original logs preserved)
# ==========================================
async def scrape_detik_batch(cfg, articles, seen_urls, seen_titles, state, quota, bar, funnel):
    if state["exhausted"] or quota <= 0:
        return 0
    portal_name = "Detik"
    f = funnel
    keywords = cfg["keywords"]
    host = cfg["detik_host"]
    origin = "https://" + host
    channels = cfg.get("detik_channels") or ["indeks"]
    base_root = cfg["detik_base"].rstrip("/")
    added = 0
    current = state["last_date"]
    print(f"\n[DETIK] Starting (multi-channel)...", flush=True)

    while current >= MIN_DATE and added < quota and len(articles) < TARGET:
        date_iso = current.strftime("%Y-%m-%d")
        date_param = current.strftime("%Y-%m-%d" if cfg.get("detik_date_iso") else "%m/%d/%Y")

        # === CHANNEL LOOP (must wrap everything below) ===
        for channel in channels:
            if added >= quota or len(articles) >= TARGET:
                break

            index_base = f"{base_root}/{channel}/indeks"
            seen_this_date = set()
            prev_page_urls = None
            page = 0

            while True:
                page += 1
                if added >= quota or len(articles) >= TARGET:
                    break

                log_fetch(portal_name, f"{index_base}?page={page}&date={date_param}")
                headers = {
                    "User-Agent": random.choice(USER_AGENTS),
                    "Accept-Language": "id,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
                try:
                    await asyncio.sleep(random.uniform(1.2, 2.5))
                    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                        res = await client.get(
                            index_base,
                            params={"page": page, "date": date_param},
                            headers=headers,
                        )
                    if res.status_code != 200:
                        print(f"[INDEX] ERROR: {portal_name} | {channel} page={page} | HTTP {res.status_code}", flush=True)
                        break

                    soup = BeautifulSoup(res.text, "html.parser")
                    links = (
                        soup.select("div.list-content__item a[href*='/d-']")
                        or soup.select("article a[href*='/d-']")
                        or soup.select(".media__text a[href*='/d-']")
                        or soup.select(".list-content a[href*='/d-']")
                        or soup.select('a[href*="/d-"]')
                    )
                    page_items = {}
                    for link in links:
                        raw_href = link.get("href")
                        if not raw_href:
                            continue
                        art_url = urllib.parse.urljoin(origin, raw_href).split("?")[0].strip()
                        if host not in art_url:
                            continue
                        txt = clean_title(re.sub(r"\s+", " ", link.get_text()).strip()) or ""
                        if art_url not in page_items or len(txt) > len(page_items[art_url]):
                            page_items[art_url] = txt

                    page_urls = set(page_items)
                    if not page_urls:
                        print(f"[INDEX] {portal_name} | {channel} Date={date_iso} page={page} | empty → next channel", flush=True)
                        break
                    if page_urls == prev_page_urls and prev_page_urls is not None:
                        print(f"[INDEX] {portal_name} | {channel} Date={date_iso} page={page} | exact repeat → next channel", flush=True)
                        break
                    prev_page_urls = page_urls
                    seen_this_date |= page_urls

                    new = 0
                    raw_links = len(page_items)
                    for art_url, title_text in page_items.items():
                        if added >= quota or len(articles) >= TARGET:
                            break
                        f["candidates"] += 1
                        if not title_text:
                            continue
                        if any(w in title_text.lower() for w in POLLING_WORDS):
                            continue
                        if not title_has_enough_words(title_text):
                            continue
                        if art_url in seen_urls or title_text.lower() in seen_titles:
                            f["duplicate"] += 1
                            continue
                        m = re.search(r"/(\d{4})/(\d{2})/(\d{2})", art_url)
                        dstr = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else date_iso
                        if not within_window(dstr):
                            f["date_rejected"] += 1
                            continue
                        if not title_has_keyword(title_text, keywords):
                            f["keyword_rejected"] += 1
                            continue
                        record_article(articles, seen_urls, seen_titles, f, bar,
                                        "Detik", art_url, title_text, dstr)
                        added += 1
                        new += 1

                    log_result(portal_name, channel, date_iso, page, raw_links, new, len(articles))
                    if new == 0 and page > 1:
                        break
                except Exception as e:
                    print(f"[DETIK] error {channel} {date_iso} page={page}: {e}", flush=True)
                    break

        current -= timedelta(days=1)
        state["last_date"] = current

    if current < MIN_DATE:
        state["exhausted"] = True
    state["collected"] += added
    return added


# ==========================================
# CNN  (original logs preserved)
# ==========================================
def scrape_cnn_batch(cfg, articles, seen_urls, seen_titles, state, quota, bar, funnel):
    if state["exhausted"] or quota <= 0:
        return 0

    portal_name = "CNN Indonesia"
    f = funnel
    keywords = cfg["keywords"]
    section = cfg["cnn_section"]
    base = f"https://www.cnnindonesia.com/{cfg['cnn_indeks']}"
    added = 0
    current = state["last_date"]

    print(f"\n[CNN] Starting (date+page method)...", flush=True)

    while current >= MIN_DATE and added < quota and len(articles) < TARGET:
        date_slash = current.strftime("%Y/%m/%d")
        date_iso = current.strftime("%Y-%m-%d")
        page = 1
        empty_streak = 0

        while added < quota and len(articles) < TARGET and empty_streak < 2:
            url = f"{base}?date={date_slash}&page={page}"
            log_fetch(portal_name, url)
            headers = {"User-Agent": random.choice(USER_AGENTS), "Accept-Language": "id,en;q=0.9"}
            try:
                time.sleep(random.uniform(1.0, 2.0))
                res = requests.get(url, headers=headers, timeout=12)
                if res.status_code != 200 or "gokuProps" in (res.text or "")[:1500]:
                    print(f"[INDEX] {portal_name} | Date={date_iso} Page={page} | blocked/empty (HTTP {res.status_code})", flush=True)
                    empty_streak += 1
                    page += 1
                    continue

                soup = BeautifulSoup(res.content, "html.parser")
                # broader selectors – CNN index is inconsistent
                article_nodes = (
                    soup.select("article")
                    or soup.select(".list-content article")
                    or soup.select(".media__text")
                    or soup.select("div[class*='list'] a[href]")
                )
                seen_in_page = set()
                candidate_anchors = []
                if article_nodes and hasattr(article_nodes[0], "find_all"):
                    for node in article_nodes:
                        for a in node.find_all("a", href=True):
                            h = a["href"].split("?")[0]
                            if h not in seen_in_page:
                                seen_in_page.add(h)
                                candidate_anchors.append(a)
                else:
                    # fallback: any link that looks like article
                    for a in soup.select(f'a[href*="{section}"]'):
                        h = a["href"].split("?")[0]
                        if h not in seen_in_page and re.search(r"/\d{4}/\d{2}/", h):
                            seen_in_page.add(h)
                            candidate_anchors.append(a)

                if not candidate_anchors:
                    log_result(portal_name, "", date_iso, page, 0, 0, len(articles))
                    empty_streak += 1
                    page += 1
                    continue

                new = 0
                raw_links = 0
                for a in candidate_anchors:
                    if added >= quota or len(articles) >= TARGET:
                        break
                    href = a["href"].split("?")[0]
                    if section not in href:
                        continue
                    raw_links += 1
                    f["candidates"] += 1

                    # prefer <h2> inside article (clean title), fallback to link text
                    title_el = None
                    parent = a.find_parent("article") or a.find_parent(class_=re.compile(r"flex|media|list"))
                    if parent:
                        title_el = parent.find("h2") or parent.find("h3")
                    if title_el:
                        title_text = clean_title(title_el.get_text(strip=True))
                    else:
                        title_text = clean_title(re.sub(r"\s+", " ", a.get_text()).strip())
                    if not title_text:
                        continue
                    if not title_has_enough_words(title_text):
                        continue
                    if href in seen_urls or title_text.lower() in seen_titles:
                        f["duplicate"] += 1
                        continue
                    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})", href)
                    dstr = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else date_iso
                    if not within_window(dstr):
                        f["date_rejected"] += 1
                        continue
                    if not title_has_keyword(title_text, keywords):
                        f["keyword_rejected"] += 1
                        continue
                    record_article(articles, seen_urls, seen_titles, f, bar,
                                   "CNN Indonesia", href, title_text, dstr)
                    added += 1
                    new += 1

                log_result(portal_name, "", date_iso, page, raw_links, new, len(articles))
                if raw_links == 0:
                    break
                page += 1

            except Exception as e:
                print(f"[CNN] error: {e}", flush=True)
                break

        state["last_date"] = current
        current -= timedelta(days=1)

    if current < MIN_DATE:
        state["exhausted"] = True
    state["collected"] += added
    return added


# ==========================================
# KOMPAS (original logs preserved)
# ==========================================
async def scrape_kompas_batch(cfg, articles, seen_urls, seen_titles, state, quota, bar, funnel):
    if state["exhausted"] or quota <= 0:
        return 0

    portal_name = "Kompas"
    f = funnel
    keywords = cfg["keywords"]
    site = cfg["kompas_site"]
    patterns = cfg.get("kompas_url_patterns") or [f"{cfg['kompas_host']}/read/"]
    added = 0
    current = state["last_date"]
    fails = 0

    print(f"\n[KOMPAS] Starting with Playwright (async)...", flush=True)

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="id-ID",
                viewport={"width": 1366, "height": 768},
            )
            page = await context.new_page()

            while current >= MIN_DATE and added < quota and len(articles) < TARGET and fails < 5:
                date_iso = current.strftime("%Y-%m-%d")
                page_num = 1

                while added < quota and len(articles) < TARGET and fails < 5:
                    url = f"https://indeks.kompas.com/?site={site}&date={date_iso}&page={page_num}"
                    log_fetch(portal_name, url)

                    try:
                        await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                        await asyncio.sleep(random.uniform(2, 3))

                        prev = 0
                        for _ in range(6):
                            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            await asyncio.sleep(1.2)
                            cur = await page.evaluate('document.querySelectorAll(\'a[href*="/read/"]\').length')
                            if cur == prev:
                                break
                            prev = cur
                        await page.evaluate("window.scrollTo(0, 0)")
                        await asyncio.sleep(0.4)

                        content = await page.content()
                        if "gokuProps" in content or len(content) < 2000:
                            fails += 1
                            print(f"[KOMPAS] blocked {date_iso} p{page_num} ({fails}/5)", flush=True)
                            break

                        fails = 0
                        css_sel = ", ".join(f'a[href*="{p}"]' for p in patterns)
                        links = await page.query_selector_all(css_sel)

                        new = 0
                        raw_links = 0
                        seen_in_page = set()
                        for link in links:
                            if added >= quota or len(articles) >= TARGET:
                                break
                            href = await link.get_attribute("href")
                            if not href:
                                continue
                            href = urllib.parse.urljoin("https://indeks.kompas.com/", href).split("?")[0]
                            if href in seen_in_page:
                                continue
                            seen_in_page.add(href)
                            if not any(p in href for p in patterns):
                                continue
                            raw_links += 1
                            f["candidates"] += 1
                            if href in seen_urls:
                                f["duplicate"] += 1
                                continue
                            # get title from h2 if exists, else from link text
                            h2 = await link.query_selector("h2")
                            if h2:
                                title_text = clean_title((await h2.inner_text()).strip())
                            else:
                                title_text = clean_title((await link.inner_text()).strip())
                            if not title_text:
                                continue
                            if not title_has_enough_words(title_text):
                                continue
                            if title_text.lower() in seen_titles:
                                f["duplicate"] += 1
                                continue
                            m = re.search(r"/(\d{4})/(\d{2})/(\d{2})", href)
                            if m:
                                dstr = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                                if not within_window(dstr):
                                    f["date_rejected"] += 1
                                    continue
                            else:
                                dstr = date_iso
                            if not title_has_keyword(title_text, keywords):
                                f["keyword_rejected"] += 1
                                continue
                            record_article(articles, seen_urls, seen_titles, f, bar,
                                           "Kompas", href, title_text, dstr)
                            added += 1
                            new += 1

                        log_result(portal_name, "", date_iso, page_num, raw_links, new, len(articles))
                        if raw_links == 0:
                            break
                        page_num += 1

                    except Exception as e:
                        print(f"[KOMPAS] error: {e}", flush=True)
                        fails += 1
                        break

                state["last_date"] = current
                current -= timedelta(days=1)

            await browser.close()
    except Exception as e:
        print(f"[KOMPAS] browser failed: {e}", flush=True)
        state["exhausted"] = True

    if current < MIN_DATE or fails >= 5:
        state["exhausted"] = True
    state["collected"] += added
    return added


# ==========================================
# TRIBUN (original logs preserved)
# ==========================================
async def tribun_pub_date(page, url):
    try:
        await page.goto(url, timeout=30000, wait_until="domcontentloaded")
        for sel in ('meta[property="article:published_time"]',
                    'meta[name="content_PublishedDate"]',
                    'meta[itemprop="datePublished"]'):
            el = await page.query_selector(sel)
            if el:
                v = (await el.get_attribute("content") or "").strip()
                m = re.match(r"(\d{4})-(\d{2})-(\d{2})", v)
                if m:
                    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        html = await page.content()
        m = re.search(r'"datePublished"\s*:\s*"(\d{4})-(\d{2})-(\d{2})', html)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    except Exception as e:
        print(f"[TRIBUN] date fetch error: {e}", flush=True)
    return None

async def scrape_tribun_batch(cfg, articles, seen_urls, seen_titles, state, quota, bar, funnel):
    if state["exhausted"] or quota <= 0:
        return 0
    portal_name = "Tribun News"
    f = funnel
    keywords = cfg["keywords"]
    marker = cfg["tribun_marker"]
    added = 0
    fails = 0
    OLD_PAGE_STOP = 2       # stop only after N pure-old pages

    chan_state = state.setdefault("channels", {})
    print(f"\n[TRIBUN] Starting with Playwright (async)...", flush=True)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="id-ID",
                viewport={"width": 1366, "height": 768},
            )
            page = await context.new_page()
            art_page = await context.new_page()

            for channel in cfg["tribun_channels"]:
                cs = chan_state.setdefault(
                    channel, {"page": 1, "prev": None, "streak": 0, "done": False}
                )
                if cs["done"]:
                    continue
                if added >= quota or len(articles) >= TARGET or fails >= 5:
                    break

                page_num = cs["page"]
                old_page_streak = cs["streak"]
                prev_page_urls = cs["prev"]

                while (
                    added < quota
                    and len(articles) < TARGET
                    and fails < 5
                ):
                    url = f"https://www.tribunnews.com/{channel}?page={page_num}&date=undefined"
                    log_fetch(portal_name, url)
                    try:
                        await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                        await asyncio.sleep(random.uniform(2.5, 4.0))
                        # light scroll to trigger lazy content
                        for _ in range(3):
                            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            await asyncio.sleep(0.8)
                        content = await page.content()
                        if "gokuProps" in content or len(content) < 2000:
                            fails += 1
                            print(f"[TRIBUN] blocked {channel} page {page_num} ({fails}/5)", flush=True)
                            continue
                        fails = 0

                        ART_ID_RE = re.compile(r"/[\w\-]+/\d{5,}/")
                        links = await page.query_selector_all("a[href]")
                        seen_in_page = set()
                        page_urls = set()
                        new = 0
                        raw_links = 0
                        dated_ok = 0
                        dated_old = 0
                        dated_new = 0          # > MAX_DATE

                        for link in links:
                            if added >= quota or len(articles) >= TARGET:
                                break
                            href = await link.get_attribute("href")
                            if not href:
                                continue
                            href = urllib.parse.urljoin("https://www.tribunnews.com/", href)
                            href = href.split("?")[0].split("#")[0]
                            if href in seen_in_page:
                                continue
                            seen_in_page.add(href)

                            if marker not in href:
                                continue
                            if not ART_ID_RE.search(href):
                                continue
                            if any(x in href for x in ("/tag/", "/topic/", "/index-news/", "/search", "/video/")):
                                continue

                            raw_links += 1
                            page_urls.add(href)
                            f["candidates"] += 1

                            if href in seen_urls:
                                f["duplicate"] += 1
                                continue
                            title_text = clean_title((await link.inner_text()).strip())
                            if not title_text:
                                continue
                            if not title_has_enough_words(title_text):
                                continue
                            if title_text.lower() in seen_titles:
                                f["duplicate"] += 1
                                continue
                            if not title_has_keyword(title_text, keywords):
                                f["keyword_rejected"] += 1
                                continue

                            await asyncio.sleep(random.uniform(0.8, 1.6))
                            dstr = await tribun_pub_date(art_page, href)
                            if dstr is None:
                                f["date_rejected"] += 1
                                continue
                            try:
                                y, m, d = map(int, dstr.split("-"))
                                art_date = date(y, m, d)
                            except Exception:
                                f["date_rejected"] += 1
                                continue

                            if art_date < MIN_DATE:
                                f["date_rejected"] += 1
                                dated_old += 1
                                continue
                            if art_date > MAX_DATE:
                                f["date_rejected"] += 1
                                dated_new += 1
                                continue
                            # inside window
                            dated_ok += 1

                            record_article(articles, seen_urls, seen_titles, f, bar,
                                           "Tribun News", href, title_text, dstr)
                            added += 1
                            new += 1

                        log_result(portal_name, channel, "-", page_num, raw_links, new, len(articles))
                        if added >= quota or len(articles) >= TARGET:
                            break

                        # === STOP CONDITIONS ===
                        if raw_links == 0:
                            cs["done"] = True
                            print(f"[TRIBUN] {channel} page={page_num} empty → channel finished", flush=True)
                            break

                        if prev_page_urls is not None and page_urls == prev_page_urls:
                            cs["done"] = True
                            print(f"[TRIBUN] {channel} page={page_num} == page={page_num-1} → channel finished", flush=True)
                            break
                        prev_page_urls = page_urls

                        # only pure-old pages count toward stop
                        if dated_ok == 0 and dated_old > 0 and dated_new == 0:
                            old_page_streak += 1
                            if old_page_streak >= OLD_PAGE_STOP:
                                cs["done"] = True
                                print(f"[TRIBUN] {channel} past date window → channel finished", flush=True)
                                break
                        else:
                            old_page_streak = 0

                        page_num += 1

                    except Exception as e:
                        print(f"[TRIBUN] error: {e}", flush=True)
                        fails += 1

                if not cs["done"]:
                    cs["page"] = page_num
                    cs["prev"] = prev_page_urls
                    cs["streak"] = old_page_streak

            await browser.close()
    except Exception as e:
        print(f"[TRIBUN] browser failed: {e}", flush=True)
        state["exhausted"] = True

    if fails >= 5:
        state["exhausted"] = True
    state["collected"] += added

    if all(chan_state.get(c, {}).get("done") for c in cfg["tribun_channels"]):
        state["exhausted"] = True
    return added

# ==========================================
# RUN ONE DOMAIN (backfill controller)
# ==========================================
async def run_domain(domain_key):
    cfg = DOMAINS[domain_key]
    articles = []
    seen_urls = set()
    seen_titles = set()
    funnels = {name: new_funnel() for name in PORTAL_ORDER}
    states = {name: new_portal_state() for name in PORTAL_ORDER}
    print(f"\n========== Starting {cfg['label']} scraper (Target: {TARGET}) ==========", flush=True)
    bar = tqdm(total=TARGET, desc=f"Harvesting {cfg['label']}", unit="rows")
    # Async scrapers only. CNN is synchronous, so it is called directly below.
    scrapers = {
        "Detik": scrape_detik_batch,
        "Kompas": scrape_kompas_batch,
        "Tribun News": scrape_tribun_batch,
    }
    max_rounds = 20          # safety; 4×250=1000 → usually 1-3 rounds enough
    for round_i in range(1, max_rounds + 1):
        if len(articles) >= TARGET:
            break
        print(f"\n----- ROUND {round_i} | collected={len(articles)}/{TARGET} -----", flush=True)
        made_progress = False
        for portal in PORTAL_ORDER:
            if len(articles) >= TARGET:
                break
            st = states[portal]
            if st["exhausted"]:
                continue
            remaining_global = TARGET - len(articles)
            # every round offer full PORTAL_CAP (250) or whatever is still needed
            quota = min(remaining_global, PORTAL_CAP)
            if quota <= 0:
                continue
            print(
                f"[{portal}] quota={quota} "
                f"(portal total {st['collected']}/{PORTAL_CAP}, global {len(articles)}/{TARGET})",
                flush=True,
            )
            if portal == "CNN Indonesia":
                added = scrape_cnn_batch(
                    cfg, articles, seen_urls, seen_titles, st, quota, bar, funnels[portal]
                )
            else:
                added = await scrapers[portal](
                    cfg, articles, seen_urls, seen_titles, st, quota, bar, funnels[portal]
                )
            if added > 0:
                made_progress = True
        if not made_progress:
            print("[ENGINE] no portal made progress → stop", flush=True)
            break

    bar.close()

    print(f"\n[FUNNEL] {cfg['label']} drop-off by portal:", flush=True)
    for name in PORTAL_ORDER:
        print_funnel(name, funnels[name])

    # Finalize
    articles = articles[:TARGET]
    for i, row in enumerate(articles, 1):
        row["id"] = i

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(articles, columns=OUTPUT_COLUMNS)
    df["label"] = pd.NA
    df.to_csv(cfg["output_file"], index=False)
    print(f"[SUCCESS] Saved {len(df)} rows → {cfg['output_file']}", flush=True)
    return df


# ==========================================
# ENTRY
# ==========================================
async def run_all():
    for key in DOMAINS:
        await run_domain(key)


# Example:
# await run_domain("technology")
# await run_domain("health")
# await run_domain("education")
await run_domain("politics")
# await run_domain("sport")
# await run_all()

# if __name__ == "__main__":
#     asyncio.run(run_domain("technology"))
