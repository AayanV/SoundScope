from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List, Optional, cast

from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

from .config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET



def _chunked(iterable: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

_PLAYLIST_ID_RE = re.compile(r"(?:playlist/|spotify:playlist:)?([A-Za-z0-9]{22})")

def _normalize_playlist_id(pid_or_url: Optional[str]) -> str:
    s = (pid_or_url or "").strip()
    m = _PLAYLIST_ID_RE.search(s)
    return m.group(1) if m else s

def _is_valid_track_id(tid: Optional[str]) -> bool:
    return bool(tid) and bool(re.fullmatch(r"[A-Za-z0-9]{22}", tid))

def _sg(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """safe-get: coerce None -> {} so `.get()` is always valid"""
    return d or {}



def make_client() -> Spotify:
    auth_mgr = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
    )
    return Spotify(auth_manager=auth_mgr, requests_timeout=30, retries=5)



def get_tracks_from_playlists(sp: Spotify, playlist_ids: List[str], limit: int = 1000) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []

    for raw in (playlist_ids or []):
        if len(tracks) >= limit:
            break

        pid = _normalize_playlist_id(raw)
        if not pid:
            continue

        try:
            # Preferred path
            results = cast(Dict[str, Any], sp.playlist_items(
                pid,
                additional_types=("track",),
                market="US",
                limit=100,
                offset=0,
            ))

            while results and _sg(results).get("items"):
                for item in _sg(results).get("items", []):
                    item_d = _sg(cast(Optional[Dict[str, Any]], item))
                    t = _sg(item_d.get("track"))
                    tid = t.get("id")
                    if tid:
                        tracks.append(t)
                        if len(tracks) >= limit:
                            return tracks

                if _sg(results).get("next") and len(tracks) < limit:
                    results = cast(Dict[str, Any], sp.next(results))
                else:
                    break

        except SpotifyException as e:
            # Fallback: fetch playlist object then paginate tracks subobject
            try:
                pl = cast(Dict[str, Any], sp.playlist(pid, market="US"))
                pl_tracks = _sg(_sg(pl).get("tracks"))
                items = pl_tracks.get("items", [])

                for it in items:
                    it_d = _sg(cast(Optional[Dict[str, Any]], it))
                    t = _sg(it_d.get("track"))
                    if t.get("id"):
                        tracks.append(t)
                        if len(tracks) >= limit:
                            return tracks

                next_url = pl_tracks.get("next")
                while next_url and len(tracks) < limit:
                    page = cast(Dict[str, Any], sp._get(next_url))  # spotipy internal pagination
                    for it in _sg(page).get("items", []):
                        it_d = _sg(cast(Optional[Dict[str, Any]], it))
                        t = _sg(it_d.get("track"))
                        if t.get("id"):
                            tracks.append(t)
                            if len(tracks) >= limit:
                                return tracks
                    next_url = _sg(page).get("next")

            except SpotifyException as e2:
                print(f"[WARN] Skipping playlist {pid}: {e2}")
                continue

        time.sleep(0.05)

    return tracks


def search_tracks(sp: Spotify, query: str, limit: int = 1000) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []
    offset = 0
    page_size = 50

    while len(tracks) < limit:
        res = cast(Dict[str, Any], sp.search(q=query, type="track", limit=page_size, offset=offset, market="US"))
        items = _sg(_sg(res).get("tracks")).get("items", [])
        if not items:
            break
        tracks.extend(items)
        offset += page_size
        time.sleep(0.1)

    return tracks[:limit]


def get_new_releases_tracks(sp: Spotify, limit: int = 1000) -> List[Dict[str, Any]]:
    albums: List[Dict[str, Any]] = []
    page_size = 50
    offset = 0

    while len(albums) < 500:
        res = cast(Dict[str, Any], sp.new_releases(limit=page_size, offset=offset))
        items = _sg(_sg(res).get("albums")).get("items", [])
        if not items:
            break
        albums.extend(items)
        offset += page_size
        time.sleep(0.1)

    albums = albums[:500]
    album_ids = [a.get("id") for a in albums if a.get("id")]

    tracks: List[Dict[str, Any]] = []
    for aid in album_ids:
        tr_res = cast(Dict[str, Any], sp.album_tracks(aid, limit=50))
        for t in _sg(tr_res).get("items", []):
            tid = _sg(cast(Optional[Dict[str, Any]], t)).get("id")
            if not _is_valid_track_id(tid):
                continue
            full = cast(Dict[str, Any], sp.track(cast(str, tid)))
            if full:
                tracks.append(full)
                if len(tracks) >= limit:
                    return tracks
        time.sleep(0.05)

    return tracks[:limit]


def enrich_with_audio_features(sp: Spotify, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fetch audio features with fallback strategy:
      - batches of 50
      - on 400/403, split to 10s
      - then singletons; only truly bad IDs map to None
    """
    def safe_audio_features(id_list: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        out: Dict[str, Optional[Dict[str, Any]]] = {}
        ids = [i for i in id_list if _is_valid_track_id(i)]
        if not ids:
            return out

        def fetch(ids_subset: List[str]) -> None:
            if not ids_subset:
                return
            try:
                feats = sp.audio_features(ids_subset)
                # feats aligns with ids_subset; may contain None
                for f_id, f in zip(ids_subset, feats): # type: ignore
                    out[f_id] = cast(Optional[Dict[str, Any]], f)
            except SpotifyException as e:
                status = getattr(e, "http_status", None)
                if status in (400, 403) and len(ids_subset) > 1:
                    for i in range(0, len(ids_subset), 10):
                        sub = ids_subset[i:i + 10]
                        try:
                            sub_feats = sp.audio_features(sub)
                            for f_id, f in zip(sub, sub_feats): # type: ignore
                                out[f_id] = cast(Optional[Dict[str, Any]], f)
                        except SpotifyException as e2:
                            status2 = getattr(e2, "http_status", None)
                            if status2 in (400, 403) and len(sub) > 1:
                                for tid in sub:
                                    try:
                                        single = sp.audio_features([tid])
                                        out[tid] = cast(Optional[Dict[str, Any]], (single[0] if single else None))
                                    except SpotifyException:
                                        out[tid] = None
                                    time.sleep(0.02)
                            else:
                                for tid in sub:
                                    out[tid] = None
                        time.sleep(0.05)
                else:
                    for tid in ids_subset:
                        out[tid] = None

        for i in range(0, len(ids), 50):
            fetch(ids[i:i + 50])
            time.sleep(0.05)
        return out

    ids = [t.get("id") for t in (tracks or []) if t and t.get("id")]
    features_map = safe_audio_features(cast(List[str], ids))

    enriched: List[Dict[str, Any]] = []
    for t in tracks:
        t_d = _sg(t)
        tid = t_d.get("id")
        feat = _sg(features_map.get(cast(str, tid)) if tid else None)

        artists_list = t_d.get("artists") or []
        artist_names = ", ".join(_sg(a).get("name", "") for a in artists_list)

        enriched.append({
            "id": tid,
            "name": t_d.get("name"),
            "artists": artist_names,
            "popularity": t_d.get("popularity"),
            "duration_ms": t_d.get("duration_ms"),
            "explicit": int(bool(t_d.get("explicit"))),

            "danceability": feat.get("danceability"),
            "energy": feat.get("energy"),
            "key": feat.get("key"),
            "loudness": feat.get("loudness"),
            "mode": feat.get("mode"),
            "speechiness": feat.get("speechiness"),
            "acousticness": feat.get("acousticness"),
            "instrumentalness": feat.get("instrumentalness"),
            "liveness": feat.get("liveness"),
            "valence": feat.get("valence"),
            "tempo": feat.get("tempo"),
            "time_signature": feat.get("time_signature"),
        })

    return enriched
