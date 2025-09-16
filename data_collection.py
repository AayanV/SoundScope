from typing import List, Dict, Any, Optional
import pandas as pd
from .spotify_client import make_client, get_tracks_from_playlists, search_tracks, get_new_releases_tracks, enrich_with_audio_features

def collect(playlist_ids: Optional[list] = None,
            query: Optional[str] = None,
            new_releases: bool = False,
            limit: int = 1000) -> pd.DataFrame:
    sp = make_client()
    tracks = []

    if playlist_ids:
        tracks.extend(get_tracks_from_playlists(sp, playlist_ids, limit))
    if query and len(tracks) < limit:
        tracks.extend(search_tracks(sp, query, limit - len(tracks)))
    if new_releases and len(tracks) < limit:
        tracks.extend(get_new_releases_tracks(sp, limit - len(tracks)))

    # Deduplicate by id
    seen = set()
    deduped = []
    for t in tracks:
        tid = t.get("id")
        if tid and tid not in seen:
            deduped.append(t)
            seen.add(tid)

    enriched = enrich_with_audio_features(sp, deduped)
    df = pd.DataFrame(enriched)
    df = df.dropna(subset=["popularity"])  # ensure target present
    return df
