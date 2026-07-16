"""Project-wide constants: teams, leagues/divisions, and the feature set."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MATCHES_CSV = DATA_DIR / "mlb_games.csv"

# The features the model trains on — rolling batting + pitching form for
# each side, league (AL/NL), plus the starting-pitcher matchup and the
# ballpark run environment (the two biggest baseball-specific drivers).
FEATURE_COLS = [
    "home_runs_for_avg", "home_runs_against_avg", "home_batting_avg",
    "home_on_base_pct", "home_slugging_pct", "home_era", "home_whip",
    "home_bullpen_era", "home_win_pct", "home_league",
    "away_runs_for_avg", "away_runs_against_avg", "away_batting_avg",
    "away_on_base_pct", "away_slugging_pct", "away_era", "away_whip",
    "away_bullpen_era", "away_win_pct", "away_league",
    # Starting-pitcher matchup — the single biggest driver of a game.
    # xFIP and K-BB% are among the most stable, predictive pitcher stats.
    "home_starter_xfip", "home_starter_k_bb_pct",
    "away_starter_xfip", "away_starter_k_bb_pct",
    # Ballpark run environment (Coors inflates, Petco suppresses, ...).
    "park_factor",
]

# League-average starter stats, used to impute when a probable pitcher
# or their season line is missing.
LEAGUE_AVG_XFIP = 4.20
LEAGUE_AVG_K_BB_PCT = 0.13

# Rolling form window (games) feeding the *_avg / rate features.
FORM_WINDOW = 20
MIN_GAMES = 5
# Max runs per side when turning Poisson means into win probabilities.
MAX_RUNS = 25
# Share of tie (equal-run) probability given to the home team — extra
# innings, with the usual small home edge.
EXTRA_INNINGS_HOME = 0.54

AL, NL = "AL", "NL"
LEAGUE_CODE = {AL: 0, NL: 1}

# 30 MLB clubs -> (league, division).
TEAMS = {
    "Baltimore Orioles": (AL, "East"),
    "Boston Red Sox": (AL, "East"),
    "New York Yankees": (AL, "East"),
    "Tampa Bay Rays": (AL, "East"),
    "Toronto Blue Jays": (AL, "East"),
    "Chicago White Sox": (AL, "Central"),
    "Cleveland Guardians": (AL, "Central"),
    "Detroit Tigers": (AL, "Central"),
    "Kansas City Royals": (AL, "Central"),
    "Minnesota Twins": (AL, "Central"),
    "Athletics": (AL, "West"),
    "Houston Astros": (AL, "West"),
    "Los Angeles Angels": (AL, "West"),
    "Seattle Mariners": (AL, "West"),
    "Texas Rangers": (AL, "West"),
    "Atlanta Braves": (NL, "East"),
    "Miami Marlins": (NL, "East"),
    "New York Mets": (NL, "East"),
    "Philadelphia Phillies": (NL, "East"),
    "Washington Nationals": (NL, "East"),
    "Chicago Cubs": (NL, "Central"),
    "Cincinnati Reds": (NL, "Central"),
    "Milwaukee Brewers": (NL, "Central"),
    "Pittsburgh Pirates": (NL, "Central"),
    "St. Louis Cardinals": (NL, "Central"),
    "Arizona Diamondbacks": (NL, "West"),
    "Colorado Rockies": (NL, "West"),
    "Los Angeles Dodgers": (NL, "West"),
    "San Diego Padres": (NL, "West"),
    "San Francisco Giants": (NL, "West"),
}

LEAGUE = {team: lg for team, (lg, _) in TEAMS.items()}
DIVISION = {team: f"{lg} {div}" for team, (lg, div) in TEAMS.items()}

# Ballpark run factors (home team's park), ~1.0 = neutral. Loosely based
# on multi-year run park factors; Coors is the famous outlier. Used both
# as a model feature and to shape the sample-data run environment.
PARK_FACTORS = {
    "Colorado Rockies": 1.15, "Boston Red Sox": 1.06, "Cincinnati Reds": 1.06,
    "Kansas City Royals": 1.03, "Arizona Diamondbacks": 1.03,
    "Baltimore Orioles": 1.02, "Texas Rangers": 1.02, "New York Yankees": 1.02,
    "Chicago Cubs": 1.01, "Philadelphia Phillies": 1.01, "Toronto Blue Jays": 1.01,
    "Atlanta Braves": 1.00, "Minnesota Twins": 1.00, "Washington Nationals": 1.00,
    "Houston Astros": 1.00, "St. Louis Cardinals": 0.99, "Los Angeles Angels": 0.99,
    "Chicago White Sox": 0.99, "Milwaukee Brewers": 0.99, "Pittsburgh Pirates": 0.98,
    "Los Angeles Dodgers": 0.98, "New York Mets": 0.97, "Detroit Tigers": 0.97,
    "Miami Marlins": 0.97, "Cleveland Guardians": 0.97, "Athletics": 0.96,
    "Tampa Bay Rays": 0.95, "San Diego Padres": 0.95, "Seattle Mariners": 0.94,
    "San Francisco Giants": 0.94,
}


# --- Team-name normalization ------------------------------------------
# The MLB Stats API uses the full names above, but abbreviations and old
# spellings ("Oakland Athletics") show up too, so incoming names are run
# through normalize_team() first.
import unicodedata


def _slug(name) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return s.lower().strip()


_TEAM_ALIASES = {
    "oakland athletics": "Athletics",
    "oakland a's": "Athletics",
    "cleveland indians": "Cleveland Guardians",
}

_SLUG_TO_CANONICAL = {_slug(team): team for team in TEAMS}
# Also index by last word (nickname) for loose matches, when unambiguous.
_NICK_TO_CANONICAL = {}
for _team in TEAMS:
    _nick = _team.rsplit(" ", 1)[-1].lower()
    _NICK_TO_CANONICAL.setdefault(_nick, []).append(_team)
_NICK_TO_CANONICAL = {n: t[0] for n, t in _NICK_TO_CANONICAL.items() if len(t) == 1}


def normalize_team(name):
    """Map any data-source spelling to a canonical MLB team name, or None
    for non-MLB entries (e.g. All-Star / exhibition placeholders)."""
    if name in TEAMS:
        return name
    slug = _slug(name)
    if slug in _SLUG_TO_CANONICAL:
        return _SLUG_TO_CANONICAL[slug]
    if slug in _TEAM_ALIASES:
        return _TEAM_ALIASES[slug]
    return _NICK_TO_CANONICAL.get(slug.rsplit(" ", 1)[-1])
