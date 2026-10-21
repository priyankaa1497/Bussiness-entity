"""Text normalisation for business names and addresses.

Design goals (derived from the v1 error analysis):
* Never drop non-Latin text.  v1 threw away Devanagari / Kannada names, which made
  ~40% of Indian records unblockable.  Here every script is transliterated to ASCII
  with ``anyascii`` (pure offline table lookup, ISC licence - no external service).
* Country-agnostic: no hard-coded {US, India} logic.  Abbreviation tables are just
  normalisation vocab and apply to any country (French tokens included).
* Produce several *views* of each record that later stages can exploit:
  - ``name_norm``  : full normalised name
  - ``name_core``  : name with legal-form tokens removed (Pvt Ltd, Inc, SARL ...)
  - ``name_compact``: letters/digits only, no spaces (matches domain-name variants
                      such as ``balajicollege.com`` vs ``Balaji College``)
  - ``name_skel``  : consonant skeleton (vowels removed) - robust to transliteration
                      spelling differences (``lakshmi`` vs ``lksmi`` -> ``lksm``)
  - ``addr_norm``  : normalised address
  - ``addr_nums``  : all digit runs found in the address (house / plot numbers ...)
  - ``postal``     : postal code candidate (5 or 6 digit run, optionally 5+4)
  - ``tokens``     : blocking tokens (name + address + compact name + skeleton)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from anyascii import anyascii

# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------

# Canonical legal-form spellings.  Keys are lower-case tokens *after* punctuation removal.
LEGAL_CANON: dict[str, str] = {
    # India / UK style
    "pvt": "private", "pvtltd": "private limited", "pvt.": "private", "p": "private",
    "ltd": "limited", "ltd.": "limited", "limitd": "limited", "lmt": "limited",
    "llp": "llp", "opc": "opc",
    "pra": "private", "li": "limited",  # transliterated "प्रा. लि."
    # US style
    "inc": "incorporated", "inc.": "incorporated", "incorp": "incorporated",
    "corp": "corporation", "corp.": "corporation", "co": "company", "cos": "companies",
    "llc": "llc", "l.l.c": "llc", "plc": "plc", "lp": "lp", "pc": "pc",
    # French / EU style
    "sarl": "sarl", "s.a.r.l": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa",
    "eurl": "eurl", "sci": "sci", "snc": "snc", "gmbh": "gmbh", "ag": "ag", "bv": "bv",
    "ltee": "limitee", "ltée": "limitee",
}

LEGAL_TOKENS: set[str] = {
    "private", "limited", "llp", "opc", "incorporated", "corporation", "company",
    "companies", "llc", "plc", "lp", "pc", "sarl", "sas", "sasu", "sa", "eurl", "sci",
    "snc", "gmbh", "ag", "bv", "limitee", "pvt", "ltd", "inc", "corp", "co", "pty",
    "group", "enterprises", "enterprise", "industries", "traders", "trading", "services",
    "solutions", "technologies", "technology", "international", "india", "associates",
    "dba", "the", "and", "of",
}

# Address abbreviations -> canonical (applies to any country; harmless when absent)
ADDR_CANON: dict[str, str] = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "ln": "lane", "dr": "drive",
    "hwy": "highway", "pkwy": "parkway", "ct": "court", "pl": "place", "sq": "square",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "bldg": "building",
    "bl": "building", "no": "number", "nos": "number", "num": "number", "n°": "number",
    "opp": "opposite", "nr": "near", "bhd": "behind", "mkt": "market", "ind": "industrial",
    "indl": "industrial", "estt": "estate", "est": "estate", "extn": "extension",
    "ext": "extension", "colony": "colony", "nagar": "nagar", "gali": "gali", "marg": "road",
    "sec": "sector", "sect": "sector", "ph": "phase", "blk": "block", "twp": "township",
    "vlg": "village", "vill": "village", "dist": "district", "distt": "district",
    "tal": "taluka", "tq": "taluka", "teh": "tehsil", "po": "post", "ps": "post",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast",
    "nw": "northwest", "se": "southeast", "sw": "southwest",
    "mt": "mount", "ft": "fort", "hts": "heights", "pk": "park", "gdn": "garden",
    "gdns": "gardens", "cir": "circle", "ter": "terrace", "trl": "trail", "xing": "crossing",
    "r": "rue", "imp": "impasse", "che": "chemin", "chem": "chemin", "all": "allee",
    "pce": "place", "rte": "route", "fbg": "faubourg", "qu": "quai", "res": "residence",
    "lot": "lotissement", "za": "zone activite", "zi": "zone industrielle", "zac": "zac",
    "cedex": "",  # French postal routing word - carries no location signal
}

# Words to drop from *blocking tokens* (kept in the normalised strings for features).
BLOCK_STOP: set[str] = {
    "the", "a", "an", "and", "of", "for", "to", "at", "in", "on", "by", "with", "near",
    "opposite", "behind", "beside", "number", "no", "floor", "road", "street", "avenue",
    "nagar", "colony", "main", "cross", "block", "sector", "phase", "building", "plot",
    "flat", "house", "shop", "office", "post", "district", "village", "city", "town",
    "private", "limited", "incorporated", "corporation", "company", "llc", "inc", "ltd",
    "pvt", "corp", "co", "sarl", "sas", "sa", "group", "enterprises", "industries",
    "services", "solutions", "india", "international", "rue", "boulevard", "place",
    "chemin", "route", "region", "zone",
}

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_TOKEN = re.compile(r"[a-z0-9]+")
_DIGITS = re.compile(r"\d+")
_POSTAL = re.compile(r"(?<!\d)(\d{5}(?:-\d{4})?|\d{6})(?!\d)")
_URL_SUFFIX = re.compile(r"\.(com|in|co|org|net|io|biz|info|fr|us|eu|co\.in|net\.in|org\.in)\b")
_VOWELS = re.compile(r"[aeiou]")
_NON_LATIN = re.compile(r"[^\x00-\x7F]")


def _split_camel_digits(s: str) -> str:
    # "wz1390" -> "wz 1390"; "12b" stays as is (letters after digits are unit suffixes)
    return re.sub(r"(?<=[a-z])(?=\d)", " ", s)


def to_ascii_lower(text: str) -> str:
    """NFKC -> anyascii transliteration -> lower-case ASCII."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    if _NON_LATIN.search(t):
        t = anyascii(t)
    return t.lower()


def _apply_vocab(tokens: list[str], vocab: dict[str, str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        rep = vocab.get(tok)
        if rep is None:
            out.append(tok)
        elif rep:
            out.extend(rep.split())
    return out


def normalize_name(raw: str) -> tuple[str, str, str, str, bool]:
    """Return (name_norm, name_core, name_compact, name_skel, was_non_latin)."""
    if raw is None:
        return "", "", "", "", False
    raw_s = str(raw)
    non_latin = bool(_NON_LATIN.search(raw_s))
    t = to_ascii_lower(raw_s)
    t = _URL_SUFFIX.sub(" ", t)
    t = t.replace("www.", " ")
    t = t.replace("&", " and ").replace("@", " at ").replace("+", " plus ")
    t = t.replace("'", "").replace("’", "")
    t = _NON_ALNUM.sub(" ", t)
    t = _split_camel_digits(t)
    toks = _TOKEN.findall(t)
    toks = _apply_vocab(toks, LEGAL_CANON)
    # "p ltd" / "private ltd" -> "private limited" handled through vocab above
    name_norm = " ".join(toks)
    core_toks = [x for x in toks if x not in LEGAL_TOKENS]
    if not core_toks:  # name was purely legal words - keep original
        core_toks = toks
    name_core = " ".join(core_toks)
    name_compact = "".join(core_toks)
    skel = _VOWELS.sub("", name_compact)
    skel = re.sub(r"(.)\1+", r"\1", skel)  # collapse doubled letters
    return name_norm, name_core, name_compact, skel, non_latin


def normalize_address(raw: str) -> tuple[str, list[str], str]:
    """Return (addr_norm, digit_runs, postal)."""
    if raw is None:
        return "", [], ""
    raw_s = str(raw)
    postal = ""
    m = _POSTAL.findall(raw_s)
    if m:
        # prefer the last 5/6 digit run that is not the first house number
        postal = m[-1].split("-")[0]
    t = to_ascii_lower(raw_s)
    t = t.replace("&", " and ")
    t = re.sub(r"[/\\,;:()\[\]{}|]", " ", t)
    t = t.replace("-", " ").replace(".", " ")
    t = _NON_ALNUM.sub(" ", t)
    t = _split_camel_digits(t)
    toks = _TOKEN.findall(t)
    toks = _apply_vocab(toks, ADDR_CANON)
    addr_norm = " ".join(toks)
    nums = _DIGITS.findall(addr_norm)
    return addr_norm, nums, postal


def _soundex_like(tok: str) -> str:
    """Tiny offline phonetic code (no extra dependency). Good enough for Indian/US spellings."""
    if len(tok) < 3:
        return tok
    first = tok[0]
    mapping = str.maketrans("bfpvcgjkqsxzdtlmnr", "111122222222334556")
    body = tok[1:].translate(mapping)
    body = re.sub(r"[aeiouhwy]", "", body)
    body = re.sub(r"(.)\1+", r"\1", body)
    return (first + body + "000")[:4]


def blocking_tokens(name_core: str, name_compact: str, name_skel: str, addr_norm: str, postal: str) -> list[str]:
    """Tokens used by the sparse TF-IDF blocker.

    Name tokens get a ``n:`` prefix, address tokens ``a:``, so the same word in the two
    fields is treated as two different features (the address word "bharat" should not
    match the name word "bharat").  Compact / skeleton / phonetic / prefix keys recover
    domain-name and transliteration variants that share few exact tokens.
    """
    out: list[str] = []
    core_toks = [t for t in name_core.split() if t not in BLOCK_STOP and len(t) > 1]
    for tok in core_toks:
        out.append("n:" + tok)
        if len(tok) >= 4:
            out.append("nx:" + _soundex_like(tok))
    for a, b in zip(core_toks, core_toks[1:]):
        out.append(f"nb:{a}_{b}")
    if len(name_compact) >= 5:
        out.append("nc:" + name_compact)
        out.append("np4:" + name_compact[:4])
        if len(name_compact) >= 6:
            out.append("ns:" + "".join(sorted(name_compact)))  # anagram / reorder of letters
    if len(name_skel) >= 4:
        out.append("nk:" + name_skel)
    if core_toks:
        initials = "".join(t[0] for t in core_toks if t)
        if len(initials) >= 2:
            out.append("ni:" + initials)
        if postal and len(core_toks[0]) >= 3:
            out.append(f"npp:{core_toks[0]}_{postal}")
    for tok in addr_norm.split():
        if tok in BLOCK_STOP or len(tok) <= 1:
            continue
        out.append("a:" + tok)
    # digit runs in address (house / plot) — high precision when combined with name tokens via TF-IDF
    for num in _DIGITS.findall(addr_norm):
        if len(num) >= 2:
            out.append("anum:" + num)
    if postal:
        out.append("p:" + postal)
    return out


def exact_keys(name_core: str, name_compact: str, name_skel: str, postal: str) -> list[str]:
    """High-precision inverted-index keys (exact / near-exact channel)."""
    keys: list[str] = []
    if name_core and len(name_core) >= 3:
        keys.append("ec:" + name_core)
    if name_compact and len(name_compact) >= 5:
        keys.append("ex:" + name_compact)
    if name_skel and len(name_skel) >= 5:
        keys.append("ek:" + name_skel)
    first = name_core.split()[0] if name_core else ""
    if postal and first and len(first) >= 3:
        keys.append(f"ep:{first}_{postal}")
    return keys


@dataclass
class Record:
    entity_id: str
    country: str
    name_norm: str
    name_core: str
    name_compact: str
    name_skel: str
    addr_norm: str
    addr_nums: list[str] = field(default_factory=list)
    postal: str = ""
    non_latin: bool = False
    tokens: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)


def prepare_record(entity_id: str, name: str, address: str, country: str) -> Record:
    name_norm, name_core, name_compact, name_skel, non_latin = normalize_name(name)
    addr_norm, nums, postal = normalize_address(address)
    toks = blocking_tokens(name_core, name_compact, name_skel, addr_norm, postal)
    keys = exact_keys(name_core, name_compact, name_skel, postal)
    return Record(
        entity_id=str(entity_id),
        country=(country or "").strip(),
        name_norm=name_norm,
        name_core=name_core,
        name_compact=name_compact,
        name_skel=name_skel,
        addr_norm=addr_norm,
        addr_nums=nums,
        postal=postal,
        non_latin=non_latin,
        tokens=toks,
        keys=keys,
    )
