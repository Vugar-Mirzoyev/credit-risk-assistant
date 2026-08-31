"""Download the policy corpus from the eCFR.

    python scripts/fetch_policies.py

Three regulations, chosen because each one governs something Layer 1 actually
built:

============  ===========================================================
Regulation B  Equal Credit Opportunity Act, 12 CFR 1002. Defines what an
              "empirically derived, demonstrably and statistically sound"
              credit scoring system is, what may not be used as a basis for
              a decision, and what an adverse action notice must contain.
              The champion's points table exists to satisfy that last one.
Regulation V  Fair Credit Reporting Act, 12 CFR 1022. Risk-based pricing
              notices and credit score disclosure - the rules around the
              bureau attributes that make up most of the feature set.
Regulation Z  Truth in Lending, 12 CFR 1026. Disclosure of rate, term and
              instalment, which are three of the model's inputs.
============  ===========================================================

Regulation Z is much the largest and much the least relevant to a credit
decision. That is deliberate. A corpus where everything is relevant makes
retrieval look good for the wrong reason; a corpus with a large adjacent-but-
different body of text is what makes recall@k mean something.

The downloaded files are committed. eCFR content changes over time, so a
corpus that is fetched fresh on every run is not reproducible: the issue date
is recorded here and the XML is kept, which means a reviewer can rerun the
retrieval evaluation without a network connection and get the same numbers.
US federal regulations are public domain.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ECFR = "https://www.ecfr.gov/api/versioner/v1"
USER_AGENT = "credit-risk-assistant/1.0 (educational portfolio project)"
TIMEOUT_SECONDS = 180

#: Part number to filename. The name records what the part is, because
#: "1002" means nothing to a reader six months later.
PARTS = {
    "1002": "12cfr1002_regulation_b.xml",
    "1022": "12cfr1022_regulation_v_fcra.xml",
    "1026": "12cfr1026_regulation_z_truth_in_lending.xml",
}

CORPUS_DIR = PROJECT_ROOT / "data" / "policies"
MANIFEST = CORPUS_DIR / "MANIFEST.json"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def latest_issue_date(title: int = 12) -> str:
    """The date the eCFR last reissued this title."""
    titles = json.loads(_get(f"{ECFR}/titles.json"))["titles"]
    entry = next(t for t in titles if t["number"] == title)
    return entry["latest_issue_date"]


def fetch(force: bool = False) -> dict:
    """Download each part, and write a manifest recording what was fetched."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    date = latest_issue_date()
    print(f"eCFR title 12, issue date {date}")

    entries = []
    for part, filename in PARTS.items():
        destination = CORPUS_DIR / filename
        if destination.exists() and not force:
            print(f"  {filename:<48} already present, skipping")
        else:
            raw = _get(f"{ECFR}/full/{date}/title-12.xml?part={part}")
            destination.write_bytes(raw)
            print(f"  {filename:<48} {len(raw) / 1024 / 1024:.2f} MB")
        entries.append(
            {
                "part": part,
                "file": filename,
                "bytes": destination.stat().st_size,
            }
        )

    manifest = {
        "source": "eCFR api.versioner.v1, title 12",
        "issue_date": date,
        "documents": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {MANIFEST}")
    return manifest


def main() -> None:
    force = "--force" in sys.argv
    manifest = fetch(force=force)
    total = sum(d["bytes"] for d in manifest["documents"])
    print(f"corpus: {len(manifest['documents'])} documents, {total / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
