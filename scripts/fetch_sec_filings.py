"""Fetch SEC EDGAR filings for sample companies.

EDGAR provides free access to all public company filings.
See: https://www.sec.gov/edgar/searchedgar/companysearch
"""

import json
import time
from pathlib import Path

import httpx

OUTPUT_DIR = Path("data/raw/filings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# EDGAR requires a User-Agent header with contact info
HEADERS = {"User-Agent": "TradeIQ research@example.com", "Accept-Encoding": "gzip, deflate"}

# Sample companies to fetch
COMPANIES = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "TSLA": "0001318605",
    "NVDA": "0001045810",
    "META": "0001326801",
    "JPM": "0000019617",
}


def fetch_filing_list(cik: str, filing_type: str = "10-K", count: int = 3) -> list[dict]:
    """Fetch recent filings metadata from EDGAR."""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{filing_type}%22&dateRange=custom&startdt=2022-01-01&enddt=2025-12-31&forms={filing_type}&from=0&size={count}"
    # Use the submissions API instead
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    resp = httpx.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == filing_type and len(filings) < count:
            filings.append({
                "form": form,
                "date": dates[i],
                "accession": accessions[i].replace("-", ""),
                "primary_doc": primary_docs[i],
                "cik": cik,
            })

    return filings


def fetch_filing_text(cik: str, accession: str, primary_doc: str) -> str:
    """Fetch the actual filing document text."""
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
    resp = httpx.get(url, headers=HEADERS, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def extract_sections_from_10k(html_text: str, ticker: str) -> dict[str, str]:
    """Extract key sections from a 10-K filing HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "lxml")
    text = soup.get_text(separator="\n", strip=True)

    # Simple section extraction based on common 10-K headers
    sections: dict[str, str] = {}
    section_markers = {
        "risk_factors": ["Item 1A", "Risk Factors"],
        "business": ["Item 1.", "Business"],
        "mda": ["Item 7.", "Management's Discussion and Analysis"],
        "financial_statements": ["Item 8.", "Financial Statements"],
    }

    for section_name, markers in section_markers.items():
        for marker in markers:
            start = text.find(marker)
            if start != -1:
                # Take a reasonable chunk after the marker
                end = min(start + 15000, len(text))
                sections[section_name] = text[start:end]
                break

    # If no sections found, take a summary of the full text
    if not sections:
        sections["full_text"] = text[:20000]

    return sections


def main() -> None:
    print("Fetching SEC EDGAR filings...")

    for ticker, cik in COMPANIES.items():
        print(f"\n--- {ticker} (CIK: {cik}) ---")
        try:
            filings = fetch_filing_list(cik, filing_type="10-K", count=2)
            print(f"  Found {len(filings)} 10-K filings")

            for filing in filings:
                try:
                    print(f"  Fetching {filing['form']} from {filing['date']}...")
                    html = fetch_filing_text(
                        cik.lstrip("0"),
                        filing["accession"],
                        filing["primary_doc"],
                    )

                    sections = extract_sections_from_10k(html, ticker)

                    for section_name, content in sections.items():
                        filename = f"{ticker}_{filing['form']}_{filing['date']}_{section_name}.md"
                        out_path = OUTPUT_DIR / filename

                        # Format as markdown
                        md_content = f"""# {ticker} {filing['form']} - {section_name.replace('_', ' ').title()}
Filing Date: {filing['date']}
Ticker: {ticker}
Filing Type: {filing['form']}
Section: {section_name}

---

{content}
"""
                        out_path.write_text(md_content, encoding="utf-8")
                        print(f"    Saved: {filename} ({len(content)} chars)")

                    # Respect EDGAR rate limits (10 requests/sec)
                    time.sleep(0.2)

                except Exception as e:
                    print(f"  Error fetching filing: {e}")

        except Exception as e:
            print(f"  Error fetching filing list for {ticker}: {e}")

        time.sleep(0.2)

    print(f"\nDone! Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
