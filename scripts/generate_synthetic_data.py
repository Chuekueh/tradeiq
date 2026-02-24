"""Generate synthetic financial documents for the TradeIQ knowledge base.

Creates:
- Earnings call transcripts
- Trading strategy documents
- Market analysis reports
"""

import json
import random
from pathlib import Path

TRANSCRIPTS_DIR = Path("data/raw/transcripts")
STRATEGIES_DIR = Path("data/raw/strategies")
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "technology", "ceo": "Tim Cook"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "technology", "ceo": "Satya Nadella"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "technology", "ceo": "Sundar Pichai"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "consumer", "ceo": "Andy Jassy"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "consumer", "ceo": "Elon Musk"},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "technology", "ceo": "Jensen Huang"},
    {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "technology", "ceo": "Mark Zuckerberg"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "finance", "ceo": "Jamie Dimon"},
]

QUARTERS = [
    ("Q1", "2024", "March 2024"),
    ("Q2", "2024", "June 2024"),
    ("Q3", "2024", "September 2024"),
    ("Q4", "2024", "December 2024"),
    ("Q1", "2025", "March 2025"),
]


def generate_earnings_transcript(company: dict, quarter: str, year: str, period: str) -> str:
    revenue = random.randint(20, 120) * 1000
    revenue_growth = round(random.uniform(-5, 35), 1)
    eps = round(random.uniform(0.5, 8.0), 2)
    margin = round(random.uniform(15, 45), 1)
    guidance_low = revenue + random.randint(-2000, 3000)
    guidance_high = guidance_low + random.randint(1000, 5000)

    highlights = random.sample([
        f"Cloud revenue grew {random.randint(15, 50)}% year-over-year",
        f"Operating margin expanded by {random.randint(100, 400)} basis points",
        f"Free cash flow reached ${random.randint(5, 30)} billion",
        f"Returned ${random.randint(2, 15)} billion to shareholders via buybacks",
        f"AI-related revenue exceeded ${random.randint(1, 10)} billion",
        f"International revenue grew {random.randint(5, 25)}% in constant currency",
        f"Customer count increased by {random.randint(10, 40)}%",
        f"R&D spending increased {random.randint(5, 20)}% to support AI initiatives",
        f"Subscription revenue now represents {random.randint(40, 75)}% of total",
        f"Launched {random.randint(2, 8)} new products during the quarter",
    ], k=5)

    risks = random.sample([
        "macroeconomic uncertainty continues to impact enterprise spending",
        "foreign exchange headwinds reduced revenue by approximately 2-3%",
        "supply chain constraints in certain product categories",
        "increased competition in key markets putting pressure on pricing",
        "regulatory developments in the EU and China",
        "rising interest rates affecting consumer demand",
        "talent acquisition costs increasing in the AI space",
    ], k=3)

    return f"""# {company['name']} ({company['ticker']}) {quarter} {year} Earnings Call Transcript
Date: {period}
Ticker: {company['ticker']}
Filing Type: Earnings Transcript
Sector: {company['sector']}

---

## Opening Remarks

Operator: Good afternoon and welcome to {company['name']}'s {quarter} fiscal year {year} earnings conference call. I would like to turn the call over to the head of Investor Relations.

IR: Thank you. Good afternoon everyone. Joining me today is our CEO, {company['ceo']}, and our CFO. Before we begin, I want to remind everyone that this call may contain forward-looking statements.

## CEO Remarks

{company['ceo']}: Thank you and good afternoon everyone. I'm pleased to report strong results for {quarter} {year}.

Total revenue came in at ${revenue:,} million, representing {revenue_growth}% growth year-over-year. Earnings per share were ${eps}, exceeding consensus estimates.

### Key Highlights

{chr(10).join(f'- {h}' for h in highlights)}

### Outlook

Looking ahead, we expect {quarter} revenue to be in the range of ${guidance_low:,} million to ${guidance_high:,} million. We remain confident in our long-term strategy and continue to invest aggressively in AI capabilities.

## CFO Remarks

Our CFO: Thank you {company['ceo'].split()[0]}. Let me walk through the financial details.

### Revenue Breakdown
- Total revenue: ${revenue:,} million ({'+' if revenue_growth > 0 else ''}{revenue_growth}% YoY)
- Gross margin: {margin}%
- Operating income: ${int(revenue * margin / 100):,} million
- EPS: ${eps} (GAAP)

### Balance Sheet
- Cash and equivalents: ${random.randint(15, 80)} billion
- Total debt: ${random.randint(10, 50)} billion
- Share repurchases: ${random.randint(2, 15)} billion during the quarter

## Q&A Session

Analyst (Goldman Sachs): Can you provide more color on AI revenue and how you see it trending?

{company['ceo']}: Absolutely. AI has been a transformative driver for our business. We're seeing strong adoption across enterprise customers, and our AI-related revenue pipeline has more than doubled over the past year.

Analyst (Morgan Stanley): What are the key risks you're monitoring?

{company['ceo']}: The main risks we're watching include:
{chr(10).join(f'- {r}' for r in risks)}

Despite these headwinds, we believe our diversified business model and strong market position put us in an excellent position to navigate these challenges.

Analyst (JP Morgan): How should we think about margins going forward?

CFO: We expect margins to remain in the {margin - 2:.0f}% to {margin + 2:.0f}% range. We're balancing investment in growth areas like AI with discipline on operating expenses.

## Closing

{company['ceo']}: Thank you all for joining us today. We're excited about the opportunities ahead and remain focused on delivering long-term value for our shareholders.

---
*This transcript is for research purposes. Forward-looking statements are subject to risks and uncertainties.*
"""


def generate_trading_strategy(name: str, category: str, description: str, details: str) -> str:
    return f"""# Trading Strategy: {name}
Category: {category}
Risk Level: {random.choice(['Low', 'Medium', 'High', 'Very High'])}
Time Horizon: {random.choice(['Intraday', 'Swing (2-10 days)', 'Position (weeks-months)', 'Long-term (months-years)'])}
Suitable Markets: {random.choice(['Equities', 'Equities & Options', 'All Markets', 'Futures & Equities'])}

---

## Overview

{description}

## Strategy Details

{details}

## Key Indicators

- Primary: {random.choice(['RSI', 'MACD', 'Moving Averages', 'Bollinger Bands', 'VWAP', 'Volume Profile'])}
- Secondary: {random.choice(['ATR', 'Stochastic', 'OBV', 'Ichimoku Cloud', 'Fibonacci Retracements'])}
- Confirmation: {random.choice(['Volume', 'Price Action', 'Market Breadth', 'Sector Rotation', 'VIX'])}

## Risk Management

- Position sizing: {random.choice(['Fixed fractional (1-2% risk per trade)', 'Kelly Criterion', 'Volatility-adjusted', 'Equal weight'])}
- Stop loss: {random.choice(['ATR-based trailing stop', 'Support/resistance levels', 'Percentage-based (5-8%)', 'Chandelier exit'])}
- Maximum drawdown tolerance: {random.randint(10, 30)}%
- Win rate target: {random.randint(40, 65)}%
- Risk-reward ratio: 1:{round(random.uniform(1.5, 3.5), 1)}

## Backtesting Results (Hypothetical)

| Metric | Value |
|--------|-------|
| Annual Return | {random.randint(8, 35)}% |
| Sharpe Ratio | {round(random.uniform(0.8, 2.5), 2)} |
| Maximum Drawdown | {random.randint(8, 25)}% |
| Win Rate | {random.randint(42, 63)}% |
| Profit Factor | {round(random.uniform(1.2, 2.8), 2)} |

*Note: Past hypothetical performance does not guarantee future results. These are illustrative backtesting results only.*
"""


STRATEGIES = [
    {
        "name": "Momentum Breakout",
        "category": "Momentum",
        "description": "This strategy identifies stocks breaking out of consolidation patterns on high volume. It capitalizes on the tendency of strong trends to continue after a period of consolidation.",
        "details": """### Entry Criteria
1. Stock must be above its 200-day moving average (confirming uptrend)
2. Price consolidates for at least 10 trading days in a range of less than 10%
3. Breakout occurs on volume at least 2x the 20-day average
4. RSI must be between 50-70 (strong but not overbought)

### Exit Criteria
1. Trailing stop at 2x ATR below the highest close since entry
2. Take partial profits (50%) at 1.5x the consolidation range
3. Exit if RSI exceeds 80 and shows bearish divergence
4. Time-based exit: close position if no progress after 20 trading days"""
    },
    {
        "name": "Mean Reversion - Oversold Bounce",
        "category": "Mean Reversion",
        "description": "This strategy buys quality stocks that have experienced sharp short-term declines, betting on a reversion to the mean. It works best in range-bound or mildly bullish markets.",
        "details": """### Entry Criteria
1. Stock is a member of the S&P 500 (quality filter)
2. RSI(2) drops below 10 (extremely oversold on short timeframe)
3. Stock is still above its 200-day moving average (long-term uptrend intact)
4. The decline is not driven by a fundamental catalyst (earnings miss, fraud, etc.)

### Exit Criteria
1. Exit when RSI(2) rises above 70
2. Maximum holding period: 5 trading days
3. Stop loss at 5% below entry price
4. Take profit if stock recovers 50% of the recent decline within 2 days"""
    },
    {
        "name": "Pairs Trading - Sector Neutrality",
        "category": "Statistical Arbitrage",
        "description": "This strategy identifies pairs of correlated stocks within the same sector and trades the spread when it diverges from historical norms. It aims to be market-neutral, profiting from the relative movement between two stocks.",
        "details": """### Pair Selection
1. Both stocks must be in the same sector and sub-industry
2. Historical correlation must exceed 0.80 over the past 252 trading days
3. The spread must be cointegrated (pass Augmented Dickey-Fuller test at 95% confidence)
4. Both stocks must have average daily volume > $10 million

### Entry Criteria
1. The z-score of the spread exceeds +/- 2.0 standard deviations
2. Long the underperformer, short the outperformer
3. Position size: dollar-neutral (equal dollar amounts long and short)

### Exit Criteria
1. Spread reverts to within 0.5 standard deviations of the mean
2. Stop loss: spread widens to 3.0 standard deviations
3. Maximum holding period: 20 trading days
4. Exit immediately if correlation drops below 0.60"""
    },
    {
        "name": "Earnings Momentum",
        "category": "Event-Driven",
        "description": "This strategy capitalizes on post-earnings announcement drift (PEAD), where stocks tend to continue moving in the direction of an earnings surprise for weeks or months after the announcement.",
        "details": """### Entry Criteria
1. Company reports earnings that beat consensus estimates by at least 10%
2. Revenue also beats estimates (dual beat)
3. Stock gaps up at least 3% on the earnings day
4. Volume on earnings day is at least 3x the 20-day average

### Position Management
1. Enter on the first pullback after the earnings gap (usually within 1-3 days)
2. Entry price should be above the pre-earnings close but below the gap high
3. Size position at 50% initially, add remaining 50% if stock holds above the gap level for 5 days

### Exit Criteria
1. Trailing stop at the pre-earnings close price
2. Take partial profits (1/3) at 10% gain from entry
3. Exit remaining position before the next earnings report
4. Exit if sector rotation signals emerge (sector ETF breaks below 50-day MA)"""
    },
    {
        "name": "Volatility Contraction Breakout",
        "category": "Volatility",
        "description": "Based on the principle that periods of low volatility are followed by periods of high volatility. This strategy identifies stocks where Bollinger Band width has contracted to extreme lows, then trades the subsequent expansion.",
        "details": """### Setup Identification
1. Calculate Bollinger Band Width (BBW) as: (Upper Band - Lower Band) / Middle Band
2. BBW must be at or below its 6-month low
3. Stock should be in a clear trend (above or below 50-day MA)
4. Average True Range should also be at 6-month lows

### Entry Criteria
1. Wait for price to close outside the contracted Bollinger Bands
2. If bullish: buy when price closes above the upper band
3. If bearish: short when price closes below the lower band
4. Volume must confirm: at least 50% above 20-day average

### Exit Criteria
1. Trail stop using the middle Bollinger Band (20-day SMA)
2. Take profits when BBW expands to its 6-month average
3. Time stop: exit after 15 trading days if no significant expansion
4. Hard stop: 3% beyond the opposite Bollinger Band at entry"""
    },
]


def main() -> None:
    print("Generating synthetic financial documents...\n")

    # Generate earnings transcripts
    count = 0
    for company in COMPANIES:
        for quarter, year, period in QUARTERS:
            transcript = generate_earnings_transcript(company, quarter, year, period)
            filename = f"{company['ticker']}_{quarter}_{year}_earnings.md"
            (TRANSCRIPTS_DIR / filename).write_text(transcript, encoding="utf-8")
            count += 1

    print(f"Generated {count} earnings call transcripts in {TRANSCRIPTS_DIR}")

    # Generate trading strategy documents
    for strategy in STRATEGIES:
        doc = generate_trading_strategy(**strategy)
        filename = f"{strategy['name'].lower().replace(' ', '_').replace('-', '_')}.md"
        (STRATEGIES_DIR / filename).write_text(doc, encoding="utf-8")

    print(f"Generated {len(STRATEGIES)} trading strategy documents in {STRATEGIES_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
