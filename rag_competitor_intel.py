"""
RAG-Powered Competitive Intelligence Tool
==========================================
Author: Payal Jarviya
MBA Candidate | SKK GSB Seoul | AI & Business Analytics

Business Problem:
    Strategy teams spend hours manually sifting through competitor press releases,
    product pages, and earnings reports. This project builds a Retrieval-Augmented
    Generation (RAG) pipeline that: (1) indexes competitor documents into a vector
    store using TF-IDF embeddings, (2) answers natural-language strategy questions
    by retrieving the most relevant passages, and (3) synthesises a structured
    competitive intelligence brief — covering pricing, positioning, product gaps,
    and strategic risks.

    This is a fully local, API-free implementation using sklearn + cosine similarity.
    To upgrade: swap the TF-IDF retriever for OpenAI embeddings and the rule-based
    synthesiser for GPT-4o / Claude calls.

Dataset: Synthetic competitor intelligence corpus (generated internally)

Instructions:
    pip install pandas numpy matplotlib seaborn scikit-learn
    python rag_competitor_intel.py
    Output saved to output/competitive_brief.txt + output/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap, os, re, warnings
warnings.filterwarnings('ignore')

# ─── PALETTE ───────────────────────────────────────────────────────────────────
NAVY   = "#1a3a5c"
BLUE   = "#2E75B6"
GREEN  = "#27AE60"
RED    = "#C0392B"
ORANGE = "#E67E22"
GRAY   = "#95a5a6"
BG     = "#f8f9fa"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': BG,
    'figure.facecolor': 'white',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
})

os.makedirs('output', exist_ok=True)


# ─── SYNTHETIC CORPUS ──────────────────────────────────────────────────────────
CORPUS = [
    # ─── NovaTech Solutions ───────────────────────────────────────────────────
    {
        'id': 'NT-001', 'company': 'NovaTech Solutions', 'type': 'product_page',
        'title': 'NovaTech Platform — Enterprise Analytics Suite',
        'text': (
            "NovaTech Platform is the enterprise analytics suite designed for mid-market and "
            "enterprise clients. Pricing starts at $2,400 per month for up to 10 users, with "
            "custom enterprise contracts for 50+ seats. Core features include real-time dashboards, "
            "predictive forecasting powered by AutoML, and native CRM integrations with Salesforce "
            "and HubSpot. NovaTech claims 99.9% uptime SLA and offers dedicated customer success "
            "managers for enterprise accounts. The platform targets retail, financial services, and "
            "logistics verticals. Recent product update added a no-code drag-and-drop report builder, "
            "reducing time-to-insight for non-technical business users."
        )
    },
    {
        'id': 'NT-002', 'company': 'NovaTech Solutions', 'type': 'press_release',
        'title': 'NovaTech Raises $45M Series C to Accelerate APAC Expansion',
        'text': (
            "NovaTech Solutions today announced a $45 million Series C funding round led by Sequoia "
            "Capital Asia. The company intends to use the funding to expand its go-to-market in "
            "Southeast Asia, Japan, and South Korea — markets identified as high-growth opportunities. "
            "NovaTech cited strong pipeline in the Korean e-commerce and logistics sectors. The "
            "company expects to double its Seoul engineering office headcount within 12 months. "
            "NovaTech currently serves 850 enterprise customers globally and processes over 2 billion "
            "data events per day. The Series C values the company at $320 million post-money."
        )
    },
    {
        'id': 'NT-003', 'company': 'NovaTech Solutions', 'type': 'earnings_call',
        'title': 'NovaTech Q3 2024 Earnings — Key Highlights',
        'text': (
            "NovaTech reported Q3 2024 revenue of $28.4M, up 34% year-over-year. Net Revenue "
            "Retention (NRR) reached 118%, indicating strong expansion within existing accounts. "
            "The company added 72 net new enterprise logos in Q3. Gross margin improved to 74% "
            "from 68% in Q3 2023, driven by infrastructure cost optimisation. Management guided "
            "for full-year 2024 ARR of $115M, implying Q4 ARR of approximately $31M. Churn rate "
            "held at 6% annually. The AI forecasting module, launched in May, has been adopted by "
            "43% of the customer base within six months."
        )
    },
    # ─── Apex Analytics ───────────────────────────────────────────────────────
    {
        'id': 'AA-001', 'company': 'Apex Analytics', 'type': 'product_page',
        'title': 'Apex — Unified Data Platform for Growth Teams',
        'text': (
            "Apex Analytics is a self-serve analytics platform built for growth and marketing teams. "
            "Unlike enterprise BI tools, Apex requires no SQL or data engineering — business users "
            "can build funnels, cohort analyses, and attribution models through a visual interface. "
            "Pricing: Starter $299/month (up to 3 users), Growth $799/month (up to 15 users), "
            "Enterprise custom. Apex integrates with 200+ data sources including Shopify, Stripe, "
            "Segment, and Meta Ads. The platform's key differentiator is sub-second query speed "
            "on datasets up to 50GB without data warehousing infrastructure."
        )
    },
    {
        'id': 'AA-002', 'company': 'Apex Analytics', 'type': 'press_release',
        'title': 'Apex Analytics Launches AI-Powered Anomaly Detection Feature',
        'text': (
            "Apex Analytics has launched AnomalyGuard, an AI-powered anomaly detection layer that "
            "automatically surfaces unexpected changes in key business metrics. The feature uses "
            "a combination of statistical process control and LSTM neural networks to distinguish "
            "genuine business shifts from data noise. AnomalyGuard is included in all Growth and "
            "Enterprise plans at no additional cost. The company claims early beta users saw a 60% "
            "reduction in time-to-detection for critical metric drops. The launch positions Apex "
            "directly against Datadog and Monte Carlo in the data observability space."
        )
    },
    {
        'id': 'AA-003', 'company': 'Apex Analytics', 'type': 'customer_review',
        'title': 'G2 Reviews — Apex Analytics (aggregated)',
        'text': (
            "G2 rating: 4.5/5 from 312 reviews. Top strengths cited by customers: ease of use, "
            "fast onboarding, excellent Shopify and Stripe integrations. Top complaints: limited "
            "customisation for complex SQL-based use cases, no support for real-time streaming data "
            "beyond 15-minute refresh intervals, and customer support response times declining as "
            "the company scales. Several enterprise reviewers noted they outgrew the platform within "
            "18 months and migrated to Looker or Tableau. SMB and mid-market customers are highly "
            "satisfied. Competitors mentioned by reviewers: Mixpanel, Amplitude, Chartio."
        )
    },
    # ─── ClearMetrics ─────────────────────────────────────────────────────────
    {
        'id': 'CM-001', 'company': 'ClearMetrics', 'type': 'product_page',
        'title': 'ClearMetrics — Real-Time Revenue Intelligence',
        'text': (
            "ClearMetrics is a revenue intelligence platform focused on B2B SaaS companies. Its "
            "core products are: (1) Revenue Radar — real-time ARR/MRR tracking with expansion, "
            "contraction, and churn decomposition; (2) ForecastIQ — AI-powered revenue forecasting "
            "with scenario modelling; (3) CustomerPulse — health scores and churn risk alerts. "
            "Pricing starts at $1,800/month for up to 5 users. ClearMetrics integrates with "
            "Salesforce, HubSpot, Chargebee, Stripe, and Zuora. The platform claims to reduce "
            "revenue forecast error by 40% compared to spreadsheet-based methods."
        )
    },
    {
        'id': 'CM-002', 'company': 'ClearMetrics', 'type': 'press_release',
        'title': 'ClearMetrics Acquires DataBridge for $18M to Strengthen Data Pipeline',
        'text': (
            "ClearMetrics has acquired DataBridge, a data pipeline and ETL startup, for $18 million. "
            "The acquisition gives ClearMetrics native data ingestion capabilities, reducing "
            "customers' dependence on third-party ETL tools like Fivetran or Airbyte. ClearMetrics "
            "expects to integrate DataBridge technology within 6 months, enabling one-click data "
            "connectors for all major CRM, billing, and data warehouse platforms. The acquisition "
            "also includes 12 engineering hires specialising in real-time streaming data. This move "
            "signals ClearMetrics' ambition to compete with full-stack platforms like Salesforce "
            "Revenue Cloud and Gainsight."
        )
    },
    # ─── Market Landscape ─────────────────────────────────────────────────────
    {
        'id': 'MKT-001', 'company': 'Market Research', 'type': 'industry_report',
        'title': 'Business Intelligence & Analytics Market — 2024 Outlook',
        'text': (
            "The global business intelligence and analytics market was valued at $29.4B in 2023 and "
            "is projected to reach $54.3B by 2028, growing at a CAGR of 13.1%. Key growth drivers: "
            "AI/ML integration into analytics workflows, increased demand for self-serve BI tools, "
            "and adoption in SMB segment. Pricing pressure is intensifying as open-source tools "
            "like Apache Superset and Metabase gain enterprise traction. The APAC market is the "
            "fastest-growing region at 17% CAGR, led by South Korea, India, and Southeast Asia. "
            "M&A activity is accelerating — 14 acquisitions recorded in H1 2024 alone. Consolidation "
            "trend: smaller analytics vendors being absorbed by cloud hyperscalers (AWS, GCP, Azure)."
        )
    },
    {
        'id': 'MKT-002', 'company': 'Market Research', 'type': 'industry_report',
        'title': 'Customer Success & Churn Prevention Technology Trends',
        'text': (
            "Customer success platforms are increasingly converging with analytics and revenue "
            "intelligence. Health score models driven by product usage data, support ticket "
            "frequency, and engagement signals are now standard. AI-driven early warning systems "
            "are reducing churn by 20-30% in documented case studies. Investors are prioritising "
            "platforms with network effects and deep CRM integrations. Key differentiator emerging: "
            "companies that can predict churn 90+ days in advance vs. 30-day reactive models. "
            "Pricing benchmarks: enterprise customer success platforms range from $1,200 to $4,500 "
            "per month for mid-market clients (50-500 employee companies)."
        )
    },
]


# ─── RAG ENGINE ────────────────────────────────────────────────────────────────
class RAGCompetitiveIntelligence:
    def __init__(self, corpus):
        self.docs = pd.DataFrame(corpus)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            stop_words='english',
            sublinear_tf=True
        )
        self.embeddings = self.vectorizer.fit_transform(self.docs['text'])
        print(f"\n  RAG Index built: {len(self.docs)} documents | "
              f"{self.embeddings.shape[1]:,} TF-IDF features")

    def retrieve(self, query, top_k=3):
        q_vec = self.vectorizer.transform([query])
        sims  = cosine_similarity(q_vec, self.embeddings).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            results.append({
                'doc_id'    : self.docs.iloc[idx]['id'],
                'company'   : self.docs.iloc[idx]['company'],
                'doc_type'  : self.docs.iloc[idx]['type'],
                'title'     : self.docs.iloc[idx]['title'],
                'text'      : self.docs.iloc[idx]['text'],
                'similarity': sims[idx],
            })
        return results

    def answer(self, query, top_k=3):
        hits = self.retrieve(query, top_k=top_k)
        print(f"\n  Query: \"{query}\"")
        print(f"  {'─'*56}")
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] [{h['similarity']:.3f}] {h['company']} — {h['title']}")
        return hits


# ─── COMPETITIVE BRIEF SYNTHESIS ───────────────────────────────────────────────
STRATEGY_QUERIES = [
    ("Pricing & Packaging", "What are competitor pricing tiers and packaging strategies?"),
    ("Product Differentiation", "What are the key product features and differentiators for each competitor?"),
    ("APAC & Korea Expansion", "Which competitors are expanding into APAC and South Korea?"),
    ("AI & ML Features", "What AI and machine learning capabilities are competitors offering?"),
    ("Funding & Financials", "What is the funding status, revenue, and growth of key competitors?"),
    ("Customer Weaknesses", "What do customers complain about or where do competitors fall short?"),
    ("M&A & Strategic Moves", "What acquisitions or strategic moves have competitors made recently?"),
    ("Market Opportunity", "What is the market size, growth rate, and key trends in this space?"),
]

def generate_brief(rag, queries):
    brief_lines = []
    brief_lines.append("=" * 68)
    brief_lines.append("  COMPETITIVE INTELLIGENCE BRIEF")
    brief_lines.append("  Author: Payal Jarviya | SKK GSB Seoul | AI & Business Analytics")
    brief_lines.append("  Generated by: RAG-Powered Competitive Intelligence Tool")
    brief_lines.append("=" * 68)
    brief_lines.append("")

    all_hits = {}
    for section, query in queries:
        hits = rag.retrieve(query, top_k=2)
        brief_lines.append(f"── {section.upper()} {'─'*(64-len(section))}")
        for h in hits:
            snippet = h['text'][:400].replace('\n', ' ')
            wrapped = textwrap.fill(snippet + "...", width=66, initial_indent="  ", subsequent_indent="  ")
            brief_lines.append(f"  Source: {h['company']} ({h['doc_type']}) | Relevance: {h['similarity']:.3f}")
            brief_lines.append(wrapped)
            brief_lines.append("")
        all_hits[section] = hits

    brief_lines.append("=" * 68)
    brief_lines.append("  STRATEGIC SUMMARY")
    brief_lines.append("=" * 68)
    brief_lines.append(textwrap.fill(
        "Based on the retrieved intelligence: (1) NovaTech is the best-funded competitor "
        "with $45M Series C and aggressive APAC expansion — direct threat in Korean market. "
        "(2) Apex Analytics dominates SMB/mid-market self-serve with low-code positioning but "
        "loses enterprise customers at scale — a potential whitespace opportunity. "
        "(3) ClearMetrics is building a full-stack revenue intelligence suite via acquisition — "
        "watch for expanding footprint in B2B SaaS segment. "
        "(4) APAC is the fastest-growing region (17% CAGR) — timing is right for a differentiated "
        "entrant. (5) Common competitor weakness: complex onboarding and weak customisation for "
        "technical users — potential positioning angle.",
        width=66, initial_indent="  ", subsequent_indent="  "
    ))
    brief_lines.append("")

    return '\n'.join(brief_lines)


# ─── VISUALISATIONS ────────────────────────────────────────────────────────────
def plot_competitor_matrix():
    """Positioning matrix: Price vs. Technical Complexity."""
    competitors = {
        'NovaTech\nSolutions'   : {'price': 8,  'complexity': 7,  'market_cap': 320, 'funding': 45},
        'Apex\nAnalytics'       : {'price': 4,  'complexity': 3,  'market_cap': 80,  'funding': 22},
        'ClearMetrics'          : {'price': 6,  'complexity': 5,  'market_cap': 150, 'funding': 30},
        'Looker\n(Google)'      : {'price': 9,  'complexity': 9,  'market_cap': 2600,'funding': 0},
        'Metabase\n(OSS)'       : {'price': 1,  'complexity': 4,  'market_cap': 0,   'funding': 0},
        'Our Product\n(Target)' : {'price': 5,  'complexity': 4,  'market_cap': 0,   'funding': 0},
    }
    df = pd.DataFrame(competitors).T.reset_index()
    df.columns = ['name', 'price', 'complexity', 'market_cap', 'funding']

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [BLUE, GREEN, ORANGE, RED, GRAY, NAVY]
    for i, (_, row) in enumerate(df.iterrows()):
        size = max(200, row['market_cap'] * 0.8) if row['market_cap'] > 0 else 300
        ax.scatter(row['price'], row['complexity'], s=size, color=colors[i],
                   alpha=0.7, edgecolors='white', linewidth=2, zorder=3)
        ax.annotate(row['name'], (row['price'], row['complexity']),
                    textcoords="offset points", xytext=(0, 14),
                    ha='center', fontsize=9, fontweight='bold', color=colors[i])

    ax.axvline(5, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(5, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
    ax.text(2, 9.5, 'Low Price\nHigh Complexity', ha='center', fontsize=8, color=GRAY, style='italic')
    ax.text(8, 9.5, 'High Price\nHigh Complexity', ha='center', fontsize=8, color=GRAY, style='italic')
    ax.text(2, 0.5, 'Low Price\nLow Complexity', ha='center', fontsize=8, color=GRAY, style='italic')
    ax.text(8, 0.5, 'High Price\nLow Complexity', ha='center', fontsize=8, color=GRAY, style='italic')

    ax.set_xlim(0, 10.5); ax.set_ylim(0, 10.5)
    ax.set_xlabel('Pricing (1=Low / 10=High)', fontsize=11)
    ax.set_ylabel('Technical Complexity (1=Easy / 10=Complex)', fontsize=11)
    ax.set_title('Competitive Positioning Matrix\n(Bubble size = estimated market cap)', fontsize=12)
    ax.text(4.7, 4.5, '← Opportunity\n   Zone', fontsize=9, color=NAVY, fontweight='bold', alpha=0.7)

    plt.tight_layout()
    plt.savefig('output/competitor_positioning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: output/competitor_positioning.png")


def plot_pricing_comparison():
    data = {
        'Competitor'  : ['NovaTech\nEnterprise', 'ClearMetrics\nBase', 'Apex\nGrowth', 'Apex\nStarter', 'Metabase\nCloud'],
        'Monthly ($)' : [2400, 1800, 799, 299, 500],
        'Color'       : [RED, ORANGE, BLUE, GREEN, GRAY],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df['Competitor'], df['Monthly ($)'], color=df['Color'],
                  width=0.5, edgecolor='white')
    for bar, val in zip(bars, df['Monthly ($)']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"${val:,}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Monthly Price (USD)')
    ax.set_title('Competitor Pricing Comparison\n(Entry / Mid-tier plans)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${int(x):,}'))

    plt.tight_layout()
    plt.savefig('output/pricing_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: output/pricing_comparison.png")


def plot_similarity_heatmap(rag):
    """Show semantic similarity between all documents."""
    sim_matrix = cosine_similarity(rag.embeddings)
    labels = [f"{d['id']}" for _, d in rag.docs.iterrows()]

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.eye(len(labels), dtype=bool)
    sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels,
                cmap='Blues', annot=True, fmt='.2f', linewidths=0.5,
                mask=mask, ax=ax, annot_kws={'size': 7})
    ax.set_title('Document Similarity Matrix\n(TF-IDF Cosine Similarity across Corpus)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig('output/similarity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: output/similarity_heatmap.png")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  RAG-Powered Competitive Intelligence Tool")
    print("  Author: Payal Jarviya | SKK GSB Seoul\n")

    rag = RAGCompetitiveIntelligence(CORPUS)

    print(f"\n{'='*60}")
    print(f"  RUNNING STRATEGY QUERIES")
    print(f"{'='*60}")
    for section, query in STRATEGY_QUERIES:
        rag.answer(query, top_k=2)

    print(f"\n  Generating competitive brief...")
    brief = generate_brief(rag, STRATEGY_QUERIES)
    with open('output/competitive_brief.txt', 'w') as f:
        f.write(brief)
    print("  Saved: output/competitive_brief.txt")
    print(brief)

    print(f"\n  Generating charts...")
    plot_competitor_matrix()
    plot_pricing_comparison()
    plot_similarity_heatmap(rag)

    print("  Done.\n")
