# RAG-Powered Competitive Intelligence Tool

**Business Question:** Which competitors pose the greatest strategic threat — and where are the gaps in their offering we can exploit?

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)

---

## What This Tool Does

This is a fully local, API-free Retrieval-Augmented Generation (RAG) pipeline for competitive intelligence. It:

1. **Indexes** a corpus of competitor documents (product pages, press releases, earnings calls, customer reviews, industry reports) using TF-IDF embeddings
2. **Retrieves** the most semantically relevant passages for any natural-language strategy question
3. **Synthesises** a structured competitive brief covering pricing, positioning, product gaps, and strategic risks

No OpenAI API key required — runs entirely on local compute using `scikit-learn`.

---

## Competitive Landscape Covered

| Company | Documents Indexed | Key Intel |
|---------|-------------------|-----------|
| NovaTech Solutions | Product page, Series C press release, Q3 earnings | $45M raised, APAC expansion, $320M valuation |
| Apex Analytics | Product page, AnomalyGuard launch, G2 reviews | Self-serve positioning, SMB-focused, weak at enterprise scale |
| ClearMetrics | Product page, DataBridge acquisition | Revenue intelligence + ETL play, B2B SaaS focus |
| Market Research | BI market report, Customer success trends | $29.4B market, 13.1% CAGR, APAC fastest-growing |

---

## Key Intelligence Findings

| Strategic Dimension | Finding |
|--------------------|---------|
| Most aggressive competitor | NovaTech — $45M Series C, doubling Seoul headcount |
| Whitespace opportunity | Apex loses enterprise clients at scale — opportunity for technical-buyer positioning |
| APAC timing | NovaTech and market reports both signal APAC as high-growth; timing is right |
| Common weakness | Weak customisation, long onboarding, declining support at scale |
| M&A signal | ClearMetrics acquiring ETL infrastructure — full-stack ambition, watch this space |

---

## Business Problem

Strategy teams at fast-growing companies spend 10-20 hours per month manually tracking competitor moves across press releases, G2 reviews, LinkedIn posts, and earnings calls. This is:

- **Slow:** By the time someone reads and synthesises 50 documents, the intelligence is already stale
- **Biased:** Manual synthesis favours recently-read sources over systematically-retrieved relevant ones  
- **Unscalable:** As the competitor set grows, the manual approach breaks down entirely

A RAG pipeline turns this into an on-demand, queryable intelligence layer — ask any strategy question, get the most relevant passages back in seconds.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENT CORPUS                          │
│  (Product pages, press releases, earnings, G2 reviews)      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    TF-IDF Vectorisation
                    (ngram 1-2, 5,000 features)
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  VECTOR INDEX (dense matrix)                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
              Natural-Language Strategy Query
                           │
                    Cosine Similarity Retrieval
                    (top-k most relevant passages)
                           │
┌──────────────────────────▼──────────────────────────────────┐
│            COMPETITIVE INTELLIGENCE BRIEF                    │
│  (Structured by: Pricing | Product | APAC | AI | Funding)   │
└─────────────────────────────────────────────────────────────┘
```

**To upgrade to production:** Replace TF-IDF vectoriser with OpenAI `text-embedding-3-large` or `sentence-transformers`, and swap the rule-based synthesiser for a GPT-4o / Claude call that generates narrative answers from retrieved passages.

---

## Sample Queries

```python
rag.answer("Which competitors are expanding into APAC and South Korea?")
rag.answer("What do customers complain about or where do competitors fall short?")
rag.answer("What AI and machine learning capabilities are competitors offering?")
rag.answer("What is the market size, growth rate, and key trends in this space?")
```

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/payaljarviya/rag-competitor-intel
cd rag-competitor-intel
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the script
```bash
python rag_competitor_intel.py
```

### 4. Add your own documents
Extend the `CORPUS` list with your own competitor documents:
```python
CORPUS.append({
    'id': 'COMP-001',
    'company': 'Your Competitor',
    'type': 'product_page',
    'title': 'Document Title',
    'text': 'Full document text here...'
})
```

---

## Output Files

| File | Description |
|------|-------------|
| `output/competitive_brief.txt` | Full structured competitive intelligence brief |
| `output/competitor_positioning.png` | Positioning matrix: Price vs. Technical Complexity |
| `output/pricing_comparison.png` | Side-by-side pricing comparison across competitors |
| `output/similarity_heatmap.png` | Document similarity matrix across the corpus |

---

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG) architecture
- TF-IDF vectorisation and cosine similarity retrieval
- Competitive strategy analysis and brief synthesis
- NLP pipeline design
- Python: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

*Author: Payal Jarviya | MBA Candidate, SKK GSB Seoul | AI & Business Analytics + Marketing Analytics*
