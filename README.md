
# VectraMark

**AI-Powered Market Intelligence and Analytics Platform**

**GitHub Repository:** [View VectraMark source code](https://github.com/shivamgupta754600-ship-it/vectramark)

VectraMark is a market intelligence project designed to analyze financial market data, identify technical trends, explore relevant news, and provide structured research insights.

The project focuses on building a modular, maintainable, and safety-aware foundation for market analysis.

> **Disclaimer:** VectraMark is an educational and analytical software project. Its outputs are not financial advice, investment recommendations, or guarantees of future market performance.

## Supported Markets

- **Gold:** XAU/USD (XAUUSD)
- **Silver:** XAG/USD (XAGUSD)
- **Nasdaq-100**
- **S&P 500**
- **Dow Jones Industrial Average**

Actual data availability depends on the configured market-data source.

## Current Features

### 1. Technical Market Analysis
- Technical bias classification.
- Indicator-based analysis using market candles.
- Support and resistance level identification.
- Analysis across configurable periods and intervals.

### 2. News Context Analysis
- News context associated with market research.
- Rule-based headline classification and confidence assessment.
- Visibility into uncertain or unclassified headlines.

### 3. Market Safety Guardrails
- Flags for uncertain news classifications.
- Review-required states when the available evidence is insufficient.
- Structured responses intended to make limitations visible.

### 4. Market Data Quality Checks
- Checks for market-data quality and completeness.
- Validation of important candle-data properties.
- Data-quality status included in the research output.

### 5. Unified Research Snapshot API
- Combines technical bias, support and resistance, news context, news safety, and data-quality results.
- Uses FastAPI to expose structured API responses.
- Provides a foundation for future market research features.

## Technology Stack

- **Language:** Python 3.12
- **Package and Environment Management:** uv
- **Backend API:** FastAPI
- **Data Analysis:** Pandas and NumPy
- **Machine Learning:** Scikit-learn
- **LLM and Agent Frameworks:** LangChain
- **LLM Provider:** Groq
- **Testing:** pytest
- **Code Quality:** Ruff
- **Planned Frontend:** React, TypeScript, and Vite

Technology listed for planned features does not necessarily mean that the corresponding feature is fully integrated yet.

## Project Architecture

VectraMark follows a modular architecture to separate responsibilities and make future changes easier.

- **API Layer:** Handles HTTP requests, routes, and response schemas.
- **Application Layer:** Coordinates market-analysis workflows.
- **Domain Layer:** Contains market-data validation and analytical logic.
- **Configuration Layer:** Centralizes application settings and market configuration.

This separation helps keep API handling, business workflows, analytical rules, and configuration independent.

## Development Setup

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Git

### 1. Clone the Repository

```bash
git clone <https://github.com/shivamgupta754600-ship-it/vectramark.git>
cd vectramark
```

If you already have the project locally, open its existing directory instead.

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run the Development Server

```bash
uv run uvicorn --app-dir src vectramark.api.main:app --reload
```

The API should be available at:

- **API base URL:** http://127.0.0.1:8000
- **Interactive API documentation:** http://127.0.0.1:8000/docs

### 4. Run Tests

```bash
uv run pytest
```

### 5. Check Code Quality

```bash
uv run ruff check .
uv run ruff format --check .
```

## Configuration and Security

- Keep API keys and other secrets in local environment variables.
- Do not commit `.env` files or credentials to Git.
- Use `.env.example` to document required environment-variable names without real secret values, when needed.
- Validate external data before using it in analytical workflows.
- Treat uncertain analysis and news classifications cautiously.
- Do not interpret an analytical score or market bias as a guaranteed prediction.

## Development Roadmap

Planned areas for future development include:

- Improved market-news relevance and classification.
- More robust analysis across different time horizons.
- Historical strategy evaluation and backtesting.
- A strategy-building workflow with measurable performance metrics.
- A React and TypeScript user interface.
- Further testing, monitoring, and reliability improvements.

Features in this roadmap are planned and should not be considered complete until implemented and validated.

## Project Status

**Status:** Active development.

VectraMark is being developed incrementally, with an emphasis on modular design, testing, validation, and explicit handling of uncertain results.
