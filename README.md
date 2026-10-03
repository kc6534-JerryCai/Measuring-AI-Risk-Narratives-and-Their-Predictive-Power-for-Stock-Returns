# Measuring AI Risk Narratives and Their Predictive Power for Stock Returns

This repository contains an end-to-end research pipeline for constructing an hourly AI-risk news index and testing whether it predicts intraday returns for AI-exposed US equities.

The implemented empirical analysis covers **AI risk**. The broader project also proposes an AI-development factor, but that factor has not yet been operationalized or tested.

## Research design

The pipeline:

1. collects AI-related X News story clusters and Guardian articles;
2. splits headlines and article summaries or bodies into sentences and clauses;
3. labels headlines and clauses independently with DeepSeek;
4. distinguishes five concrete AI-risk categories:
   - loss of human control;
   - malicious use;
   - job displacement;
   - discrimination;
   - privacy violation;
5. assigns topic relevance values of `0`, `0.2`, or `1` and risk-expression directions of `-1`, `0`, or `1`;
6. constructs six hourly net indices and six corresponding average-tone indices using strictly prior-only RMS scaling;
7. aligns lagged risk with exact 10:00-16:00 New York trading windows; and
8. runs panel, equal-weight portfolio, category, individual-stock, and trading-strategy tests.

Granger-style tests are interpreted as predictive-association tests, not structural causal estimates.

## Main files

### Collection and preprocessing

- `download_x_ai_news.py`: X News collection entry point.
- `download_guardian_ai_september.py`: Guardian Content API collection.
- `preprocess_news_clauses.py`: sentence and clause preprocessing.
- `download_hourly_returns.py`: provider-fallback intraday market-data downloader and exact hourly aggregation.

### Annotation and index construction

- `annotate_deepseek_risk.py`: resumable DeepSeek annotation with schema and evidence validation.
- `construct_hourly_ai_risk_index.py`: document scoring, prior-only RMS scaling, deduplication, and 12 hourly measures.
- `PDF/deepseek_risk_annotation_prompt_v11.md`: versioned annotation prompt.

### Empirical notebooks

- `granger_main_regression.ipynb`: 20-stock panel specification.
- `granger_equal_weight_portfolio.ipynb`: fixed equal-weight AI portfolio.
- `granger_risk_categories_equal_weight.ipynb`: five-category tests.
- `granger_individual_stock_regressions.ipynb`: exploratory stock-level tests.
- `risk_high_long_qqq_short_ai_backtest.ipynb`: high-risk long-QQQ/short-AI strategy.
- `risk_high_long_short_return_curves.ipynb`: gross, net, and benchmark return curves.

The final academic report is available at [output/pdf/AI_Risk_and_AI_Company_Returns_Report.pdf](output/pdf/AI_Risk_and_AI_Company_Returns_Report.pdf).

## Current empirical summary

- X News sample: 1,882 story clusters, 1,882 labeled headlines, and 13,284 labeled body clauses.
- Hourly index: 12 measures over an observed-sample grid; all 33 X queries reached their result cap, so the index is not a complete news census.
- Market sample: 20 AI-exposed stocks, QQQ, 23 NYSE trading days, and six regular-session windows per day.
- Main panel and equal-weight portfolio tests: no statistically reliable next-hour relationship for one to three risk lags.
- Category tests: no result is significant after multiple-testing control.
- Individual stocks: four nominal signals, none significant after Benjamini-Hochberg or Holm correction.
- High-risk long-QQQ/short-AI strategy: `-0.38%` gross and `-2.45%` after a 5 bp one-way turnover cost in the available sample.
- Always-on comparison: `+0.79%` gross; the daily-flat implementation becomes `-1.02%` after modeled costs.

These results do not establish that AI risk is generally irrelevant to asset prices. They show that the present short-sample, X-based hourly factor is not yet a robust predictor or tradable signal.

## Environment setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add only the credentials required for the services you intend to use. Keys are read from the environment and must never be committed.

Supported variables include:

- `X_BEARER_TOKEN`
- `GUARDIAN_API_KEY`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL`
- `DEEPSEEK_BASE_URL`
- `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`
- `TWELVE_DATA_API_KEY` or `TWELVEDATA_API_KEY`
- `FMP_API_KEY`

## Reproduction outline

Each command exposes detailed options through `--help`.

```bash
python download_x_ai_news.py --help
python download_guardian_ai_september.py --help
python preprocess_news_clauses.py --help
python annotate_deepseek_risk.py --help
python construct_hourly_ai_risk_index.py --help
python download_hourly_returns.py --help
```

An example market-data run for the six regular-session windows is:

```bash
python download_hourly_returns.py \
  --start 2026-09-01 \
  --end 2026-10-02 \
  --window-mode clock_10_16 \
  --out data/hourly_10_16
```

The notebooks are stored with their executed outputs. Run them from the repository root after rebuilding the required files under `data/`.

Run the test suite with:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Data policy

Downloaded API responses, provider caches, full article text, market bars, model caches, and generated working datasets under `data/` are intentionally excluded from Git. This prevents accidental credential leakage, avoids redistributing provider content, and keeps the repository manageable. The scripts, configuration template, executed notebooks, prompt, and final report are included so the analysis can be inspected and the data can be regenerated by an authorized user.

## Important limitations

- All X keyword queries reached the request result cap.
- X News timestamps are story-cluster update times rather than verified publisher release times.
- Guardian full text was collected and preprocessed but did not enter the completed regressions.
- The return sample contains only 23 trading days.
- Several category series are sparse, especially discrimination.
- Trading simulations omit latency, market impact, borrow fees, financing costs, and dynamic spreads.
- QQQ has economic overlap with the AI-stock basket.
