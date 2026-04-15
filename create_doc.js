const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageNumber, PageBreak, LevelFormat } = require("docx");
const fs = require("fs");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellM = { top: 80, bottom: 80, left: 120, right: 120 };

function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, children: [new TextRun({ text, bold: true })] });
}
function para(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text, size: 24, ...opts })] });
}
function boldPara(label, value) {
  return new Paragraph({ spacing: { after: 80 }, children: [
    new TextRun({ text: label, bold: true, size: 24 }),
    new TextRun({ text: value, size: 24 }),
  ]});
}

function makeRow(cells, isHeader = false) {
  return new TableRow({
    children: cells.map((text, i) => new TableCell({
      borders, width: { size: i === 0 ? 3500 : 2930, type: WidthType.DXA },
      margins: cellM,
      shading: isHeader ? { fill: "1E1E2E", type: ShadingType.CLEAR } : undefined,
      children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: isHeader, size: 22, color: isHeader ? "FFFFFF" : "333333" })] })],
    })),
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1a1a2e" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "6C5CE7" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ]
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        children: [new TextRun({ text: "True Markets \u2014 Technical Documentation", size: 18, color: "999999", italics: true })],
        alignment: AlignmentType.RIGHT,
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Page ", size: 18, color: "999999" }), new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "999999" })],
      })] }),
    },
    children: [
      // ══════════════════════════════════════════
      // TITLE PAGE
      // ══════════════════════════════════════════
      new Paragraph({ spacing: { before: 3000 }, alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "True Markets", size: 56, bold: true, font: "Arial", color: "1a1a2e" }),
      ]}),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "AI-Powered BTC Prediction Engine", size: 32, color: "6C5CE7" }),
      ]}),
      new Paragraph({ spacing: { before: 400 }, alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "Technical Documentation & Backtest Results", size: 24, color: "666666" }),
      ]}),
      new Paragraph({ spacing: { before: 200 }, alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "April 2026", size: 22, color: "999999" }),
      ]}),
      new Paragraph({ children: [new PageBreak()] }),

      // ══════════════════════════════════════════
      // 1. EXECUTIVE SUMMARY
      // ══════════════════════════════════════════
      heading("1. Executive Summary"),
      para("True Markets is an AI-powered BTC prediction platform that combines 6 real-time signals into a weighted Buy/Sell recommendation. The platform uses on-chain data from BGeometrics, real exchange order flow from Binance.US and Coinbase, prediction market probabilities from Polymarket, and live technical indicators from TrueMarkets MCP."),
      para("Our ML model achieves 67.4% directional accuracy on 359 out-of-sample test days (April 2025 \u2013 April 2026), using 30 Boruta-selected features from 216 engineered on-chain and price-derived features."),
      new Paragraph({ children: [new PageBreak()] }),

      // ══════════════════════════════════════════
      // 2. SYSTEM ARCHITECTURE
      // ══════════════════════════════════════════
      heading("2. System Architecture"),
      heading("2.1 Tech Stack", HeadingLevel.HEADING_2),
      boldPara("Backend: ", "FastAPI + Python (scikit-learn, PyTorch, pandas, numpy)"),
      boldPara("Frontend: ", "Next.js 14, React 18, TailwindCSS"),
      boldPara("Data Sources: ", "BGeometrics Premium, Binance.US, Coinbase, Polymarket Gamma API, TrueMarkets MCP, alternative.me"),
      boldPara("Refresh: ", "Every 30 seconds, all 6 signals"),

      heading("2.2 The 6 Signals", HeadingLevel.HEADING_2),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2200, 1000, 3160, 1500, 1500],
        rows: [
          new TableRow({ children: ["Signal", "Weight", "Source", "Endpoint", "Update"].map((t, i) =>
            new TableCell({ borders, width: { size: [2200,1000,3160,1500,1500][i], type: WidthType.DXA }, margins: cellM,
              shading: { fill: "6C5CE7", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, size: 20, color: "FFFFFF" })] })] })) }),
          ...([
            ["Polymarket", "20%", "Gamma API", "gamma-api.polymarket.com", "30s"],
            ["Order Flow", "15%", "Binance.US + Coinbase", "api.binance.us/api/v3", "30s"],
            ["Our Model", "20%", "BGeometrics on-chain", "api.bitcoin-data.com", "30s"],
            ["Technical", "20%", "TrueMarkets MCP", "get_price_history", "30s"],
            ["TM Sentiment", "10%", "TrueMarkets MCP", "get_asset_summary", "30s"],
            ["Fear & Greed", "15%", "alternative.me", "/fng", "30s"],
          ].map(row => new TableRow({ children: row.map((t, i) =>
            new TableCell({ borders, width: { size: [2200,1000,3160,1500,1500][i], type: WidthType.DXA }, margins: cellM,
              children: [new Paragraph({ children: [new TextRun({ text: t, size: 20 })] })] })) }))),
        ],
      }),
      new Paragraph({ children: [new PageBreak()] }),

      // ══════════════════════════════════════════
      // 3. ML MODEL
      // ══════════════════════════════════════════
      heading("3. ML Model \u2014 On-Chain Ensemble"),
      heading("3.1 Architecture", HeadingLevel.HEADING_2),
      para("The model is a GradientBoosting classifier (scikit-learn) with Boruta feature selection, trained on 5 years of daily on-chain data from BGeometrics Premium API. The architecture was informed by Omole & Enke (2024) \u201CDeep learning for Bitcoin price direction prediction\u201D, Financial Innovation 10:117, which demonstrated that on-chain features with proper feature selection significantly outperform price-only models."),
      boldPara("Algorithm: ", "GradientBoostingClassifier (n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8)"),
      boldPara("Feature Selection: ", "Boruta (Random Forest wrapper, 20 iterations)"),
      boldPara("Horizon: ", "Next-day direction (binary: up=1, down=0)"),
      boldPara("Normalization: ", "Min-Max scaling from training set only (no data leakage)"),
      boldPara("Split: ", "80/20 temporal (no shuffle, preserves time order)"),

      heading("3.2 Feature Engineering", HeadingLevel.HEADING_2),
      para("The key innovation is rate-of-change feature engineering. Raw on-chain values (e.g., MVRV ratio = 1.3) are weakly predictive by themselves. However, computing the 1-day, 3-day, 7-day, and 14-day percentage changes of each metric creates highly predictive features. For example, a 5% drop in MVRV ratio over 1 day is a strong bearish signal."),
      boldPara("Base features (42): ", "27 BGeometrics on-chain endpoints including HODL waves (13 age bands), MVRV ratio, MVRV Z-score, NUPL, NVT ratio, Puell multiple, reserve risk, exchange netflow/inflow/outflow, supply in profit/loss, adjusted SOPR, CDD, difficulty, hash ribbons, active addresses, liveliness, realized loss."),
      boldPara("Rate-of-change features (168): ", "4 lags (1d, 3d, 7d, 14d) \u00d7 42 base features"),
      boldPara("Price-derived features (6): ", "log_return, volatility_5d, volatility_20d, momentum_5d, momentum_20d, RSI"),
      boldPara("Total: ", "216 features \u2192 Boruta selects 30"),

      heading("3.3 Backtest Results", HeadingLevel.HEADING_2),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [4680, 4680],
        rows: [
          makeRow(["Metric", "Value"], true),
          makeRow(["Test Accuracy", "67.4%"]),
          makeRow(["Precision", "67.2%"]),
          makeRow(["Recall", "66.9%"]),
          makeRow(["F1 Score", "67.0%"]),
          makeRow(["AUC-ROC", "0.7365"]),
          makeRow(["MCC (Matthews)", "0.348"]),
          makeRow(["Train Samples", "1,435 days (May 2021 \u2013 Apr 2025)"]),
          makeRow(["Test Samples", "359 days (Apr 2025 \u2013 Apr 2026)"]),
          makeRow(["Base Rate", "49.6% (balanced)"]),
          makeRow(["Features Selected", "30 / 216"]),
        ],
      }),

      heading("3.4 Confusion Matrix", HeadingLevel.HEADING_2),
      new Table({
        width: { size: 5000, type: WidthType.DXA },
        columnWidths: [1800, 1600, 1600],
        rows: [
          makeRow(["", "Pred Down", "Pred Up"], true),
          makeRow(["Actual Down", "123 (TN)", "58 (FP)"]),
          makeRow(["Actual Up", "59 (FN)", "119 (TP)"]),
        ],
      }),

      heading("3.5 Top 10 Features by Importance", HeadingLevel.HEADING_2),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [600, 4380, 2190, 2190],
        rows: [
          makeRow(["#", "Feature", "Importance", "Category"], true),
          makeRow(["1", "mvrv_ratio_chg1d", "14.3%", "On-chain rate-of-change"]),
          makeRow(["2", "hash_ribbons (SMA60)", "10.4%", "Mining"]),
          makeRow(["3", "hash_ribbons (SMA30)", "8.0%", "Mining"]),
          makeRow(["4", "puell_multiple_chg1d", "4.8%", "On-chain rate-of-change"]),
          makeRow(["5", "hodl_age_10y", "4.5%", "HODL waves"]),
          makeRow(["6", "avg_spent_output_lifespan_chg14d", "4.0%", "On-chain rate-of-change"]),
          makeRow(["7", "avg_dormancy", "3.7%", "On-chain"]),
          makeRow(["8", "momentum_5d", "3.4%", "Price-derived"]),
          makeRow(["9", "puell_multiple_chg7d", "3.1%", "On-chain rate-of-change"]),
          makeRow(["10", "hash_ribbons_chg7d", "3.0%", "Mining rate-of-change"]),
        ],
      }),
      new Paragraph({ children: [new PageBreak()] }),

      // ══════════════════════════════════════════
      // 4. DATA SOURCES
      // ══════════════════════════════════════════
      heading("4. Data Sources"),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2000, 2500, 3360, 1500],
        rows: [
          makeRow(["Source", "Data", "Endpoint", "Auth"], true),
          makeRow(["BGeometrics", "27 on-chain metrics", "api.bitcoin-data.com/v1/", "Bearer token"]),
          makeRow(["Binance.US", "BTC trades + order book", "api.binance.us/api/v3/", "None"]),
          makeRow(["Coinbase", "BTC trades", "api.exchange.coinbase.com/", "None"]),
          makeRow(["Polymarket", "18 BTC thresholds", "gamma-api.polymarket.com/", "None"]),
          makeRow(["TrueMarkets", "Price, sentiment, charts", "MCP get_price_history", "MCP"]),
          makeRow(["alternative.me", "Fear & Greed Index", "api.alternative.me/fng", "None"]),
          makeRow(["CryptoCompare", "5yr OHLCV (training)", "Cached btc_5Y_1d.json", "None"]),
        ],
      }),

      heading("4.1 BGeometrics On-Chain Endpoints", HeadingLevel.HEADING_2),
      para("All 27 endpoints downloaded as CSV with full history. Data merged on date, forward-filled for MNAR missing values, NaN rows dropped."),
      ...([
        "asopr, average-dormancy, asol, cdd, supply-adjusted-cdd",
        "exchange-netflow-btc, exchange-inflow-btc, exchange-outflow-btc",
        "mvrv, mvrv-zscore, nupl, nvt-ratio",
        "active-addresses, puell-multiple, reserve-risk",
        "utxos-in-profit-pct, utxos-in-loss-pct, supply-profit, supply-loss",
        "nrpl-btc, realized-loss-usd, rpv, liveliness",
        "difficulty-btc, hashribbons, hodl-waves-supply, btc-ohlc",
      ].map(t => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: t, size: 20 })] }))),
      new Paragraph({ children: [new PageBreak()] }),

      // ══════════════════════════════════════════
      // 5. SIGNAL DETAILS
      // ══════════════════════════════════════════
      heading("5. Signal Implementation Details"),

      heading("5.1 Polymarket (20%)", HeadingLevel.HEADING_2),
      para("Fetches the April 2026 BTC price event via Gamma API. Extracts 18 threshold markets ($20K\u2013$150K) with yes_price (probability), volume, and liquidity. Compares probability-weighted expected price to current price for directional signal."),

      heading("5.2 Order Flow (15%)", HeadingLevel.HEADING_2),
      para("Fetches 1,000 recent trades from Binance.US and 100 from Coinbase in parallel. Computes buy vs sell aggressor volume ratio (60% weight) and order book bid/ask depth imbalance (40% weight). Combined into -1 to +1 signal mapped to strong_sell/sell/neutral/buy/strong_buy."),

      heading("5.3 Our Model (20%)", HeadingLevel.HEADING_2),
      para("GradientBoosting ensemble trained on 5 years of BGeometrics on-chain data. 216 features (42 base + 168 rate-of-change + 6 price). Boruta selects 30. Predicts next-day direction probability. 67.4% test accuracy on 359 OOS days."),

      heading("5.4 Technical Indicators (20%)", HeadingLevel.HEADING_2),
      para("RSI (14-period): mean-reversion at extremes (<30 oversold, >70 overbought). MACD histogram (12,26,9): trend momentum. Bollinger position: where price sits in bands. All computed from TrueMarkets MCP price data (same source as market page)."),

      heading("5.5 TM Sentiment (10%)", HeadingLevel.HEADING_2),
      para("AI-generated sentiment analysis from TrueMarkets MCP get_asset_summary. Analyzes 30+ crypto news sources. Returns bullish/bearish/neutral classification with summary text."),

      heading("5.6 Fear & Greed (15%)", HeadingLevel.HEADING_2),
      para("Contrarian signal from alternative.me Fear & Greed Index (0\u2013100). Extreme fear (<20) = bullish (contrarian buy). Extreme greed (>80) = bearish (contrarian sell). Middle zone follows momentum."),
      new Paragraph({ children: [new PageBreak()] }),

      // ══════════════════════════════════════════
      // 6. PRICE CONSISTENCY
      // ══════════════════════════════════════════
      heading("6. Price Consistency"),
      para("All BTC prices across the platform come from a single function: _get_btc_price() in routes.py. This ensures the market page, prediction page, technical indicators, and model input all show the same price from the same source (TrueMarkets MCP). The prediction page also polls /api/price/bitcoin every 15 seconds (same endpoint as the market page) to override any cached price differences."),

      // ══════════════════════════════════════════
      // 7. REFERENCES
      // ══════════════════════════════════════════
      heading("7. References"),
      para("Omole, O. and Enke, D. (2024). Deep learning for Bitcoin price direction prediction: models and trading strategies empirically compared. Financial Innovation, 10:117.", { italics: true }),
      para("Khattak, B.H.A. et al. (2026). AI crypto trading: multi-class multi-granular analysis for boosting high-frequency trade predictions with Fibonacci and hybrid convolutional neural networks. Journal of Big Data, 13(1).", { italics: true }),
      para("Wang, J., Feng, K. and Qiao, G. (2024). A hybrid deep learning model for Bitcoin price prediction: data decomposition and feature selection. Applied Economics, 56(53), 6890-6905.", { italics: true }),
    ],
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/Users/saieshagupta/Desktop/claude/truemarkets-predictions/TrueMarkets_Technical_Doc.docx", buffer);
  console.log("Saved: TrueMarkets_Technical_Doc.docx");
});
