# True Markets Prediction Engine

An independent, ensemble ML prediction engine for crypto price targets. Replaces Polymarket scraping with first-principles probabilistic forecasts.

## Architecture

**Backend** (Python/FastAPI): Ensemble of 3 models predicting probability of hitting price thresholds.

| Model | Weight | Input | Technique |
|-------|--------|-------|-----------|
| LSTM Neural Network | 40% | 30-day price sequences + technical indicators | PyTorch LSTM, 2 layers |
| XGBoost Feature Model | 45% | On-chain metrics, fear/greed, volatility, sentiment | Gradient-boosted features |
| Sentiment Analyzer | 15% | Reddit sentiment + Fear & Greed Index | VADER NLP + weighted scoring |

**Frontend** (Next.js/TypeScript): Dark-themed dashboard with prediction cards, probability charts, model signal breakdown, and Polymarket comparison.

**Data Sources** (all free, no auth required):
- CoinGecko API — prices, volume, market cap
- Alternative.me — Fear & Greed Index
- blockchain.info — on-chain metrics (hash rate, tx volume)
- Reddit — social sentiment via VADER NLP
- Polymarket Gamma API — comparison baseline

## Quick Start

### Backend
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Optional: train models with historical data
python -m train.train_models
# Start server
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` for the dashboard.
API docs at `http://localhost:8000/docs`.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/predictions/{coin}` | Ensemble predictions with model breakdown |
| `GET /api/signals/{coin}` | Individual model signals |
| `GET /api/market-data/{coin}` | Real-time market data + indicators |
| `GET /api/comparison/{coin}` | Our model vs Polymarket comparison |
| `GET /api/health` | Health check |

## Deployment

- **Frontend**: Deploy to Vercel — connect GitHub repo, set `NEXT_PUBLIC_API_URL` env var
- **Backend**: Deploy to Railway/Render — use Dockerfile, set `FRONTEND_URL` env var

## Key Differentiators

1. **Independent Signal** — generates predictions from first principles, not scraping
2. **Transparent** — shows which signals (LSTM vs XGBoost vs sentiment) drive each prediction
3. **Backtestable** — includes training pipeline with historical data validation
4. **Extensible** — add new coins, data sources, or models easily
5. **Comparison** — shows where our model diverges from Polymarket (potential alpha)
