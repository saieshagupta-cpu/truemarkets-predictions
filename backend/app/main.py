from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import FRONTEND_URL

app = FastAPI(
    title="True Markets Prediction Engine",
    description="Independent ensemble ML predictions for crypto price targets",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "True Markets Prediction Engine",
        "docs": "/docs",
        "endpoints": {
            "predictions": "/api/predictions/{coin}",
            "signals": "/api/signals/{coin}",
            "market_data": "/api/market-data/{coin}",
            "comparison": "/api/comparison/{coin}",
            "health": "/api/health",
        },
    }
