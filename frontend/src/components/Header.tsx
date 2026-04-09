"use client";

interface HeaderProps {
  activeTab: "market" | "prediction";
  onTabChange: (tab: "market" | "prediction") => void;
  lastUpdated: Date | null;
  nextRefresh: number;
  onRefresh: () => void;
}

export default function Header({ activeTab, onTabChange, lastUpdated, nextRefresh, onRefresh }: HeaderProps) {
  const m = Math.floor(nextRefresh / 60);
  const s = nextRefresh % 60;

  return (
    <header className="border-b border-tm-border bg-tm-card/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-14">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-md bg-tm-accent flex items-center justify-center">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                  <polyline points="22,7 13.5,15.5 8.5,10.5 2,17" />
                  <polyline points="16,7 22,7 22,13" />
                </svg>
              </div>
              <span className="font-bold text-sm">True Markets</span>
            </div>

            <div className="h-5 w-px bg-tm-border" />

            {/* Market / Prediction toggle */}
            <div className="flex items-center bg-tm-bg rounded-lg p-0.5">
              <button
                onClick={() => onTabChange("market")}
                className={`px-4 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === "market"
                    ? "bg-tm-card text-tm-text shadow-sm"
                    : "text-tm-muted hover:text-tm-text"
                }`}
              >
                Market
              </button>
              <button
                onClick={() => onTabChange("prediction")}
                className={`px-4 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === "prediction"
                    ? "bg-tm-card text-tm-text shadow-sm"
                    : "text-tm-muted hover:text-tm-text"
                }`}
              >
                Prediction
              </button>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs text-tm-muted">
            <span className="px-2 py-0.5 rounded-full bg-tm-green/10 text-tm-green border border-tm-green/20 text-[10px]">
              LIVE
            </span>
            {lastUpdated && <span>{lastUpdated.toLocaleTimeString()}</span>}
            <span className="text-tm-border">|</span>
            <span>{m}:{s.toString().padStart(2, "0")}</span>
            <button
              onClick={onRefresh}
              className="px-2 py-1 rounded bg-tm-accent/20 text-tm-accent hover:bg-tm-accent/30 transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
