from __future__ import annotations
import asyncio, time, math, random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable, Iterable
import aiohttp
import pandas as pd
import numpy as np
from dateutil import tz

# ---------- Provider Adapters ----------

@dataclass
class ProviderConfig:
    name: str
    base_url: str
    concurrency: int = 5                # Max in-flight requests to this provider
    rpm: Optional[int] = None           # Rate limit: requests per minute (AlphaVantage e.g. 5/min free)
    timeout_s: float = 15.0
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)

class ProviderAdapter:
    """
    Each adapter must implement:
        build_url_and_params(ticker: str) -> (url, params)
        parse(json: dict) -> pd.DataFrame with columns: ['ticker','date','open','high','low','close','volume','provider']
    """
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg

    def build_url_and_params(self, ticker: str) -> Tuple[str, Dict[str, str]]:
        raise NotImplementedError

    def parse(self, ticker: str, payload: Dict[str, Any]) -> pd.DataFrame:
        raise NotImplementedError

class YahooChartAdapter(ProviderAdapter):
    """
    Public Yahoo Finance chart API.
    Example: https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=5y&interval=1d
    """
    def __init__(self, range_: str = "2y", interval: str = "1d"):
        super().__init__(ProviderConfig(
            name="yahoo",
            base_url="https://query1.finance.yahoo.com/v8/finance/chart",
            concurrency=8,
            rpm=None,
            timeout_s=15.0,
            headers={"User-Agent": "Mozilla/5.0"}
        ))
        self.range_ = range_
        self.interval = interval

    def build_url_and_params(self, ticker: str) -> Tuple[str, Dict[str, str]]:
        url = f"{self.cfg.base_url}/{ticker}"
        params = {"range": self.range_, "interval": self.interval}
        return url, params

    def parse(self, ticker: str, payload: Dict[str, Any]) -> pd.DataFrame:
        result = payload.get("chart", {}).get("result")
        if not result:
            return pd.DataFrame(columns=["ticker","date","open","high","low","close","volume","provider"])
        r = result[0]
        ts = r.get("timestamp", [])
        indicators = r.get("indicators", {})
        quote = indicators.get("quote", [{}])[0]
        o = quote.get("open", [])
        h = quote.get("high", [])
        l = quote.get("low", [])
        c = quote.get("close", [])
        v = quote.get("volume", [])
        # Convert unix epoch to UTC pandas datetime
        if ts:
            idx = pd.to_datetime(pd.Series(ts, dtype="float64"), unit="s", utc=True)
            df = pd.DataFrame({
                "ticker": ticker,
                "date": idx,
                "open": pd.Series(o, dtype="float64"),
                "high": pd.Series(h, dtype="float64"),
                "low":  pd.Series(l, dtype="float64"),
                "close":pd.Series(c, dtype="float64"),
                "volume": pd.Series(v, dtype="float64"),
            })
            df["provider"] = self.cfg.name
            df = df.dropna(subset=["close"])  # drop empty rows
            return df
        return pd.DataFrame(columns=["ticker","date","open","high","low","close","volume","provider"])

class AlphaVantageDailyAdjAdapter(ProviderAdapter):
    """
    Alpha Vantage TIME_SERIES_DAILY_ADJUSTED
    Free tier is 5 req/min and 500/day. Provide api_key.
    """
    def __init__(self, api_key: str):
        super().__init__(ProviderConfig(
            name="alpha_vantage",
            base_url="https://www.alphavantage.co/query",
            concurrency=1,      # stay conservative with AV free tier
            rpm=5,              # 5 req/min
            timeout_s=20.0,
        ))
        self.api_key = api_key

    def build_url_and_params(self, ticker: str) -> Tuple[str, Dict[str, str]]:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": self.api_key
        }
        return self.cfg.base_url, params

    def parse(self, ticker: str, payload: Dict[str, Any]) -> pd.DataFrame:
        key = "Time Series (Daily)"
        if key not in payload:
            return pd.DataFrame(columns=["ticker","date","open","high","low","close","volume","provider"])
        ts = payload[key]
        # ts is dict of date -> fields
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.index.name = "date"
        df = df.rename(columns={
            "1. open":"open","2. high":"high","3. low":"low","4. close":"close","6. volume":"volume","5. adjusted close":"adj_close"
        })
        df = df[["open","high","low","close","volume"]].astype(float)
        df.reset_index(inplace=True)
        # AlphaVantage dates are in local naive date strings (YYYY-MM-DD). Convert to UTC midnight.
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["ticker"] = ticker
        df["provider"] = self.cfg.name
        return df

# ---------- Async Fetch Core ----------

@dataclass
class FetchResult:
    ticker: str
    provider: str
    df: Optional[pd.DataFrame]
    error: Optional[str] = None
    status: Optional[int] = None

class RateLimiter:
    """Token bucket per minute; simple and robust for small client-side rate limits."""
    def __init__(self, rpm: Optional[int]):
        self.rpm = rpm or 10**9
        self.tokens = self.rpm
        self.last = time.monotonic()

    async def acquire(self):
        if self.rpm >= 10**8:
            return  # unlimited
        while True:
            now = time.monotonic()
            # refill per second
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.rpm, self.tokens + elapsed * (self.rpm/60.0))
            if self.tokens >= 1:
                self.tokens -= 1
                return
            # sleep just enough to get ~1 token
            await asyncio.sleep(max(0.05, 60.0/self.rpm/2))

class ConcurrentPriceFetcher:
    def __init__(self, adapters: List[ProviderAdapter], per_request_timeout: Optional[float] = None):
        self.adapters = adapters
        self.per_request_timeout = per_request_timeout
        # Per-provider semaphores and rate limiters
        self._sems: Dict[str, asyncio.Semaphore] = {
            a.cfg.name: asyncio.Semaphore(a.cfg.concurrency) for a in adapters
        }
        self._limiters: Dict[str, RateLimiter] = {
            a.cfg.name: RateLimiter(a.cfg.rpm) for a in adapters
        }

    async def _fetch_one(
        self, session: aiohttp.ClientSession, adapter: ProviderAdapter, ticker: str,
        max_retries: int = 3, backoff_base: float = 0.6
    ) -> FetchResult:
        url, params = adapter.build_url_and_params(ticker)
        sem = self._sems[adapter.cfg.name]
        limiter = self._limiters[adapter.cfg.name]

        attempt = 0
        while True:
            attempt += 1
            await limiter.acquire()
            async with sem:
                try:
                    timeout = aiohttp.ClientTimeout(total=self.per_request_timeout or adapter.cfg.timeout_s)
                    async with session.get(url, params=params, headers=adapter.cfg.headers, timeout=timeout) as resp:
                        status = resp.status
                        if status >= 500:
                            raise aiohttp.ClientResponseError(request_info=resp.request_info, history=(), status=status)
                        payload = await resp.json(content_type=None)
                        # Provider-specific error signals
                        if adapter.cfg.name == "alpha_vantage" and "Note" in payload:
                            # Throttled; wait a bit more
                            await asyncio.sleep(12)
                            raise RuntimeError("AlphaVantage throttle note: " + payload.get("Note", ""))
                        df = adapter.parse(ticker, payload)
                        return FetchResult(ticker=ticker, provider=adapter.cfg.name, df=df, status=status)
                except Exception as e:
                    if attempt <= max_retries:
                        # exponential backoff with jitter
                        sleep_s = (backoff_base ** attempt) + random.uniform(0, 0.25)
                        await asyncio.sleep(sleep_s)
                        continue
                    return FetchResult(ticker=ticker, provider=adapter.cfg.name, df=None, error=str(e))

    async def fetch_many_async(
        self, tickers: Iterable[str], providers: Optional[List[str]] = None
    ) -> List[FetchResult]:
        chosen = [a for a in self.adapters if (providers is None or a.cfg.name in providers)]
        conn = aiohttp.TCPConnector(limit=32, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=conn) as session:
            tasks = []
            for adapter in chosen:
                for t in tickers:
                    tasks.append(self._fetch_one(session, adapter, t))
            return await asyncio.gather(*tasks)

    def fetch_many(
        self, tickers: Iterable[str], providers: Optional[List[str]] = None
    ) -> List[FetchResult]:
        return asyncio.run(self.fetch_many_async(tickers, providers))

# ---------- Utilities to combine results ----------

def combine_results_to_long(results: List[FetchResult]) -> pd.DataFrame:
    frames = []
    for r in results:
        if r.df is not None and not r.df.empty:
            frames.append(r.df)
    if not frames:
        return pd.DataFrame(columns=["ticker","date","open","high","low","close","volume","provider"])
    df = pd.concat(frames, ignore_index=True)
    # Ensure sorted and proper dtypes
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["ticker","provider","date"])
    return df

def pivot_to_wide(df_long: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """
    Wide matrix: index=date, columns=MultiIndex[(provider, ticker)] with selected field.
    """
    cols_needed = {"date","ticker","provider",field}
    missing = cols_needed - set(df_long.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    wide = df_long.pivot_table(index="date", columns=["provider","ticker"], values=field, aggfunc="last").sort_index()
    return wide

# ---------- Example usage with your FinancePreprocessor ----------

if __name__ == "__main__":
    # 1) Configure providers
    yahoo = YahooChartAdapter(range_="1y", interval="1d")
    # For Alpha Vantage, pass your key; comment out if not available.
    # av = AlphaVantageDailyAdjAdapter(api_key="YOUR_ALPHA_VANTAGE_KEY")

    fetcher = ConcurrentPriceFetcher(adapters=[yahoo])  # or [yahoo, av]

    tickers = ["AAPL", "MSFT", "NVDA"]

    results = fetcher.fetch_many(tickers, providers=None)
    # Inspect errors, if any
    for r in results:
        if r.error:
            print(f"[{r.provider}] {r.ticker} -> ERROR: {r.error}")

    df_long = combine_results_to_long(results)
    print("Long shape:", df_long.shape, df_long.head())

    # 2) Get a clean closing-price panel and preprocess per ticker
    close_wide = pivot_to_wide(df_long, field="close")

    # Example: preprocess a single (provider, ticker) series with your FinancePreprocessor
    from dataclasses import asdict

    from typing import Optional
    # Suppose we choose Yahoo/AAPL column
    target_col = ("yahoo", "AAPL")
    if target_col in close_wide.columns:
        tmp = close_wide[[target_col]].rename(columns={target_col: "close"}).reset_index().rename(columns={"date":"date"})
        # Use your preprocessor
        pp_cfg = PreprocessConfig(
            date_col="date",
            price_col="close",
            index_utc=True,
            missing_strategy="ffill_bfill",
            outlier_strategy="iqr",
            outlier_threshold=3.0,
            normalize_strategy="zscore",
            train_end=None
        )
        pp = FinancePreprocessor(pp_cfg)
        out = pp.fit_transform(tmp)
        print("Preprocessed sample:")
        print(out.tail())
