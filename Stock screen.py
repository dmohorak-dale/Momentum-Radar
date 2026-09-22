import webbrowser
import threading
import time
import requests
import pandas as pd
import numpy as np
from typing import List
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Momentum Radar Standalone Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BatchRequest(BaseModel):
    tickers: List[str]

# -------------------------------------------------------------------
# INDIKATOREN & PROVIDER ENGINE
# -------------------------------------------------------------------
def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def fetch_from_yahoo_api(ticker: str):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1y&interval=1d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            result = res.json()['chart']['result'][0]
            timestamps = result['timestamp']
            closes = result['indicators']['quote'][0]['close']
            clean_data = [(pd.to_datetime(ts, unit='s'), cl) for ts, cl in zip(timestamps, closes) if cl is not None]
            if len(clean_data) > 50:
                dates, values = zip(*clean_data)
                return pd.Series(values, index=dates)
    except Exception:
        pass
    return None

def fetch_from_stooq(ticker: str):
    try:
        symbol = ticker.lower() if "." in ticker else f"{ticker.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        df = pd.read_csv(url)
        if not df.empty and 'Close' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            return df['Close']
    except Exception:
        pass
    return None

def fetch_from_binance(ticker: str):
    try:
        pair = f"{ticker.upper()}USDT"
        url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1d&limit=250"
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            raw_data = res.json()
            closes = [float(item[4]) for item in raw_data]
            dates = pd.to_datetime([item[0] for item in raw_data], unit='ms')
            return pd.Series(closes, index=dates)
    except Exception:
        pass
    return None

def get_historical_prices(ticker: str):
    prices = fetch_from_yahoo_api(ticker)
    if prices is not None:
        return prices, "Yahoo API"
    prices = fetch_from_stooq(ticker)
    if prices is not None:
        return prices, "Stooq"
    prices = fetch_from_binance(ticker)
    if prices is not None:
        return prices, "Binance"
    return None, "Offline"

def process_single_ticker(ticker: str, msci_prices: pd.Series):
    ticker = ticker.upper()
    close_prices, provider_used = get_historical_prices(ticker)
    if close_prices is None or len(close_prices) < 20:
        return None

    latest_price = float(close_prices.iloc[-1])
    lookback_3m = min(63, len(close_prices) - 1)
    m3_price = float(close_prices.iloc[-lookback_3m])
    momentum_3m = ((latest_price - m3_price) / m3_price) * 100

    if msci_prices is not None and len(msci_prices) >= lookback_3m:
        msci_latest = float(msci_prices.iloc[-1])
        msci_3m = float(msci_prices.iloc[-lookback_3m])
        msci_mom = ((msci_latest - msci_3m) / msci_3m) * 100
        vs_msci = momentum_3m - msci_mom
    else:
        vs_msci = 0.0

    returns = close_prices.pct_change().dropna()
    volatility = float(returns.tail(30).std() * np.sqrt(252) * 100)

    sma_period = min(200, len(close_prices))
    sma200 = float(close_prices.rolling(sma_period).mean().iloc[-1])
    dist_sma200 = ((latest_price - sma200) / sma200) * 100
    sma_status = f"Über SMA200 (+{dist_sma200:.1f}%)" if dist_sma200 >= 0 else f"Unter SMA200 ({dist_sma200:.1f}%)"

    rsi_val = calculate_rsi(close_prices)
    signal = "LONG" if (momentum_3m > 0 and dist_sma200 > 0 and rsi_val > 45) else "SHORT"

    low_52 = float(close_prices.tail(252).min())
    high_52 = float(close_prices.tail(252).max())

    return {
        "ticker": ticker,
        "name": ticker,
        "price": round(latest_price, 2),
        "signal": signal,
        "momentum3m": f"{momentum_3m:+.1f}%",
        "momentumVsMsci": f"{vs_msci:+.1f}%",
        "volatility": f"{volatility:.1f}%",
        "liquidity": f"Aktiv ({provider_used})",
        "sma200": sma_status,
        "rsi": round(rsi_val, 1),
        "range52w": f"${low_52:.2f} -${high_52:.2f}",
        "chartPoints": close_prices.tail(12).round(2).tolist(),
        "provider": provider_used,
        "userNote": ""
    }

@app.post("/api/stocks/batch")
def get_batch_analysis(payload: BatchRequest):
    msci_prices, _ = get_historical_prices("URTH")
    if msci_prices is None:
        msci_prices, _ = get_historical_prices("^GSPC")

    results = []
    for ticker in payload.tickers:
        res = process_single_ticker(ticker, msci_prices)
        if res:
            results.append(res)

    return {"data": results, "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")}

@app.get("/", response_class=HTMLResponse)
def render_mobile_ui():
    return HTML_CONTENT

# -------------------------------------------------------------------
# FRONTEND TEMPLATE (KOMPRAKT GEHALTEN FÜR ANROID PASTE)
# -------------------------------------------------------------------
HTML_CONTENT = """<!DOCTYPE html><html lang="de"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no"><title>Momentum Radar</title><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet"><style>*{box-sizing:border-box;margin:0;padding:0;font-family:'Inter',sans-serif}body{background:radial-gradient(circle at top,#141923 0%,#080a0f 100%);color:#F0F4F8;display:flex;justify-content:center;min-height:100vh}.mobile-frame{width:100%;max-width:480px;background:rgba(13,17,23,0.95);border-left:1px solid rgba(255,255,255,0.05);border-right:1px solid rgba(255,255,255,0.05);padding:16px;min-height:100vh}.header{display:flex;justify-content:space-between;align-items:center;padding-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:12px}.brand{display:flex;align-items:center;gap:8px}.live-dot{width:8px;height:8px;background:#00e676;border-radius:50%;box-shadow:0 0 8px #00e676}.header h1{font-size:15px;font-weight:700;color:#FFF;text-transform:uppercase}.header-sub{font-size:10px;color:#6E7681;margin-top:2px}.refresh-btn{background:linear-gradient(135deg,#1f293d 0%,#111827 100%);color:#38ef7d;border:1px solid rgba(56,239,125,0.3);padding:6px 12px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer}.add-bar{display:flex;gap:8px;margin-bottom:14px}.add-input{flex:1;background:rgba(22,27,34,0.8);color:#FFF;border:1px solid rgba(255,255,255,0.1);border-radius:6px;padding:8px 10px;font-size:12px;text-transform:uppercase;outline:none}.add-btn{background:#00f2fe;color:#080a0f;border:none;padding:8px 12px;border-radius:6px;font-size:11px;font-weight:700;cursor:pointer}.section-label{color:#8B949E;font-size:10px;font-weight:700;text-transform:uppercase;margin-bottom:8px}.multi-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px}.stock-card{background:rgba(22,27,34,0.7);border-radius:10px;padding:10px;border:1px solid rgba(255,255,255,0.06);cursor:pointer}.stock-card.active{border-color:#00f2fe;background:linear-gradient(135deg,rgba(0,242,254,0.1) 0%,rgba(18,22,32,0.8) 100%)}.card-top{display:flex;justify-content:space-between}.card-symbol{font-weight:700;font-size:13px;color:#FFF}.remove-btn{color:#6E7681;font-size:11px;cursor:pointer}.card-price{font-size:12px;font-weight:600;color:#00f2fe;margin-top:2px}.card-mom{font-size:10px;font-weight:700;margin-top:2px}.card-chart{height:30px;margin-top:6px}.detail-card{background:rgba(22,27,34,0.85);border-radius:12px;padding:14px;border:1px solid rgba(255,255,255,0.08)}.detail-header{display:flex;justify-content:space-between;margin-bottom:12px}.stock-title{font-size:18px;font-weight:700;color:#FFF}.stock-price{font-size:20px;font-weight:700;color:#00f2fe}.signal-pill{padding:4px 10px;border-radius:12px;font-size:10px;font-weight:700}.bg-long{background:rgba(0,230,118,0.12);color:#00e676;border:1px solid rgba(0,230,118,0.4)}.bg-short{background:rgba(255,82,82,0.12);color:#ff5252;border:1px solid rgba(255,82,82,0.4)}.param-label{color:#8B949E;font-size:10px;font-weight:700;text-transform:uppercase;margin:12px 0 6px}.chart-container{height:80px;background:rgba(13,17,23,0.6);border-radius:8px;padding:6px;border:1px solid rgba(255,255,255,0.03);margin-bottom:12px;display:flex;align-items:center;justify-content:center}.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.grid-item,.full-item{background:rgba(13,17,23,0.5);padding:8px 10px;border-radius:6px;border:1px solid rgba(255,255,255,0.03)}.full-item{grid-column:span 2}.grid-key{color:#6E7681;font-size:9px;font-weight:600;text-transform:uppercase}.grid-val{color:#F0F4F8;font-size:12px;font-weight:600;margin-top:2px}.note-section{margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08)}.note-title{color:#FFD54F;font-size:10px;font-weight:700;margin-bottom:6px}.note-input{width:100%;background:rgba(13,17,23,0.7);color:#F0F4F8;border:1px solid rgba(255,255,255,0.1);border-radius:6px;padding:8px;font-size:11px;min-height:50px;resize:none;outline:none}</style></head><body><div class="mobile-frame"><div class="header"><div class="brand"><div class="live-dot"></div><div><h1>MOMENTUM RADAR</h1><div class="header-sub" id="lastUpdated">Lade Daten...</div></div></div><button class="refresh-btn" onclick="fetchAllData()">🔄 REFRESH</button></div><div class="add-bar"><input type="text" id="newTicker" class="add-input" placeholder="Ticker (z.B. MSFT)..." onkeydown="if(event.key==='Enter') addTicker()"><button class="add-btn" onclick="addTicker()">+ Hinzufügen</button></div><div class="section-label">Marktinstrumente Übersicht</div><div class="multi-grid" id="multiGrid"></div><div class="detail-card"><div class="detail-header"><div><div class="stock-title" id="stockTitle">-</div><div class="stock-price" id="stockPrice">$0.00</div></div><div class="signal-pill" id="signalPill">🟢 LONG</div></div><div class="param-label">Trendverlauf (Sparkline)</div><div class="chart-container" id="chartContainer"></div><div class="param-label">Indikatoren & Metriken</div><div class="grid"><div class="grid-item"><div class="grid-key">Momentum 3M</div><div class="grid-val" id="m3m">-</div></div><div class="grid-item"><div class="grid-key">vs. MSCI World</div><div class="grid-val" id="mVsMsci">-</div></div><div class="grid-item"><div class="grid-key">Volatilität</div><div class="grid-val" id="volatility">-</div></div><div class="grid-item"><div class="grid-key">Quelle</div><div class="grid-val" id="liquidity">-</div></div><div class="grid-item"><div class="grid-key">200-Tage-SMA</div><div class="grid-val" id="sma200">-</div></div><div class="grid-item"><div class="grid-key">RSI (14)</div><div class="grid-val" id="rsi">-</div></div><div class="full-item"><div class="grid-key">52-Wochen-Range</div><div class="grid-val" id="range52w">-</div></div></div><div class="note-section"><div class="note-title">📝 Notizen</div><textarea class="note-input" id="userNote" placeholder="Persönliche Anmerkungen..." oninput="saveNote()"></textarea></div></div></div><script>let tickers=JSON.parse(localStorage.getItem('m_tickers'))||["AAPL","TSLA","NVDA","MSFT","AMZN","BTC"];let stocksData=[];let selectedTicker=tickers[0]||"AAPL";let userNotes=JSON.parse(localStorage.getItem('m_notes')||'{}');const DEMO=[{ticker:"AAPL",name:"AAPL",price:224.50,signal:"LONG",momentum3m:"+14.2%",momentumVsMsci:"+5.8%",volatility:"16.4%",liquidity:"Aktiv (Demo)",sma200:"Über SMA200 (+8.1%)",rsi:61.2,range52w:"$164.08 - $237.23",chartPoints:[205,208,207,212,215,218,220,222,221,224.50]},{ticker:"TSLA",name:"TSLA",price:212.10,signal:"SHORT",momentum3m:"-18.5%",momentumVsMsci:"-22.1%",volatility:"42.8%",liquidity:"Aktiv (Demo)",sma200:"Unter SMA200 (-11.4%)",rsi:34.5,range52w:"$138.80 - $271.00",chartPoints:[250,245,240,232,228,220,215,210,208,212.10]}];async function fetchAllData(){document.getElementById('lastUpdated').innerText='Lade Kurse...';try{const res=await fetch('/api/stocks/batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tickers:tickers})});const json=await res.json();if(json.data&&json.data.length>0){stocksData=json.data;document.getElementById('lastUpdated').innerText='Stand '+json.timestamp;}else{useFallback();}}catch(e){useFallback();}render();}function useFallback(){stocksData=DEMO;document.getElementById('lastUpdated').innerText='Demo-Modus';}function addTicker(){const input=document.getElementById('newTicker');const val=input.value.trim().toUpperCase();if(val&&!tickers.includes(val)){tickers.push(val);localStorage.setItem('m_tickers',JSON.stringify(tickers));selectedTicker=val;input.value='';fetchAllData();}}function removeTicker(sym,e){e.stopPropagation();tickers=tickers.filter(t=>t!==sym);localStorage.setItem('m_tickers',JSON.stringify(tickers));if(selectedTicker===sym&&tickers.length>0)selectedTicker=tickers[0];fetchAllData();}function render(){const grid=document.getElementById('multiGrid');grid.innerHTML='';stocksData.forEach(s=>{const card=document.createElement('div');card.className=`stock-card ${s.ticker===selectedTicker?'active':''}`;card.onclick=()=>{selectedTicker=s.ticker;render();};const isUp=s.signal==='LONG';const sigColor=isUp?'#00e676':'#ff5252';card.innerHTML=`<div class="card-top"><span class="card-symbol">${s.ticker}</span><span class="remove-btn" onclick="removeTicker('${s.ticker}', event)">✕</span></div><div class="card-price">$${s.price}</div><div class="card-mom" style="color:${sigColor}">${s.signal==='LONG'?'🟢 LONG':'🔴 SHORT'} (${s.momentum3m})</div><div class="card-chart" id="spark_${s.ticker}"></div>`;grid.appendChild(card);renderMiniSparkline(`spark_${s.ticker}`,s.chartPoints,isUp);});const active=stocksData.find(s=>s.ticker===selectedTicker)||stocksData[0];if(!active)return;document.getElementById('stockTitle').innerText=active.name;document.getElementById('stockPrice').innerText='$'+active.price;const pill=document.getElementById('signalPill');pill.innerText=active.signal==='LONG'?'🟢 LONG':'🔴 SHORT';pill.className=`signal-pill ${active.signal==='LONG'?'bg-long':'bg-short'}`;document.getElementById('m3m').innerText=active.momentum3m;document.getElementById('mVsMsci').innerText=active.momentumVsMsci;document.getElementById('volatility').innerText=active.volatility;document.getElementById('liquidity').innerText=active.liquidity;document.getElementById('sma200').innerText=active.sma200;document.getElementById('rsi').innerText=active.rsi;document.getElementById('range52w').innerText=active.range52w;document.getElementById('userNote').value=userNotes[active.ticker]||'';renderSVGChart(active.chartPoints);}function renderMiniSparkline(elemId,points,isUp){const container=document.getElementById(elemId);if(!container||!points||points.length===0)return;const min=Math.min(...points);const max=Math.max(...points);const range=(max-min)||1;const width=120;const height=26;const coords=points.map((val,idx)=>{const x=(idx/(points.length-1))*width;const y=height-((val-min)/range)*(height-6)-3;return `${x.toFixed(1)},${y.toFixed(1)}`;});const color=isUp?'#00e676':'#ff5252';container.innerHTML=`<svg viewBox="0 0 ${width} ${height}" style="width:100%;height:100%;"><path d="M ${coords.join(' L ')}" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round"/></svg>`;}function renderSVGChart(points){const container=document.getElementById('chartContainer');if(!points||points.length===0)return;const min=Math.min(...points);const max=Math.max(...points);const range=(max-min)||1;const width=320;const height=65;const coords=points.map((val,idx)=>{const x=(idx/(points.length-1))*width;const y=height-((val-min)/range)*(height-14)-7;return `${x.toFixed(1)},${y.toFixed(1)}`;});const pathD=`M ${coords.join(' L ')}`;const areaD=`M 0,${height} L ${coords.join(' L ')} L ${width},${height} Z`;const isUp=points[points.length-1]>=points[0];const strokeColor=isUp?'#00e676':'#ff5252';container.innerHTML=`<svg viewBox="0 0 ${width} ${height}" style="width:100%;height:100%;"><defs><linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${strokeColor}" stop-opacity="0.25"/><stop offset="100%" stop-color="${strokeColor}" stop-opacity="0.0"/></linearGradient></defs><path d="${areaD}" fill="url(#chartGrad)" /><path d="${pathD}" fill="none" stroke="${strokeColor}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;}function saveNote(){const text=document.getElementById('userNote').value;userNotes[selectedTicker]=text;localStorage.setItem('m_notes',JSON.stringify(userNotes));}window.onload=fetchAllData;</script></body></html>"""

def open_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
