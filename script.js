/* ── STRIAX · script.js ─────────────────────────────────── */

/* ================================================================
   DATA INTEGRATION GUIDE
   ─────────────────────
   Replace the placeholder functions below with real API calls
   from your Python backend. Your teammate's ML output should
   provide data in the same shape shown in PLACEHOLDER_DATA.

   Example API call (once your backend is running):
   ────────────────────────────────────────────────
   async function fetchRealData(range = '1m') {
     const res = await fetch(`/api/rblx/history?range=${range}`);
     const data = await res.json();
     return data;
     // Expected shape:
     // {
     //   labels:    ['Apr 1', 'Apr 2', ...],  ← date strings
     //   prices:    [35.10, 36.42, ...],       ← closing prices
     //   ma7:       [34.80, 35.20, ...],       ← 7-day moving avg
     //   ma30:      [33.50, 33.90, ...],       ← 30-day moving avg
     //   volumes:   [12400000, 9800000, ...],  ← daily volume
     //   latest:    { price, change, changePct, date, volume },
     //   signal:    { label, trend, volatility, maNote },
     //   backtest:  { trades, winRate, avgReturn, maxDrawdown, sharpe, benchmark }
     // }
   }
   ================================================================ */


/* ── PLACEHOLDER DATA ─────────────────────────────────────── */

function generatePlaceholderPrices(n, start, volatility) {
  const prices = [start];
  for (let i = 1; i < n; i++) {
    const delta = (Math.random() - 0.47) * volatility;
    prices.push(Math.max(18, +(prices[i - 1] + delta).toFixed(2)));
  }
  return prices;
}

function movingAverage(prices, window) {
  return prices.map((_, i) => {
    const slice = prices.slice(Math.max(0, i - window + 1), i + 1);
    return +(slice.reduce((a, b) => a + b, 0) / slice.length).toFixed(2);
  });
}

function generateLabels(n, endDate = new Date()) {
  const labels = [];
  const d = new Date(endDate);
  for (let i = n - 1; i >= 0; i--) {
    const day = new Date(d);
    day.setDate(d.getDate() - i);
    // skip weekends
    if (day.getDay() === 0 || day.getDay() === 6) continue;
    labels.push(day.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
  }
  return labels.slice(-n);
}

const RANGE_DAYS = { '1w': 7, '1m': 30, '6m': 126 };

function buildPlaceholderData(range = '1m') {
  const n = RANGE_DAYS[range] || 30;
  const prices = generatePlaceholderPrices(n, 38.5, 1.2);
  const ma7    = movingAverage(prices, 7);
  const ma30   = movingAverage(prices, 30);
  const volumes = prices.map(() => Math.round((8 + Math.random() * 14) * 1e6));
  const labels  = generateLabels(n + 30).slice(-prices.length);

  const latest    = prices[prices.length - 1];
  const prev      = prices[prices.length - 2] || latest;
  const change    = +(latest - prev).toFixed(2);
  const changePct = +((change / prev) * 100).toFixed(2);

  return {
    labels, prices, ma7, ma30, volumes,
    latest: {
      price: latest,
      change,
      changePct,
      date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
      volume: volumes[volumes.length - 1]
    },
    signal: {
      label:      'Watch',
      trend:      'Neutral — no clear directional momentum detected yet.',
      volatility: 'Medium — daily returns within ±2% range over last 10 sessions.',
      maNote:     'Price hovering near 7-day MA — no confirmed crossover yet.',
      signalNote: 'Watch — model confidence below threshold for a strong buy/sell call.'
    },
    signalDistribution: { bullish: 54, neutral: 26, bearish: 20 },
    backtest: {
      trades: 248, winRate: '67.4%', avgReturn: '+2.3%',
      maxDrawdown: '-18.6%', sharpe: '1.14', benchmark: '-4.2%'
    }
  };
}


/* ── CHART INSTANCES ──────────────────────────────────────── */
let priceChartInst = null;
let volumeChartInst = null;
let signalChartInst = null;

const chartDefaults = {
  color: {
    grid:   'rgba(255,255,255,0.05)',
    tick:   '#545d70',
    accent: '#60a5fa',
    orange: '#fb923c',
    green:  '#4ade80',
    amber:  '#facc15',
    red:    '#f87171',
  }
};


/* ── PRICE CHART ──────────────────────────────────────────── */
function renderPriceChart(data) {
  const ctx = document.getElementById('priceChart').getContext('2d');
  if (priceChartInst) priceChartInst.destroy();

  const { labels, prices, ma7, ma30 } = data;
  const skip = Math.max(1, Math.floor(labels.length / 8));
  const skipped = labels.map((l, i) => i % skip === 0 ? l : '');

  priceChartInst = new Chart(ctx, {
    type: 'line',
    data: {
      labels: skipped,
      datasets: [
        {
          label: 'Close Price',
          data: prices,
          borderColor: chartDefaults.color.accent,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35,
          fill: {
            target: 'origin',
            above: 'rgba(96,165,250,0.06)'
          }
        },
        {
          label: '7-Day MA',
          data: ma7,
          borderColor: chartDefaults.color.orange,
          borderWidth: 1.5,
          borderDash: [5, 4],
          pointRadius: 0,
          tension: 0.35,
          fill: false
        },
        {
          label: '30-Day MA',
          data: ma30,
          borderColor: chartDefaults.color.green,
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          tension: 0.35,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index', intersect: false,
          backgroundColor: '#0f1318',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#8b93a7',
          bodyColor: '#f0f4ff',
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: chartDefaults.color.grid, drawBorder: false },
          ticks: { color: chartDefaults.color.tick, font: { size: 11 }, maxRotation: 0 }
        },
        y: {
          grid: { color: chartDefaults.color.grid, drawBorder: false },
          ticks: {
            color: chartDefaults.color.tick, font: { size: 11 },
            callback: v => '$' + v.toFixed(0)
          }
        }
      },
      interaction: { mode: 'index', intersect: false }
    }
  });
}


/* ── VOLUME CHART ─────────────────────────────────────────── */
function renderVolumeChart(data) {
  const ctx = document.getElementById('volumeChart').getContext('2d');
  if (volumeChartInst) volumeChartInst.destroy();

  const { labels, volumes } = data;
  const skip = Math.max(1, Math.floor(labels.length / 7));
  const skipped = labels.map((l, i) => i % skip === 0 ? l : '');

  volumeChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: skipped,
      datasets: [{
        label: 'Volume',
        data: volumes,
        backgroundColor: 'rgba(96,165,250,0.35)',
        hoverBackgroundColor: 'rgba(96,165,250,0.6)',
        borderRadius: 3,
        borderSkipped: false
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f1318',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#8b93a7',
          bodyColor: '#f0f4ff',
          callbacks: {
            label: ctx => ` Volume: ${(ctx.parsed.y / 1e6).toFixed(1)}M`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: chartDefaults.color.tick, font: { size: 10 }, maxRotation: 0 }
        },
        y: {
          grid: { color: chartDefaults.color.grid, drawBorder: false },
          ticks: {
            color: chartDefaults.color.tick, font: { size: 10 },
            callback: v => (v / 1e6).toFixed(0) + 'M'
          }
        }
      }
    }
  });
}


/* ── SIGNAL DONUT CHART ───────────────────────────────────── */
function renderSignalChart(dist) {
  const ctx = document.getElementById('signalChart').getContext('2d');
  if (signalChartInst) signalChartInst.destroy();

  signalChartInst = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Bullish', 'Neutral', 'Bearish'],
      datasets: [{
        data: [dist.bullish, dist.neutral, dist.bearish],
        backgroundColor: [
          chartDefaults.color.green,
          chartDefaults.color.amber,
          chartDefaults.color.red
        ],
        borderWidth: 0,
        hoverOffset: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f1318',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#8b93a7',
          bodyColor: '#f0f4ff',
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed}%`
          }
        }
      }
    }
  });
}


/* ── UPDATE UI ────────────────────────────────────────────── */
function updateUI(data) {
  const { latest, signal, backtest, signalDistribution } = data;

  /* ── Ticker strip ── */
  document.getElementById('latestPrice').textContent = `$${latest.price.toFixed(2)}`;
  document.getElementById('lastUpdated').textContent  = `Last updated: ${latest.date}`;

  const changeEl  = document.getElementById('priceChange');
  const direction = latest.change >= 0 ? 'up' : 'down';
  const sign      = latest.change >= 0 ? '+' : '';
  changeEl.textContent  = `${sign}${latest.change.toFixed(2)} (${sign}${latest.changePct.toFixed(2)}%)`;
  changeEl.className    = `ticker-change ${direction}`;

  /* ── Signal pill ── */
  const pill = document.getElementById('signalPill');
  const lbl  = signal.label.toLowerCase();
  pill.className = `signal-pill ${lbl === 'bullish' ? 'bullish' : lbl === 'bearish' ? 'bearish' : 'neutral'}`;
  document.getElementById('signalLabel').textContent = `AI Signal: ${signal.label}`;

  /* ── Metric cards ── */
  document.getElementById('cardPrice').textContent   = `$${latest.price.toFixed(2)}`;
  document.getElementById('cardChange').textContent  = `${sign}${latest.change.toFixed(2)} today`;
  document.getElementById('cardChange').className    = `card-sub ${direction}`;
  document.getElementById('cardVolume').textContent  = `${(latest.volume / 1e6).toFixed(1)}M`;
  document.getElementById('cardDate').textContent    = latest.date;

  /* ── AI Insights ── */
  document.getElementById('insightTrend').textContent      = signal.trend;
  document.getElementById('insightVolatility').textContent = signal.volatility;
  document.getElementById('insightSignal').textContent     = signal.signalNote;
  document.getElementById('insightMA').textContent         = signal.maNote;

  /* ── Backtest ── */
  document.getElementById('btTrades').textContent    = backtest.trades;
  document.getElementById('btWinRate').textContent   = backtest.winRate;
  document.getElementById('btAvgReturn').textContent = backtest.avgReturn;
  document.getElementById('btDrawdown').textContent  = backtest.maxDrawdown;
  document.getElementById('btSharpe').textContent    = backtest.sharpe;
  document.getElementById('btBenchmark').textContent = backtest.benchmark;
}


/* ── RANGE SWITCHER ───────────────────────────────────────── */
window.setRange = function (range, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');

  const data = buildPlaceholderData(range); // ← swap with fetchRealData(range) later
  renderPriceChart(data);
  renderVolumeChart(data);
};


/* ── INIT ─────────────────────────────────────────────────── */
(function init() {
  /* ── Replace buildPlaceholderData() with your real API call once ready ──
     Example:
       const data = await fetchRealData('1m');
     For now, placeholder data is used. */

  const data = buildPlaceholderData('1m');
  updateUI(data);
  renderPriceChart(data);
  renderVolumeChart(data);
  renderSignalChart(data.signalDistribution);
})();
