# How to Run StockAI

## 1. Prerequisites

- Python 3.12
- Node.js 18+ and npm
- PowerShell or a compatible terminal
- Optional DeepSeek/OpenAI API key for LLM analysis

## 2. Install Python Dependencies

```powershell
cd C:\repos\StockAI
python -m pip install -r requirements.txt
```

## 3. Configure Environment

The framework loads `.env`. Copy the example if needed:

```powershell
Copy-Item .env.example .env
```

Recommended LLM setting:

```text
DEEPSEEK_API_KEY=your_key
DEEPSEEK_MODEL=deepseek-v4-pro
RUN_TIMEOUT_SECONDS=180
STOCK_CACHE_DB=state/stock_cache.db
US_STOCK_CACHE_DB=state/us_stock_cache.db
SEC_EDGAR_USER_AGENT=StockAI Research contact@example.com
```

If no LLM key is set, `stock_analyst`/`us_validator` use deterministic fallbacks.

Fetched AkShare daily history is stored incrementally in `state/stock_cache.db`; only missing date ranges are downloaded on later runs.

## 4. Build the Portal Frontend

```powershell
cd C:\repos\StockAI\portal-ui
npm install
npm run build
cd ..

cd C:\repos\StockAI\stockportal\ui
npm install
npm run build
cd ..\..
```

For frontend development:

```powershell
cd C:\repos\StockAI\portal-ui
npm run dev
```

Stock portal UI development:

```powershell
cd C:\repos\StockAI\stockportal\ui
npm run dev
```

Vite runs at `http://127.0.0.1:5173` and proxies API calls to `http://127.0.0.1:8030`.

## 5. Start the Full Stack

```powershell
cd C:\repos\StockAI
python -m main run
```

This starts:

| Service | Default URL |
|---|---|
| Registry | `http://127.0.0.1:8001` |
| Stock data agent | `http://127.0.0.1:8021` |
| Stock news agent | `http://127.0.0.1:8022` |
| Stock analyst agent | `http://127.0.0.1:8023` |
| Stock quant agent | `http://127.0.0.1:8024` |
| US data agent | `http://127.0.0.1:8026` |
| US fundamental agent | `http://127.0.0.1:8027` |
| US validator agent | `http://127.0.0.1:8028` |
| A-share Orchestrator | `http://127.0.0.1:8020` |
| US Orchestrator | `http://127.0.0.1:8029` |
| Portal backend | `http://127.0.0.1:8030` |

Open the system monitoring portal at `http://127.0.0.1:8030/`.

Open the dedicated stock analysis portal at `http://127.0.0.1:8040/`.

## 6. Start Services Individually

```powershell
python -m main registry
python -m main agent --manifest plugins/stock_data/agent.yaml
python -m main agent --manifest plugins/stock_news/agent.yaml
python -m main agent --manifest plugins/stock_quant/agent.yaml
python -m main agent --manifest plugins/stock_analyst/agent.yaml
python -m main agent --manifest plugins/us_data/agent.yaml
python -m main agent --manifest plugins/us_fundamental/agent.yaml
python -m main agent --manifest plugins/us_validator/agent.yaml
python -m main orchestrator --manifest config/orchestration.yaml
python -m main orchestrator --manifest config/orchestration_us.yaml --port 8029 --checkpoint-db state/orchestrator_us.db --queue-db state/orchestrator_queue_us.db
python -m main portal
python -m main stockportal
```

## 7. Submit and Poll a Run

```powershell
$run = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8020/runs `
  -ContentType application/json `
  -Body '{"query":"600519"}'

$runId = $run.run_id

do {
  Start-Sleep -Seconds 2
  $result = Invoke-RestMethod -Uri "http://127.0.0.1:8020/runs/$runId"
} while ($result.status -eq "running")

$result.outputs.report
```

Submit a US stock to the independent US orchestrator:

```powershell
$run = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8029/runs `
  -ContentType application/json `
  -Body '{"query":"AAPL"}'
```

Refresh a batch in the backend per market:

```powershell
python -m main refresh-reports --symbols AAPL,MSFT --market us
python -m main refresh-reports --all --market us
```

Cancel a running job:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8020/runs/$runId/cancel"
```

## 8. Run Tests

```powershell
python -m pytest tests -q
python smoke_test.py
```

`smoke_test.py` requires live access to AkShare/Eastmoney/CNINFO.

## 9. Troubleshooting

### Port Already in Use

```powershell
Get-NetTCPConnection -LocalPort 8001,8020,8021,8022,8023,8024,8030,8040
```

Stop only the project Python processes before restarting.

### AkShare or News Endpoint Failure

The news agent continues if at least one source succeeds. The run fails only when both Eastmoney and CNINFO fail.

### LLM Not Configured

The final report falls back to deterministic MACD and LightGBM signals and states that LLM analysis was unavailable.

### Slow Quant Model

LightGBM results are cached for 15 minutes. Set `RUN_TIMEOUT_SECONDS=180` or higher for very slow networks.
