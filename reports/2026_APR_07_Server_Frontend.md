Bash Verify tornado is installed
IN
python3 -c "import tornado; print(tornado.version)" 2>&1

OUT
6.5.2
Tornado is installed — this is just a Pylance static analysis warning because tornado lacks type stubs. It won't affect runtime. No fix needed.

Let's verify the server starts correctly:

Bash Test server starts without import errors
IN
timeout 5 python3 src/web/server.py --port 8080 2>&1 || true

OUT
(eval):1: command not found: timeout
Bash Test server module imports cleanly
IN
python3 -c "
import sys
sys.path.insert(0, 'src')
from web.server import make_app
print('make_app import OK')
app = make_app()
print('make_app() OK')
" 2>&1

OUT
make_app import OK
make_app() OK
Clean. The warning is purely a Pylance resolver issue — tornado has no PEP 561 type stubs so VS Code flags it as unresolved. Runtime is fine.

To suppress it you can add a # type: ignore comment, but it's harmless as-is. To start the dashboard:


# From the repo root:
python src/main.py --web

# or directly:
python src/web/server.py
Then open http://localhost:8080.

What was built:

File	Purpose
src/web/server.py	Tornado backend with 9 REST endpoints, 5-min TTL cache
src/web/static/index.html	Vue 3 SPA (CDN, no build step)
Dashboard tabs:

Dashboard — regime cards (bullish/cautious/bearish coloring + score bar), valuation P/E + Fed Model spread, recent orders
US Market — 6-signal grid, Buffett Indicator, earnings yield spread
Korea — KOSPI regime, KR decision maker output, stock table with Kiwoom codes + scores
Crypto — Live Coinone BTC/ETH/XRP/SOL/ADA/DOGE tickers in KRW (1-min cache)
Reports — list of saved report .md files with inline viewer
