from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import csv
import secrets
import pandas as pd
import numpy as np



accounts: list[str] = []
tickets: np.ndarray = np.array([], dtype=np.int64)
winners_data: list[dict] = []

# first-draw "grand winner" pool = first 100 rows of the uploaded CSV
first_accounts: list[str] = []
first_tickets: np.ndarray = np.array([], dtype=np.int64)
first_draw_used: bool = False


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_column(columns, candidates):
    """Return the first column whose lowercased name matches a candidate."""
    lower = {c.lower().strip(): c for c in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None


def weighted_draw(acc_list, tk_array, count):
    """Draw `count` winners, weighted by ticket count, without replacement.

    The logic is identical to drawing one physical ticket out of the drum,
    seeing whose name is on it, setting that person aside, and drawing again:

      1. Build a running total of tickets (cumulative sum).
      2. Pick a random ticket number between 1 and the grand total, using
         `secrets` (cryptographically strong randomness).
      3. searchsorted finds, in O(log n), which account owns that ticket.
      4. Remove that account, then repeat for the next winner.

    NumPy does steps 1 and 3 in compiled C, so even at 300k accounts each
    draw is a fraction of a millisecond.
    """
    acc = list(acc_list)
    tk = tk_array.astype(np.int64).copy()
    count = min(count, len(acc))

    drawn = []
    for _ in range(count):
        cumulative = np.cumsum(tk)
        total = int(cumulative[-1])
        if total <= 0:
            break

        # secrets.randbelow(total) -> 0..total-1, so +1 gives 1..total
        winning_ticket = secrets.randbelow(total) + 1

        # first index where the running total reaches the winning ticket
        idx = int(np.searchsorted(cumulative, winning_ticket, side="left"))

        drawn.append({"account": acc[idx], "tickets": int(tk[idx])})

        # set this winner aside before the next draw
        del acc[idx]
        tk = np.delete(tk, idx)

    return drawn


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global accounts, tickets

    df = pd.read_csv(file.file)

    account_col = _find_column(df.columns, ["account", "name", "username", "user"])
    ticket_col = _find_column(df.columns, ["tickets", "ticket", "entries", "entry", "count"])

    if account_col is None or ticket_col is None:
        return JSONResponse(
            status_code=400,
            content={
                "detail": (
                    "CSV needs an account column (account/name) and a tickets "
                    f"column (tickets/entries). Found: {list(df.columns)}"
                )
            },
        )

    # Clean the data: numeric tickets, drop blanks / zero / negative entries.
    df = df[[account_col, ticket_col]].copy()
    df.columns = ["account", "tickets"]
    df["account"] = df["account"].astype(str).str.strip()
    df["tickets"] = pd.to_numeric(df["tickets"], errors="coerce").fillna(0)
    df = df[df["tickets"] > 0]

    # grab the first 100 rows (in file order) for the grand-winner draw
    global first_accounts, first_tickets, first_draw_used
    first_df = df.head(100).groupby("account", as_index=False)["tickets"].sum()
    first_accounts = first_df["account"].tolist()
    first_tickets = first_df["tickets"].astype(np.int64).to_numpy()
    first_draw_used = False

    # Merge duplicate accounts so each person is a single entry in the drum.
    df = df.groupby("account", as_index=False)["tickets"].sum()

    accounts = df["account"].tolist()
    tickets = df["tickets"].astype(np.int64).to_numpy()

    return {
        "players": int(len(accounts)),
        "tickets": int(tickets.sum()),
    }


@app.get("/draw")
async def draw():
    if len(accounts) == 0:
        return JSONResponse(status_code=400, content={"detail": "Upload a CSV first"})
    result = weighted_draw(accounts, tickets, 1)
    return result[0] if result else JSONResponse(status_code=400, content={"detail": "No valid entries"})


@app.get("/draw-multiple")
async def draw_multiple(count: int = Query(1)):
    global winners_data, first_draw_used

    if len(accounts) == 0:
        return JSONResponse(status_code=400, content={"detail": "Upload a CSV first"})

    # first draw after upload -> only the first 100 rows; every draw after -> everyone
    if not first_draw_used and len(first_accounts) > 0:
        pool_acc, pool_tk = first_accounts, first_tickets
        first_draw_used = True
    else:
        pool_acc, pool_tk = accounts, tickets

    winners_data = weighted_draw(pool_acc, pool_tk, max(1, count))
    return winners_data


@app.get("/sample-names")
async def sample_names(count: int = Query(60)):
    """A small random sample of account names, used only to make the
    on-screen 'shuffle' reel look real. Never used to pick winners."""
    if len(accounts) == 0:
        return []
    n = min(count, len(accounts))
    idx = np.random.choice(len(accounts), size=n, replace=False)
    return [accounts[int(i)] for i in idx]


@app.get("/stats")
async def stats():
    return {"players": int(len(accounts)), "tickets": int(tickets.sum()) if len(tickets) else 0}


@app.get("/download-winners")
async def download_winners():
    filename = "winners.csv"
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Account", "Tickets"])
        for index, winner in enumerate(winners_data, start=1):
            writer.writerow([index, winner["account"], winner["tickets"]])

    return FileResponse(filename, media_type="text/csv", filename=filename)