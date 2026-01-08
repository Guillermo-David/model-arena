import os
import csv
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score

from zoneinfo import ZoneInfo

# =======================
# CONFIG
# =======================
GRACE_SECONDS = 30
ADMIN_PIN = os.getenv("ADMIN_PIN", "1234")
COUNTDOWN_SECONDS = 15
CV_FOLDS = 5

LOCAL_TZ = ZoneInfo("Europe/Madrid")

DB_PATH = "arena.sqlite3"
RESULTS_CSV = Path("results_backup.csv")
GAMES_CSV = Path("games_backup.csv")


# =======================
# APP + DB
# =======================
app = FastAPI()
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row

def now():
    return datetime.now(tz=LOCAL_TZ)

def to_dt(x):
    if isinstance(x, datetime):
        return x
    return datetime.fromisoformat(x)


def init_db():
    conn.execute("""
    CREATE TABLE IF NOT EXISTS games (
  game_id INTEGER PRIMARY KEY,
  created_at TEXT,
  level_id TEXT,
  complexity_level INTEGER,
  duration_sec INTEGER,
  starts_at TEXT,
  ends_at TEXT
);

    """)

    conn.execute("""
CREATE TABLE IF NOT EXISTS results (
  id INTEGER PRIMARY KEY,
  time TEXT,
  game_id INTEGER,
  player_id TEXT,
  name TEXT,
  score REAL,
  is_clutch INTEGER,
  config_json TEXT,
  metrics_json TEXT
);

""")
    conn.commit()



init_db()

# =======================
# DATA
# =======================
data = load_diabetes(as_frame=True)
X_base = data.data
y = data.target


def add_noise(X, n, seed=42):
    if n <= 0:
        return X
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(len(X), n))
    return pd.concat(
        [X, pd.DataFrame(noise, columns=[f"noise_{i}" for i in range(n)], index=X.index)],
        axis=1
    )


# =======================
# GAME HELPERS
# =======================
def get_current_game():
    r = conn.execute("SELECT * FROM games ORDER BY game_id DESC LIMIT 1").fetchone()
    return dict(r) if r else None

def render_form_schema(schema: dict, values: dict) -> str:
    html = ""

    for name, field in schema.items():
        ftype = field["type"]
        enabled = field.get("enabled", True)
        default = values.get(name, field.get("default", ""))

        disabled = "disabled" if not enabled else ""
        hidden = not enabled


        if ftype == "select":
            label = field.get("label", name.capitalize())
            req = "required" if field.get("required", False) else ""

            html += f"<label>{label}: "
            html += f"<select name='{name}' {disabled} {req}>"
            for c in field["choices"]:
                if isinstance(c, tuple):
                    val, label = c
                else:
                    val, label = c, c
                sel = "selected" if str(default) == str(val) else ""
                html += f"<option value='{val}' {sel}>{label}</option>"
            html += "</select></label> "

            if hidden:
                html += f"<input type='hidden' name='{name}' value='{default}'>"

        elif ftype == "int":
            html += f"{name}: <input type='number' name='{name}' value='{default}' {disabled}> "
            if hidden:
                html += f"<input type='hidden' name='{name}' value='{default}'>"


        elif ftype == "float":
            html += f"{name}: <input type='number' step='0.1' name='{name}' value='{default}' {disabled}> "
            if hidden:
                html += f"<input type='hidden' name='{name}' value='{default}'>"


    return html



def game_status(game):
    if not game:
        return "no_game"

    start = to_dt(game["starts_at"])
    end   = to_dt(game["ends_at"])
    grace = end + timedelta(seconds=GRACE_SECONDS)

    if now() < start:
        return "countdown"
    if start <= now() <= end:
        return "running"
    if end < now() <= grace:
        return "grace"
    return "ended"


def remaining_seconds(game):
    return max(0, int((to_dt(game["ends_at"]) - now()).total_seconds()))


def remaining_grace_seconds(game):
    end = to_dt(game["ends_at"])
    grace_end = end + timedelta(seconds=GRACE_SECONDS)
    return max(0, int((grace_end - now()).total_seconds()))


from abc import ABC, abstractmethod

# ToDo
# Todo a un repo de github
# Sobreescribir esta clase para implementar un juego nuevo
# localhost:8000
# Añadir el nuevo juego a LEVELS = {}
# En la parte donde define el html <select name=level_id> hay que añadirlo también
# Investigar FastApi, interesante
# pip install uvicorn, fastapi, pandas, numpy...
# crear venv...
# lanzar con uvicorn app:app --reload
# Una vez funciona, vamos a al hosting: OnRender (free)
#   - dashboard 
#   - create project 
#   - Add new 
#   - webservice 
#   - gitProvider 
#   - enganchar el repositorio de github
#   - comando de ejecución en el formulario de creación de la app
#       https://docs.google.com/document/d/1zsBvUSuUu6o9FSGgmHFB8fPbnJrSJKqNrGNW1ZnMt0Q/edit?usp=sharing

class GameLevel(ABC):
    id: str
    name: str
    metric_order: list[str] = []

    def allowed_form(self, game: dict) -> dict:
        return {}

    @abstractmethod
    def description_html(self) -> str: ...

    @abstractmethod
    def form_schema(self, game: dict) -> dict: ...

    @abstractmethod
    def run(self, config: dict) -> dict: ...

    @abstractmethod
    def score(self, result: dict) -> float: ...

class RegressionR2Level(GameLevel):
    id = "regression_r2_penalized"
    name = "Regresión (R² penalizado)"
    metric_order = ["r2", "n_used", "noise"]


    def allowed_form(self, game: dict) -> dict:
        level = game.get("complexity_level", 2) if game else 2
        if level == 1:
            return dict(
                model=False,
                anova=False,
                k=False,
                alpha=False,
                noise=False,
            )
        return dict(
            model=True,
            anova=True,
            k=True,
            alpha=True,
            noise=True,
        )


    def description_html(self) -> str:
        return """
        <div class="card">
          <h3>Reto</h3>
          <p>Entrena un modelo (Ridge/Lasso). Score = R² - 0.01·k (si ANOVA).</p>
        </div>
        """

    def form_schema(self, game: dict) -> dict:
        flags = self.allowed_form(game)


        return {
            "model": {"type": "select", "choices": ["ridge", "lasso"], "enabled": flags["model"], "default": "ridge"},
            "anova": {"type": "select", "choices": [("1", "Sí"), ("0", "No")], "enabled": True, "default": "1"},
            "k": {"type": "int", "enabled": flags["k"], "default": "10"},
            "alpha": {"type": "float", "enabled": flags["alpha"], "default": "1.0"},
            "noise": {"type": "int", "enabled": flags["noise"], "default": "40"},
        }

    def run(self, config: dict) -> dict:
        model = config["model"]
        anova = int(config["anova"])
        k = int(config["k"])
        alpha = float(config["alpha"])
        noise = int(config["noise"])

        X = add_noise(X_base, noise)
        steps = [("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]

        if anova:
            n_used = min(k, X.shape[1])
            steps.append(("anova", SelectKBest(f_regression, k=n_used)))
        else:
            n_used = X.shape[1]

        model_obj = Ridge(alpha=alpha) if model == "ridge" else Lasso(alpha=alpha, max_iter=20000)
        steps.append(("model", model_obj))

        pipe = Pipeline(steps)
        r2 = cross_val_score(pipe, X, y, cv=CV_FOLDS, scoring="r2").mean()

        return {
            "r2": float(r2),
            "n_used": int(n_used),
            "model": model,
            "anova": int(anova),
            "k": int(k),
            "alpha": float(alpha),
            "noise": int(noise),
        }

    def score(self, result: dict) -> float:
        if result["anova"]:
            return result["r2"] - 0.01 * result["n_used"]
        return result["r2"]

class GuessTargetLevel(GameLevel):
    id = "guess_target"
    name = "Adivina el objetivo"
    metric_order = ["error"]

    def description_html(self) -> str:
        return """
        <div class="card">
          <h3>Reto</h3>
          <p>
            Elige un valor numérico.
            El score será mejor cuanto más cerca esté del objetivo oculto.
          </p>
        </div>
        """

    def form_schema(self, game: dict) -> dict:
        return {
            "guess": {
                "type": "float",
                "default": "100.0",
            }
        }

    def run(self, config: dict) -> dict:
        target = 150.0  # fijo a propósito
        guess = float(config["guess"])
        error = abs(guess - target)

        return {
            "error": error
        }

    def score(self, result: dict) -> float:
        # Score alto = mejor (invertimos el error)
        return max(0.0, 100.0 - result["error"])

class RobustezR2Level(GameLevel):
    id = "robust_r2"
    name = "Robustez (R² bajo ruido)"
    metric_order = ["r2_min", "r2_mean", "r2_std", "worst_noise", "n_used"]

    def allowed_form(self, game: dict) -> dict:
        level = game.get("complexity_level", 2) if game else 2
        if level == 1:
            return dict(model=False, anova=False, k=True, alpha=True)
        return dict(model=True, anova=True, k=True, alpha=True)

    def description_html(self) -> str:
        return """
        <div class="card">
          <h3>Reto</h3>
          <p>
            Ajusta un modelo que funcione bien incluso cuando añadimos columnas de ruido.
            El servidor lo prueba con varios <b>noise</b> y puntúa por tu rendimiento en el peor caso.
          </p>
          <p class="muted">
            Métricas: r2_min (peor), r2_mean (media), r2_std (variación).<br>
            Score = 0.7 * r2_min + 0.3 * r2_mean - 0.01 * k (si usas ANOVA).
          </p>
        </div>
        """

    def form_schema(self, game: dict) -> dict:
        flags = self.allowed_form(game)

        seed = str((game["game_id"] * 1337) if game else 1337)

        return {
            "model": {"type": "select", "choices": ["ridge", "lasso"], "enabled": flags["model"], "default": "ridge"},
            "anova": {"type": "select", "choices": [("1", "Sí"), ("0", "No")], "enabled": flags["anova"], "default": "1"},
            "k": {"type": "int", "enabled": flags["k"], "default": "10"},
            "alpha": {"type": "float", "enabled": flags["alpha"], "default": "1.0"},
            "seed": {"type": "int", "enabled": False, "default": seed},
        }

    def run(self, config: dict) -> dict:
        model = config.get("model", "ridge")
        anova = int(config.get("anova", 1))
        k = int(config.get("k", 10))
        alpha = float(config.get("alpha", 1.0))
        seed = int(config.get("seed", 1337))
        noises = [0, 20, 40, 80]

        steps = [("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]

        n_used = k if anova else None

        model_obj = Ridge(alpha=alpha) if model == "ridge" else Lasso(alpha=alpha, max_iter=20000)

        r2_by_noise = {}
        for n in noises:
            Xn = add_noise(X_base, n, seed=seed + n * 101)

            local_steps = list(steps)

            if anova:
                n_used_eff = min(k, Xn.shape[1])
                local_steps.append(("anova", SelectKBest(f_regression, k=n_used_eff)))
            else:
                n_used_eff = Xn.shape[1]

            pipe = Pipeline(local_steps + [("model", model_obj)])

            X_train, X_test, y_train, y_test = train_test_split(
                Xn, y, test_size=0.25, random_state=seed
            )
            pipe.fit(X_train, y_train)
            r2_by_noise[n] = float(pipe.score(X_test, y_test))

        r2_vals = np.array(list(r2_by_noise.values()), dtype=float)
        worst_idx = int(np.argmin(r2_vals))
        worst_noise = noises[worst_idx]

        if not anova:
            n_used = add_noise(X_base, worst_noise, seed=seed + worst_noise * 101).shape[1]

        return {
            "r2_min": float(r2_vals.min()),
            "r2_mean": float(r2_vals.mean()),
            "r2_std": float(r2_vals.std()),
            "worst_noise": int(worst_noise),
            "n_used": int(n_used),
            "model": model,
            "anova": int(anova),
            "k": int(k),
            "alpha": float(alpha),
        }

    def score(self, result: dict) -> float:
        base = 0.7 * result["r2_min"] + 0.3 * result["r2_mean"]
        if result["anova"]:
            base -= 0.01 * result["n_used"]
        return float(base)


LEVELS = {
    RegressionR2Level.id: RegressionR2Level(),
    GuessTargetLevel.id: GuessTargetLevel(),
    RobustezR2Level.id: RobustezR2Level(),
}

def get_level(game: dict | None) -> GameLevel:
    level_id = game.get("level_id") if game else RegressionR2Level.id
    return LEVELS.get(level_id, LEVELS[RegressionR2Level.id])


# =======================
# UI HELPERS
# =======================
def page(title, body):
    return f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
      margin: 24px;
      background: #fafafa;
    }}

    nav {{
      margin-bottom: 16px;
    }}

    nav a {{
      margin-right: 12px;
      text-decoration: none;
      font-weight: 600;
      color: #333;
    }}

    nav a:hover {{
      text-decoration: underline;
    }}

    h2 {{
      margin-bottom: 8px;
    }}

    .card {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 18px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }}

    .danger {{
      border-color: #f3a6a6;
      background: #fff1f1;
    }}

    .muted {{
      color: #555;
    }}

    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 8px;
    }}

    th, td {{
      border-bottom: 1px solid #eee;
      padding: 8px;
      text-align: left;
      font-size: 14px;
    }}

    th {{
      background: #fafafa;
    }}

    input, select, button {{
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid #ccc;
      font-size: 14px;
    }}

    button {{
      cursor: pointer;
      font-weight: 600;
      background: #2563eb;
      color: white;
      border: none;
    }}

    button:hover {{
      background: #1e4fd6;
    }}

    form {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}

    tr.gold td {{
        background-color: #fff6cc !important;
        }}
        tr.silver td {{
        background-color: #f0f0f0 !important;
        }}
        tr.bronze td {{
        background-color: #ffe4cc !important;
        }}


    .best-attempt {{
        background: #e8fff0;
        font-weight: 600;
        }}

    .progress {{
        width: 100%;
        height: 14px;
        background: #eee;
        border-radius: 8px;
        overflow: hidden;
        margin: 6px 0;
        }}

        .progress-bar {{
        height: 100%;
        background: linear-gradient(90deg, #22c55e, #2563eb);
        transition: width 0.5s ease;
        }}

            .status {{
      position: sticky;
      top: 0;
      z-index: 20;
      padding: 10px 12px;
      border-radius: 12px;
      margin-bottom: 12px;
      font-weight: 800;
      border: 1px solid #ddd;
      background: white;
      box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }}
    .status.countdown {{ background: #fef9c3; border-color: #f59e0b; }}
    .status.running   {{ background: #e0f2fe; border-color: #38bdf8; }}
    .status.grace     {{ background: #fff1f1; border-color: #fca5a5; }}
    .status.ended     {{ background: #f3f4f6; border-color: #cbd5e1; color: #334155; }}

    .last-attempt {{
        outline: 2px solid #2563eb;
        outline-offset: -2px;
        animation: flash-last 1.5s ease-out;
    }}

    .last-badge {{
        display: inline-block;
        margin-left: 8px;
        padding: 2px 8px;
        font-size: 12px;
        font-weight: 700;
        border-radius: 999px;
        background: #2563eb;
        color: white;
    }}


    @keyframes flash-last {{
        0% {{
            background-color: #dbeafe;
        }}
        100% {{
            background-color: transparent;
        }}
    }} 

    .best-attempt.last-attempt {{
        background-color: #e8fff0;
    }}


  </style>
</head>
<body>

<h2>{title}</h2>

<nav>
  <a href="/">Submit</a>
  <a href="/leaderboard">Leaderboard</a>
  <a href="/admin">Admin</a>
</nav>

{body}

<script>
  if (window.location.pathname === "/leaderboard") {{
    setTimeout(() => window.location.reload(), 5000);
  }}
</script>

    <script>
    if (window.location.pathname === "/admin") {{
        if (window.ADMIN_STATUS === "countdown") {{
        setTimeout(() => location.reload(), 1000);
        }}
        if (window.ADMIN_STATUS === "grace") {{
        setTimeout(() => location.reload(), 1000);
        }}
    }}
    </script>


<script>
(function () {{
  if (!["/", "/leaderboard"].includes(window.location.pathname)) return;

  const bar = document.getElementById("progress-bar");
  const label = document.getElementById("remaining-time");
  if (!bar || !label) return;

  let startsAt = parseInt(bar.dataset.startsAtTs);
  let endsAt   = parseInt(bar.dataset.endsAtTs);
  const duration = parseInt(bar.dataset.duration);

  if (isNaN(startsAt) || isNaN(endsAt) || isNaN(duration)) {{
    console.error("Progress bar dataset inválido:", bar.dataset);
    return;
  }}

  // 🔒 Normalización de unidades
  if (startsAt < 1e12) startsAt *= 1000;
  if (endsAt   < 1e12) endsAt   *= 1000;

  function update() {{
    const now = Date.now();
    const elapsed = Math.max(0, now - startsAt);
    const total = duration * 1000;

    const remaining = Math.max(0, Math.ceil((endsAt - now) / 1000));
    const percent = Math.max(0, 100 - (elapsed / total) * 100);

    bar.style.width = percent + "%";
    label.textContent = remaining + " s";

    if (remaining > 0) {{
      requestAnimationFrame(update);
    }} else {{
      bar.style.width = "0%";
      label.textContent = "⛔ finalizado";
      setTimeout(() => location.reload(), 500);
    }}
  }}

  update();
}})();
</script>


<script>
(function () {{
  if (window.location.pathname !== "/") return;

  let lastStatus = null;

  async function poll() {{
    try {{
      const res = await fetch("/game/status");
      const data = await res.json();

      if (lastStatus && lastStatus !== data.status) {{
        location.reload();
        return;
      }}

      lastStatus = data.status;
    }} catch (e) {{}}

    setTimeout(poll, 1000);
  }}

  poll();
}})();
</script>


</body>
</html>
"""

@app.get("/game/status")
def game_status_api():
    game = get_current_game()
    return {"status": game_status(game)}

# =======================
# ADMIN
# =======================
@app.get("/admin", response_class=HTMLResponse)
def admin():
    game = get_current_game()
    status = game_status(game)
    grace_remaining = remaining_grace_seconds(game) if game and status == "grace" else None
    is_grace = game and status == "grace"

    body = "<div class='card'>"
    if game:
        extra = ""
        if status == "grace":
            extra = f" | ⏱ Tiempo extra: <b>{grace_remaining}s</b>"

        body += f"""
        <p>
        Partida #{game['game_id']} |
        Nivel {game['complexity_level']} |
        Estado: <b>{status}</b>{extra}
        </p>
        """
    else:
        body += "<p>No hay partida activa</p>"
    body += "</div>"

    # ---------- START GAME ----------
    body += """
    <div class="card">
      <form method="post" action="/admin/start">
        PIN: <input type="password" name="pin">
        Tipo de partida:
<select name="level_id">
  <option value="regression_r2_penalized">
    Regresión (R² penalizado)
  </option>
  <option value="guess_target">
    Adivina el objetivo
  </option>
  <option value="robust_r2">
    Robustez (R² bajo ruido)
  </option>
</select>

Nivel:
<select name="complexity_level">
  <option value="1">Nivel 1</option>
  <option value="2" selected>Nivel 2</option>
</select>
        Duración (min):
        <select name="duration_min">
          <option value="10">10</option>
          <option value="20" selected>20</option>
        </select>
        <button>Iniciar partida</button>
      </form>
    </div>
    """

    # ---------- RECUPERAR ALUMNO ----------
    names = conn.execute(
        """
        SELECT name, COUNT(*) AS n
        FROM results
        WHERE name IS NOT NULL AND name != ''
        GROUP BY name
        ORDER BY name
        """
    ).fetchall()

    recovery_rows = ""

    if len(names) > 1:
        canonical = names[0]["name"]

        for n in names[1:]:
            bad = n["name"]
            recovery_rows += f"""
            <form method="post" action="/admin/rename" style="margin-bottom:8px">
            <b>{bad}</b> → <b>{canonical}</b><br>
            PIN: <input type="password" name="pin">
            <input type="hidden" name="from_name" value="{bad}">
            <input type="hidden" name="to_name" value="{canonical}">
            <button>Unificar nombre</button>
            </form>
            """

    body += f"""    
    <div class="card danger">
    <form method="post" action="/admin/reset">
        PIN: <input type="password" name="pin">
        <button>Reset leaderboard</button>
    </form>

    {""
    if not game else f"""
    <form method="post" action="/admin/end">
        <input type="hidden" name="game_id" value="{game['game_id']}">
        PIN: <input type="password" name="pin">
        <button {"disabled" if status == "countdown" else ""}>
        ⛔ Terminar partida ahora
        </button>
    </form>
    """}
    </div>
    """




    body += f"""
    <div class="card">
        <h3>Unificar nombres</h3>
        <p class="muted">
        Corrige nombres duplicados o mal escritos (Juan / juan / Juan Pérez).
        </p>
       {recovery_rows if recovery_rows else "<p class='muted'>No hay duplicados detectados.</p>"}
    </div>
    """

    body += f"""
    <script>
        window.ADMIN_STATUS = "{status}";
    </script>
    """


    return page("Admin", body)

@app.post("/admin/rename")
def admin_rename(
    pin: str = Form(...),
    from_name: str = Form(...),
    to_name: str = Form(...)
):
    if pin != ADMIN_PIN:
        return HTMLResponse(page(
            "Admin",
            "<div class='card danger'>PIN incorrecto</div>"
        ))

    conn.execute(
        """
        UPDATE results
        SET name=?
        WHERE name=?
        """,
        (to_name, from_name)
    )
    conn.commit()

    return RedirectResponse("/admin", status_code=303)

#9.22

@app.post("/admin/end")
def admin_end(pin: str = Form(...), game_id: int = Form(...)):
    if pin != ADMIN_PIN:
        return HTMLResponse(page(
            "Admin",
            "<div class='card danger'>PIN incorrecto</div>"
        ))

    forced_end = now() - timedelta(seconds=1)
    conn.execute(
        "UPDATE games SET ends_at=? WHERE game_id=?",
        (forced_end.isoformat(), game_id)
    )
    conn.commit()

    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/start")
def admin_start(
    pin: str = Form(...),
    level_id: str = Form(...),
    complexity_level: int = Form(...),
    duration_min: int = Form(...)
):


    if pin != ADMIN_PIN:
        return HTMLResponse(page("Admin", "<div class='card danger'>PIN incorrecto</div>"))

    start = now() + timedelta(seconds=COUNTDOWN_SECONDS)
    end   = start + timedelta(minutes=int(duration_min))

    last = conn.execute("SELECT MAX(game_id) FROM games").fetchone()[0]
    game_id = (last or 0) + 1

    row = (
  game_id,
  now().isoformat(),
  level_id,
  complexity_level,
  duration_min * 60,
  start.isoformat(),
  end.isoformat()
)

    conn.execute("""INSERT INTO games (
  game_id,
  created_at,
  level_id,
  complexity_level,
  duration_sec,
  starts_at,
  ends_at
)
 VALUES (?,?,?,?,?,?,?)""", row)
    conn.commit()

    with GAMES_CSV.open("a", newline="") as f:
        csv.writer(f).writerow(row)

    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/reset")
def admin_reset(pin: str = Form(...)):
    if pin != ADMIN_PIN:
        return HTMLResponse(page("Admin", "<div class='card danger'>PIN incorrecto</div>"))

    conn.execute("DELETE FROM results")
    conn.execute("DELETE FROM games")
    conn.commit()

    if RESULTS_CSV.exists():
        RESULTS_CSV.unlink()
    if GAMES_CSV.exists():
        GAMES_CSV.unlink()

    return RedirectResponse("/admin", status_code=303)


# =======================
# SUBMIT
# =======================
@app.get("/", response_class=HTMLResponse)
def submit_page(request: Request):
    game = get_current_game()

    if not game:
        return page("Submit", "<div class='card danger'>No hay partida activa</div>")

    level_obj = get_level(game)
    
    q = request.query_params
    form_model = q.get("model", "ridge")
    form_anova = int(q.get("anova", 1))
    form_k = q.get("k", "10")
    form_alpha = q.get("alpha", "1.0")
    form_noise = q.get("noise", "40")

    player_id = request.cookies.get("player_id")

    # 🔐 Crear identidad si no existe (primer acceso)
    if not player_id:
        player_id = os.urandom(16).hex()
    
    # Nombre del jugador para ESTA partida (se fija al primer envío)
    existing = conn.execute(
        """
        SELECT name
        FROM results
        WHERE game_id=? AND player_id=?
        LIMIT 1
        """,
        (game["game_id"], player_id)
    ).fetchone()

    player_name = existing["name"] if existing else None


    status = game_status(game)
    is_grace = status == "grace"

    used_clutch = False
    if is_grace:
        used_clutch = conn.execute(
            """
            SELECT COUNT(*) 
            FROM results
            WHERE game_id=? AND player_id=? AND is_clutch=1
            """,
            (game["game_id"], player_id)
        ).fetchone()[0] > 0

    banner = ""
    if status == "countdown":
        start_in = max(0, int((to_dt(game["starts_at"]) - now()).total_seconds()))
        banner = f"<div class='status countdown'>⏳ Empieza en <b>{start_in}s</b></div>"
    elif status == "running":
        banner = "<div class='status running'>▶ Partida en curso</div>"
    elif status == "grace":
        banner = "<div class='status grace'>⭐ Última jugada · 1 intento · sin penalización</div>"
    else:
        banner = "<div class='status ended'>⛔ Partida finalizada</div>"


    start = to_dt(game["starts_at"])
    end   = to_dt(game["ends_at"])

    if is_grace:
        bar_start = end
        bar_end   = end + timedelta(seconds=GRACE_SECONDS)
        duration  = GRACE_SECONDS
    else:
        bar_start = start
        bar_end   = end
        duration  = int((end - start).total_seconds())



    starts_at_ts = int(bar_start.timestamp())
    ends_at_ts   = int(bar_end.timestamp())

    msg = ""
    last_flag = request.query_params.get("last") == "1"

    if status == "ended":
        body = """
        <div class="status ended">
        ⛔ Partida finalizada
        </div>
        <div class="card">
        Esperando nueva partida…
        </div>
        """

        # ======================
        # RESUMEN ÚLTIMA PARTIDA
        # ======================
        rows = conn.execute(
            """
            SELECT
                player_id,
                MAX(name) AS name,
                MAX(COALESCE(score, 0)) AS score
            FROM results
            WHERE game_id=?
            GROUP BY player_id
            ORDER BY score DESC, id DESC
            """,
            (game["game_id"],)
        ).fetchall()

        ranking = list(rows)
        if not ranking:
            body += """
            <div class="card muted">
                No se ha realizado ninguna jugada en esta partida.
            </div>
            """
        else:
            my_index = None
            for i, r in enumerate(ranking):
                if r["player_id"] == player_id:
                    my_index = i
                    break

            visible = set(range(min(3, len(ranking))))

            if my_index is not None:
                for i in range(my_index - 2, my_index + 3):
                    if 0 <= i < len(ranking):
                        visible.add(i)

            visible = sorted(visible)

            body += """
            <div class="card">
            <h3>Resumen última partida</h3>
            <table>
                <tr>
                <th>#</th>
                <th>Jugador</th>
                <th>Score</th>
                </tr>
            """

            prev = None
            for i in visible:
                if prev is not None and i > prev + 1:
                    hidden = i - prev - 1
                    body += f"""
                    <tr>
                    <td colspan="3" class="muted" style="text-align:center">
                        … {hidden} jugadores ocultos …
                    </td>
                    </tr>
                    """

                r = ranking[i]
                prev = i

                cls = ""
                medal = i + 1
                if i == 0:
                    cls, medal = "gold", "🥇"
                elif i == 1:
                    cls, medal = "silver", "🥈"
                elif i == 2:
                    cls, medal = "bronze", "🥉"

                classes = [cls]
                if r["player_id"] == player_id:
                    classes += ["best-attempt", "last-attempt"]

                body += f"""
                <tr class="{' '.join(classes).strip()}">
                <td>{medal}</td>
                <td>{r["name"]}</td>
                <td><b>{r["score"]:.3f}</b></td>
                </tr>
                """

            if not visible:
                body += "</table></div>"
                return page("Submit", body)

            last_visible = visible[-1]
            hidden_after = len(ranking) - last_visible - 1
            if hidden_after > 0:
                body += f"""
                <tr>
                <td colspan="3" class="muted" style="text-align:center">
                    … {hidden_after} jugadores ocultos …
                </td>
                </tr>
                """

            body += "</table></div>"


        # ======================
        # RANKING GENERAL
        # ======================
        rows = conn.execute(
            """
            SELECT
                player_id,
                MAX(name) AS name,
                SUM(COALESCE(score, 0)) AS total
            FROM results
            GROUP BY player_id
            ORDER BY total DESC
            """
        ).fetchall()

        ranking = list(rows)
        # Asegurar que el jugador actual aparece aunque no tenga jugadas
        if player_id and all(r["player_id"] != player_id for r in ranking):
            ranking.append({
                "player_id": player_id,
                "name": player_name or "—",
                "total": 0.0
            })

        # Reordenar tras añadirlo
        ranking.sort(key=lambda r: r["total"], reverse=True)

        if not ranking:
            body += """
            <div class="card">
                <h3>Resumen del Ranking general</h3>
                <p class="muted">
                    Aún no se ha registrado ninguna jugada en el sistema.
                </p>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Jugador</th>
                        <th>Total</th>
                    </tr>
                </table>
            </div>
            """
        else:
            my_index = None
            for i, r in enumerate(ranking):
                if r["player_id"] == player_id:
                    my_index = i
                    break

            visible = set(range(min(3, len(ranking))))

            if my_index is not None:
                for i in range(my_index - 2, my_index + 3):
                    if 0 <= i < len(ranking):
                        visible.add(i)

            visible = sorted(visible)

            body += """
            <div class="card">
            <h3>Resumen del Ranking general</h3>
            <table>
                <tr>
                <th>#</th>
                <th>Jugador</th>
                <th>Total</th>
                </tr>
            """

            prev = None
            for i in visible:
                if prev is not None and i > prev + 1:
                    hidden = i - prev - 1
                    body += f"""
                    <tr>
                    <td colspan="3" class="muted" style="text-align:center">
                        … {hidden} jugadores ocultos …
                    </td>
                    </tr>
                    """

                r = ranking[i]
                prev = i

                cls = ""
                medal = i + 1
                if i == 0:
                    cls, medal = "gold", "🥇"
                elif i == 1:
                    cls, medal = "silver", "🥈"
                elif i == 2:
                    cls, medal = "bronze", "🥉"

                classes = [cls]
                if r["player_id"] == player_id:
                    classes += ["best-attempt", "last-attempt"]

                body += f"""
                <tr class="{' '.join(classes).strip()}">
                <td>{medal}</td>
                <td>{r["name"]}</td>
                <td><b>{r["total"]:.3f}</b></td>
                </tr>
                """

            if not visible:
                body += "</table></div>"
                return page("Submit", body)

            last_visible = visible[-1]
            hidden_after = len(ranking) - last_visible - 1
            if hidden_after > 0:
                body += f"""
                <tr>
                <td colspan="3" class="muted" style="text-align:center">
                    … {hidden_after} jugadores ocultos …
                </td>
                </tr>
                """

            body += "</table></div>"


        body += """
        <script>
        setTimeout(function() {
            location.reload();
        }, 5000);
        </script>
        """

        return page("Submit", body)




    if status == "countdown":
        start_ts = int(to_dt(game["starts_at"]).timestamp())


        body = f"""
        <div class="status countdown">
        ⏳ Empieza en <b><span id="countdown"></span>s</b>
        </div>

        <script>
        (function() {{
        const startTs = {start_ts} * 1000;

        function tick() {{
            const now = Date.now();
            const remaining = Math.max(0, Math.ceil((startTs - now) / 1000));
            const el = document.getElementById("countdown");

            if (el) {{
            el.textContent = remaining;
            }}

            if (remaining <= 0) {{
            location.reload();
            }} else {{
            setTimeout(tick, 1000);
            }}
        }}

        tick();
        }})();
        </script>
        """

        return page("Submit", body)

    body = banner
    body += level_obj.description_html()
    schema = level_obj.form_schema(game)
    body += msg

    body += f"""
    <div class="progress">
    <div
        id="progress-bar"
        class="progress-bar"
        data-starts-at-ts="{starts_at_ts}"
        data-ends-at-ts="{ends_at_ts}"
        data-duration="{duration}"
        data-grace="{str(is_grace).lower()}">
    </div>
    </div>
    <small id="remaining-time"></small>
    """

    if is_grace and used_clutch:
        body += """
        <div class="card danger">
            ⛔ Ya has utilizado tu última jugada en esta partida
        </div>
        """
    else:
        if is_grace and not used_clutch:
            body += """
            <div class="card danger">
            🕒 <b>Última jugada</b><br>
            Un único envío sin penalizaciones.
            </div>
            """
        body += "<div class='card'>"

        # Si ya hay nombre guardado para (game_id, player_id), no se vuelve a pedir
        if player_name:
            body += f"""
            <p class="muted">Jugador</p>
            <p><b>{player_name}</b></p>
            """
            # Para poder entrenar, seguimos necesitando el form (pero sin pedir nombre)
            body += f"""
            <form method="post" action="/submit">
            <input type="hidden" name="name" value="{player_name}">
            """
        else:
            body += """
            <form method="post" action="/submit">
            Nombre: <input name="name" required>
            """

        body += f"""
        <input type="hidden" name="player_id" value="{player_id}">
"""

        values = {
            "model": form_model,
            "anova": form_anova,
            "k": form_k,
            "alpha": form_alpha,
            "noise": form_noise,
        }

        body += render_form_schema(schema, values)
        
        body += f"""
        <button name="mode" value="normal"
            {"disabled" if is_grace and used_clutch else ""}>
            Entrenar
        </button>

        {f"""
<button name="mode" value="clutch_best"
    {"disabled" if used_clutch else ""}>
    ⭐ Rejugar mejor intento
</button>
""" if is_grace else ""}

        </form>
        """


        body += """        
                </div>
                """

    last_id = None
    if player_id and last_flag:
        last_row = conn.execute(
            """
            SELECT id, score, is_clutch, config_json, metrics_json
            FROM results
            WHERE game_id=? AND player_id=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (game["game_id"], player_id)
        ).fetchone()

        last_id = last_row["id"] if last_row else None

        prev_best = conn.execute(
            """
            SELECT score
            FROM results
            WHERE game_id=? AND player_id=? AND id < ?
            ORDER BY score DESC
            LIMIT 1
            """,
            (game["game_id"], player_id, last_row["id"])
        ).fetchone() if last_row else None

        if last_row:
            delta_txt = ""
            if prev_best:
                delta = last_row["score"] - prev_best["score"]
                arrow = "↑" if delta >= 0 else "↓"
                delta_txt = f" <span class='muted'>({arrow} {delta:+.3f})</span>"

            import json

            clutch = " ⭐" if last_row["is_clutch"] else ""

            metrics = json.loads(last_row["metrics_json"])
            config  = json.loads(last_row["config_json"])

            metrics_html = "".join(
                f"<li><b>{k}</b>: {metrics.get(k, '-')}</li>"
                for k in level_obj.metric_order
            )


            visible_keys = [k for k, f in schema.items() if f.get("enabled", True)]
            config_html = ", ".join(f"{k}={config[k]}" for k in visible_keys if k in config)


            body += f"""
            <div class="card">
            <h3>Último intento{clutch}</h3>

            <p><b>Score:</b> {last_row['score']:.3f}</p>

            <p class="muted">Configuración: {config_html}</p>

            <ul class="muted">
                {metrics_html}
            </ul>
            </div>
            """


    # Historial del jugador (SIEMPRE por player_id)
    if player_id:
        rows = conn.execute(
            """
            SELECT
                id,
                score,
                is_clutch,
                config_json,
                metrics_json
            FROM results
            WHERE game_id=? AND player_id=?
            ORDER BY score DESC
            """,
            (game["game_id"], player_id)
        ).fetchall()


        body += """
            <p class="muted">Identidad activa · historial persistente</p>
        """
        if rows:

            import json

            metric_keys = []
            if rows:
                metric_keys = level_obj.metric_order or []

                body += """
                <div class='card'>
                <h3>Tus intentos</h3>
                <table>
                <tr>
                <th>#</th>
                <th>Total</th>
                """
                for k in metric_keys:
                    body += f"<th>{k}</th>"
                body += "</tr>"


            for i, r in enumerate(rows, 1):
                metrics = json.loads(r["metrics_json"])
                classes = []

                if i == 1:
                    classes.append("best-attempt")
                if last_flag and last_id and r["id"] == last_id:
                    classes.append("last-attempt")

                is_last = last_flag and last_id and r["id"] ==  last_id
                
                body += f"<tr class='{' '.join(classes)}'>"
                badge = " <span class='last-badge'>última</span>" if is_last else ""
                body += f"<td>{i}{badge}</td>"

                body += f"<td><b>{r['score']:.3f}</b></td>"

                

                for k in metric_keys:
                    body += f"<td>{metrics.get(k, '-')}</td>"

                body += "</tr>"


            body += "</table></div>"

        else:
            body += "<div class='card'><p class='muted'>Aún no tienes envíos en esta partida.</p></div>"

    response = HTMLResponse(page("Submit", body))
    if player_id:
        response.set_cookie(
            "player_id",
            player_id,
            max_age=60 * 60 * 24 * 365
        )

    return response


from fastapi import Request

@app.post("/submit")
async def submit_action(
    request: Request,
    name: str = Form(...),
    player_id: str = Form(...),
    mode: str = Form("normal"),
):

    game = get_current_game()
    level_obj = get_level(game)

    form = await request.form()
    schema = level_obj.form_schema(game)

    config = {}
    for field in schema.keys():
        if field in form:
            config[field] = form[field]


    status = game_status(game)


    # Si ya hay nombre para (game_id, player_id), lo imponemos (nombre fijo por partida)
    existing = conn.execute(
        """
        SELECT name
        FROM results
        WHERE game_id=? AND player_id=?
        LIMIT 1
        """,
        (game["game_id"], player_id)
    ).fetchone()

    if existing and existing["name"]:
        name = existing["name"]

    if status not in ("running", "grace"):
        return RedirectResponse("/", status_code=303)

    is_clutch = status == "grace"
    if is_clutch:
        used = conn.execute(
            """
            SELECT COUNT(*) FROM results
            WHERE game_id=? AND player_id=? AND time > ?
            """,
            (game["game_id"], player_id, game["ends_at"])
        ).fetchone()[0]


        if used:
            return HTMLResponse(
                page("Submit", "<div class='card danger'>⛔ Ya usaste tu última jugada</div>")
            )
    
    if is_clutch and mode == "clutch_best":
        best = conn.execute(
            """
            SELECT config_json
            FROM results
            WHERE game_id=? AND player_id=?
            ORDER BY score DESC
            LIMIT 1
            """,
            (game["game_id"], player_id)
        ).fetchone()

        if not best:
            return HTMLResponse(
                page("Submit", "<div class='card danger'>No tienes intentos previos</div>")
            )

        import json
        config = json.loads(best["config_json"])
    
    result = level_obj.run(config)
    score = level_obj.score(result)

    import json

    conn.execute(
    """
    INSERT INTO results
    (time, game_id, player_id, name, score, is_clutch, config_json, metrics_json)
    VALUES (?,?,?,?,?,?,?,?)
    """,
    (
        now().isoformat(),
        game["game_id"],
        player_id,
        name,
        score,
        1 if is_clutch else 0,
        json.dumps(config),
        json.dumps(result),
    )
    )

    return RedirectResponse("/?last=1", status_code=303)




# =======================
# LEADERBOARD
# =======================
@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard():
    game = get_current_game()
    if not game:
        return page("Leaderboard", "<div class='card danger'>No hay partidas</div>")

    status = game_status(game)
    is_grace = status == "grace"

    if is_grace:
        ends_at = to_dt(game["ends_at"]) + timedelta(seconds=GRACE_SECONDS)

        duration = GRACE_SECONDS
    else:
        ends_at = to_dt(game["ends_at"])

        duration = game["duration_sec"]

    starts_at_ts = int(to_dt(game["starts_at"]).timestamp())

    ends_at_ts   = int(ends_at.timestamp())

    body = f"""
    <div class="progress">
    <div
        id="progress-bar"
        class="progress-bar"
        data-starts-at-ts="{starts_at_ts}"
        data-ends-at-ts="{ends_at_ts}"
        data-duration="{duration}"
        data-grace="{str(is_grace).lower()}">
    </div>
    </div>
    <small id="remaining-time"></small>
    """

    if is_grace:
        body += """
        <div class="card danger">
        🕒 <b>Última jugada</b><br>
        Un único envío sin penalizaciones.
        </div>
        """

    title = (
        "Ranking última partida"
        if status == "ended"
        else f"Ranking partida actual (Partida #{game['game_id']})"
    )

    body += f"<div class='card'><h3>{title}</h3><table>"


    rows = conn.execute(
        """
        SELECT r.player_id, r.name, r.score, r.is_clutch, r.metrics_json
        FROM results r
        WHERE r.game_id=?
        AND r.id = (
            SELECT id
            FROM results
            WHERE game_id=? AND player_id=r.player_id
            ORDER BY score DESC, id DESC
            LIMIT 1
        )
        ORDER BY r.score DESC
        """,
        (game["game_id"], game["game_id"])
    ).fetchall()


    import json

    level_obj = get_level(game)
    metric_keys = level_obj.metric_order or []


    body += "<tr><th>#</th><th>Jugador</th><th>Score</th>"
    for k in metric_keys:
        body += f"<th>{k}</th>"
    body += "</tr>"


    for i, r in enumerate(rows, 1):
        cls = "gold" if i == 1 else "silver" if i == 2 else "bronze" if i == 3 else ""
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else i
        star = ' <span title="Última jugada sin penalizaciones">⭐</span>' if r["is_clutch"] else ""



        body += f"""
        <tr class="{cls}">
        <td>{medal}</td>
        <td>{r['name']}{star}</td>
        <td><b>{r['score']:.3f}</b></td>
        """

        metrics = json.loads(r["metrics_json"])
        for k in metric_keys:
            body += f"<td>{metrics.get(k, '-')}</td>"

        body += "</tr>"


    body += "</table></div>"

    # -------- Ranking acumulado --------
    body += "<div class='card'><h3>Ranking acumulado</h3><table>"
    body += "<tr><th>#</th><th>Jugador</th><th>Total</th></tr>"

    
    
    rows = conn.execute(
        """
       SELECT
    player_id,
    MAX(name) AS name,
    SUM(score) AS total
FROM results
GROUP BY player_id
ORDER BY total DESC
        """
    ).fetchall()
    




    for i, r in enumerate(rows, 1):
        cls = "gold" if i == 1 else "silver" if i == 2 else "bronze" if i == 3 else ""
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else i

        body += f"""
        <tr class='{cls}'>
            <td>{medal}</td>
            <td>{r['name']}</td>
            <td><b>{r['total']:.3f}</b></td>
        </tr>
        """


    body += "</table></div>"

    return page("Leaderboard", body)





