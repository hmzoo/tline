#!/usr/bin/env python3
"""tline — serveur web minimal (bibliothèque standard uniquement).

Sert le dossier `static/` sur http://127.0.0.1:8000 par défaut.

    python server.py            # démarrage simple
    python server.py --open     # ouvre le navigateur
    python server.py -p 9000    # autre port
"""

from __future__ import annotations

import argparse
import errno
import functools
import http.server
import json
import pathlib
import re
import socketserver
import threading
import urllib.parse
import webbrowser

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "dataset.jsonl"
MODELS_DIR = DATA_DIR / "models"
PORT_TRIES = 10  # nombre de ports successifs essayés si le port demandé est occupé
MAX_SAMPLES = 50_000  # taille maximale du jeu de données conservé
TRAIN_LIMIT = 20_000  # nombre de couples renvoyés au navigateur
MAX_MODEL_POINTS = 60_000  # points max dans un modèle sauvegardé
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DATA_LOCK = threading.Lock()


def load_dataset() -> list[list[float]]:
    """Relit le jeu de données (une paire [vx, vy] par ligne)."""
    if not DATA_FILE.exists():
        return []
    out: list[list[float]] = []
    with DATA_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(pair, list) and len(pair) == 2:
                out.append([float(pair[0]), float(pair[1])])
    return out


def append_dataset(items: list) -> int:
    """Ajoute des couples (vx, vy) et renvoie la taille totale conservée."""
    clean: list[list[float]] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                clean.append([float(item[0]), float(item[1])])
            except (TypeError, ValueError):
                continue
    if not clean:
        return len(load_dataset())

    with _DATA_LOCK:
        data = load_dataset() + clean
        data = data[-MAX_SAMPLES:]
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = DATA_FILE.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for vx, vy in data:
                handle.write(f"[{vx:.3f}, {vy:.3f}]\n")
        tmp.replace(DATA_FILE)
        return len(data)


def clear_dataset() -> None:
    with _DATA_LOCK:
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILE.write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# modèles sauvegardés (data/models/<nom>.json)
# ---------------------------------------------------------------------------


def model_path(name: str) -> pathlib.Path | None:
    """Chemin du fichier de modèle, ou None si le nom est invalide."""
    if not MODEL_NAME_RE.match(name or ""):
        return None
    return MODELS_DIR / f"{name}.json"


def list_models() -> list[dict]:
    """Métadonnées de tous les modèles enregistrés (sans les vitesses)."""
    if not MODELS_DIR.is_dir():
        return []
    out: list[dict] = []
    for path in MODELS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        velocities = data.get("velocities") or []
        out.append(
            {
                "name": str(data.get("name") or path.stem),
                "samples": int(data.get("samples") or len(velocities)),
                "createdAt": str(data.get("createdAt") or ""),
                "error": data.get("error"),
                "signature": data.get("signature"),
            }
        )
    out.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
    return out


def read_model(name: str) -> dict | None:
    path = model_path(name)
    if not path or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_model(name: str, payload: dict) -> int:
    """Écrit le modèle de façon atomique et renvoie le nombre de points."""
    path = model_path(name)
    if path is None:
        raise ValueError("nom invalide")

    velocities = []
    for item in payload.get("velocities") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                velocities.append([round(float(item[0]), 3), round(float(item[1]), 3)])
            except (TypeError, ValueError):
                continue
        if len(velocities) >= MAX_MODEL_POINTS:
            break
    if not velocities:
        raise ValueError("aucune vitesse")

    record = {
        "version": int(payload.get("version") or 1),
        "name": name,
        "createdAt": payload.get("createdAt") or "",
        "samplingHz": payload.get("samplingHz"),
        "window": payload.get("window") or {},
        "samples": len(velocities),
        "error": payload.get("error"),
        "signature": payload.get("signature"),
        "velocities": velocities,
    }

    with _DATA_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
    return len(velocities)


def delete_model(name: str) -> bool:
    path = model_path(name)
    if not path or not path.is_file():
        return False
    with _DATA_LOCK:
        path.unlink(missing_ok=True)
    return True


class Handler(http.server.SimpleHTTPRequestHandler):
    """Handler statique sans cache + mini API (dataset et modèles)."""

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            length = 0
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return None
        return body if isinstance(body, dict) else None

    def _route(self) -> str:
        return urllib.parse.urlparse(self.path).path

    def _model_name(self) -> str:
        route = self._route()
        return urllib.parse.unquote(route[len("/api/models/") :])

    def do_GET(self) -> None:  # noqa: N802
        route = self._route()
        if route == "/api/dataset":
            data = load_dataset()
            self._send_json({"count": len(data), "samples": data[-TRAIN_LIMIT:]})
            return
        if route == "/api/models":
            self._send_json({"models": list_models()})
            return
        if route.startswith("/api/models/"):
            model = read_model(self._model_name())
            if model is None:
                self.send_error(404, "Modèle introuvable")
                return
            self._send_json(model)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        route = self._route()

        if route == "/api/dataset":
            body = self._read_json()
            if body is None:
                self.send_error(400, "JSON invalide")
                return
            samples = body.get("samples", [])
            total = append_dataset(samples if isinstance(samples, list) else [])
            self._send_json({"saved": len(samples), "total": total})
            return

        if route.startswith("/api/models/"):
            body = self._read_json()
            if body is None:
                self.send_error(400, "JSON invalide")
                return
            name = self._model_name()
            try:
                count = write_model(name, body)
            except ValueError as exc:
                self.send_error(400, f"Modèle refusé : {exc}")
                return
            self._send_json({"saved": True, "name": name, "samples": count}, status=201)
            return

        self.send_error(404, "Not Found")

    def do_DELETE(self) -> None:  # noqa: N802
        route = self._route()
        if route == "/api/dataset":
            clear_dataset()
            self._send_json({"total": 0})
            return
        if route.startswith("/api/models/"):
            name = self._model_name()
            if not delete_model(name):
                self.send_error(404, "Modèle introuvable")
                return
            self._send_json({"deleted": name})
            return
        self.send_error(404, "Not Found")

    def end_headers(self) -> None:  # noqa: D102
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # noqa: D102
        print(f"  · {self.address_string()} — {fmt % args}")


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def bind(host: str, port: int, handler) -> Server:
    """Ouvre le serveur, en glissant sur les ports suivants si besoin."""
    if port == 0:  # port libre choisi par l'OS
        return Server((host, 0), handler)

    for candidate in range(port, port + PORT_TRIES):
        try:
            return Server((host, candidate), handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            print(f"  · port {candidate} déjà utilisé, essai suivant…")

    raise SystemExit(
        f"Aucun port libre entre {port} et {port + PORT_TRIES - 1}. "
        "Relancez avec -p <port>."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="tline — interface web de vitesse du pointeur")
    parser.add_argument("-H", "--host", default="127.0.0.1", help="adresse d'écoute")
    parser.add_argument("-p", "--port", type=int, default=8000, help="port d'écoute (8000 par défaut)")
    parser.add_argument("--open", action="store_true", help="ouvre le navigateur au démarrage")
    args = parser.parse_args()

    if not STATIC_DIR.is_dir():
        raise SystemExit(f"Dossier introuvable : {STATIC_DIR}")

    handler = functools.partial(Handler, directory=str(STATIC_DIR))

    with bind(args.host, args.port, handler) as httpd:
        host, port = httpd.server_address[0], httpd.server_address[1]
        url = f"http://{host}:{port}/"
        print(f"tline → {url}   (Ctrl+C pour arrêter)")
        if args.open:
            threading.Timer(0.4, webbrowser.open, [url]).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\narrêt.")


if __name__ == "__main__":
    main()
