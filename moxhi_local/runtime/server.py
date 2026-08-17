from __future__ import annotations
import asyncio
import os
from pathlib import Path
from typing import Any
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Paths
# ============================================================

RUNTIME_DIR = Path(__file__).resolve().parent
MOXHI_ROOT = RUNTIME_DIR.parent

INDEX_FILE = MOXHI_ROOT / "index.html"

HOST = os.getenv(
    "MOXHI_HOST",
    "127.0.0.1",
)

PORT = int(
    os.getenv(
        "MOXHI_PORT",
        "8090",
    )
)

MOXHI_URL = f"http://{HOST}:{PORT}/"


# ============================================================
# Moxhi Runtime
# ============================================================
class TranslateRequest(BaseModel):
    text: str

class MoxhiRuntime:

    def __init__(self) -> None:
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

        self.ready = False
        self.starting = False
        self.webgpu = False
        self.error: str | None = None

        self._lock = asyncio.Lock()

    async def inspect_engine(self):
        if self.page is None:
            raise RuntimeError("Moxhi page is not running")

        modules = [
            "/assets/generate-jUtMzJoV.js",
            "/assets/weights-Co7eGWJH.js",
            "/assets/tokenizer-Duj9RLAT.js",
            "/assets/engine-CHD2bVyS.js",
            "/assets/models-CvHl_Lx7.js",
        ]

        return await self.page.evaluate(
            """
            async (modules) => {

                const result = {};

                for (const path of modules) {

                    try {

                        const mod = await import(path);

                        const exports = {};

                        for (const key of Object.keys(mod)) {

                            const value = mod[key];

                            exports[key] = {
                                type: typeof value,
                                source:
                                    typeof value === "function"
                                        ? value.toString()
                                        : null
                            };
                        }

                        result[path] = {
                            ok: true,
                            exports
                        };

                    } catch (error) {

                        result[path] = {
                            ok: false,
                            error: String(error)
                        };
                    }
                }

                return result;
            }
            """,
            modules,
        )

    async def start(self) -> None:

        async with self._lock:

            if self.page is not None:
                return

            if self.starting:
                return

            self.starting = True
            self.error = None

        try:

            print()
            print("=" * 72)
            print(" Moxhi Runtime")
            print("=" * 72)
            print(f"Root    : {MOXHI_ROOT}")
            print(f"URL     : {MOXHI_URL}")
            print("Starting Playwright...")

            self.playwright = await async_playwright().start()

            browser_args = [
                "--enable-unsafe-webgpu",
                "--enable-features=Vulkan",
            ]

            browser_name = os.getenv(
                "MOXHI_BROWSER",
                "chrome",
            ).lower()

            # ------------------------------------------------
            # Browser
            # ------------------------------------------------

            browser_name = os.getenv(
                "MOXHI_BROWSER",
                "chrome",
            ).lower()

            headless = os.getenv(
                "MOXHI_HEADLESS",
                "true",
            ).lower() in {
                           "1",
                           "true",
                           "yes",
                           "on",
                       }

            print(f"Browser : {browser_name}")
            print(f"Headless: {headless}")

            if browser_name == "chromium":

                print("Browser : Playwright Chromium")

                self.browser = await self.playwright.chromium.launch(
                    headless=headless,
                    args=browser_args,
                )

            else:

                print("Browser : Google Chrome")

                self.browser = await self.playwright.chromium.launch(
                    channel="chrome",
                    headless=headless,
                    args=browser_args,
                )

            self.context = await self.browser.new_context()

            self.page = await self.context.new_page()

            # ------------------------------------------------
            # Browser logging
            # ------------------------------------------------

            self.page.on(
                "console",
                lambda msg: print(
                    f"[BROWSER:{msg.type}] {msg.text}"
                ),
            )

            self.page.on(
                "pageerror",
                lambda exc: print(
                    f"[BROWSER ERROR] {exc}"
                ),
            )

            self.page.on(
                "requestfailed",
                lambda request: print(
                    f"[REQUEST FAILED] "
                    f"{request.url} "
                    f"{request.failure}"
                ),
            )

            # ------------------------------------------------
            # Load local Moxhi
            # ------------------------------------------------

            print(f"Loading  : {MOXHI_URL}")

            await self.page.goto(
                MOXHI_URL,
                wait_until="domcontentloaded",
                timeout=120_000,
            )

            print("Page loaded.")

            # ------------------------------------------------
            # WebGPU
            # ------------------------------------------------

            self.webgpu = await self.page.evaluate(
                """
                () => {
                    return !!(
                        navigator.gpu &&
                        typeof navigator.gpu.requestAdapter === "function"
                    );
                }
                """
            )

            print(f"WebGPU   : {self.webgpu}")

            if not self.webgpu:
                raise RuntimeError(
                    "WebGPU is not available in Chromium."
                )

            # ------------------------------------------------
            # Temporary wait.
            #
            # Phase 1 only needs the Moxhi page to be alive.
            # Later we will replace this with the actual
            # Moxhi engine-ready signal.
            # ------------------------------------------------

            await self.page.wait_for_timeout(3000)

            self.ready = True

            print("-" * 72)
            print(" Moxhi Runtime READY")
            print("-" * 72)
            print()

        except Exception as exc:

            self.ready = False
            self.error = str(exc)

            print()
            print("=" * 72)
            print(" Moxhi Runtime FAILED")
            print("=" * 72)
            print(exc)
            print()

        finally:

            self.starting = False

    async def stop(self) -> None:

        print("Stopping Moxhi Runtime...")

        self.ready = False

        if self.context is not None:

            try:
                await self.context.close()
            except Exception:
                pass

            self.context = None

        if self.browser is not None:

            try:
                await self.browser.close()
            except Exception:
                pass

            self.browser = None

        if self.playwright is not None:

            try:
                await self.playwright.stop()
            except Exception:
                pass

            self.playwright = None

        self.page = None

    def status(self) -> dict[str, Any]:

        return {
            "ready": self.ready,
            "starting": self.starting,
            "webgpu": self.webgpu,
            "browser": os.getenv(
                "MOXHI_BROWSER",
                "chrome",
            ),
            "root": str(MOXHI_ROOT),
            "error": self.error,
        }


runtime = MoxhiRuntime()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Moxhi Translation Runtime",
    version="0.1.0",
)


# ============================================================
# Startup / Shutdown
# ============================================================

@app.on_event("startup")
async def startup_event():

    # IMPORTANT:
    #
    # Do NOT await runtime.start() here.
    #
    # Uvicorn must first start accepting HTTP requests
    # because Chromium itself will request localhost:8090.
    #
    asyncio.create_task(
        runtime.start()
    )


@app.on_event("shutdown")
async def shutdown_event():

    await runtime.stop()


# ============================================================
# Health
# ============================================================

@app.get("/health")
async def health():

    return runtime.status()


@app.get("/runtime")
async def runtime_status():

    return runtime.status()


# ============================================================
# Moxhi UI
# ============================================================

@app.get("/")
async def index():

    return FileResponse(
        INDEX_FILE
    )


# ============================================================
# Static resources
# ============================================================

app.mount(
    "/assets",
    StaticFiles(
        directory=MOXHI_ROOT / "assets"
    ),
    name="assets",
)

app.mount(
    "/model",
    StaticFiles(
        directory=MOXHI_ROOT / "model"
    ),
    name="model",
)

app.mount(
    "/weights",
    StaticFiles(
        directory=MOXHI_ROOT / "weights"
    ),
    name="weights",
)

app.mount(
    "/model-d8",
    StaticFiles(
        directory=MOXHI_ROOT / "model-d8"
    ),
    name="model-d8",
)

app.mount(
    "/cdn-cgi",
    StaticFiles(
        directory=MOXHI_ROOT / "cdn-cgi"
    ),
    name="cdn-cgi",
)

@app.get("/debug/engine")
async def debug_engine():

    if runtime.page is None:
        return {
            "ok": False,
            "error": "Moxhi runtime is not started"
        }

    try:

        result = await runtime.inspect_engine()

        return {
            "ok": True,
            "modules": result
        }

    except Exception as exc:

        return {
            "ok": False,
            "error": str(exc)
        }

@app.get("/debug/window")
async def debug_window():

    if runtime.page is None:
        return {
            "ok": False,
            "error": "Moxhi runtime is not running",
        }

    return await runtime.page.evaluate(
        """
        () => {

            const result = {};

            for (const key of Object.keys(window)) {

                const lower = key.toLowerCase();

                if (
                    lower.includes("engine") ||
                    lower.includes("model") ||
                    lower.includes("translate") ||
                    lower.includes("tokenizer") ||
                    lower.includes("weight") ||
                    lower.includes("moxhi")
                ) {

                    let value;

                    try {
                        value = window[key];
                    } catch {
                        value = null;
                    }

                    result[key] = {
                        type: typeof value,
                        constructor:
                            value?.constructor?.name ?? null,
                        keys:
                            value &&
                            typeof value === "object"
                                ? Object.keys(value).slice(0, 100)
                                : null
                    };
                }
            }

            return result;
        }
        """
    )
@app.get("/debug/moxhi-api")
async def debug_moxhi_api():

    if runtime.page is None:
        return {
            "ok": False,
            "error": "Moxhi page is not running",
        }

    return await runtime.page.evaluate("""
        () => {

            const m = window.__moxhi;

            if (!m) {
                return {
                    ok: false,
                    error: "window.__moxhi not found"
                };
            }

            const inspect = (value) => {

                const result = {
                    type: typeof value,
                    constructor: value?.constructor?.name ?? null,
                };

                if (typeof value === "function") {
                    result.source = value.toString();
                }

                if (
                    value &&
                    typeof value === "object"
                ) {
                    result.keys = Object.keys(value);
                }

                return result;
            };

            return {
                ok: true,

                moxhi: inspect(m),

                engine: inspect(m.engine),

                build: inspect(m.build),

                translate: inspect(m.translate),

                engineKeys:
                    m.engine &&
                    typeof m.engine === "object"
                        ? Object.keys(m.engine)
                        : [],

                buildKeys:
                    m.build &&
                    typeof m.build === "object"
                        ? Object.keys(m.build)
                        : []
            };
        }
    """)

@app.post("/translate")
async def translate(request: TranslateRequest):

    if runtime.page is None:
        raise HTTPException(
            status_code=503,
            detail="Moxhi runtime is not running"
        )

    if not request.text.strip():
        return {
            "ok": True,
            "text": "",
            "stats": None,
        }

    async with runtime._lock:

        result = await runtime.page.evaluate(
            """
            async (text) => {

                if (!window.__moxhi) {
                    throw new Error("__moxhi is not initialized");
                }

                if (!window.__moxhi.engine) {
                    throw new Error(
                        "Moxhi engine is not initialized"
                    );
                }

                if (window.__moxhi.engine.isLost?.()) {
                    throw new Error(
                        "Moxhi WebGPU device is lost"
                    );
                }

                const result =
                    await window.__moxhi.engine.translateText(text);

                return {
                    text: result.text,
                    rowTexts: result.rowTexts,
                    cancelled: result.cancelled,
                    stats: result.stats
                };
            }
            """,
            request.text,
        )

        return {
            "ok": True,
            **result,
        }
# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=False,
    )