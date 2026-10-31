import io
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from supabase import Client, create_client

from services.ai_services import analyze_accident_image

# Load backend/.env explicitly, before reading any variable
load_dotenv(Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
BUCKET = os.getenv("STORAGE_BUCKET", "alert-image")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or the Supabase service key in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

ALLOWED_SOURCES = {"app", "sms", "witness", "simulation"}

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/alerts/process")
async def process_alert(
    user_id: str = Form(...),
    lat: float = Form(...),
    lng: float = Form(...),
    impact_g: float = Form(...),
    speed_before_kmh: float = Form(...),
    source: str = Form("app"),
    image: UploadFile | None = File(None),
):
    vehicles_count = 0
    people_count = 0
    accident_score = 0.0
    image_url = None
    storage_path = None
    warnings: list[str] = []

    # 1) Photo: AI analysis and upload. A failure here must NOT kill the alert
    contents = await image.read() if image is not None else None

    if contents:
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            ai = await run_in_threadpool(analyze_accident_image, pil_img)
            vehicles_count = int(ai.get("vehicles_count", 0))
            people_count = int(ai.get("people_count", 0))
            accident_score = float(ai.get("accident_score", 0.0))
        except Exception as e:
            print(f"AI analysis failed (alert kept): {e}")
            warnings.append(f"ai_failed: {e}")

        try:
            storage_path = f"alerts/{uuid.uuid4()}.jpg"
            supabase.storage.from_(BUCKET).upload(
                path=storage_path,
                file=contents,
                file_options={"content-type": "image/jpeg"},
            )
            image_url = supabase.storage.from_(BUCKET).get_public_url(storage_path)
        except Exception as e:
            print(f"Image upload failed (alert kept): {e}")
            warnings.append(f"upload_failed: {e}")
            storage_path = None
            image_url = None

    # 2) Clean values for the database
    try:
        valid_user_id = str(uuid.UUID(user_id))
    except ValueError:
        valid_user_id = None  # 'anonymous' or invalid id

    src = source.strip().lower()
    if src not in ALLOWED_SOURCES:
        src = "app"

    # The AI supports the decision, it never confirms an accident by itself
    status = "unconfirmed"

    # 3) Insert the alert: this one must succeed
    try:
        res = supabase.table("alerts").insert({
            "user_id": valid_user_id,
            "lat": lat,
            "lng": lng,
            "impact_g": impact_g,
            "speed_before_kmh": speed_before_kmh,
            "source": src,
            "status": status,
            "vehicles_count": vehicles_count,
            "people_count": people_count,
            "accident_score": accident_score,
        }).execute()
        alert_id = res.data[0]["id"]
    except Exception as e:
        print(f"Alert insert error: {e}")
        raise HTTPException(status_code=500, detail=f"alert insert failed: {e}")

    # 4) Link the image to the alert (failure is not fatal)
    if storage_path:
        try:
            supabase.table("alert_images").insert({
                "alert_id": alert_id,
                "storage_path": storage_path,
                "image_url": image_url,
            }).execute()
        except Exception as e:
            print(f"alert_images insert failed: {e}")
            warnings.append(f"alert_images_failed: {e}")

    ai_data = {
        "accident_score": accident_score,
        "vehicles_count": vehicles_count,
        "people_count": people_count,
        "image_url": image_url,
        "status": status,
    }
    return {
        "status": "success",
        "alert_id": alert_id,
        "warnings": warnings,
        **ai_data,        # top level: what the Flutter AI page reads
        "ai_data": ai_data,  # nested copy for compatibility
    }