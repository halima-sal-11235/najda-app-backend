import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

_client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)


def update_alert(alert_id: str, data: dict):
    _client.table("alerts").update(data).eq("id", alert_id).execute()


def save_image(alert_id: str, filename: str, content: bytes) -> str:
    path = f"{alert_id}/{filename}"
    _client.storage.from_("alert-image").upload(
        path, content, {"content-type": "image/jpeg"}
    )
    _client.table("alert_images").insert(
        {"alert_id": alert_id, "storage_path": path}
    ).execute()
    return path