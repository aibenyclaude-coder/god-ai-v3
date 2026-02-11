"""
God AI v3.0 - Google Drive アップロードモジュール

使い方:
  from gdrive import upload_file, upload_text, list_files

セットアップ:
  1. Google Cloud Console で Drive API を有効化
  2. OAuth 2.0 クライアントID（デスクトップアプリ）を作成
  3. client_secret.json を core/ に配置
  4. 初回実行時にブラウザ認証 → token.json が自動生成
"""

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("god.gdrive")

CORE_DIR = Path(__file__).resolve().parent
CLIENT_SECRET_PATH = CORE_DIR / "client_secret.json"
TOKEN_PATH = CORE_DIR / "drive_token.json"
FOLDER_ID_PATH = CORE_DIR / "drive_folder_id.txt"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# God AI専用フォルダ名
GOD_AI_FOLDER_NAME = "God AI"


def _get_credentials():
    """Google Drive API の認証情報を取得。
    token.json があればそれを使い、なければ client_secret.json からOAuth認証。
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
    except ImportError:
        log.error("google-auth パッケージがインストールされていません")
        log.error("pip3 install google-api-python-client google-auth-oauthlib")
        return None

    creds = None

    # 既存トークンがあればロード
    if TOKEN_PATH.exists():
        try:
            token_data = json.loads(TOKEN_PATH.read_text())
            # Benyのスクリプト形式に対応
            if "token" in token_data and "refresh_token" in token_data:
                creds = Credentials(
                    token=token_data["token"],
                    refresh_token=token_data["refresh_token"],
                    client_id=token_data["client_id"],
                    client_secret=token_data["client_secret"],
                    token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
                    scopes=SCOPES,
                )
            else:
                creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        except Exception as e:
            log.warning(f"drive_token.json の読み込みに失敗: {e}")

    # トークンが無効 or 期限切れ → リフレッシュ or 再認証
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            log.info("Google Drive トークンをリフレッシュしました")
        except Exception as e:
            log.warning(f"トークンリフレッシュ失敗: {e}")
            creds = None

    if not creds or not creds.valid:
        if not CLIENT_SECRET_PATH.exists():
            log.error(f"client_secret.json が見つかりません: {CLIENT_SECRET_PATH}")
            log.error("Google Cloud Console で OAuth 2.0 クライアントIDを作成してください")
            return None
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRET_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)
            log.info("Google Drive 認証成功（ブラウザ認証完了）")
        except Exception as e:
            log.error(f"Google Drive 認証失敗: {e}")
            return None

    # トークン保存（次回から再認証不要）
    try:
        TOKEN_PATH.write_text(creds.to_json())
        log.info(f"トークン保存: {TOKEN_PATH}")
    except Exception as e:
        log.warning(f"トークン保存失敗: {e}")

    return creds


def _get_service():
    """Google Drive API サービスオブジェクトを取得"""
    try:
        from googleapiclient.discovery import build
    except ImportError:
        log.error("googleapiclient がインストールされていません")
        return None

    creds = _get_credentials()
    if not creds:
        return None

    try:
        service = build("drive", "v3", credentials=creds)
        return service
    except Exception as e:
        log.error(f"Drive API サービス構築失敗: {e}")
        return None


def _get_or_create_folder(service, folder_name: str = GOD_AI_FOLDER_NAME) -> Optional[str]:
    """God AI専用フォルダを取得。なければ作成。"""
    # 保存済みフォルダIDがあればそれを使う
    if FOLDER_ID_PATH.exists():
        try:
            folder_id = FOLDER_ID_PATH.read_text().strip()
            if folder_id:
                log.info(f"保存済みフォルダID使用: {folder_id}")
                return folder_id
        except Exception:
            pass

    try:
        # フォルダ検索
        query = (
            f"name='{folder_name}' and "
            f"mimeType='application/vnd.google-apps.folder' and "
            f"trashed=false"
        )
        results = service.files().list(
            q=query, spaces="drive", fields="files(id, name)"
        ).execute()
        files = results.get("files", [])

        if files:
            folder_id = files[0]["id"]
            log.info(f"既存フォルダ使用: {folder_name} (ID: {folder_id})")
            return folder_id

        # フォルダ作成
        file_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = service.files().create(
            body=file_metadata, fields="id"
        ).execute()
        folder_id = folder.get("id")
        log.info(f"フォルダ作成: {folder_name} (ID: {folder_id})")
        return folder_id

    except Exception as e:
        log.error(f"フォルダ操作失敗: {e}")
        return None


def upload_file(
    file_path: str,
    folder_name: str = GOD_AI_FOLDER_NAME,
    mime_type: Optional[str] = None,
) -> Optional[dict]:
    """ファイルをGoogle Driveにアップロード。

    Args:
        file_path: アップロードするファイルのパス
        folder_name: Drive上のフォルダ名
        mime_type: MIMEタイプ（Noneなら自動検出）

    Returns:
        {"id": "...", "name": "...", "webViewLink": "..."} or None
    """
    from googleapiclient.http import MediaFileUpload

    path = Path(file_path)
    if not path.exists():
        log.error(f"ファイルが見つかりません: {file_path}")
        return None

    service = _get_service()
    if not service:
        return None

    folder_id = _get_or_create_folder(service, folder_name)

    # MIME自動検出
    if not mime_type:
        suffix = path.suffix.lower()
        mime_map = {
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".json": "application/json",
            ".py": "text/x-python",
            ".csv": "text/csv",
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        mime_type = mime_map.get(suffix, "application/octet-stream")

    try:
        file_metadata = {"name": path.name}
        if folder_id:
            file_metadata["parents"] = [folder_id]

        media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)

        # 既存ファイルチェック（同名なら上書き）
        existing = _find_file(service, path.name, folder_id)
        if existing:
            # 更新
            file = service.files().update(
                fileId=existing["id"],
                body={"name": path.name},
                media_body=media,
                fields="id, name, webViewLink, modifiedTime",
            ).execute()
            log.info(f"Drive更新: {file.get('name')} (ID: {file.get('id')})")
        else:
            # 新規作成
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, name, webViewLink, modifiedTime",
            ).execute()
            log.info(f"Driveアップロード: {file.get('name')} (ID: {file.get('id')})")

        return file

    except Exception as e:
        log.error(f"アップロード失敗: {e}")
        return None


def upload_text(
    content: str,
    filename: str,
    folder_name: str = GOD_AI_FOLDER_NAME,
    mime_type: str = "text/plain",
) -> Optional[dict]:
    """テキスト内容を直接Google Driveにアップロード。

    Args:
        content: アップロードするテキスト内容
        filename: Drive上のファイル名
        folder_name: Drive上のフォルダ名
        mime_type: MIMEタイプ

    Returns:
        {"id": "...", "name": "...", "webViewLink": "..."} or None
    """
    from googleapiclient.http import MediaInMemoryUpload

    service = _get_service()
    if not service:
        return None

    folder_id = _get_or_create_folder(service, folder_name)

    try:
        file_metadata = {"name": filename}
        if folder_id:
            file_metadata["parents"] = [folder_id]

        media = MediaInMemoryUpload(
            content.encode("utf-8"), mimetype=mime_type, resumable=True
        )

        existing = _find_file(service, filename, folder_id)
        if existing:
            file = service.files().update(
                fileId=existing["id"],
                body={"name": filename},
                media_body=media,
                fields="id, name, webViewLink, modifiedTime",
            ).execute()
            log.info(f"Drive更新: {file.get('name')}")
        else:
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, name, webViewLink, modifiedTime",
            ).execute()
            log.info(f"Driveアップロード: {file.get('name')}")

        return file

    except Exception as e:
        log.error(f"テキストアップロード失敗: {e}")
        return None


def _find_file(service, name: str, folder_id: Optional[str] = None) -> Optional[dict]:
    """Drive上のファイルを名前で検索"""
    try:
        query = f"name='{name}' and trashed=false"
        if folder_id:
            query += f" and '{folder_id}' in parents"

        results = service.files().list(
            q=query, spaces="drive", fields="files(id, name)"
        ).execute()
        files = results.get("files", [])
        return files[0] if files else None
    except Exception:
        return None


def list_files(folder_name: str = GOD_AI_FOLDER_NAME) -> list:
    """God AIフォルダ内のファイル一覧を取得"""
    service = _get_service()
    if not service:
        return []

    folder_id = _get_or_create_folder(service, folder_name)
    if not folder_id:
        return []

    try:
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name, mimeType, modifiedTime, size)",
            orderBy="modifiedTime desc",
        ).execute()
        return results.get("files", [])
    except Exception as e:
        log.error(f"ファイル一覧取得失敗: {e}")
        return []


def is_configured() -> bool:
    """Google Drive APIが利用可能かチェック"""
    return TOKEN_PATH.exists() or CLIENT_SECRET_PATH.exists()


# ─── CLI テスト ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not CLIENT_SECRET_PATH.exists():
        print(f"❌ client_secret.json が見つかりません: {CLIENT_SECRET_PATH}")
        print("Google Cloud Console で OAuth 2.0 クライアントID を作成してください")
        exit(1)

    print("🔑 Google Drive 認証テスト...")
    service = _get_service()
    if service:
        print("✅ Drive API 接続成功")

        # テストアップロード
        result = upload_text("Hello from God AI v3.0!", "test_upload.txt")
        if result:
            print(f"✅ テストアップロード成功: {result.get('name')} (ID: {result.get('id')})")
        else:
            print("❌ テストアップロード失敗")

        # ファイル一覧
        files = list_files()
        print(f"\n📁 God AI v3 フォルダ内: {len(files)} ファイル")
        for f in files:
            print(f"  - {f['name']} ({f.get('modifiedTime', 'N/A')})")
    else:
        print("❌ Drive API 接続失敗")
