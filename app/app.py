import traceback
from types import TracebackType
import torch
from pathlib import Path
from absl import app, flags, logging
from lib.layers.convir_layers import build_net, ConvIR
import zipfile
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from io import BytesIO
import os
import kaggle

import window
import sys

PROJECT_ID = "first-project-389416"
CLIENT_ID = "1041884767277-03qbb9uing3bepgj3712827qlt22149d.apps.googleusercontent.com"
CLIENT_SECRET = "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"

# command line arguments:
FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    name="window",
    default=None,
    help="Whether to run the app in window mode (meaning the command line tool already generated a mode3)",
)


# scopes required per Google Drive API (read only)
def google_drive_download_and_extract_zip(file_id: str, folder_name: str) -> None:
    """Download a zip from google drive and extract it. Skips download if `folder_name` already exists"""
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    path = Path.cwd() / folder_name
    if path.exists():
        logging.info(
            f"folder {folder_name} already exists, skipping download of {file_id}"
        )
        return

    tokenPath = Path.cwd() / "token.json"
    creds = None
    # se ci sono credenziali valide, usale
    if tokenPath.exists():
        creds = Credentials.from_authorized_user_file(str(tokenPath), SCOPES)
    # altrimenti apri una sessione di login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logging.info()
            creds.refresh(Request())
        else:
            logging.info("Opening browser for Google login...")
            # https://google-auth-oauthlib.readthedocs.io/en/latest/reference/google_auth_oauthlib.flow.html
            client_config: dict[str, any] = {
                "installed": {
                    "client_id": CLIENT_ID,
                    "project_id": PROJECT_ID,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_secret": CLIENT_SECRET,
                    "redirect_uris": ["http://localhost:3000"],
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=3000)
        # salva le credenziali per uso futuro
        with open(tokenPath, "w") as token_file:
            token_file.write(creds.to_json())

    # costruisci l'API Google Drive con l'utente corrente
    try:
        service = build("drive", "v3", credentials=creds)

        # scaricati i metadati del richiesto (https://developers.google.com/drive/api/guides/manage-downloads#python)
        request = service.files().get_media(fileId=file_id)
        buffer = BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            logging.info(f"Download {int(status.progress() * 100)}.")
    except HttpError as err:
        logging.error("An error occurred", err)
        buffer = None

    # estrai lo zip
    if buffer is not None:
        buffer.seek(0)
        logging.info(f"extracting to '{folder_name}'")
        with zipfile.ZipFile(buffer, "r") as zip_ref:
            zip_ref.extractall(path)

        logging.info(f"Extraction complete. files available at '{folder_name}'")


def sayHello() -> None:
    logging.info(Path.cwd())
    logging.info(f"torch version: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info("CUDA is available")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logging.info(f"Current GPU: {torch.cuda.current_device()}")
        logging.info(
            f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    else:
        logging.info("CUDA is not available")


def kaggle_download_and_extract_zip(dataset_name: str, output_path: Path) -> None:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset_name, path=str(output_path), quiet=False, unzip=True
    )


def logging_exception_hook(exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: TracebackType | None) -> None:
    # alternative: logging.exception
    logging.error("Exeption %s [%s]\n\tTraceback:\n%s", exc_value, exc_type, traceback.format_exception(exc_value, tb))


def main(args: list[str]) -> None:
    sys.excepthook = logging_exception_hook
    if FLAGS.window:
        with window.Window("EDCNN", width=800, height=600) as w:
            w.run_render_loop()
    else:
        sayHello()
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        model: ConvIR = build_net().to(device)
        file_id = "1y_wQ5G5B65HS_mdIjxKYTcnRys_AGh5v"
        target_folder = "data"
        # google_drive_download_and_extract_zip(file_id, target_folder)
        kaggle_download_and_extract_zip(
            "rahulbhalley/gopro-deblur", Path.cwd() / target_folder
        )


if __name__ == "__main__":
    app.run(main)
