from pathlib import Path
from typing import NamedTuple
import sys
import utils
import zipfile
from io import BytesIO
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from absl import logging
from utils import AtomicBool
import importlib

from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.api_client import ApiClient
import os
from datetime import datetime
import time


class CustomTqdm(tqdm):
    def update(self, n=1):
        super().update(n)
        progress = self.n / self.total if self.total else 0
        logging.info("Progress: %f / 1.000000", progress)


class CustomKaggleApi(KaggleApi):
    def __init__(self):
        super().__init__(ApiClient())
        self.authenticate()

    def download_file(
        self, response, outfile, quiet=True, resume=False, chunk_size=1048576
    ):
        outpath = os.path.dirname(outfile)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        size = int(response.headers["Content-Length"])
        size_read = 0
        open_mode = "wb"
        remote_date = datetime.strptime(
            response.headers["Last-Modified"], "%a, %d %b %Y %H:%M:%S %Z"
        )
        remote_date_timestamp = time.mktime(remote_date.timetuple())

        if not quiet:
            print("Downloading " + os.path.basename(outfile) + " to " + outpath)

        file_exists = os.path.isfile(outfile)
        resumable = (
            "Accept-Ranges" in response.headers
            and response.headers["Accept-Ranges"] == "bytes"
        )

        if resume and resumable and file_exists:
            size_read = os.path.getsize(outfile)
            open_mode = "ab"

            if not quiet:
                print(
                    "... resuming from %d bytes (%d bytes left) ..."
                    % (
                        size_read,
                        size - size_read,
                    )
                )

            request_history = response.retries.history[0]
            response = self.api_client.request(
                request_history.method,
                request_history.redirect_location,
                headers={"Range": "bytes=%d-" % (size_read,)},
                _preload_content=False,
            )

        with CustomTqdm(
            total=size,
            initial=size_read,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=quiet,
        ) as pbar:
            with open(outfile, open_mode) as out:
                while True:
                    data = response.read(chunk_size)
                    if not data:
                        break
                    out.write(data)
                    os.utime(
                        outfile,
                        times=(remote_date_timestamp - 1, remote_date_timestamp - 1),
                    )
                    size_read = min(size, size_read + chunk_size)
                    pbar.update(len(data))
            if not quiet:
                print("\n", end="")

            os.utime(outfile, times=(remote_date_timestamp, remote_date_timestamp))


def _google_drive_download_and_extract_zip(file_id: str, zip_path: Path) -> None:
    """Download a zip from google drive and extract it. Skips download if `folder_name` already exists"""
    PROJECT_ID = "first-project-389416"
    CLIENT_ID = (
        "1041884767277-03qbb9uing3bepgj3712827qlt22149d.apps.googleusercontent.com"
    )
    CLIENT_SECRET = "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"
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    if not zip_path.name.endswith(".zip"):
        raise ValueError("ZIP File Path should end with a .zip file name")

    target_path = zip_path.parent / zip_path.name.removesuffix(".zip")
    # TODO check that the path has the correct file structure?
    if target_path.exists() and target_path.is_dir():
        logging.info(
            "Path %s already exists, skipping dataset download", str(target_path)
        )
        return

    buffer: BytesIO | Path = None
    if not zip_path.exists():
        tokenPath = Path.cwd() / "token.json"
        creds = None
        # se ci sono credenziali valide, usale
        if tokenPath.exists():
            creds = Credentials.from_authorized_user_file(str(tokenPath), SCOPES)
        # altrimenti apri una sessione di login
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
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
                flow: InstalledAppFlow = InstalledAppFlow.from_client_config(
                    client_config, SCOPES
                )
                creds = flow.run_local_server(port=3000, timeout_seconds=40)
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
    else:
        logging.info("ZIP File %s already exists, extracting it", str(zip_path))
        buffer = zip_path

    # estrai lo zip
    if buffer is not None:
        buffer.seek(0)
        logging.info("extracting to '%s'", str(target_path))
        with zipfile.ZipFile(buffer, "r") as zip_ref:
            zip_ref.extractall(target_path)

        logging.info("Extraction complete. files available at '%s'", str(target_path))


def _kaggle_datasets_make_available(dataset_str: str, datasets_path: Path) -> None:
    CustomKaggleApi().dataset_download_files(
        dataset_str, path=datasets_path, quiet=False, unzip=True
    )


def datasets_make_available_gopro_deblur(datasets_path: Path, mode: int = 1) -> None:
    """mode: 0 for kaggle, 1 for google drive, anything else for mega (still todo)"""
    if mode == 0:
        if (datasets_path / "gopro_deblur").exists():
            logging.info("Kaggle gopro deblur dataset already downloaded")
        else:
            _kaggle_datasets_make_available("rahulbhalley/gopro-deblur", datasets_path)
            # TODO post processing in /train e /test
    elif mode == 1:
        _google_drive_download_and_extract_zip(
            "1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK",
            datasets_path / "GOPRO_Large.zip",
        )
    else:
        raise NotImplementedError("TODO")


def datasets_make_available_blur_dataset(datasets_path: Path) -> None:
    _kaggle_datasets_make_available(
        "kwentar/blur-dataset",
        datasets_path / "blur_dataset",
    )


def test_work(pos: str, *, key: str):
    import time

    logging.info("Simulating work, arg: %s", pos)
    time.sleep(2)
    logging.info("Work simulated, karg: %s", key)
