from pathlib import Path
from urllib.request import urlretrieve
import zipfile

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ZIP_PATH = DATA_DIR / "student.zip"

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

DATA_DIR.mkdir(exist_ok=True)

print("Downloading dataset...")

urlretrieve(URL, ZIP_PATH)

print("Extracting...")

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(DATA_DIR)

ZIP_PATH.unlink()

print("Dataset downloaded successfully.")
