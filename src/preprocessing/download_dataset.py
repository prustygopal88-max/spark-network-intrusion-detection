import os
import sys
import time
import requests

DATA_URL = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

def download_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get total size from HEAD request
    head_resp = requests.head(DATA_URL, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True, timeout=30)
    total_size = int(head_resp.headers.get("content-length", 0))
    print(f"Target dataset: Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    print(f"Expected size: {total_size / (1024 * 1024):.2f} MB ({total_size} bytes)")

    max_retries = 10
    retry_count = 0

    while retry_count < max_retries:
        downloaded = os.path.getsize(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else 0
        if total_size > 0 and downloaded >= total_size:
            print(f"Download complete: {OUTPUT_FILE} ({downloaded / (1024 * 1024):.2f} MB)")
            return OUTPUT_FILE

        headers = {"User-Agent": "Mozilla/5.0"}
        mode = "wb"
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"
            mode = "ab"
            print(f"Resuming download from byte {downloaded} ({downloaded / (1024 * 1024):.2f} MB)...")
        else:
            print("Starting fresh download...")

        try:
            with requests.get(DATA_URL, headers=headers, stream=True, timeout=30) as r:
                if r.status_code not in (200, 206):
                    print(f"Unexpected status code {r.status_code}, retrying...")
                    time.sleep(2)
                    retry_count += 1
                    continue
                
                # If server doesn't support Range, start fresh
                if downloaded > 0 and r.status_code == 200:
                    mode = "wb"
                    downloaded = 0

                chunk_size = 1024 * 512
                start_time = time.time()
                last_print = downloaded

                with open(OUTPUT_FILE, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded - last_print >= 10 * 1024 * 1024:
                                pct = (downloaded / total_size) * 100 if total_size > 0 else 0
                                print(f"Progress: {downloaded / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB ({pct:.1f}%)")
                                last_print = downloaded

        except (requests.RequestException, Exception) as e:
            print(f"Connection issue encountered ({e}). Retrying in 2 seconds...")
            retry_count += 1
            time.sleep(2)

    final_size = os.path.getsize(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else 0
    print(f"Final downloaded size: {final_size / (1024 * 1024):.2f} MB")
    return OUTPUT_FILE

if __name__ == "__main__":
    download_dataset()
