import os
import urllib.request
import zipfile
import sys

def download_file(url, target_path):
    if os.path.exists(target_path):
        print(f"[*] {target_path} already exists. Skipping download.")
        return target_path

    print(f"[*] Downloading {url} to {target_path}...")
    
    def reporthook(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            percent = int(downloaded * 100 / total_size)
            sys.stdout.write(f"\rDownloading: {percent}% ({downloaded / (1024*1024*1024):.2f} GB / {total_size / (1024*1024*1024):.2f} GB)")
        else:
            sys.stdout.write(f"\rDownloaded {(downloaded) / (1024 * 1024):.2f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, target_path, reporthook)
        print("\n[*] Download complete!")
    except Exception as e:
        print(f"\n[!] Error downloading {url}: {e}")
    return target_path

def extract_zip(zip_path, extract_to):
    print(f"[*] Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("[*] Extraction complete!")
    except Exception as e:
        print(f"[!] Error extracting {zip_path}: {e}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    coco_dir = os.path.join(base_dir, 'datasets', 'raw', 'coco')
    os.makedirs(coco_dir, exist_ok=True)

    # COCO dataset files required
    coco_files = {
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip"
    }

    # Step 2: Download and Extract COCO
    for filename, url in coco_files.items():
        target_path = os.path.join(coco_dir, filename)
        download_file(url, target_path)
        extract_zip(target_path, coco_dir)
        
    print("\n[+] COCO dataset download and extraction process finished.")

    # Step 5: Information about running OpenImages downloader
    openimages_classes_file = os.path.join(base_dir, 'scripts', 'netra_classes.txt')
    openimages_script = os.path.join(base_dir, 'scripts', 'downloader.py')
    openimages_out = os.path.join(base_dir, 'datasets', 'raw', 'openimages')

    print("\n--- Next Steps for OpenImages ---")
    print("To download the remaining specific classes via OpenImages, run the following command in your terminal:")
    print(f"python \"{openimages_script}\" \"{openimages_classes_file}\" --download_folder=\"{openimages_out}\" --limit=2000")
    print("Note: You must 'pip install -r requirements.txt' first to resolve dependencies like boto3 and pandas.")

if __name__ == "__main__":
    main()
