#!/usr/bin/env python3
"""
HuggingFace Collection Downloader CLI

A reliable tool to download all models from a HuggingFace collection using the
official huggingface_hub library. This version explicitly handles public collections
for both fetching and downloading to avoid authentication issues from bad local tokens.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from huggingface_hub import get_collection, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# --- Setup Logging ---
def setup_logging(verbose: bool = False):
    """Configures the logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=log_format, datefmt="%H:%M:%S")
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

# --- Main Class ---
class CollectionDownloader:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def extract_slug_from_url(self, collection_url: str) -> str:
        """Extracts the collection slug from a full URL."""
        if collection_url.startswith("http"):
            parts = collection_url.strip("/").split("/")
            if len(parts) >= 3 and parts[-3] == "collections":
                namespace = parts[-2]
                slug_id = parts[-1]
                slug = f"{namespace}/{slug_id}"
                self.logger.debug(f"Extracted slug '{slug}' from URL.")
                return slug
        self.logger.debug(f"Input '{collection_url}' is not a full URL. Assuming it's already a slug.")
        return collection_url

    def get_model_ids_from_collection(self, collection_slug: str, use_auth: bool) -> Optional[List[str]]:
        """Fetches a collection and returns a list of all model IDs within it."""
        self.logger.info(f"🔍 Fetching collection with slug: '{collection_slug}'...")
        auth_token = True if use_auth else False
        if not use_auth:
            self.logger.info("Attempting to access collection publicly. No token will be sent.")

        try:
            collection = get_collection(collection_slug, token=auth_token)
            self.logger.info(f"✅ Successfully fetched collection: '{collection.title}'")
            model_ids = [item.item_id for item in collection.items if item.item_type == "model"]
            if not model_ids:
                self.logger.warning("Collection was found, but it contains no models.")
                return []
            self.logger.info(f"Found {len(model_ids)} models in the collection.")
            return model_ids

        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                self.logger.error("❌ Authentication failed (401 Unauthorized).")
                self.logger.error("💡 If this is a private collection, please use the --use-auth-token flag and run 'huggingface-cli login'.")
            elif e.response.status_code == 404:
                self.logger.error(f"❌ Collection not found (404). Please check the URL or slug.")
            else:
                self.logger.error(f"❌ An HTTP error occurred: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ An unexpected error occurred while fetching the collection: {e}")
            return None

    def download_model(
        self, model_id: str, storage_dir: str, use_auth: bool, allow_patterns: Optional[List[str]] = None
    ) -> bool:
        """Downloads a single model repository."""
        self.logger.info(f"📥 Starting download for: {model_id}")

        # --- THE FINAL FIX ---
        # Apply the same token logic to snapshot_download.
        auth_token = True if use_auth else False

        try:
            snapshot_download(
                repo_id=model_id,
                token=auth_token, # This tells the function to ignore local tokens if False
                cache_dir=storage_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                user_agent=f"hf-collection-downloader/1.0",
            )
            self.logger.info(f"✅ Successfully downloaded: {model_id}")
            return True
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                 self.logger.error(f"❌ Authentication failed for model {model_id}. Your token may be invalid for this repository.")
            else:
                self.logger.error(f"❌ Failed to download {model_id} (HTTP Error): {e}")
        except Exception as e:
            self.logger.error(f"❌ An unexpected error occurred during download of {model_id}: {e}")
        return False

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Download all models from a HuggingFace collection.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("collection", help="The full URL or slug of the HuggingFace collection.")
    parser.add_argument("-d", "--download-dir", help="Directory to download models to.", default=None)
    parser.add_argument("-f", "--file-patterns", nargs="*", help="Optional file patterns to download.", default=None)
    parser.add_argument("--dry-run", action="store_true", help="List models without downloading.")
    parser.add_argument("--use-auth-token", action="store_true", help="Force the use of the machine's Hugging Face token (for private items).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (debug) logging.")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    downloader = CollectionDownloader(logger)

    collection_slug = downloader.extract_slug_from_url(args.collection)
    model_ids = downloader.get_model_ids_from_collection(collection_slug, use_auth=args.use_auth_token)

    if model_ids is None: sys.exit(1)
    if not model_ids: sys.exit(0)

    if args.dry_run:
        logger.info("--- 🔍 DRY RUN MODE ---")
        logger.info(f"The following {len(model_ids)} models would be downloaded:")
        for i, model_id in enumerate(model_ids, 1): print(f"  {i:2d}. {model_id}")
        sys.exit(0)

    storage_dir = Path(args.download_dir or collection_slug.replace("/", "_") + "_models")
    storage_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"🚀 Models will be downloaded to: {storage_dir.resolve()}")

    successful, failed = [], []
    for i, model_id in enumerate(model_ids, 1):
        logger.info("-" * 40 + f"\nProcessing model {i}/{len(model_ids)}: {model_id}")
        # Pass the authentication flag down to the download function
        if downloader.download_model(model_id, str(storage_dir), args.use_auth_token, args.file_patterns):
            successful.append(model_id)
        else:
            failed.append(model_id)

    logger.info("\n" + "=" * 20 + " SUMMARY " + "=" * 20)
    logger.info(f"✅ Successful downloads: {len(successful)}")
    logger.info(f"❌ Failed downloads:     {len(failed)}")
    if failed:
        logger.warning("Failed models: " + ", ".join(failed))
        sys.exit(1)
    logger.info("🎉 All models downloaded successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()
