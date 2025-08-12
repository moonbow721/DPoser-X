import argparse
import os
from huggingface_hub import hf_hub_download, HfFolder
from typing import List

# --- Configuration ---
REPO_ID = "Moon-bow/DPoser-X"
LOCAL_DIR = "pretrained_models"

# Mapping from model type to its file path on the Hugging Face Hub
MODEL_FILES = {
    "body": "body/BaseMLP/last.ckpt",
    "hand": "hand/BaseMLP/last.ckpt",
    "face": "face/BaseMLP/last.ckpt",
    "face_shape": "face_shape/BaseMLP/last.ckpt",
    "wholebody": "wholebody/mixed/last.ckpt",
}

def download_models(model_types: List[str]):
    """
    Downloads the specified models from the Hugging Face Hub.
    
    Args:
        model_types (List[str]): A list of model types to download.
    """
    # Ensure the target directory exists
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"Models will be saved to: ./{LOCAL_DIR}")

    # Use a set to handle unique models, especially for the 'face' case
    models_to_download = set(model_types)

    if "all" in models_to_download:
        # If 'all' is specified, the set becomes all available models
        models_to_download = set(MODEL_FILES.keys())
    elif "face" in models_to_download:
        # If 'face' is specified, ensure 'face_shape' is also included
        models_to_download.add("face_shape")
    print(f"models_to_download: {models_to_download}")
    
    # Get a token if available, otherwise downloads will be anonymous
    token = HfFolder.get_token()

    for model_type in sorted(list(models_to_download)):
        if model_type not in MODEL_FILES:
            print(f"⚠️  Warning: Unknown model type '{model_type}' skipped.")
            continue

        file_path = MODEL_FILES[model_type]
        print(f"\nDownloading '{model_type}' model...")
        
        try:
            # Download the file
            hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                local_dir=LOCAL_DIR,
                token=token,
            )
            print(f"✅ Successfully downloaded to ./{LOCAL_DIR}/{file_path}")
        except Exception as e:
            print(f"❌ Failed to download '{model_type}'. Please check your connection and the repository path.")
            print(f"   Error: {e}")

    print("\n✨ All requested downloads are complete.")

def main():
    """
    Main function to parse arguments and initiate downloads.
    """
    parser = argparse.ArgumentParser(
        description="Download pre-trained models for DPoser-X from the Hugging Face Hub.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "models",
        nargs='*',
        default=["all"],
        choices=["body", "hand", "face", "wholebody", "all"],
        help=(
            "Specify which model(s) to download.\n"
            "Options: 'body', 'hand', 'face', 'wholebody'.\n"
            "Use 'all' to download everything (this is the default).\n"
            "Note: Specifying 'face' will automatically download both 'face' and 'face_shape' models."
        )
    )
    
    args = parser.parse_args()
    download_models(args.models)

if __name__ == "__main__":
    main()