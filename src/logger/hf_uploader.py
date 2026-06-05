# HuggingFace Hub checkpoint uploader.
# Decoupled from trainer: trainer holds an optional reference and calls
# `upload(path)` after each torch.save. Repo is created lazily at __init__
# under the name `<username>/<run_name>_<timestamp>` (prefix configurable).
from datetime import datetime
from pathlib import Path

from src.logger.utils import get_secret


class HFCheckpointUploader:
    """Create a HF model repo at init, upload checkpoint files on demand."""

    def __init__(
        self,
        username,
        run_name,
        private=True,
        repo_prefix="",
        timestamp_fmt="%Y%m%d_%H%M%S",
        logger=None,
    ):
        from huggingface_hub import HfApi

        # Validate HF auth info
        self.logger = logger
        self.token = get_secret("HF_WRITE_TOKEN")
        if not self.token:
            raise RuntimeError(
                "HFCheckpointUploader: no token. "
                "Pass `token` or set HF_WRITE_TOKEN (env / Kaggle secret / .env)."
            )
        if not username:
            raise RuntimeError("HFCheckpointUploader: `username` is required")
        if not run_name:
            raise RuntimeError("HFCheckpointUploader: `run_name` is required")

        ts = datetime.now().strftime(timestamp_fmt)
        self.repo_id = f"{username}/{repo_prefix}{run_name}_{ts}"
        self.private = private
        self.api = HfApi(token=self.token)
        self.api.create_repo(
            repo_id=self.repo_id,
            private=private,
            exist_ok=True,
            repo_type="model",
        )
        self._log(f"HF repo ready: https://huggingface.co/{self.repo_id}")

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def upload(self, local_path, path_in_repo=None, commit_message=None):
        local_path = Path(local_path)
        if not local_path.exists():
            self._log(f"HF upload skipped (missing): {local_path}")
            return
        path_in_repo = path_in_repo or local_path.name
        commit_message = commit_message or f"upload {path_in_repo}"
        try:
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                commit_message=commit_message,
            )
            self._log(f"HF uploaded: {path_in_repo}")
        except Exception as exc:
            self._log(f"HF upload failed for {local_path}: {exc}")
