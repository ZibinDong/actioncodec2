"""Audit and upload the merged tokenizer catalog and six official AC2 policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from actioncodec2.publishing import audit_public_files


def buffer_commit_metadata(request) -> None:
    """Normalize mirror URLs and give commit metadata an explicit length."""
    if request.url.host in {"hf-mirror.org", "huggingface.co"}:
        request.url = request.url.copy_with(host="hf-mirror.com")
        request.headers["Host"] = "hf-mirror.com"
    if request.method == "POST" and "/commit/" in request.url.path:
        content = request.read()
        request.headers.pop("transfer-encoding", None)
        request.headers["Content-Length"] = str(len(content))


def upload_client():
    import httpx
    from huggingface_hub.utils._http import default_client_factory

    client = default_client_factory()
    client.timeout = httpx.Timeout(120, connect=20)
    client.event_hooks["request"].append(buffer_commit_metadata)
    return client


def commit_files(api, repo_id, directory, names, removed=()):
    """Upload an audited batch, retrying temporary Hub transport failures."""
    import httpx
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    from huggingface_hub.errors import HfHubHTTPError

    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=directory / name)
                  for name in names]
    operations.extend(CommitOperationDelete(path_in_repo=name) for name in removed)
    for attempt in range(6):
        try:
            return api.create_commit(
                repo_id=repo_id, operations=operations,
                commit_message="Publish audited ActionCodec2 release files",
            )
        except (HfHubHTTPError, httpx.TransportError) as error:
            status = error.response.status_code if isinstance(error, HfHubHTTPError) else None
            if attempt == 5 or (status is not None and status not in (429, 500, 502, 503, 504)):
                raise
            delay = min(30, 2 ** attempt)
            print(f"retrying: {repo_id}: HTTP {status or 'transport'}, in {delay}s", flush=True)
            time.sleep(delay)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/releases"))
    parser.add_argument("--owner", default="ZibinDong")
    parser.add_argument("--tag", default="v0.1.0", help="release tag for the audited files")
    parser.add_argument("--upload", action="store_true", help="upload after all local checks pass")
    parser.add_argument("--replace-tag", action="store_true", help="move an existing release tag to the uploaded revision")
    args = parser.parse_args(argv)
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(f"{args.owner}/ActionCodec2")
    from actioncodec2.core.catalog import VARIANT_ALIASES

    expected = {
        f"G05_ActionCodec2-{order}_{dataset}": VARIANT_ALIASES[f"paper-{order}"]
        for dataset in ("LIBERO", "SimplerEnv", "DROID") for order in ("1st", "2nd")
    }
    manifest = json.loads((args.root / "policies/manifest.json").read_text())
    policies = manifest.get("policies")
    if (manifest.get("format_version") != 1 or not isinstance(policies, list) or
            len(policies) != len(expected) or
            any(not isinstance(record, dict) for record in policies) or
            {record.get("directory") for record in policies} != set(expected)):
        raise ValueError("release manifest must contain exactly the six official AC2 policies")
    catalog = args.root / "ActionCodec2"
    index = json.loads((catalog / "variants.json").read_text())
    from actioncodec2.tokenizers.hub_catalog import _check_inventory

    expected_variants = set(VARIANT_ALIASES.values())
    if index.get("format_version") != 1 or set(index.get("variants", {})) != expected_variants:
        raise ValueError("tokenizer catalog must contain the 30 AC2 and 4 Sparse variants")
    for name, record in index["variants"].items():
        _check_inventory(catalog / "variants" / name, record)
    uploads = [(f"{args.owner}/ActionCodec2", catalog, audit_public_files(catalog))]
    for record in policies:
        name = record["directory"]
        if record.get("repo_id") != f"{args.owner}/{name}" or VARIANT_ALIASES.get(record.get("variant"), record.get("variant")) != expected[name]:
            raise ValueError("release manifest contains an unexpected policy repository or variant")
        bundle = args.root / "policies" / name
        policy = json.loads((bundle / "policy_config.json").read_text())
        settings = policy["tokenizer"]
        fingerprint = index["variants"][expected[name]]["fingerprint"]
        if (settings.get("name") != "actioncodec2" or VARIANT_ALIASES.get(settings.get("variant"), settings.get("variant")) != expected[name] or
                settings.get("artifact") != "./assets/tokenizer_artifact" or
                settings.get("variant_fingerprint") != fingerprint or
                record.get("variant_fingerprint") != fingerprint):
            raise ValueError(f"policy is not bound to its bundled official variant: {name}")
        inventory = audit_public_files(bundle)
        if inventory != record.get("files"):
            raise ValueError(f"policy files changed after release preparation: {name}")
        uploads.append((record["repo_id"], bundle, inventory))
    for repo_id, directory, files in uploads:
        size = sum((directory / name).stat().st_size for name in files)
        print(f"audited: {repo_id}: {len(files)} files, {size} bytes", flush=True)
    if not args.upload:
        return
    from huggingface_hub import HfApi, get_token, set_client_factory

    if not get_token():
        parser.error("Hugging Face authentication is missing; run `hf auth login` before uploading")
    set_client_factory(upload_client)
    api = HfApi(endpoint="https://hf-mirror.com")
    api.whoami()
    receipts = []
    for repo_id, directory, files in uploads:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        from huggingface_hub import RepoFile

        remote = {entry.path: entry for entry in api.list_repo_tree(repo_id, recursive=True)
                  if isinstance(entry, RepoFile)}
        names = []
        for name, sha256 in files.items():
            previous = remote.get(name)
            if previous is not None and previous.lfs is not None and previous.lfs.sha256 == sha256:
                continue
            if previous is not None and previous.lfs is None:
                payload = (directory / name).read_bytes()
                blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
                if blob == previous.blob_id:
                    continue
            names.append(name)
        removed = sorted(set(remote) - set(files) - {".gitattributes"})
        if names or removed:
            commit = commit_files(api, repo_id, directory, names, removed)
            revision = commit.oid
        else:
            revision = api.repo_info(repo_id).sha
        receipts.append({"repo_id": repo_id, "revision": revision})
        (args.root / "upload_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
        if args.replace_tag:
            tag = next((ref for ref in api.list_repo_refs(repo_id).tags if ref.name == args.tag), None)
            if tag is not None and tag.target_commit != revision:
                api.delete_tag(repo_id, tag=args.tag)
        api.create_tag(repo_id, tag=args.tag, revision=revision, exist_ok=True)
        print(f"uploaded: {repo_id}@{revision}; changed {len(names)}, removed {len(removed)}", flush=True)


if __name__ == "__main__":
    main()
