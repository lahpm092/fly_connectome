"""Download scripts for all three connectome datasets.

Data sources
============
1. Frankenbrain SQLite (FAFB + MANC fused via BANC neck bridge)
   - Harvard Dataverse, ~10-30 GB
   - Already integrates brain and VNC

2. FAFB synapse data (Zenodo)
   - Individual synapse table with neurotransmitter predictions
   - Edge list by neuropil

3. MANC flat files (neuPrint / Google Cloud)
   - Neuron list, synapse list, adjacency

4. Optic Lobe flat files (neuPrint / Google Cloud)
   - Neuron list, synapse list, adjacency
"""

import os
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm

from fly_connectome.config import (
    DATA_DIR,
    FAFB_EDGES_URL,
    FAFB_SYNAPSE_URL,
    MANC_FLAT_FILES_BUCKET,
    OPTIC_LOBE_BUCKET,
)


def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download a file from a URL with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already exists: {dest}")
        return dest

    print(f"  Downloading {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def _gsutil_download(bucket_path: str, dest_dir: Path) -> None:
    """Download from Google Cloud Storage using gsutil."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading from {bucket_path} -> {dest_dir}")
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", bucket_path, str(dest_dir)],
        check=True,
    )


def download_fafb(data_dir: Path | None = None) -> Path:
    """Download FAFB connectome data from Zenodo.

    Downloads:
    - Synapse table (~130M synapses with NT predictions)
    - Edge list summarized by neuropil
    """
    out = (data_dir or DATA_DIR) / "fafb"
    out.mkdir(parents=True, exist_ok=True)
    print("=== Downloading FAFB (Female Adult Fly Brain) ===")

    _download_file(FAFB_SYNAPSE_URL, out / "synapses.feather")
    _download_file(FAFB_EDGES_URL, out / "edges_by_neuropil.feather")

    print(f"  FAFB data saved to {out}")
    return out


def download_manc(data_dir: Path | None = None) -> Path:
    """Download MANC connectome flat files from Google Cloud.

    Falls back to neuPrint programmatic access if gsutil unavailable.
    """
    out = (data_dir or DATA_DIR) / "manc"
    out.mkdir(parents=True, exist_ok=True)
    print("=== Downloading MANC (Male Adult Nerve Cord) ===")

    try:
        _gsutil_download(f"{MANC_FLAT_FILES_BUCKET}/*", out)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  gsutil not available. Use neuprint-python instead:")
        print("    from neuprint import Client")
        print(f'    c = Client("neuprint.janelia.org", dataset="manc:v1.2.1")')
        print('    neurons, roi_counts = c.fetch_neurons(NC(...))')
        print("  Or download manually from:")
        print("    https://www.janelia.org/project-team/flyem/manc-connectome")

    return out


def download_optic_lobe(data_dir: Path | None = None) -> Path:
    """Download optic lobe connectome from Google Cloud.

    Falls back to neuPrint programmatic access if gsutil unavailable.
    """
    out = (data_dir or DATA_DIR) / "optic_lobe"
    out.mkdir(parents=True, exist_ok=True)
    print("=== Downloading Optic Lobe (Male Right) ===")

    try:
        _gsutil_download(f"{OPTIC_LOBE_BUCKET}/*", out)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  gsutil not available. Use neuprint-python instead:")
        print("    from neuprint import Client")
        print('    c = Client("neuprint.janelia.org", dataset="optic-lobe:v1.0")')
        print("  Or download manually from:")
        print("    https://www.janelia.org/project-team/flyem/optic-lobe")

    return out


def download_all(data_dir: Path | None = None) -> dict[str, Path]:
    """Download all datasets."""
    return {
        "fafb": download_fafb(data_dir),
        "manc": download_manc(data_dir),
        "optic_lobe": download_optic_lobe(data_dir),
    }


def download_via_neuprint(
    server: str,
    dataset: str,
    token: str | None = None,
) -> "neuprint.Client":
    """Create a neuPrint client for programmatic dataset access.

    Works for MANC, optic lobe, and the hemibrain.
    Requires a neuPrint account token for some queries.

    Usage::

        client = download_via_neuprint(
            "neuprint.janelia.org", "manc:v1.2.1"
        )
        # Fetch all neurons
        neurons_df, roi_counts = client.fetch_neurons(
            neuprint.NeuronCriteria(status="Traced")
        )
        # Fetch adjacency
        adj_df = client.fetch_adjacencies(
            neuprint.NeuronCriteria(status="Traced"),
            neuprint.NeuronCriteria(status="Traced"),
        )
    """
    from neuprint import Client

    token = token or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    return Client(server, dataset=dataset, token=token)
