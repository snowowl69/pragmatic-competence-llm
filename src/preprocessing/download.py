"""
Dataset download script for the pragmatic competence benchmark.

Downloads source datasets for all three pragmatic phenomena:
  - Sarcasm: MUStARD, iSarcasm
  - Implicature: Circa
  - Indirect speech acts: (hand-curated — generated via curate_indirect.py)

All downloads are saved to data/raw/ with subdirectories per source.
Licenses must be verified at download time (see data/raw/LICENSES.md).

Usage:
    python src/preprocessing/download.py [--all | --mustard | --isarcasm | --circa]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------


def download_mustard() -> None:
    """Download the MUStARD (Multimodal Sarcasm Detection) dataset.

    MUStARD provides sarcasm-labeled dialogue utterances with context.
    Source: https://github.com/soujanyaporia/MUStARD

    The dataset is a JSON file with dialogue context, utterance, and sarcasm labels.
    """
    output_dir = RAW_DATA_DIR / "mustard"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        print("[MUStARD] Attempting to load via Hugging Face datasets...")
        # Try HF Hub first — if a community upload exists
        try:
            ds = load_dataset("soujanyaporia/MUStARD", trust_remote_code=True)
            output_path = output_dir / "mustard_hf.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for split_name in ds:
                    for item in ds[split_name]:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[MUStARD] ✓ Saved to {output_path}")
            return
        except Exception:
            print("[MUStARD] Not found on HF Hub, trying direct GitHub download...")

    except ImportError:
        print("[MUStARD] `datasets` library not installed, trying direct download...")

    # Fallback: direct download from GitHub
    try:
        import urllib.request

        url = "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json"
        output_path = output_dir / "sarcasm_data.json"
        print(f"[MUStARD] Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"[MUStARD] ✓ Saved to {output_path}")

    except Exception as e:
        print(f"[MUStARD] ✗ Download failed: {e}")
        print("[MUStARD] Manual download required:")
        print("  1. Go to https://github.com/soujanyaporia/MUStARD")
        print(f"  2. Download sarcasm_data.json to {output_dir}/")
        print("  3. Verify license terms and update data/raw/LICENSES.md")


def download_isarcasm() -> None:
    """Download the iSarcasm dataset.

    iSarcasm provides sarcasm-labeled tweets with irony type annotations.
    Source: https://github.com/silviu-oprea/iSarcasm
    """
    output_dir = RAW_DATA_DIR / "isarcasm"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        print("[iSarcasm] Attempting to load via Hugging Face datasets...")
        try:
            ds = load_dataset("silviuo/iSarcasm", trust_remote_code=True)
            output_path = output_dir / "isarcasm_hf.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for split_name in ds:
                    for item in ds[split_name]:
                        item["_split"] = split_name
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[iSarcasm] ✓ Saved to {output_path}")
            return
        except Exception:
            print("[iSarcasm] Not found on HF Hub, trying direct download...")

    except ImportError:
        pass

    # Fallback: direct download from GitHub candidate repositories
    candidate_base_urls = [
        "https://raw.githubusercontent.com/dmbavkar/iSarcasm/master",
        "https://raw.githubusercontent.com/HeadTails/iSarcasmEval/main/train",
        "https://raw.githubusercontent.com/silviu-oprea/iSarcasm/main",
    ]

    for base_url in candidate_base_urls:
        try:
            import urllib.request

            files = ["isarcasm_train.csv", "isarcasm_test.csv"]
            if "iSarcasmEval" in base_url:
                files = ["train-task1.csv", "task_1_test_labels.csv"]

            print(f"[iSarcasm] Trying download from {base_url}...")
            for fname in files:
                url = f"{base_url}/{fname}"
                output_path = output_dir / fname
                print(f"[iSarcasm] Downloading {fname} from {url}...")
                urllib.request.urlretrieve(url, output_path)
                print(f"[iSarcasm] ✓ Saved to {output_path}")

            return  # Success
        except Exception as e:
            print(f"[iSarcasm] Attempt from {base_url} failed: {e}")

    print("[iSarcasm] Manual download required:")
    print("  1. Go to https://github.com/silviu-oprea/iSarcasm or https://github.com/HeadTails/iSarcasmEval")
    print(f"  2. Download training/testing CSV files to {output_dir}/")


def download_circa() -> None:
    """Download the Circa dataset.

    Circa provides indirect answers to polar (yes/no) questions with
    implicature labels — exactly what we need for the implicature category.
    Source: https://github.com/google-research-datasets/circa
    """
    output_dir = RAW_DATA_DIR / "circa"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        print("[Circa] Attempting to load via Hugging Face datasets...")
        try:
            ds = load_dataset("google-research-datasets/circa", trust_remote_code=True)
            output_path = output_dir / "circa_hf.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for split_name in ds:
                    for item in ds[split_name]:
                        item["_split"] = split_name
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[Circa] ✓ Saved to {output_path}")
            return
        except Exception:
            print("[Circa] Not found on HF Hub with that name, trying alternatives...")

    except ImportError:
        pass

    # Fallback: direct download
    try:
        import urllib.request

        url = "https://raw.githubusercontent.com/google-research-datasets/circa/main/circa-data.tsv"
        output_path = output_dir / "circa-data.tsv"
        print(f"[Circa] Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"[Circa] ✓ Saved to {output_path}")

    except Exception as e:
        print(f"[Circa] ✗ Download failed: {e}")
        print("[Circa] Manual download required:")
        print("  1. Go to https://github.com/google-research-datasets/circa")
        print(f"  2. Download circa-data.tsv to {output_dir}/")


def create_indirect_speech_act_seed() -> None:
    """Create seed examples for indirect speech act curation.

    Since no large-scale indirect speech act dataset exists with the coverage
    we need, this generates a starter set of template-derived examples that
    the team should expand and validate manually per the annotation guidelines.
    """
    output_dir = RAW_DATA_DIR / "indirect_speech_acts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed examples — each has a direct and indirect version of the same intent
    seed_examples = [
        # Indirect requests
        {
            "context": "Two coworkers in a meeting room.",
            "speaker": "coworker",
            "utterance": "It's really stuffy in here.",
            "literal_meaning": "A statement about the air quality of the room.",
            "intended_meaning": "A request to open a window or turn on the AC.",
            "pragmatic_category": "indirect_speech_act",
            "label": "request",
            "answer_options": ["statement", "request", "question", "suggestion", "complaint"],
            "difficulty": "easy",
            "requires_context": False,
            "source": "hand-curated",
            "license": "CC-BY-4.0",
        },
        {
            "context": "A customer at a restaurant, looking at the menu.",
            "speaker": "customer",
            "utterance": "Do you have anything without nuts?",
            "literal_meaning": "A question about the availability of nut-free options.",
            "intended_meaning": "A request for nut-free food, likely due to an allergy.",
            "pragmatic_category": "indirect_speech_act",
            "label": "request",
            "answer_options": ["statement", "request", "question", "suggestion", "complaint"],
            "difficulty": "easy",
            "requires_context": True,
            "source": "hand-curated",
            "license": "CC-BY-4.0",
        },
        {
            "context": "A student talking to a professor after class.",
            "speaker": "student",
            "utterance": "I didn't quite follow the part about eigenvalues.",
            "literal_meaning": "The student states they didn't understand eigenvalues.",
            "intended_meaning": "A request for the professor to re-explain eigenvalues.",
            "pragmatic_category": "indirect_speech_act",
            "label": "request",
            "answer_options": ["statement", "request", "question", "suggestion", "complaint"],
            "difficulty": "medium",
            "requires_context": True,
            "source": "hand-curated",
            "license": "CC-BY-4.0",
        },
        # Indirect suggestions
        {
            "context": "Two friends deciding where to eat.",
            "speaker": "friend",
            "utterance": "I hear that new Italian place is really good.",
            "literal_meaning": "The speaker heard positive things about an Italian restaurant.",
            "intended_meaning": "A suggestion to go to the Italian restaurant.",
            "pragmatic_category": "indirect_speech_act",
            "label": "suggestion",
            "answer_options": ["statement", "request", "question", "suggestion", "complaint"],
            "difficulty": "easy",
            "requires_context": True,
            "source": "hand-curated",
            "license": "CC-BY-4.0",
        },
        {
            "context": "A manager reviewing an employee's presentation draft.",
            "speaker": "manager",
            "utterance": "Some of these slides have a lot of text.",
            "literal_meaning": "An observation about text density on slides.",
            "intended_meaning": "A suggestion to reduce the text on the slides.",
            "pragmatic_category": "indirect_speech_act",
            "label": "suggestion",
            "answer_options": ["statement", "request", "question", "suggestion", "complaint"],
            "difficulty": "medium",
            "requires_context": True,
            "source": "hand-curated",
            "license": "CC-BY-4.0",
        },
        # Indirect complaints
        {
            "context": "A passenger on a delayed flight, talking to a flight attendant.",
            "speaker": "passenger",
            "utterance": "We've been sitting on the tarmac for two hours now.",
            "literal_meaning": "A factual statement about elapsed time.",
            "intended_meaning": "A complaint about the delay and an implicit demand for information or action.",
            "pragmatic_category": "indirect_speech_act",
            "label": "complaint",
            "answer_options": ["statement", "request", "question", "suggestion", "complaint"],
            "difficulty": "easy",
            "requires_context": True,
            "source": "hand-curated",
            "license": "CC-BY-4.0",
        },
        {
            "context": "A customer calling tech support.",
            "speaker": "customer",
            "utterance": "This is the third time I've had to call about this issue.",
            "literal_meaning": "A statement about the number of previous calls.",
            "intended_meaning": "A complaint about unresolved issues and poor service quality.",
            "pragmatic_category": "indirect_speech_act",
            "label": "complaint",
            "answer_options": ["statement", "request", "question", "suggestion", "complaint"],
            "difficulty": "medium",
            "requires_context": False,
            "source": "hand-curated",
            "license": "CC-BY-4.0",
        },
        # Genuine statements (controls — not indirect speech acts)
        {
            "context": "A weather reporter delivering the forecast on TV.",
            "speaker": "reporter",
            "utterance": "It's going to be cold out there today.",
            "literal_meaning": "An informational statement about the weather.",
            "intended_meaning": "The speaker is informing the audience — this is a genuine statement, not an indirect request.",
            "pragmatic_category": "indirect_speech_act",
            "label": "statement",
            "answer_options": ["statement", "request", "question", "suggestion", "complaint"],
            "difficulty": "easy",
            "requires_context": True,
            "source": "control-item",
            "license": "CC-BY-4.0",
        },
        {
            "context": "A doctor examining a patient.",
            "speaker": "doctor",
            "utterance": "Can you raise your left arm above your head?",
            "literal_meaning": "A question about the patient's physical ability.",
            "intended_meaning": "A genuine diagnostic question, not a request — the doctor wants to know if the patient physically can.",
            "pragmatic_category": "indirect_speech_act",
            "label": "question",
            "answer_options": ["statement", "request", "question", "suggestion", "complaint"],
            "difficulty": "medium",
            "requires_context": True,
            "source": "control-item",
            "license": "CC-BY-4.0",
        },
        # Indirect request — "I wouldn't..." pattern
        {
            "context": "A colleague notices another about to sit in a broken chair.",
            "speaker": "colleague",
            "utterance": "I wouldn't sit there if I were you.",
            "literal_meaning": "A conditional statement about the speaker's hypothetical behavior.",
            "intended_meaning": "A warning not to sit in the chair (likely broken or reserved).",
            "pragmatic_category": "indirect_speech_act",
            "label": "suggestion",
            "answer_options": ["statement", "request", "question", "suggestion", "complaint"],
            "difficulty": "easy",
            "requires_context": True,
            "source": "hand-curated",
            "license": "CC-BY-4.0",
        },
    ]

    output_path = output_dir / "indirect_seed_examples.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(seed_examples):
            example["id"] = f"ind_{i + 1:04d}"
            example["split"] = "train"  # Will be reassigned during splitting
            example["annotator_notes"] = ""
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"[Indirect Speech Acts] ✓ Created {len(seed_examples)} seed examples at {output_path}")
    print("[Indirect Speech Acts] ⚠ These are SEED examples only.")
    print("  The team must expand to ~150–250 examples following docs/annotation_guidelines.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download source datasets for the pragmatic competence benchmark."
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--mustard", action="store_true", help="Download MUStARD")
    parser.add_argument("--isarcasm", action="store_true", help="Download iSarcasm")
    parser.add_argument("--circa", action="store_true", help="Download Circa")
    parser.add_argument("--indirect", action="store_true", help="Create indirect speech act seed examples")

    args = parser.parse_args()

    # Default to --all if no specific dataset is requested
    if not any([args.all, args.mustard, args.isarcasm, args.circa, args.indirect]):
        args.all = True

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data dir: {RAW_DATA_DIR}")
    print()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.mustard:
        download_mustard()
        print()

    if args.all or args.isarcasm:
        download_isarcasm()
        print()

    if args.all or args.circa:
        download_circa()
        print()

    if args.all or args.indirect:
        create_indirect_speech_act_seed()
        print()

    print("=" * 60)
    print("IMPORTANT: Verify all license terms and update data/raw/LICENSES.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
