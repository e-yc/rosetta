#!/usr/bin/env python3
"""Run the full activation experiment pipeline sequentially."""
import subprocess
import sys
import time

steps = [
    ("CORPUS BUILD", "corpus_builder.py"),
    ("PASS 1: GLM EXTRACTION", "pass1_extract.py"),
    ("PASS 2: LLAMA DIFFERENTIALS", "pass2_differential.py"),
    ("PASS 3: EIGENDECOMPOSITION & PROJECTION", "pass3_project.py"),
    ("ANALYSIS", "analysis.py"),
    ("VISUALIZATION", "visualize.py"),
]

total_start = time.time()
for title, script in steps:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd=r"C:\Users\Eric\Documents\rosetta\rosetta\activation_experiment",
    )
    elapsed = time.time() - t0
    # On Windows, large memmap processes can crash during cleanup with 0xC0000409.
    # Check if the expected output was produced rather than just the exit code.
    if result.returncode != 0:
        print(f"\n  WARNING: {script} exited with code {result.returncode} after {elapsed/60:.1f} min")
        # Check if it actually completed by looking for output files
        import os
        expected_outputs = {
            "corpus_builder.py": r"data\corpus.jsonl",
            "pass1_extract.py": r"data\pass1_metadata.json",
            "pass2_differential.py": r"results\pass2_metadata.json",
            "pass3_project.py": r"results\atlas\projections.npy",
        }
        check = expected_outputs.get(script)
        base = r"C:\Users\Eric\Documents\rosetta\rosetta\activation_experiment"
        if check and os.path.exists(os.path.join(base, check)):
            print(f"  Output {check} exists — treating as success (crash during cleanup)")
        else:
            print(f"\n*** {script} FAILED ***")
            sys.exit(1)
    print(f"\n  [{script} completed in {elapsed/60:.1f} min]", flush=True)

total = time.time() - total_start
print(f"\n{'='*70}")
print(f"  ALL STEPS COMPLETE — total time: {total/60:.1f} min")
print(f"{'='*70}")
