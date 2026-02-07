# this file just has a subprocess.run for the other 4 files. nothing special.
# AI GENERATED CODE BEGINS HERE !!!!!!!!!!!!!!!!!

import subprocess, sys
import os

DATA = sys.argv[1] if len(sys.argv) > 1 else 'data/train.txt'

BASE_MODULE = "src.task3"

steps = [
    ('Step 1: Inductive mining (inverse + two-hop)', f'{BASE_MODULE}.rule_miner_amie'),
    ('Step 2: Curated domain rules', f'{BASE_MODULE}.ik_these_relations_already'),
    ('Step 3: Standard confidence', f'{BASE_MODULE}.standard_confidence'),
    ('Step 4: PCA confidence', f'{BASE_MODULE}.pca_confidence'),
]

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

for label, module in steps:

    print("--------------------------------------------------------------")
    print(label)
    print("--------------------------------------------------------------")

    subprocess.run(
        [sys.executable, "-m", module, DATA],
        cwd=PROJECT_ROOT,
        check=True
    )

# AI GENERATED CODE ENDS HERE !!!!!!!!!!!!!!!!!