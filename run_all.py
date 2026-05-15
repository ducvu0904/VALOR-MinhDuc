import subprocess
import os

def run_cmd(cmd_list):
    print(f"\n>>> Executing: {' '.join(cmd_list)}")
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run all backbones on a specific dataset.")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "synthetic", "hillstrom-men", "hillstrom-women"],
                        help="Which dataset to run (default: all)")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["hillstrom-men", "hillstrom-women"]
    else:
        datasets = [args.dataset]
        
    # Hardcoded backbones as requested
    backbones = ["TARNet", "DragonNet", "CFR-MMD", "CFR-WASS"]

    print(f"Targeting Datasets: {datasets}")
    print(f"Targeting Backbones: {backbones}")

    for d in datasets:
        for b in backbones:
            print(f"\n{'='*60}\n  Processing Backbone: {b} on Dataset: {d}\n{'='*60}")
            
            # 1. Base
            run_cmd(["python", "main.py", "--model", b, "--dataset", d])
            
            # 2. + ZILN
            run_cmd(["python", "main.py", "--model", b, "--dataset", d, "--use_ziln"])
            
            # 3. + ZILN + Focal
            run_cmd(["python", "main.py", "--model", b, "--dataset", d, "--use_ziln", "--use_focal"])
            
            # 4. + ZILN + Focal + WR (Ranking)
            run_cmd(["python", "main.py", "--model", b, "--dataset", d, "--use_ziln", "--use_focal", "--use_ranking"])
            
            # 5. + ZILN + Focal + GTI (Gating)
            run_cmd(["python", "main.py", "--model", b, "--dataset", d, "--use_ziln", "--use_focal", "--use_gating"])
            
            # 6. + ZILN + Focal + GTI + WR (Full VALOR)
            run_cmd(["python", "main.py", "--model", b, "--dataset", d, "--use_ziln", "--use_focal", "--use_gating", "--use_ranking"])

            # 7. RERUM (Backbone + Uplift Ranking only)
            run_cmd(["python", "main.py", "--model", b, "--dataset", d, "--rerum", "--use_uplift_ranking"])

            # 8. RERUM Full (Backbone + Uplift Ranking + Response Ranking)
            run_cmd(["python", "main.py", "--model", b, "--dataset", d, "--rerum", "--use_uplift_ranking", "--use_response_ranking"])

if __name__ == "__main__":
    main()
