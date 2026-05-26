import json
import numpy as np
import argparse


def compute_averages(file_path):
    session_metrics = {}
    turn_metrics = {}
    counts = {"session": {}, "turn": {}}

    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)

            metrics = entry.get("retrieval_results", {}).get("metrics", {})

            for level in ["session", "turn"]:
                if level not in metrics:
                    continue

                for k, v in metrics[level].items():
                    if k not in session_metrics and level == "session":
                        session_metrics[k] = []
                    if k not in turn_metrics and level == "turn":
                        turn_metrics[k] = []

                    if level == "session":
                        session_metrics[k].append(v)
                    else:
                        turn_metrics[k].append(v)

    # Compute means
    avg_session = {k: float(np.mean(v)) for k, v in session_metrics.items() if v}
    avg_turn = {k: float(np.mean(v)) for k, v in turn_metrics.items() if v}

    return {"session": avg_session, "turn": avg_turn}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    results = compute_averages(args.file)
    print(json.dumps(results, indent=2))