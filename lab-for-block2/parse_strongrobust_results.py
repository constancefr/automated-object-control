import re
import csv
from pathlib import Path

results_dir = Path("results")
output_csv = Path("strongrobust_summary.csv")

# Regex patterns
filename_pattern = re.compile(r"eta_(\d*\.?\d+)_eps_(\d*\.?\d+)\.log")
verified_pattern = re.compile(r"verified:\s+(\d+)/(\d+)")
falsified_pattern = re.compile(r"falsified:\s+(\d+)/(\d+)")
timeout_pattern = re.compile(r"timed-out:\s+(\d+)/(\d+)")
errored_pattern = re.compile(r"errored:\s+(\d+)/(\d+)")

rows = []

for log_file in sorted(results_dir.glob("*.log")):
    match = filename_pattern.search(log_file.name)
    if not match:
        print(f"Skipping unrecognized file name: {log_file.name}")
        continue

    eta, eps = match.groups()
    text = log_file.read_text(errors="ignore")

    def extract(pattern):
        m = pattern.search(text)
        return int(m.group(1)) if m else 0, int(m.group(2)) if m else 0

    verified, total = extract(verified_pattern)
    falsified, _ = extract(falsified_pattern)
    timed_out, _ = extract(timeout_pattern)
    errored, _ = extract(errored_pattern)

    rows.append({
        "eta": float(eta),
        "epsilon": float(eps),
        "verified": verified,
        "falsified": falsified,
        "timed_out": timed_out,
        "errored": errored,
        "total": total,
        "verified_ratio": verified / total if total > 0 else 0.0
    })

# Sort results by eta then epsilon for nicer presentation
rows.sort(key=lambda r: (r["eta"], r["epsilon"]))

# Write CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["eta", "epsilon", "verified", "falsified", "timed_out", "errored", "total", "verified_ratio"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Parsed {len(rows)} log files. Results saved to {output_csv}")
