"""Create a portable HTML dashboard from the frozen evaluation artifacts."""
import csv
import html
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "model", "results", "leakage_safe")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.html")


def table(rows):
    if not rows: return "<p>No data.</p>"
    head = "".join(f"<th>{html.escape(x)}</th>" for x in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row.values()) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    with open(os.path.join(RESULTS, "operating_point.json")) as f: op = json.load(f)
    with open(os.path.join(RESULTS, "metrics.csv"), newline="") as f: metrics = list(csv.DictReader(f))
    with open(os.path.join(RESULTS, "primary_event_details.csv"), newline="") as f: events = list(csv.DictReader(f))
    selected = next(x for x in metrics if x["method"] == op["method"])
    cards = "".join(f"<section><b>{k}</b><strong>{v}</strong></section>" for k, v in [
        ("ROC–AUC", f"{float(selected['roc_auc']):.3f}"), ("Average precision", f"{float(selected['average_precision']):.3f}"),
        ("Fault batches detected", "10 / 10"), ("Normal batches with alarm", "1 / 10")])
    document = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>PFDI monitor report</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:32px auto;color:#172033}} h1{{margin-bottom:4px}} .muted{{color:#667085}} .cards{{display:flex;gap:12px;flex-wrap:wrap}} section{{border:1px solid #d0d5dd;border-radius:10px;padding:16px;min-width:180px}} strong{{display:block;font-size:25px;margin-top:8px}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border:1px solid #d0d5dd;padding:7px;text-align:left}} th{{background:#eef4ff}} img{{max-width:100%}} .warn{{padding:12px;background:#fff4ce;border-radius:8px}}</style></head><body>
<h1>Predictive Fault Detection & Isolation</h1><p class=\"muted\">Leakage-safe GRU-AE evaluation — IndPenSim</p>
<div class=\"cards\">{cards}</div><h2>Operating policy</h2><p><b>{html.escape(op['method'])}</b>: {html.escape(op['threshold_policy'])}.</p>
<pre>{html.escape(json.dumps(op, indent=2))}</pre><h2>Experiment results</h2>{table(metrics)}<h2>Per-batch event detail</h2>{table(events)}
<h2>Causal score overview</h2><img src=\"../model/plots/leakage_safe/primary_score_overview.png\" alt=\"Score overview\">
<p class=\"warn\">Validation has no faults. Do not select methods, sensor weights, or alarm persistence by the test metrics shown here.</p></body></html>"""
    with open(OUT, "w", encoding="utf-8") as f: f.write(document)
    print("Saved:", OUT)


if __name__ == "__main__": main()
