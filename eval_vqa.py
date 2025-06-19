import argparse, json, math
from pathlib import Path
import pandas as pd

def canonical(cid: str) -> str:
    fname = Path(cid).name
    if fname.endswith(".nii.gz"):
        return fname[:-7]          
    return Path(fname).stem        


def norm(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip().lower()

def global_accuracy(df: pd.DataFrame, gt: dict) -> float:
    glob = df[df.scope == "global"]
    tot, n = 0.0, 0
    for _, r in glob.iterrows():
        cid = canonical(r.case_id)
        preds = [norm(p) for p in str(r.prediction).split(",") if norm(p)]
        gts   = [norm(a) for a in gt[cid]["global_vqa"][0]["answer"]]

        if not gts and not preds:
            acc = 1.0
        else:
            acc = sum(p in gts for p in preds) / max(len(preds), len(gts))
        tot += acc
        n  += 1
    return 0.0 if n == 0 else tot / n

def build_chains(local_gt):
    """Return {root_id: [root, child1, child2, ...]} with children ordered by id."""
    by_id = {q["id"]: q for q in local_gt}
    chains = {}
    for q in local_gt:
        root = q["id"]
        while by_id[root]["follow_up"] != -1:
            root = by_id[root]["follow_up"]
        chains.setdefault(root, set()).add(q["id"])
    return {r: [r] + sorted(ids - {r}) for r, ids in chains.items()}


def local_accuracy(df: pd.DataFrame, gt: dict) -> float:
    loc = df[df.scope == "local"]
    chain_score_sum, chain_cnt = 0.0, 0

    for cid_csv, rows in loc.groupby("case_id"):
        cid        = canonical(cid_csv)
        local_gt   = gt[cid]["local_vqa"]
        chains     = build_chains(local_gt)
        gt_answers = {q["id"]: norm(q["answer"]) for q in local_gt}

        for _, r in rows.iterrows():
            root_id = int(r.question_id)
            if root_id not in chains:      
                continue

            preds_list = [norm(p) for p in str(r.prediction).split(",")]
            qids       = chains[root_id]

            chain_cnt += 1
            if not preds_list or preds_list[0] != gt_answers[root_id]: # main question wrong
                continue                    

            correct = 0
            for idx, qid in enumerate(qids):
                pred = preds_list[idx] if idx < len(preds_list) else ""
                if pred == gt_answers[qid]:
                    correct += 1
            chain_score_sum += correct / len(qids)

    return 0.0 if chain_cnt == 0 else chain_score_sum / chain_cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="CSV of model predictions")
    ap.add_argument("--val_json",  required=True, help="Ground-truth JSON")
    ap.add_argument("--out_json", default="vqa_results.json")
    args = ap.parse_args()

    preds = pd.read_csv(args.pred_csv)
    with open(args.val_json) as f:
        gt = {canonical(d["case_id"]): d for d in json.load(f)}

    scores = {
        "global_accuracy": round(global_accuracy(preds, gt), 6),
        "local_accuracy":  round(local_accuracy(preds, gt), 6)
    }

    print(f"Global VQA accuracy: {scores['global_accuracy']:.4f}")
    print(f"Local  VQA accuracy: {scores['local_accuracy']:.4f}")

    with open(args.out_json, "w") as f:
        json.dump(scores, f, indent=2)

if __name__ == "__main__":
    main()
