
import argparse, csv, time, math
from pathlib import Path
import numpy as np

# Project imports
from src.core.rag import RAGStore
from src.core.rulebook import Rulebook
from src.core.teleology import Teleology
from src.core.j_metrics import social_coherence, j4_empowerment_infonce, j4_empowerment_proxy, j5_ahimsa_proxy
from src.persona.persona import PersonaAgent
from src.ui.radar import plot_radar
from src.ui.teleology_dashboard import plot_series

# We import helpers from our causal module
from src.core.j5_causal import compute_j5_with_fallbacks, build_context_features, action_from_actor

try:
    import yaml
except Exception:
    yaml = None

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "out"
OUT.mkdir(exist_ok=True)

def load_scenario(path: Path):
    txt = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json
        return json.loads(txt)
    else:
        if yaml is None:
            raise RuntimeError("PyYAML not available; please install pyyaml or supply a JSON scenario.")
        return yaml.safe_load(txt)

def make_state(scn, rag):
    actors = []
    styles_export = []
    for a in scn["actors"]:
        passages = rag.query(a)
        ag = PersonaAgent(a, passages)
        actors.append(ag)
        styles_export.append({"id": ag.id, "name": ag.name, "style_spec": a.get('style'), "derived": ag.style})
    (OUT / "persona_styles.json").write_text(
        __import__("json").dumps(styles_export, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return {
        "actors": actors,
        "esc": scn.get("initial_state",{}).get("escalation","Tension"),
        "scenario": scn,
        "replay_A_cont": [], "replay_A_disc": [], "replay_C": [], "replay_harm": [],
        "ablation": "none"
    }

def simulate_turn_once(state, rulebook, teleology, rag, j5_mode="neural"):
    actors = state["actors"]
    # Collect candidate outputs (shortened: single policy step per actor)
    candidates = []
    for idx, ag in enumerate(actors):
        move = ag.policy(None)
        actor_spec = state["scenario"]["actors"][idx]
        ok, reason = rulebook.plausibility_mask(ag.id, {"core_interests": ag.core_interests}, move)
        cite = ""
        line = f"{ag.name}: {move}{cite}" if ok else f"{ag.name}: [blocked] {reason}"
        candidates.append(line)

    # Hidden states for metrics
    H_list = [ag.hidden for ag in actors]
    # J1
    from src.core.j_metrics import j1_integration, j2_diversity, j3_robustness
    J1 = j1_integration(np.vstack(H_list))
    J2 = j2_diversity(np.vstack(H_list))
    def f_recon_scalar(z):
        return float(np.sum(z*z))
    J3 = j3_robustness(f_recon_scalar, np.mean(np.vstack(H_list), axis=0))

    # J4 InfoNCE (fallback to proxy)
    actions = np.vstack([getattr(ag, "last_action_vec", np.mean(ag.hidden, axis=0)) for ag in actors])
    S_next = np.vstack([getattr(ag, "hidden", np.zeros_like(actors[0].hidden)).mean(axis=0) for ag in actors])
    S_cond = np.vstack([np.mean(ag.hidden, axis=0) for ag in actors])
    try:
        J4 = j4_empowerment_infonce(actions, S_next, S_cond)
    except Exception:
        J4 = j4_empowerment_proxy(S_next)

    # Context & Replay for J5
    a_cont = action_from_actor(actors[0])
    a_disc = int(np.clip(np.round((a_cont.mean()-a_cont.min())/max(1e-6,(a_cont.max()-a_cont.min()))*2), 0, 2))
    C_feat = build_context_features(actors, state)
    harm_obs = -j5_ahimsa_proxy(actors[0].hidden, actors[1].hidden, actors[1].decoder) if len(actors)>=2 else 0.0

    state["replay_A_cont"].append(a_cont)
    state["replay_A_disc"].append(a_disc)
    state["replay_C"].append(C_feat)
    state["replay_harm"].append(harm_obs)
    for k in ["replay_A_cont","replay_A_disc","replay_C","replay_harm"]:
        if len(state[k])>256: state[k] = state[k][-256:]

    # Compute J5 with ablation handling
    if state.get("ablation","none") == "H2":
        J5 = 0.0
        fb = "forced_zero"
    else:
        J5, fb = compute_j5_with_fallbacks(j5_mode, state, actors, a_cont, a_disc, C_feat)

    SCM = social_coherence(H_list)
    return {"J1":J1,"J2":J2,"J3":J3,"J4":J4,"J5":J5,"SCM":SCM, "fallback":fb}

def run_one(scn_path: Path, j5_mode: str, ablation: str, turns: int, seed: int):
    np.random.seed(seed)
    rag = RAGStore(BASE / "data/corpus")
    rulebook = Rulebook()
    teleology = Teleology()
    scn = load_scenario(scn_path)
    state = make_state(scn, rag)
    state["ablation"] = ablation

    series = {"J1":[], "J2":[], "J3":[], "J4":[], "J5":[], "SCM":[], "fallbacks":[]}
    for t in range(turns):
        metrics = simulate_turn_once(state, rulebook, teleology, rag, j5_mode=j5_mode)
        for k in ["J1","J2","J3","J4","J5","SCM"]:
            series[k].append(metrics[k])
        series["fallbacks"].append(metrics["fallback"] if metrics["fallback"] is not None else "none")
    return series

def aggregate(series):
    out = {}
    for k,v in series.items():
        if k=="fallbacks":
            out["fallback_rate"] = float(np.mean([1.0 if x!="none" else 0.0 for x in v]))
        else:
            arr = np.asarray(v, dtype=float)
            out[f"{k}_mean"] = float(arr.mean())
            out[f"{k}_std"]  = float(arr.std(ddof=1) if arr.size>1 else 0.0)
    return out

def main():
    ap = argparse.ArgumentParser(description="Ablation Benchmark for V6.2")
    ap.add_argument("--scenario", type=str, default="data/scenarios/dayton_3party.yaml")
    ap.add_argument("--j5", type=str, default="neural", choices=["neural","aipw","ridge","proxy"])
    ap.add_argument("--modes", type=str, default="none,H1,H2,H3")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--out_csv", type=str, default="out/ablation_results.csv")
    ap.add_argument("--out_png", type=str, default="out/ablation_results.png")
    ap.add_argument("--per_turn_csv", type=str, default="out/ablation_per_turn.csv")
    ap.add_argument("--per_turn_png", type=str, default="out/ablation_per_turn.png")
    args = ap.parse_args()

    scn_path = Path(args.scenario)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    rows = []
    per_turn_rows = []
    for ab in modes:
        for s in range(args.seeds):
            series = run_one(scn_path, args.j5, ab, args.turns, seed=1000 + s)
            agg = aggregate(series)
            row = {"ablation":ab, "seed":s, "j5_mode":args.j5}
            row.update(agg)
            rows.append(row)
            # store per-turn series as long-form rows
            T = len(series['J5'])
            for t in range(T):
                per_turn_rows.append({
                    'ablation': ab,
                    'seed': s,
                    'turn': t+1,
                    'J4': series['J4'][t],
                    'J5': series['J5'][t],
                    'SCM': series['SCM'][t]
                })

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)

    # Write per-turn CSV
    pt_csv = Path(args.per_turn_csv)
    pt_csv.parent.mkdir(exist_ok=True, parents=True)
    if per_turn_rows:
        import csv as _csv
        with pt_csv.open('w', newline='', encoding='utf-8') as f:
            wr = _csv.DictWriter(f, fieldnames=list(per_turn_rows[0].keys()))
            wr.writeheader(); wr.writerows(per_turn_rows)
    # Plot (matplotlib; bar chart + per-turn line chart)
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.DataFrame(rows)
        # aggregate by ablation
        g = df.groupby("ablation").agg({
            "J4_mean":"mean",
            "J5_mean":"mean",
            "SCM_mean":"mean",
            "fallback_rate":"mean"
        }).reset_index()
        fig = plt.figure()  # bar chart
        x = np.arange(len(g))
        w = 0.2
        plt.bar(x - 1.5*w, g["J4_mean"], width=w, label="J4")
        plt.bar(x - 0.5*w, g["J5_mean"], width=w, label="J5")
        plt.bar(x + 0.5*w, g["SCM_mean"], width=w, label="SCM")
        plt.bar(x + 1.5*w, g["fallback_rate"], width=w, label="Fallback rate")
        plt.xticks(x, g["ablation"])
        plt.title("Ablation Benchmark (means over seeds)")
        plt.legend()
        fig.savefig(args.out_png, dpi=160, bbox_inches="tight")
        # Per-turn line plot
        try:
            df2 = pd.DataFrame(per_turn_rows)
            fig2 = plt.figure()
            for ab in sorted(df2['ablation'].unique()):
                d = df2[df2['ablation']==ab].groupby('turn').agg({'J5':'mean'})
                plt.plot(d.index.values, d['J5'].values, label=f"{ab}")
            plt.title('Ablation per-turn J5 (mean across seeds)')
            plt.xlabel('Turn'); plt.ylabel('J5 (mean)'); plt.legend()
            fig2.savefig(args.per_turn_png, dpi=160, bbox_inches='tight')
        except Exception as e:
            print('Per-turn plotting skipped:', e)
    except Exception as e:
        print("Plotting skipped:", e)

    print(f"Done. Wrote {out_csv} and {args.out_png}")

if __name__ == "__main__":
    main()
