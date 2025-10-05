
import os, json, argparse
from pathlib import Path
import numpy as np

from src.core.rag import RAGStore
from src.core.rulebook import Rulebook
from src.core.socratic import apply as socratic_apply
from src.core.j_metrics import (
    j1_integration, j2_diversity, j3_robustness, j4_empowerment_infonce,
    j4_empowerment_proxy, j5_ahimsa_proxy, social_coherence
)
from src.core.teleology import Teleology
from src.core.validator import validate_yaml_file, ValidationError
from src.persona.persona import PersonaAgent
from src.ui.radar import plot_radar

def build_parser():
    parser = argparse.ArgumentParser(description="Geopolitik-Simulator V6.2")
    parser.add_argument('--scenario', type=str, default='data/scenarios/dayton_3party.yaml')
    parser.add_argument('--turns', type=int, default=5)
    parser.add_argument('--use_sde', action='store_true')
    parser.add_argument('--j5', type=str, default='neural', choices=['neural','aipw','ridge','proxy'], help='Ahimsa estimator (J5)')
    parser.add_argument('--ablation', type=str, default='none', choices=['none','H1','H2','H3'], help='Ablation: H1=Fixed Teleology, H2=No Ahimsa, H3=Static Reality')
    return parser

from src.ui.teleology_dashboard import plot_series
from src.core.sde_engine import HamiltonianSDE, SymplecticIntegrator

BASE = Path(__file__).parent
OUT = BASE / "out"
OUT.mkdir(exist_ok=True)

def load_scenario(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def _inline_citations_for_actor(store, actor_spec, k=2):
    cites = store.fetch_passages(actor_spec.get("corpus",""), k=k)
    if not cites:
        return ""
    return " " + "[" + ",".join("#"+c["id"] for c in cites) + "]"

def _extract_keywords(lines):
    txt = " ".join(lines)
    kws = []
    for k in ["Pufferzone","Inspektion","Inspektionsregime","Sanktionslockerungen","Snapback",
              "Grenzpatrouillen","Wirtschaft","Stationierung","Überwachung","Quoten","Notfallkanäle",
              "Navigationsfreiheit","Fischerei","Sicherheitszone"]:
        if k.lower() in txt.lower():
            kws.append(k)
    return list(dict.fromkeys(kws))

def make_compact_proposal(turn_text, rag, scenario):
    kws = _extract_keywords(turn_text)
    tracks = []
    if any(k in kws for k in ["Pufferzone","Inspektion","Inspektionsregime"]):
        tracks.append("Track 1: Pufferzone + internationales Inspektionsregime (Sequencing, Monitoring).")
    if any(k in kws for k in ["Sanktionslockerungen","Snapback","Wirtschaft"]):
        tracks.append("Track 2: Stufenweise Sanktionslockerungen ↔  Snapback bei Verstößen.")
    if any(k in kws for k in ["Grenzpatrouillen","Überwachung","Stationierung"]):
        tracks.append("Track 3: Gemeinsames Überwachungs-/Patrouillenkommando; keine Stationierung schwerer Waffen in Kernzonen.")
    if any(k in kws for k in ["Navigationsfreiheit","Fischerei","Sicherheitszone","Quoten","Notfallkanäle"]):
        tracks.append("Track 4: Navigationsfreiheit + saisonale Quoten & Notfallkanäle; definierte Sicherheitszonen ohne Militärübungen.")
    if not tracks:
        tracks.append("Track 1: Vertrauensbildende Maßnahmen + schrittweise gegenseitige Zugeständnisse.")
    ev = []
    for a in scenario["actors"]:
        ev += rag.fetch_passages(a.get("corpus",""), k=1)
    prop = "Kompaktvorschlag (destilliert):\n- " + "\n- ".join(tracks)
    if kws:
        prop += "\nStichworte: " + ", ".join(kws)
    if ev:
        prop += "\n\nQuellen (RAG):\n" + "\n".join([f"- {e['id']}: {e['text']}" for e in ev])
    return prop

def make_state(scn, rag):
    actors = []
    styles_export = []
    for a in scn["actors"]:
        passages = rag.query(a)
        ag = PersonaAgent(a, passages)
        actors.append(ag)
        styles_export.append({"id": ag.id, "name": ag.name, "style_spec": a.get('style'), "derived": ag.style})
    (OUT / "persona_styles.json").write_text(json.dumps(styles_export, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"actors": actors, "esc": scn.get("initial_state",{}).get("escalation","Tension"), "scenario": scn, "replay_A_cont": [], "replay_A_disc": [], "replay_C": [], "replay_harm": [] }


def _apply_ablation_preactors(state, actors):
    mode = state.get('ablation','none')
    if mode == 'H3':
        for ag in actors:
            if hasattr(ag, 'last_action_vec') and isinstance(ag.last_action_vec, np.ndarray):
                ag.last_action_vec[:] = 0.0
    return mode

def simulate_turn(state, rulebook, teleology, rag, stage3=True, emit_proposal=False, turn_idx=1, ripe_window=2, use_sde=False, sde=None, integ=None):
    actors = state["actors"]
    candidates = []
    for idx, ag in enumerate(actors):
        move = ag.policy(None)
        actor_spec = state["scenario"]["actors"][idx]
        ok, reason = rulebook.plausibility_mask(ag.id, {"core_interests": ag.core_interests}, move)
        cite = _inline_citations_for_actor(rag, actor_spec, k=1)
        line = f"{ag.name}: {move}{cite}" if ok else f"{ag.name}: [BLOCKED: {reason}]"
        candidates.append(line)
    # Socratic moderation
    candidates = socratic_apply(candidates, {"actors":[{"id":a.id, "name":a.name, "role":getattr(a,'role',None)} for a in actors]}, stage3=stage3 )
    # Simple hidden states for metrics
    H_list = [ag.hidden for ag in actors]
    J1 = j1_integration(np.vstack(H_list))
    J2 = j2_diversity(np.vstack(H_list))
    # j3 requires a scalar recon f; use small quadratic form
    def f_recon_scalar(z):
        return float(np.sum(z*z))
    J3 = j3_robustness(f_recon_scalar, np.mean(np.vstack(H_list), axis=0))
    # J4 via InfoNCE: craft toy A, S_next, S_cond from agent buffers
    actions = np.vstack([ag.last_action_vec for ag in actors])
    S_next = np.vstack([ag.observe_next() for ag in actors])
    S_cond = np.vstack([ag.observe_cond() for ag in actors])
    try:
        J4 = j4_empowerment_infonce(actions, S_next, S_cond)
    except Exception:
        J4 = j4_empowerment_proxy(S_next)  # fallback
    # J5 (causal with fallbacks)
    from src.core.j5_causal import compute_j5_with_fallbacks, build_context_features, action_from_actor
    if len(actors)>=2:
        # collect replay based on actor 0 -> actor 1 impact
        a_cont = action_from_actor(actors[0])
        a_disc = int(np.clip(np.round((a_cont.mean() - a_cont.min())/max(1e-6,(a_cont.max()-a_cont.min()))*2),0,2))
        C_feat = build_context_features(actors, state)
        harm_obs = -j5_ahimsa_proxy(actors[0].hidden, actors[1].hidden, actors[1].decoder)
        state["replay_A_cont"].append(a_cont); state["replay_A_disc"].append(a_disc)
        state["replay_C"].append(C_feat); state["replay_harm"].append(harm_obs)
        state["replay_A_cont"] = state["replay_A_cont"][-256:]
        state["replay_A_disc"] = state["replay_A_disc"][-256:]
        state["replay_C"] = state["replay_C"][-256:]
        state["replay_harm"] = state["replay_harm"][-256:]
        J5, fb = compute_j5_with_fallbacks(args.j5, state, actors, a_cont, a_disc, C_feat)
        if fb is not None:
            print(f'[J5] Fallback → {fb}')
    else:
        J5 = 0.0
    SCM = social_coherence(H_list)
    score = float(J1 + 0.2*J2 + 0.2*J3 + 0.2*J4 + 0.2*J5 + 0.2*SCM)

    turn_text = candidates
    ripe = teleology.is_ripe(J1,J2,J3,J4,J5,SCM, window=ripe_window)
    readiness = teleology.readiness(J1,J2,J3,J4,J5,SCM)
    proposal = make_compact_proposal(turn_text, rag, state["scenario"]) if (emit_proposal and ripe) else None

    # Optional SDE update of a global latent (toy)
    if use_sde and sde is not None and integ is not None:
        # embed a small phase variable into agents
        import torch
        q = torch.tensor(np.vstack(H_list)[:, :4], dtype=torch.float32)
        p = torch.zeros_like(q)
        for _ in range(2):
            q, p = integ.step(q, p, 1e-2)
        qn = q.numpy()
        for i, ag in enumerate(actors):
            ag.hidden[:4] = qn[i]

    data = {
        "turn": turn_idx,
        "turn_text": turn_text,
        "J": {"J1": J1, "J2": J2, "J3": J3, "J4": J4, "J5": J5},
        "SCM": SCM,
        "score": score,
        "ripe": ripe,
        "readiness": readiness,
        "proposal": proposal,
        "milestone_index": teleology.milestone_index,
        "milestone_total": teleology.milestone_total,
        "milestone_current": teleology.current_milestone()
    }
    (OUT / f"turn_{turn_idx:03d}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(BASE / "data/scenarios/dayton_3party.yaml"))
    parser.add_argument('--j5', type=str, default='neural', choices=['neural','aipw','ridge','proxy'], help='Ahimsa estimator (J5)')
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--no-stage3", action="store_true")
    ap.add_argument("--ripe-proposals", action="store_true")
    ap.add_argument("--use-sde", action="store_true")
    args = ap.parse_args()

    try:
        validate_yaml_file(args.scenario)
    except ValidationError as e:
        print("Scenario validation failed:", e)
        return

    rag = RAGStore(BASE / "data")
    scn = load_scenario(args.scenario)
    rulebook = Rulebook(scn)
    teleology = Teleology(scn)

    state = make_state(scn, rag)

    
    state['ablation'] = args.ablation
sde = integ = None
    if args.use_sde:
        import torch
        sde = HamiltonianSDE(q_dim=4, sigma=0.0)
        integ = SymplecticIntegrator(sde)

    series_J4, series_SCM = [], []
    for t in range(1, args.turns+1):
        data = simulate_turn(state, rulebook, teleology, rag, stage3=(not args.no_stage3), emit_proposal=args.ripe_proposals, turn_idx=t, use_sde=args.use_sde, sde=sde, integ=integ)
        series_J4.append(data["J"]["J4"]); series_SCM.append(data["SCM"])
        teleology.update_series(series_J4, series_SCM)

    # plots
    plot_radar(data["J"], data["SCM"], OUT / "radar.png")
    plot_series(series_J4, series_SCM, OUT / "teleology.png")

    print("Done. See 'out/' for artifacts.")

if __name__ == "__main__":
    main()