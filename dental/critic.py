"""Reinforcement-learned critic for the dental specialist (synopsis Section 4.4).

Policy pi(a | s) with actions  a in {Accept, Reground, Reroute, Abstain}, at most K=3 steps, and reward

    R = 1[correct] + alpha*G - beta*n_steps - gamma*1[wrong and Accept] + delta*1[wrong-if-accepted and Abstain]

G = box IoU when gold boxes exist (grounding questions), with a box-area penalty against reward hacking.

The environment is built from OFFLINE rollouts (scripts/13_collect_rollouts.py). For every question it holds
three candidate answers:
  orig      the fine-tuned VLM's greedy answer (+ token confidences)
  reground  an answer rebuilt from the YOLO detectors' boxes (disease + tooth numbering)
  reroute   a second, sampled answer from the specialist (dental-only proxy; in the full multi-domain
            system Reroute sends the query to another specialist)
Reground / Reroute replace the current answer and cost one step; Accept / Abstain end the episode.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

import numpy as np

from .common import (DIAGNOSES, QUADRANT_NAME, boxes_to_answer, extract_all_labels, extract_label, iou,
                     parse_boxes, parse_number, parse_quadrants, parse_teeth, quadrant_from_position,
                     quadrant_of_tooth)

ACTIONS = ["accept", "reground", "reroute", "abstain"]
ACCEPT, REGROUND, REROUTE, ABSTAIN = range(4)
CAND_NAMES = ["orig", "reground", "reroute"]
QTYPES = ["presence", "which_teeth", "which_quadrant", "tooth_diagnosis", "caries_depth", "count_quadrant",
          "tooth_present", "ground_diagnosis", "ground_all", "ground_tooth", "absent"]
GROUND = {"ground_diagnosis", "ground_all", "ground_tooth"}
CLOSED = {"presence", "tooth_present", "caries_depth"}
GROUP = {"caries": {"caries", "deep caries"}, "deep caries": {"deep caries"},
         "periapical lesion": {"periapical lesion"}, "impacted": {"impacted"}}
QNAME_TO_Q = {v: k for k, v in QUADRANT_NAME.items()}

DEFAULT_REWARD = dict(alpha=0.5, beta=0.05, gamma=1.0, delta=0.5, area_frac=0.25, area_penalty=1.0)
DEFAULT_DET = dict(disease_conf=0.25, tooth_conf=0.5)


# =================================================================== scoring one answer
def _cap(s):
    return s[:1].upper() + s[1:]


def quadrant_of_question(row):
    q = row["question"].lower()
    for tok in q.replace("?", " ").replace(".", " ").split():
        if tok.isdigit() and int(tok) in (1, 2, 3, 4):
            return int(tok)
    for name, qq in QNAME_TO_Q.items():
        if name in q:
            return qq
    return None


def is_correct(row, text):
    qt, lab = row["qtype"], row["label"]
    if qt in CLOSED:
        return extract_label(text, row["options"]) == lab
    if qt == "absent":
        return extract_label(text, row["options"]) == "not present"
    if qt == "which_teeth":
        return set(parse_teeth(text)) == set(lab)
    if qt == "which_quadrant":
        return set(parse_quadrants(text)) == set(lab)
    if qt == "tooth_diagnosis":
        return set(extract_all_labels(text, DIAGNOSES)) == set(lab)
    if qt == "count_quadrant":
        n = parse_number(text)
        return n is not None and round(n) == lab
    if qt in GROUND:
        pred = [b for b, _ in parse_boxes(text)]
        gold = [g[0] for g in row["gold_boxes"]]
        if not gold:
            return not pred
        if not pred:
            return False
        rec = np.mean([any(iou(p, g) >= 0.5 for p in pred) for g in gold])
        prec = np.mean([any(iou(p, g) >= 0.5 for g in gold) for p in pred])
        return bool(rec >= 0.5 and prec >= 0.5)
    return False


def grounding_G(row, text, area_frac=0.25, area_penalty=1.0):
    """IoU-based grounding quality in [0,1] (minus an area penalty); 0 for non-grounding questions."""
    if row["qtype"] not in GROUND:
        return 0.0
    pred = [b for b, _ in parse_boxes(text)]
    gold = [g[0] for g in row["gold_boxes"]]
    if not gold:
        return 1.0 if not pred else 0.0
    g = float(np.mean([max((iou(p, gb) for p in pred), default=0.0) for gb in gold]))
    img = float(row["width"] * row["height"])
    big = max(((b[2] - b[0]) * (b[3] - b[1]) / img for b in pred), default=0.0)
    if big > area_frac:                       # safeguard: huge boxes are penalised
        g -= area_penalty * (big - area_frac)
    return g


def answer_acs(row, text):
    """Quadrant ACS of the boxes in an answer (None if no box names a tooth)."""
    sc = []
    for b, lab in parse_boxes(text):
        t = parse_teeth(lab) or ([row["tooth"]] if row.get("tooth") else [])
        if t:
            sc.append(float(quadrant_from_position(b, row["midline_x"], row["occlusal_y"]) == quadrant_of_tooth(t[0])))
    return float(np.mean(sc)) if sc else None


def answer_key(row, text):
    """Normalised form of an answer, used to check whether two candidates agree."""
    qt = row["qtype"]
    if qt in CLOSED or qt == "absent":
        return extract_label(text, row["options"])
    if qt == "which_teeth":
        return tuple(parse_teeth(text))
    if qt == "which_quadrant":
        return tuple(parse_quadrants(text))
    if qt == "tooth_diagnosis":
        return tuple(extract_all_labels(text, DIAGNOSES))
    if qt == "count_quadrant":
        n = parse_number(text)
        return None if n is None else round(n)
    if qt in GROUND:
        return [b for b, _ in parse_boxes(text)]
    return text.strip().lower()


def agree(row, a, b):
    if a is None or b is None:
        return 0.0
    if row["qtype"] in GROUND:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        m = np.mean([max(iou(x, y) for y in b) for x in a])
        return float(m >= 0.5)
    return float(a == b)


# =================================================================== detector-derived answers
def assign_tooth(box, teeth):
    """Tooth number of the tooth box that best covers a disease box (None if no teeth detected)."""
    best, bt = 0.0, None
    for tb, t, _ in teeth:
        v = iou(box, tb)
        if v > best:
            best, bt = v, t
    if bt is None and teeth:                      # fall back to nearest tooth centre
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        bt = min(teeth, key=lambda x: ((x[0][0] + x[0][2]) / 2 - cx) ** 2 + ((x[0][1] + x[0][3]) / 2 - cy) ** 2)[1]
    return bt


def detector_answer(row, det, cfg=DEFAULT_DET):
    """Answer rebuilt from YOLO detections, phrased like the training answers. Returns (text, conf) or (None, 0)."""
    if det is None:
        return None, 0.0
    teeth = [(b, int(t), c) for b, t, c in det.get("teeth", []) if c >= cfg["tooth_conf"]]
    dis = [(b, c, conf, assign_tooth(b, teeth)) for b, c, conf in det.get("disease", []) if conf >= cfg["disease_conf"]]
    present = {t for _, t, _ in teeth}
    qt, dx, tooth = row["qtype"], row.get("diagnosis"), row.get("tooth")
    conf_of = lambda xs: float(max((x for x in xs), default=0.0))  # noqa: E731

    if qt == "presence":
        hit = [d for d in dis if d[1] in GROUP[dx]]
        return ("Yes." if hit else "No."), conf_of(d[2] for d in hit) if hit else 1 - conf_of(
            c for b, cl, c in det.get("disease", []) if cl in GROUP[dx])
    if qt in ("which_teeth", "which_quadrant", "ground_diagnosis"):
        hit = [d for d in dis if d[1] in GROUP[dx]]
        if qt == "ground_diagnosis":
            items = [(d[0], f"{d[1]} {d[3]}" if d[3] else d[1]) for d in hit]
            return (boxes_to_answer(items) if items else "None found."), conf_of(d[2] for d in hit)
        teeth_hit = sorted({d[3] for d in hit if d[3]})
        if not teeth_hit:
            return "None found.", 0.0
        if qt == "which_teeth":
            return ", ".join(map(str, teeth_hit)) + ".", conf_of(d[2] for d in hit)
        qs = sorted({t // 10 for t in teeth_hit})
        return _cap(", ".join(f"quadrant {q} ({QUADRANT_NAME[q]})" for q in qs)) + ".", conf_of(d[2] for d in hit)
    if qt == "ground_all":
        items = [(d[0], f"{d[1]} {d[3]}" if d[3] else d[1]) for d in dis]
        return (boxes_to_answer(items) if items else "None found."), conf_of(d[2] for d in dis)
    if qt in ("tooth_diagnosis", "caries_depth"):
        on = [d for d in dis if d[3] == tooth]
        if qt == "tooth_diagnosis":
            names = sorted({d[1] for d in on})
            return (_cap(" and ".join(names)) + "." if names else "None found."), conf_of(d[2] for d in on)
        car = [d for d in on if d[1] in ("caries", "deep caries")]
        if not car:
            return None, 0.0
        return ("Deep." if any(d[1] == "deep caries" for d in car) else "Not deep."), conf_of(d[2] for d in car)
    if qt == "tooth_present":
        c = conf_of(c for b, t, c in teeth if t == tooth)
        return ("Yes." if tooth in present else f"No, tooth {tooth} is not present."), c
    if qt == "count_quadrant":
        q = quadrant_of_question(row)
        if q is None:
            return None, 0.0
        n = sum(1 for t in present if t // 10 == q)
        return f"{n} teeth.", float(np.mean([c for b, t, c in teeth if t // 10 == q] or [0.0]))
    if qt == "ground_tooth":
        bx = [(b, c) for b, t, c in teeth if t == tooth]
        if not bx:
            return f"Tooth {tooth} is not present.", 0.0
        b, c = max(bx, key=lambda x: x[1])
        return boxes_to_answer([(b, f"tooth {tooth}")]), c
    if qt == "absent":
        if tooth in present:
            return f"Tooth {tooth} is present.", conf_of(c for b, t, c in teeth if t == tooth)
        return f"Tooth {tooth} is not present in this radiograph.", 1.0
    return None, 0.0


# =================================================================== building the environment
def build_items(rollouts, detections, rcfg=DEFAULT_REWARD, dcfg=DEFAULT_DET):
    """One item per question: three candidates with correctness, G and features."""
    items = []
    for r in rollouts:
        if r["qtype"] not in QTYPES:
            continue
        det = detections.get(r["image_id"])
        cands = {}
        for name in ("orig", "reroute"):
            c = r["cands"].get(name)
            if c:
                cands[name] = dict(text=c["text"], conf=[c["conf_first"], c["conf_mean"], c["conf_min"]])
        dtext, dconf = detector_answer(r, det, dcfg)
        if dtext is not None:
            cands["reground"] = dict(text=dtext, conf=[dconf, dconf, dconf])
        if "orig" not in cands:
            continue
        for c in cands.values():
            c["correct"] = bool(is_correct(r, c["text"]))
            c["G"] = grounding_G(r, c["text"], rcfg["area_frac"], rcfg["area_penalty"])
            c["key"] = answer_key(r, c["text"])
            c["acs"] = answer_acs(r, c["text"])
            bxs = [b for b, _ in parse_boxes(c["text"])]
            img = float(r["width"] * r["height"])
            c["n_boxes"] = len(bxs)
            c["max_area"] = max(((b[2] - b[0]) * (b[3] - b[1]) / img for b in bxs), default=0.0)
        names = list(cands)
        ag = {(a, b): agree(r, cands[a]["key"], cands[b]["key"]) for a in names for b in names if a != b}
        items.append(dict(row=r, cands=cands, agree=ag))
    return items


def state_features(item, cur, step, used, K=3):
    r, cands = item["row"], item["cands"]
    c = cands[cur]
    f = [float(r["qtype"] == q) for q in QTYPES]
    f += [r["level"] / 5.0]
    f += [float(cur == n) for n in CAND_NAMES]
    f += list(c["conf"])
    f += [c["acs"] if c["acs"] is not None else 0.0, float(c["acs"] is None)]
    f += [min(c["n_boxes"], 10) / 10.0, c["max_area"]]
    for other in CAND_NAMES:
        if other == cur:
            continue
        has = other in cands
        f += [float(has), item["agree"].get((cur, other), 0.0) if has else 0.0]
        f += [cands[other]["conf"][0] if has else 0.0]
    f += [step / K, float(used[REGROUND]), float(used[REROUTE])]
    return np.asarray(f, dtype=np.float32)


def action_mask(item, step, used, K=3):
    m = np.ones(4, dtype=bool)
    if step >= K - 1:                               # last step: must stop
        m[REGROUND] = m[REROUTE] = False
    if used[REGROUND] or "reground" not in item["cands"]:
        m[REGROUND] = False
    if used[REROUTE] or "reroute" not in item["cands"]:
        m[REROUTE] = False
    return m


def terminal_reward(item, cur, action, n_steps, rc):
    c = item["cands"][cur]
    wrong = not c["correct"]
    r = -rc["beta"] * n_steps
    if action == ACCEPT:
        r += float(c["correct"]) + rc["alpha"] * c["G"] - rc["gamma"] * float(wrong)
    else:                                            # abstain; delta < 1 caps the abstention reward
        r += rc["delta"] * float(wrong)
    return r


def run_episode(item, choose, rc, K=3):
    """choose(state_vec, mask, item, cur, step, used) -> action. Returns the episode record."""
    cur, step, used, traj = "orig", 0, [False] * 4, []
    while True:
        s = state_features(item, cur, step, used, K)
        m = action_mask(item, step, used, K)
        a = choose(s, m, item, cur, step, used)
        traj.append((s, m, a))
        if a in (ACCEPT, ABSTAIN):
            R = terminal_reward(item, cur, a, step, rc)
            return dict(traj=traj, R=R, final=cur, action=a, steps=step)
        used[a] = True
        cur = "reground" if a == REGROUND else "reroute"
        step += 1


# =================================================================== evaluation
def evaluate(items, choose, rc, K=3):
    eps = [run_episode(it, choose, rc, K) for it in items]
    n = len(eps)
    acc = [e for e in eps if e["action"] == ACCEPT]
    abst = [e for e in eps if e["action"] == ABSTAIN]
    acc_ok = sum(it["cands"][e["final"]]["correct"] for it, e in zip(items, eps) if e["action"] == ACCEPT)
    orig_wrong = sum(not it["cands"]["orig"]["correct"] for it in items)
    acc_wrong = sum((not it["cands"][e["final"]]["correct"]) for it, e in zip(items, eps) if e["action"] == ACCEPT)
    ab_prec = [not it["cands"][e["final"]]["correct"] for it, e in zip(items, eps) if e["action"] == ABSTAIN]
    absent = [(it, e) for it, e in zip(items, eps) if it["row"]["qtype"] == "absent"]
    ground = [(it, e) for it, e in zip(items, eps) if it["row"]["qtype"] in GROUND and e["action"] == ACCEPT]
    res = dict(
        n=n,
        mean_reward=float(np.mean([e["R"] for e in eps])),
        accuracy_overall=acc_ok / n,                         # abstentions count as not answered
        coverage=len(acc) / n,
        selective_accuracy=acc_ok / max(len(acc), 1),         # accuracy on answered questions
        abstention_rate=len(abst) / n,
        abstention_precision=float(np.mean(ab_prec)) if ab_prec else None,   # abstained on answers that were wrong
        missed_error_rate=acc_wrong / max(orig_wrong, 1),     # wrong answers that got through / original errors
        wrong_accepted_rate=acc_wrong / n,
        mean_steps=float(np.mean([e["steps"] for e in eps])),
        reground_rate=float(np.mean([any(t[2] == REGROUND for t in e["traj"]) for e in eps])),
        reroute_rate=float(np.mean([any(t[2] == REROUTE for t in e["traj"]) for e in eps])),
        hallucination_rate=(float(np.mean([e["action"] == ACCEPT and not it["cands"][e["final"]]["correct"]
                                            for it, e in absent])) if absent else None),
        mean_G_accepted_grounding=(float(np.mean([it["cands"][e["final"]]["G"] for it, e in ground]))
                                   if ground else None),
    )
    by_q = defaultdict(list)
    for it, e in zip(items, eps):
        by_q[it["row"]["qtype"]].append(e["action"] == ACCEPT and it["cands"][e["final"]]["correct"])
    res["accuracy_by_qtype"] = {k: float(np.mean(v)) for k, v in sorted(by_q.items())}
    return res, eps


# =================================================================== baseline critics
def policy_no_critic(s, m, item, cur, step, used):
    return ACCEPT


def policy_tools_rule(s, m, item, cur, step, used):
    """'VLM + tools' fixed pipeline (MedRAX-style): grounding questions always use the detector's boxes."""
    if item["row"]["qtype"] in GROUND and m[REGROUND] and cur == "orig":
        return REGROUND
    return ACCEPT


def make_rule_policy(tau_abstain, tau_acs):
    """Hand-written critic: abstain on low confidence; reground grounding answers with poor ACS."""
    def pol(s, m, item, cur, step, used):
        c = item["cands"][cur]
        if (cur == "orig" and m[REGROUND] and item["row"]["qtype"] in GROUND
                and (c["acs"] is None or c["acs"] < tau_acs)):
            return REGROUND
        if c["conf"][0] < tau_abstain:
            return ABSTAIN
        return ACCEPT
    return pol


def tune_rule(items, rc, K=3):
    best = None
    for ta in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        for tc in [0.0, 0.5, 0.75, 1.01]:
            r, _ = evaluate(items, make_rule_policy(ta, tc), rc, K)
            if best is None or r["mean_reward"] > best[0]:
                best = (r["mean_reward"], ta, tc)
    return dict(tau_abstain=best[1], tau_acs=best[2])


# =================================================================== supervised (SFT-style) critic
class SupervisedCritic:
    """Predicts P(correct) and G for every candidate from its features (trained on rollouts), then picks
    the option with the highest expected reward. This is the supervised counterpart of the RL critic."""

    def __init__(self, rc, K=3):
        self.rc, self.K = rc, K

    def _feat(self, item, name):
        used = [False] * 4
        return state_features(item, name, 0, used, self.K)

    def fit(self, items, epochs=300, lr=0.05, seed=0):
        import torch
        X, y, g = [], [], []
        for it in items:
            for name, c in it["cands"].items():
                X.append(self._feat(it, name))
                y.append(float(c["correct"]))
                g.append(c["G"])
        X = torch.tensor(np.stack(X))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = (X - self.mu) / self.sd
        y, g = torch.tensor(y), torch.tensor(g, dtype=torch.float32)
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr / 10)
        for _ in range(epochs):
            out = self.net(X)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out[:, 0], y) + \
                torch.nn.functional.mse_loss(out[:, 1], g)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def predict(self, item, name):
        import torch
        x = (torch.tensor(self._feat(item, name)) - self.mu) / self.sd
        with torch.no_grad():
            o = self.net(x[None])[0]
        return float(torch.sigmoid(o[0])), float(o[1])

    def policy(self):
        rc = self.rc

        def value(item, name, steps):
            p, g = self.predict(item, name)
            acc = p + rc["alpha"] * g - rc["gamma"] * (1 - p)
            ab = rc["delta"] * (1 - p)
            return max(acc, ab) - rc["beta"] * steps, (ACCEPT if acc >= ab else ABSTAIN)

        def pol(s, m, item, cur, step, used):
            here, act = value(item, cur, step)
            best, choice = here, act
            for a, name in ((REGROUND, "reground"), (REROUTE, "reroute")):
                if m[a]:
                    v, _ = value(item, name, step + 1)
                    if v > best:
                        best, choice = v, a
            return choice
        return pol


# =================================================================== PPO critic
class PPOCritic:
    def __init__(self, n_in, rc, K=3, hidden=64, seed=0):
        import torch
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.rc, self.K = rc, K
        self.pi = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.Tanh(),
                                      torch.nn.Linear(hidden, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 4))
        self.v = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.Tanh(),
                                     torch.nn.Linear(hidden, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        self.mu = torch.zeros(n_in)
        self.sd = torch.ones(n_in)

    def set_norm(self, items):
        import torch
        S = []
        for it in items[:5000]:
            for cur in it["cands"]:
                for step in range(self.K):
                    S.append(state_features(it, cur, step, [False] * 4, self.K))
        S = torch.tensor(np.stack(S))
        self.mu, self.sd = S.mean(0), S.std(0) + 1e-6

    def _logits(self, S, M):
        import torch
        lg = self.pi((S - self.mu) / self.sd)
        return lg.masked_fill(~M, -1e9)

    def act(self, s, m, greedy):
        import torch
        S, M = torch.tensor(s)[None], torch.tensor(m)[None]
        with torch.no_grad():
            lg = self._logits(S, M)[0]
        if greedy:
            return int(lg.argmax())
        return int(torch.distributions.Categorical(logits=lg).sample())

    def policy(self, greedy=True):
        return lambda s, m, item, cur, step, used: self.act(s, m, greedy)

    def train(self, items, val_items=None, iters=150, batch=1024, epochs=4, minibatch=256, lr=3e-4,
              clip=0.2, ent=0.01, log=print, eval_every=10):
        import torch
        self.set_norm(items)
        opt = torch.optim.Adam(list(self.pi.parameters()) + list(self.v.parameters()), lr=lr)
        best, best_state, hist = -1e9, None, []
        for it_n in range(1, iters + 1):
            S, M, A, LP, RET = [], [], [], [], []
            for item in random.sample(items, min(batch, len(items))):
                ep = run_episode(item, self.policy(greedy=False), self.rc, self.K)
                for s, m, a in ep["traj"]:          # undiscounted, reward only at the end of an episode
                    S.append(s); M.append(m); A.append(a); RET.append(ep["R"])
            S, M = torch.tensor(np.stack(S)), torch.tensor(np.stack(M))
            A, RET = torch.tensor(A), torch.tensor(RET, dtype=torch.float32)
            with torch.no_grad():
                LP = torch.distributions.Categorical(logits=self._logits(S, M)).log_prob(A)
                V = self.v((S - self.mu) / self.sd).squeeze(-1)
            ADV = RET - V
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            n = len(A)
            for _ in range(epochs):
                perm = torch.randperm(n)
                for i in range(0, n, minibatch):
                    j = perm[i:i + minibatch]
                    dist = torch.distributions.Categorical(logits=self._logits(S[j], M[j]))
                    ratio = torch.exp(dist.log_prob(A[j]) - LP[j])
                    pl = -torch.min(ratio * ADV[j], torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[j]).mean()
                    vl = torch.nn.functional.mse_loss(self.v((S[j] - self.mu) / self.sd).squeeze(-1), RET[j])
                    loss = pl + 0.5 * vl - ent * dist.entropy().mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            if val_items is not None and (it_n % eval_every == 0 or it_n == iters):
                r, _ = evaluate(val_items, self.policy(greedy=True), self.rc, self.K)
                hist.append(dict(iter=it_n, val_reward=r["mean_reward"], val_coverage=r["coverage"],
                                 val_selective_acc=r["selective_accuracy"]))
                log(f"  iter {it_n:4d}  val reward {r['mean_reward']:.4f}  coverage {r['coverage']:.3f}  "
                    f"selective acc {r['selective_accuracy']:.3f}  steps {r['mean_steps']:.2f}")
                if r["mean_reward"] > best:            # keep the best policy on validation
                    best = r["mean_reward"]
                    best_state = ({k: v.clone() for k, v in self.pi.state_dict().items()},
                                  {k: v.clone() for k, v in self.v.state_dict().items()})
        if best_state:
            self.pi.load_state_dict(best_state[0])
            self.v.load_state_dict(best_state[1])
        return hist

    def save(self, path):
        import torch
        torch.save(dict(pi=self.pi.state_dict(), v=self.v.state_dict(), mu=self.mu, sd=self.sd, rc=self.rc,
                        K=self.K), path)


def oracle_policy(s, m, item, cur, step, used):
    """Upper bound: accepts whichever reachable candidate is correct, else abstains."""
    cands = item["cands"]
    if cands[cur]["correct"]:
        return ACCEPT
    for a, name in ((REGROUND, "reground"), (REROUTE, "reroute")):
        if m[a] and cands[name]["correct"]:
            return a
    return ABSTAIN


def load_rollouts(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows


def load_detections(path):
    return json.load(open(path))
