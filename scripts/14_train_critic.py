"""Train and evaluate the critics on the offline rollouts (synopsis 4.4 and Table 5 'critic' ablation).

Critics compared on the test split:
  no_critic   always accept the specialist's answer
  tools_rule  'VLM + tools' fixed pipeline: grounding answers always take the detector's boxes (MedRAX-style)
  rule        hand-written thresholds on confidence and ACS (tuned on validation)
  supervised  network predicting correctness/G of each option, picks the best expected reward (SFT-style)
  rl_ppo      PPO policy over state features (the proposed critic)
  oracle      upper bound

CPU only.
  python scripts/14_train_critic.py                      # critic trained on the train-split rollouts
  python scripts/14_train_critic.py --train_on val       # trained on held-out validation rollouts
  python scripts/14_train_critic.py --beta_sweep 0 0.05 0.1 0.2   # accuracy-cost curve
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.critic import (DEFAULT_REWARD, PPOCritic, SupervisedCritic, build_items, evaluate,  # noqa: E402
                           load_detections, load_rollouts, make_rule_policy, oracle_policy, policy_no_critic,
                           policy_tools_rule, state_features, tune_rule)

SHOW = ["accuracy_overall", "coverage", "selective_accuracy", "abstention_rate", "abstention_precision",
        "missed_error_rate", "hallucination_rate", "mean_G_accepted_grounding", "mean_steps", "reground_rate",
        "reroute_rate", "mean_reward"]


def load_split(d, split, rc):
    return build_items(load_rollouts(Path(d) / f"{split}.jsonl"),
                       load_detections(Path(d) / f"{split}_detections.json"), rc)


def split_by_image(items, frac, seed):
    ids = sorted({it["row"]["image_id"] for it in items})
    random.Random(seed).shuffle(ids)
    a = set(ids[:int(len(ids) * frac)])
    return [it for it in items if it["row"]["image_id"] in a], [it for it in items if it["row"]["image_id"] not in a]


def table(results):
    names = list(results)
    print("%-28s" % "metric" + "".join("%13s" % n for n in names))
    for k in SHOW:
        cells = []
        for n in names:
            v = results[n].get(k)
            cells.append("%13s" % ("-" if v is None else "%.3f" % v))
        print("%-28s" % k + "".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="rollouts")
    ap.add_argument("--train_on", choices=["train", "val"], default="train")
    ap.add_argument("--alpha", type=float, default=DEFAULT_REWARD["alpha"])
    ap.add_argument("--beta", type=float, default=DEFAULT_REWARD["beta"])
    ap.add_argument("--gamma", type=float, default=DEFAULT_REWARD["gamma"])
    ap.add_argument("--delta", type=float, default=DEFAULT_REWARD["delta"])
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--beta_sweep", type=float, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/critic")
    args = ap.parse_args()
    assert args.delta < 1, "delta must be < 1 (capped abstention reward)"
    rc = dict(DEFAULT_REWARD, alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    val_all = load_split(args.rollouts, "val", rc)
    test = load_split(args.rollouts, "test", rc)
    if args.train_on == "train":
        train, val = load_split(args.rollouts, "train", rc), val_all
    else:                       # train on 70% of validation images, select on the other 30%
        train, val = split_by_image(val_all, 0.7, args.seed)
    print(f"items: train {len(train)}  val {len(val)}  test {len(test)}  (critic trained on '{args.train_on}')")
    acc0 = sum(it["cands"]["orig"]["correct"] for it in test) / len(test)
    print(f"specialist accuracy on these test questions: {acc0:.3f}")

    results, K = {}, args.K
    results["no_critic"], _ = evaluate(test, policy_no_critic, rc, K)
    results["tools_rule"], _ = evaluate(test, policy_tools_rule, rc, K)
    th = tune_rule(val, rc, K)
    print("rule thresholds (tuned on validation):", th)
    results["rule"], _ = evaluate(test, make_rule_policy(th["tau_abstain"], th["tau_acs"]), rc, K)
    sup = SupervisedCritic(rc, K).fit(train, seed=args.seed)
    results["supervised"], _ = evaluate(test, sup.policy(), rc, K)
    n_in = len(state_features(train[0], "orig", 0, [False] * 4, K))
    ppo = PPOCritic(n_in, rc, K, seed=args.seed)
    print("training PPO critic ...")
    hist = ppo.train(train, val, iters=args.iters)
    ppo.save(out / f"ppo_critic_{args.train_on}.pt")
    results["rl_ppo"], eps = evaluate(test, ppo.policy(greedy=True), rc, K)
    results["oracle"], _ = evaluate(test, oracle_policy, rc, K)

    print("\n== Critic comparison on the test split")
    table(results)
    print("\nPPO accuracy by question type (accepted and correct):")
    for k, v in results["rl_ppo"]["accuracy_by_qtype"].items():
        print(f"  {k:18s} no critic {results['no_critic']['accuracy_by_qtype'][k]:.3f}   PPO {v:.3f}")

    sweep = []
    for b in args.beta_sweep:          # accuracy-cost curve: how the policy trades steps for accuracy
        rcb = dict(rc, beta=b)
        p = PPOCritic(n_in, rcb, K, seed=args.seed)
        p.train(train, val, iters=args.iters, log=lambda *a: None)
        r, _ = evaluate(test, p.policy(greedy=True), rcb, K)
        sweep.append(dict(beta=b, **{k: r[k] for k in SHOW}))
        print(f"beta={b:<5} accuracy {r['accuracy_overall']:.3f}  selective {r['selective_accuracy']:.3f}  "
              f"coverage {r['coverage']:.3f}  steps {r['mean_steps']:.2f}")

    json.dump(dict(reward=rc, K=K, train_on=args.train_on, rule_thresholds=th, results=results,
                   ppo_history=hist, beta_sweep=sweep),
              open(out / f"critic_results_{args.train_on}.json", "w"), indent=2)
    with open(out / f"ppo_decisions_{args.train_on}.jsonl", "w") as f:
        for it, e in zip(test, eps):
            f.write(json.dumps(dict(qid=it["row"]["qid"], qtype=it["row"]["qtype"],
                                    actions=[["accept", "reground", "reroute", "abstain"][t[2]] for t in e["traj"]],
                                    final=e["final"], correct_if_accepted=it["cands"][e["final"]]["correct"],
                                    reward=e["R"])) + "\n")
    print(f"\nsaved {out}/critic_results_{args.train_on}.json")


if __name__ == "__main__":
    main()
