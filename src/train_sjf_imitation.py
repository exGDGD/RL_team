from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.rl import collect_sjf_examples
from src.train_acac import evaluate_policy, make_env, serialize_args


IMITATION_CHECKPOINT_VERSION = "sjf_actor_imitation_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the actor to imitate SJF decisions.")
    parser.add_argument("--train-episodes", type=int, default=128)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--episode-time", type=float, default=40.0)
    parser.add_argument("--max-tasks", type=int, default=32)
    parser.add_argument("--progress-work", type=float, default=0.0)
    parser.add_argument("--completion", type=float, default=0.0)
    parser.add_argument("--completion-work", type=float, default=0.0)
    parser.add_argument("--lambda-energy", type=float, default=0.1)
    parser.add_argument("--lambda-starvation", type=float, default=0.05)
    parser.add_argument("--lambda-latency", type=float, default=0.5)
    parser.add_argument("--starvation-max-wait-weight", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sjf_imitation/actors.pt"),
    )
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F

        from src.rl.trainer import ACACConfig, TorchACACPolicy
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is required for imitation training. Run this script in Colab."
            ) from exc
        raise

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = TorchACACPolicy(
        ACACConfig(
            hidden_dim=args.hidden_dim,
            actor_learning_rate=args.learning_rate,
        ),
        device=args.device,
    )
    optimizer = torch.optim.Adam(policy.actors.parameters(), lr=args.learning_rate)
    train_examples = collect_dataset(
        args,
        seeds=range(args.seed, args.seed + args.train_episodes),
    )
    validation_examples = collect_dataset(
        args,
        seeds=range(args.eval_seed, args.eval_seed + args.validation_episodes),
    )
    if not train_examples or not validation_examples:
        raise SystemExit("SJF imitation dataset is empty. Increase workload density.")

    print(
        "SJF actor imitation "
        f"train_examples={len(train_examples)} "
        f"validation_examples={len(validation_examples)}"
    )
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            policy=policy,
            optimizer=optimizer,
            examples=train_examples,
            batch_size=args.batch_size,
            rng=rng,
            torch=torch,
            F=F,
        )
        train_accuracy = imitation_accuracy(policy, train_examples, torch=torch)
        validation_accuracy = imitation_accuracy(policy, validation_examples, torch=torch)
        print(
            f"epoch={epoch} loss={loss:.4f} "
            f"train_accuracy={train_accuracy:.3f} "
            f"validation_accuracy={validation_accuracy:.3f}"
        )

    summary = evaluate_policy(
        policy,
        args,
        base_seed=args.eval_seed,
        episodes=args.eval_episodes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": IMITATION_CHECKPOINT_VERSION,
            "actors_state_dict": policy.actors.state_dict(),
            "model_state_dict": policy.state_dict(),
            "config": asdict(policy.config),
            "args": serialize_args(args),
            "validation_accuracy": imitation_accuracy(
                policy,
                validation_examples,
                torch=torch,
            ),
            "eval_summary": summary,
        },
        args.output,
    )
    print(
        f"saved actors={args.output} "
        f"eval_reward={summary['reward']:.3f} "
        f"sampled_eval_reward={summary['sampled']['reward']:.3f} "
        f"sjf_reward={summary['baselines']['sjf_like']['reward']:.3f}"
    )


def collect_dataset(args: argparse.Namespace, *, seeds) -> list:
    examples = []
    for seed in seeds:
        examples.extend(
            collect_sjf_examples(
                make_env(args, seed=seed),
                seed=seed,
            )
        )
    return examples


def train_epoch(*, policy, optimizer, examples, batch_size: int, rng, torch, F) -> float:
    policy.train()
    indices = rng.permutation(len(examples))
    losses = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        logits = torch.stack(
            [
                policy.imitation_logits(
                    batch=examples[index].obs,
                    agent_index=examples[index].agent_index,
                    action_mask=examples[index].action_mask,
                )
                for index in batch_indices
            ]
        )
        targets = torch.tensor(
            [examples[index].action for index in batch_indices],
            dtype=torch.long,
            device=policy.device,
        )
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def imitation_accuracy(policy, examples, *, torch) -> float:
    policy.eval()
    correct = 0
    with torch.no_grad():
        for example in examples:
            logits = policy.imitation_logits(
                batch=example.obs,
                agent_index=example.agent_index,
                action_mask=example.action_mask,
            )
            correct += int(int(torch.argmax(logits).item()) == example.action)
    return correct / max(len(examples), 1)


if __name__ == "__main__":
    main()
