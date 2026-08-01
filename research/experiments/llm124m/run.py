"""124M run driver: trunk training + spike-window short forks on Kaggle.

Budgeted for free compute: one command == one Kaggle session's worth of work (<12h).
Full-convergence forks are credits-gated and not launched here.
"""

from research.harness.preflight import check_env


def main():
    check_env()  # every entrypoint verifies the env + determinism before touching CUDA.
    # TODO: parse cmd ("trunk" | "forks"), load llm124m config, run the requested stage,
    #       push snapshots/logs to HF Hub + wandb.
    raise SystemExit("TODO: implement 124M run driver")


if __name__ == "__main__":
    main()
