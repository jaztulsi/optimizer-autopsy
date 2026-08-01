"""Proxy smoke test: end-to-end tiny run to prove the pipeline wires together.

Trunk -> induce a spike -> snapshot -> localize -> repair -> fork -> eval, on the 1-3M proxy.
Runs in minutes on a free T4. This is the keep/kill signal (BUILD_PLAN §2 Phase 2).
"""

# TODO: main() -> load proxy config, run the full trunk->fork->eval loop on one induced spike,
#       assert the localizer beats the naive reset baseline (else fail loudly).

if __name__ == "__main__":
    raise SystemExit("TODO: implement proxy smoke test")
