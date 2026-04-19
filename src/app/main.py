# main.py
from src.config.config import Config
from src.app.runner import run

def main():
    config = Config(
        use_new=True,
        learn=False,
        nprob=1,
        ninit=5,
        iseed=42,
        method="BFGS",
        verbose=1,
        maxiter=10000,
        readmode=False,
        backprop=False,
    )

    run(config)

if __name__ == "__main__":
    main()
