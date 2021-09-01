import os
import numpy as np
import optuna
import wandb


class Objective(object):
    def __init__(self, seed):
        # Setting the seed to always get the same regression.
        np.random.seed(seed)

        # Building the data generating process.
        self.nobs = 1000
        self.epsilon = np.random.uniform(size=(self.nobs, 1))
        self.real_beta = 5.0
        self.real_alpha = 1.0 / 5.0
        self.X1 = np.random.normal(loc=10, scale=3, size=(self.nobs, 1))
        self.X2 = np.random.normal(loc=10, scale=3, size=(self.nobs, 1))
        self.y = self.X1 * self.real_alpha + \
                 self.X2 * self.real_beta + \
                 self.epsilon


    def __call__(self, trial):
        # Parameters.
        trial_alpha = trial.suggest_uniform("alpha", low=-10, high=10)
        trial_beta = trial.suggest_uniform("beta", low=-10, high=10)

        # Starting WandrB run.
        config = {"trial_alpha": trial_alpha,
                  "trial_beta": trial_beta}
        run = wandb.init(project="optuna",
                         name=f"trial_",
                         group="sampling",
                         config=config,
                         reinit=True)

        # Prediction and loss.
        y_hat = self.X1 * trial_alpha + self.X2 * trial_beta
        mse = ((self.y - y_hat) ** 2).mean()

        # WandB logging.
        with run:
            run.log({"mse": mse}, step=trial.number)

        return mse


def main():
    # Execute an optimization by using an `Objective` instance.
    black_box = Objective(seed=4444)
    sampler = optuna.samplers.TPESampler(seed=4444)
    study = optuna.create_study(direction="minimize",
                                sampler=sampler)
    study.optimize(black_box,
                   n_trials=100,
                   show_progress_bar=True)
    print(f"True alpha: {black_box.real_alpha}")
    print(f"True beta: {black_box.real_beta}")
    print(f"Best params: {study.best_params}")

    # Create the summary run.
    summary = wandb.init(project="optuna",
                         name="summary",
                         job_type="logging")

    # Getting the study trials.
    trials = study.trials

    # WandB summary.
    for step, trial in enumerate(trials):
        # Logging the loss.
        summary.log({"mse": trial.value}, step=step)

        # Logging the parameters.
        for k, v in trial.params.items():
            summary.log({k: v}, step=step)


if __name__ == "__main__":
    main()
