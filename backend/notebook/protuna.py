# %%
import os
import random

import numpy
import pandas
import seaborn as sns
from matplotlib import pyplot


def set_seeds(seed: int):
    assert seed > 0
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    numpy.random.seed(seed)


set_seeds(seed=42)

sns.set_theme(font="IPAexGothic", font_scale=2)

pyplot.rcParams["figure.figsize"] = (20, 10)


# %%
import optuna
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


# %%
class ModelBuilder(object):
    def __init__(self, df: pandas.DataFrame, freq: str = "W") -> None:
        assert freq in ["D", "W", "MS", "M"] + [
            "W-" + w for w in ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        ]
        self.df: pandas.DataFrame = df
        self.freq: str = freq

    def calc_changepoints(self):
        n_changepoints = len(self.df)
        # NOTE: self.freq に応じて、n_changepoints を定める
        #       - freq == "D": 1週間単位に変更
        #       - freq == "W": 3か月単位に変更
        #       - freq == "M": 1か月単位
        if self.freq[:1] == "D":
            n_changepoints //= 1 * 7
        elif self.freq[:1] == "W":
            n_changepoints //= 3 * 4
        elif self.freq[:1] == "M":
            # n_changepoints //= 3 * 1
            pass
        print(f"{n_changepoints=}")
        return n_changepoints

    def build_prophet_model(self, is_longterm=True, **params) -> Prophet:
        print(f"build_prophet_model: {is_longterm=} {params}")

        # model
        n_changepoints = self.calc_changepoints()
        model = Prophet(
            **params,
            n_changepoints=n_changepoints,
            changepoint_range=0.95,
            yearly_seasonality=4,
        )

        # 3 years
        model.add_seasonality(name="triennial", period=365.25 * 3, fourier_order=1)

        # 40 months
        model.add_seasonality(name="kitchen", period=365.25 / 12 * 40, fourier_order=1)

        # long term cycles
        if is_longterm:
            # 5 years
            model.add_seasonality(
                name="quinquennial", period=365.25 * 5, fourier_order=1
            )
            # juglar cycle 9-10 years
            # model.add_seasonality(name="juglar_09", period=365.25 * 9, fourier_order=1)
            model.add_seasonality(name="juglar_10", period=365.25 * 10, fourier_order=1)

        return model


# %%
class Evaluator(object):
    def __init__(
        self,
        df: pandas.DataFrame,
        n_horizon: int = 365.25,
        freq="3MS",
        horizon_scaler: float = 3,
    ) -> None:
        self.df: pandas.DataFrame = df.copy()
        self.n_horizon: int = n_horizon  # cv prediction range
        self.freq = freq  # cutoff freq
        self.horizon_scaler = horizon_scaler

    def objective_value(self, trial: optuna.Trial) -> float:
        params = {
            "growth": trial.suggest_categorical("growth", ["linear", "logistic"]),
            "changepoint_prior_scale": trial.suggest_float(
                "changepoint_prior_scale", 0.001, 10
            ),
            "seasonality_prior_scale": trial.suggest_float(
                "seasonality_prior_scale", 0.01, 10
            ),
            "seasonality_mode": trial.suggest_categorical(
                "seasonality_mode", ["additive", "multiplicative"]
            ),
        }
        cap_scaler = trial.suggest_float("cap_scaler", 0, 3)
        self.df["cap"] = self.df.y.max() + cap_scaler * self.df.y.std()

        mb = ModelBuilder(df=self.df)
        model: Prophet = mb.build_prophet_model(**params)
        model.fit(self.df)
        __df_cv, df_pm = self.run_cross_validation(model=model)
        n = df_pm.shape[0]
        # NOTE:
        #     - rmse, mae: horizon が長くなる(index が後になる)とエラー幅が増加するので、
        #             差に敏感な rmse の後半を多めに評価するように逆順で累積する
        #     - xxx : rmse と同じく、horizon が短い間も精度が高くないと困るので、
        #             前半を多めに評価するように mae を累積する
        score = (
            numpy.cumsum(df_pm["rmse"].values[::-1]).mean()
            + numpy.cumsum(df_pm["mae"].values[::-1]).mean()
            + numpy.cumsum(df_pm["smape"].values[::-1]).mean()
        )
        score /= n  # for intepretability of `score` in optuna.visualizaion

        return score

    def run_cross_validation(
        self, model: Prophet
    ) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        n_horizon = self.n_horizon
        date_start = self.df.ds.max() - pandas.Timedelta(
            days=n_horizon * self.horizon_scaler
        )
        date_end = self.df.ds.max() - pandas.Timedelta(days=n_horizon)
        cutoffs = pandas.date_range(start=date_start, end=date_end, freq=self.freq)

        # run cv and metrics
        df_cv = cross_validation(
            model,
            cutoffs=cutoffs,
            horizon=f"{n_horizon} days",
            parallel="processes",  # parallel=None,
        )
        df_pm = performance_metrics(df_cv)

        # store context
        self.date_start = date_start
        self.date_end = date_end
        self.cutoffs = cutoffs

        return df_cv, df_pm


# %%
class ProphetModelAnalyser(object):
    def __init__(self, model: Prophet, df: pandas.DataFrame) -> None:
        self.model: Prophet = model
        self.df: pandas.DataFrame = df

    def pickup_beta(self, component: str):
        (
            seasonal_features,
            _,
            component_cols,
            _,
        ) = self.model.make_all_seasonality_features(self.df)
        mask = component_cols[component].values
        beta_c: numpy.ndarray = self.model.params["beta"] * mask
        beta = [beta_c.ravel()[idx] for idx, v in enumerate(mask) if bool(v)]
        return beta


# %%
from dataclasses import dataclass


# %%
@dataclass
class BestEstimator(object):
    df: pandas.DataFrame
    model: Prophet
    evaluator: Evaluator
    study: optuna.Study
    df_cv: pandas.DataFrame | None = None
    df_pm: pandas.DataFrame | None = None
    future: pandas.DataFrame | None = None
    forecast: pandas.DataFrame | None = None


# %%
def convert_wareki_to_seireki(wareki_date):
    """
    和暦を西暦に変換する関数。
    """
    era = wareki_date[0]
    year, month, day = map(int, wareki_date[1:].split("."))

    if era == "S":  # 昭和
        seireki_year = 1925 + year
    elif era == "H":  # 平成
        seireki_year = 1988 + year
    elif era == "R":  # 令和
        seireki_year = 2018 + year
    else:
        raise ValueError(f"Unknown era: {era}")

    return f"{seireki_year}-{month:02}-{day:02}"


# %%
def optuna_visualization(study: optuna.Study):
    optuna.visualization.plot_contour(study).show()
    optuna.visualization.plot_edf(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_parallel_coordinate(study).show()
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_slice(study).show()
