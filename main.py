import data_processing as dp
import ada_boost as ab
import xg_boost as xgb
import evaluator as ev
import boostInterface as bi
from nicegui import ui
from matplotlib import pyplot as plt
from pathlib import Path

class main:
    def __init__(self):
        self.value = "Hello, World!"
        self.path = Path.cwd() / f"Assets"
        self.path.mkdir(parents=True, exist_ok=True)

    def run_ui(self):

        @ui.page("/")
        def main_page():
            ui.page_title("PMA Mini Project- Group 5")

            with ui.left_drawer(top_corner=False, bottom_corner=False, bordered=True, elevated=True) as drawer:
                ui.label("Default Parameter")
                ui.label("Optimal Parameter Using Grid Search CV")
        
            with ui.header(wrap=False, value=True):
                ui.button('☰', on_click=drawer.toggle).classes('button--flat')
                ui.label('AdaBoost and XGBoost Mini Project - Group 5').style('font-weight: bold; font-size: 25pt; margin-left: 16pt;')

            with ui.tabs() as tab:
                default_param = ui.tab("Default Parameter")
                optimize_grid = ui.tab("Optimal Parameter Using Grid Search CV")
                optimize_ga = ui.tab("Optimal Parameter Using Genetic Algorithm")

            with ui.tab_panels(tabs=tab, value=default_param) as tab_panels:
                with ui.tab_panel(default_param):
                    ui.label("this is plot for MAE comparison with default parameter")
                    mae_plot_path = self.path / 'mae_comparison_default_parameter.png'
                    ui.image(mae_plot_path).style("max-width:100%; height:700px; width:1200px;")
                    ui.label("this is plot for MAPE comparison with default parameter")
                    mape_plot_path = self.path / 'mape_comparison_default_parameter.png'
                    ui.image(mape_plot_path).style("max-width:100%; height:700px; width:1200px;")
                    ui.label("this is plot for MSE comparison with default parameter")
                    mse_plot_path = self.path / 'mse_comparison_default_parameter.png'
                    ui.image(mse_plot_path).style("max-width:100%; height:700px; width:1200px;")
                    ui.label("this is plot for RMSE comparison with default parameter")
                    rmse_plot_path = self.path / 'rmse_comparison_default_parameter.png'
                    ui.image(rmse_plot_path).style("max-width:100%; height:700px; width:1200px;")
                    ui.label("this is plot for R2 Score comparison with default parameter")
                    r2_score_plot_path = self.path / 'r2_score_comparison_default_parameter.png'
                    ui.image(r2_score_plot_path).style("max-width:100%; height:700px; width:1200px;")
                with ui.tab_panel(optimize_grid):
                    ui.label("this is plot for MAE comparison with grid search cv tuning parameter")
                    mae_plot_path = self.path / 'mae_comparison_tuning_gridcv_parameter.png'
                    ui.image(mae_plot_path).style("max-width:100%; height:700px; width:1200px;")
                    ui.label("this is plot for MAPE comparison with grid search cv tuning parameter")
                    mape_plot_path = self.path / 'mape_comparison_tuning_gridcv_parameter.png'
                    ui.image(mape_plot_path).style("max-width:100%; height:700px; width:1200px;")
                    ui.label("this is plot for MSE comparison with grid search cv tuning parameter")
                    mse_plot_path = self.path / 'mse_comparison_tuning_gridcv_parameter.png'
                    ui.image(mse_plot_path).style("max-width:100%; height:700px; width:1200px;")
                    ui.label("this is plot for RMSE comparison with grid search cv tuning parameter")
                    rmse_plot_path = self.path / 'rmse_comparison_tuning_gridcv_parameter.png'
                    ui.image(rmse_plot_path).style("max-width:100%; height:700px; width:1200px;")
                    ui.label("this is plot for R2 Score comparison with grid search cv tuning parameter")
                    r2_score_plot_path = self.path / 'r2_score_comparison_tuning_gridcv_parameter.png'
                    ui.image(r2_score_plot_path).style("max-width:100%; height:700px; width:1200px;")
                with ui.tab_panel(optimize_ga):
                    ui.label("This is optimize ga")
        ui.run()
        pass

    def run_single_param(self):
        boost = bi.BoostInterface()

        scenarios = [
            {
                "scenario": "free",
                "n_estimator": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_sample_split": 2,
                "loss": "linear",
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "gamma": 0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
            {
                "scenario": "x1",
                "n_estimator": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_sample_split": 2,
                "loss": "linear",
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "gamma": 0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
            {
                "scenario": "x2",
                "n_estimator": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_sample_split": 2,
                "loss": "linear",
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "gamma": 0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
            {
                "scenario": "x1x2",
                "n_estimator": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_sample_split": 2,
                "loss": "linear",
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "gamma": 0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
        ]

        scenario_value = []
        print("hasil tanpa parameter tuning untuk adaboost dan xgboost")
        data = dp.DataProcessing()
        data.initialize_data()
        for scenario in scenarios:
            boost = ab.Adaboost(data)
            boost.initialize(scenario)
            print(f"ada boost successfully trained for scenario {scenario}")
            self.evaluator = ev.Evaluator(boost.prediction_value(), data.get_y_test())
            # RMSE
            print(f"nilai rmse untuk ada boost : {self.evaluator.get_rmse()}")
            # MSE
            print(f"nilai mse untuk ada boost : {self.evaluator.get_mse()}")
            # MAE
            print(f"nilai mae untuk ada boost : {self.evaluator.get_mae()}")
            # MAPE
            print(f"nilai mape untuk ada boost : {self.evaluator.get_mape()}")
            # R2 Score
            print(f"nilai r2 score untuk ada boost : {self.evaluator.get_r2_score()}")
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            evaluator_value = {"scenario": scenario["scenario"],
                               "algorithm": "AdaBoost",
                               "rmse": self.evaluator.get_rmse(),
                               "mse": self.evaluator.get_mse(),
                               "mae": self.evaluator.get_mae(),
                               "mape": self.evaluator.get_mape(),
                               "r2_score": self.evaluator.get_r2_score()}
            scenario_value.append(evaluator_value)
            boost = xgb.XGBoost(data)
            boost.initialize(scenario)
            print(f"xg boost successfully trained for scenario {scenario}")
            self.evaluator = ev.Evaluator(boost.prediction_value(), data.get_y_test())
            evaluator_value = {}
            # RMS7E
            print(f"nilai rmse untuk xg boost : {self.evaluator.get_rmse()}")
            # MSE
            print(f"nilai mse untuk xg boost : {self.evaluator.get_mse()}")
            # MAE
            print(f"nilai mae untuk xg boost : {self.evaluator.get_mae()}")
            # MAPE
            print(f"nilai mape untuk xg boost : {self.evaluator.get_mape()}")
            # R2 Score
            print(f"nilai r2 score untuk xg boost : {self.evaluator.get_r2_score()}")
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            print("\n\n")
            evaluator_value = {"scenario": scenario["scenario"],
                               "algorithm": "XGBoost",
                               "rmse": self.evaluator.get_rmse(),
                               "mse": self.evaluator.get_mse(),
                               "mae": self.evaluator.get_mae(),
                               "mape": self.evaluator.get_mape(),
                               "r2_score": self.evaluator.get_r2_score()}
            scenario_value.append(evaluator_value)

        #RMSE
        algorithm = ["AdaBoost", "XGBoost"]
        scenario_names = {scen["scenario"] for scen in scenario_value}
        x = range(len(scenario_names))
        width = 0.35
        offsets = [i - (len(algorithm)-1)*width/2 for i in x]
        plt.figure(figsize=(10, 6))
        for algo in algorithm:
            algo_values = [s for s in scenario_value if s["algorithm"] == algo]
            plt.bar([o + algorithm.index(algo)*width for o in offsets],
                    [s['rmse'] for s in algo_values],
                    width=width,
                    label=algo)
        plt.title('RMSE')
        plt.xlabel('Scenario')
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.tight_layout()
        plot_path = self.path / 'rmse_comparison_default_parameter.png'
        plt.savefig(plot_path)
        #MAE
        plt.figure(figsize=(10, 6))
        for algo in algorithm:
            algo_values = [s for s in scenario_value if s["algorithm"] == algo]
            plt.bar([o + algorithm.index(algo)*width for o in offsets],
                    [s['mae'] for s in algo_values],
                    width=width,
                    label=algo)
        plt.title('MAE')
        plt.xlabel('Scenario')
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.tight_layout()
        plot_path = self.path / 'mae_comparison_default_parameter.png'
        plt.savefig(plot_path)
        # MAPE
        plt.figure(figsize=(10, 6))
        for algo in algorithm:
            algo_values = [s for s in scenario_value if s["algorithm"] == algo]
            plt.bar([o + algorithm.index(algo)*width for o in offsets],
                    [s['mape'] for s in algo_values],
                    width=width,
                    label=algo)
        plt.title('MAPE')
        plt.xlabel('Scenario')
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.tight_layout()
        plot_path = self.path / 'mape_comparison_default_parameter.png'
        plt.savefig(plot_path)
        #MSE
        plt.figure(figsize=(10, 6))
        for algo in algorithm:
            algo_values = [s for s in scenario_value if s["algorithm"] == algo]
            plt.bar([o + algorithm.index(algo)*width for o in offsets],
                    [s['mse'] for s in algo_values],
                    width=width,
                    label=algo)
        plt.title('MSE')
        plt.xlabel('Scenario')
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.tight_layout()
        plot_path = self.path / 'mse_comparison_default_parameter.png'
        plt.savefig(plot_path)
        #R2 Score
        plt.figure(figsize=(10, 6))
        for algo in algorithm:
            algo_values = [s for s in scenario_value if s["algorithm"] == algo]
            plt.bar([o + algorithm.index(algo)*width for o in offsets],
                    [s['r2_score'] for s in algo_values],
                    width=width,
                    label=algo)
        plt.title('R2 Score')
        plt.xlabel('Scenario')
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.tight_layout()
        plot_path = self.path / 'r2_score_comparison_default_parameter.png'
        plt.savefig(plot_path)
        
    def run_multi_param(self):
        scenarios = [
            {
                "scenario": "free",
                "n_estimator": [50, 100, 200],
                "learning_rate": [0.1, 0.01, 0.05],
                "max_depth": [3, 5, 7],
                "min_sample_split": [1, 4, 6],
                "loss": ["linear", "square", "exponential"],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.6],
                "gamma": [0, 0.2, 0.3],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
            {
                "scenario": "x1",
                "n_estimator": [50, 100, 200],
                "learning_rate": [0.1, 0.01, 0.05],
                "max_depth": [3, 5, 7],
                "min_sample_split": [1, 4, 6],
                "loss": ["linear", "square", "exponential"],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.6],
                "gamma": [0, 0.2, 0.3],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
            {
                "scenario": "x2",
                "n_estimator": [50, 100, 200],
                "learning_rate": [0.1, 0.01, 0.05],
                "max_depth": [3, 5, 7],
                "min_sample_split": [1, 4, 6],
                "loss": ["linear", "square", "exponential"],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.6],
                "gamma": [0, 0.2, 0.3],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
            {
                "scenario": "x1x2",
                "n_estimator": [50, 100, 200],
                "learning_rate": [0.1, 0.01, 0.05],
                "max_depth": [3, 5, 7],
                "min_sample_split": [1, 4, 6],
                "loss": ["linear", "square", "exponential"],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.6],
                "gamma": [0, 0.2, 0.3],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
        ]

        scenario_value = []
        print("hasil menggunakan parameter tuning menggunakan gridCV")
        data = dp.DataProcessing()
        data.initialize_data()
        for scenario in scenarios:
            boost = ab.Adaboost(data)
            boost.initialize_parameter_tunning(scenario)
            print(
                f"ada boost successfully trained for scenario {scenario} with parameter tuning using gridCV dan model AdaBoost"
            )
            self.evaluator = ev.Evaluator(boost.prediction_value(), data.get_y_test())
            # RMSE
            print(f"nilai rmse untuk ada boost : {self.evaluator.get_rmse()}")
            # MSE
            print(f"nilai mse untuk ada boost : {self.evaluator.get_mse()}")
            # MAE
            print(f"nilai mae untuk ada boost : {self.evaluator.get_mae()}")
            # MAPE
            print(f"nilai mape untuk ada boost : {self.evaluator.get_mape()}")
            # R2 Score
            print(f"nilai r2 score untuk ada boost : {self.evaluator.get_r2_score()}")
            evaluator_value = {"scenario": scenario["scenario"],
                               "algorithm": "AdaBoost",
                               "rmse": self.evaluator.get_rmse(),
                               "mse": self.evaluator.get_mse(),
                               "mae": self.evaluator.get_mae(),
                               "mape": self.evaluator.get_mape(),
                               "r2_score": self.evaluator.get_r2_score()}
            scenario_value.append(evaluator_value)
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            boost = xgb.XGBoost(data)
            boost.initialize_parameter_tunning(scenario)
            print(
                f"ada boost successfully trained for scenario {scenario} with parameter tuning using gridCV dan model XGBoost"
            )
            self.evaluator = ev.Evaluator(boost.prediction_value(), data.get_y_test())
            # RMSE
            print(f"nilai rmse untuk ada boost : {self.evaluator.get_rmse()}")
            # MSE
            print(f"nilai mse untuk ada boost : {self.evaluator.get_mse()}")
            # MAE
            print(f"nilai mae untuk ada boost : {self.evaluator.get_mae()}")
            # MAPE
            print(f"nilai mape untuk ada boost : {self.evaluator.get_mape()}")
            # R2 Score
            print(f"nilai r2 score untuk ada boost : {self.evaluator.get_r2_score()}")
            evaluator_value = {"scenario": scenario["scenario"],
                               "algorithm": "XGBoost",
                               "rmse": self.evaluator.get_rmse(),
                               "mse": self.evaluator.get_mse(),
                               "mae": self.evaluator.get_mae(),
                               "mape": self.evaluator.get_mape(),
                               "r2_score": self.evaluator.get_r2_score()}
            scenario_value.append(evaluator_value)

        #RMSE
        algorithm = ["AdaBoost", "XGBoost"]
        scenario_names = {scen["scenario"] for scen in scenario_value}
        x = range(len(scenario_names))
        width = 0.35
        offsets = [i - (len(algorithm)-1)*width/2 for i in x]
        plt.figure(figsize=(10, 6))
        for algo in algorithm:
            algo_values = [s for s in scenario_value if s["algorithm"] == algo]
            plt.bar([o + algorithm.index(algo)*width for o in offsets],
                    [s['rmse'] for s in algo_values],
                    width=width,
                    label=algo)
        plt.title('RMSE')
        plt.xlabel('Scenario')
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.tight_layout()
        plot_path = self.path / 'rmse_comparison_tuning_gridcv_parameter.png'
        plt.savefig(plot_path)
        #MAE
        plt.figure(figsize=(10, 6))
        for algo in algorithm:
            algo_values = [s for s in scenario_value if s["algorithm"] == algo]
            plt.bar([o + algorithm.index(algo)*width for o in offsets],
                    [s['mae'] for s in algo_values],
                    width=width,
                    label=algo)
        plt.title('MAE')
        plt.xlabel('Scenario')
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.tight_layout()
        plot_path = self.path / 'mae_comparison_tuning_gridcv_parameter.png'
        plt.savefig(plot_path)
        # MAPE
        plt.figure(figsize=(10, 6))
        for algo in algorithm:
            algo_values = [s for s in scenario_value if s["algorithm"] == algo]
            plt.bar([o + algorithm.index(algo)*width for o in offsets],
                    [s['mape'] for s in algo_values],
                    width=width,
                    label=algo)
        plt.title('MAPE')
        plt.xlabel('Scenario')
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.tight_layout()
        plot_path = self.path / 'mape_comparison_tuning_gridcv_parameter.png'
        plt.savefig(plot_path)
        #MSE
        plt.figure(figsize=(10, 6))
        for algo in algorithm:
            algo_values = [s for s in scenario_value if s["algorithm"] == algo]
            plt.bar([o + algorithm.index(algo)*width for o in offsets],
                    [s['mse'] for s in algo_values],
                    width=width,
                    label=algo)
        plt.title('MSE')
        plt.xlabel('Scenario')
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.tight_layout()
        plot_path = self.path / 'mse_comparison_tuning_gridcv_parameter.png'
        plt.savefig(plot_path)
        #R2 Score
        plt.figure(figsize=(10, 6))
        for algo in algorithm:
            algo_values = [s for s in scenario_value if s["algorithm"] == algo]
            plt.bar([o + algorithm.index(algo)*width for o in offsets],
                    [s['r2_score'] for s in algo_values],
                    width=width,
                    label=algo)
        plt.title('R2 Score')
        plt.xlabel('Scenario')
        plt.xticks(x, scenario_names)
        plt.legend()
        plt.tight_layout()
        plot_path = self.path / 'r2_score_comparison_tuning_gridcv_parameter.png'
        plt.savefig(plot_path)


if __name__ in {"__main__", "__mp_main__"}:
    app = main()
    # app.run_single_param()
    # app.run_multi_param()
    app.run_ui()
