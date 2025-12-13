import data_processing as dp
import ada_boost as ab
import xg_boost as xgb
import evaluator as ev
import boostInterface as bi
import parameter_tuning as pt

class main : 
    def __init__(self):
        self.value = "Hello, World!"
        # self.data = dp.DataProcesssing()
        # self.ada.initialize_model()
        # self.ada = ab.AdaBoost(self.data)
        # self.xgboost = xgb.XGBoost(self.data)
        # self.evaluator = None

    def run(self):

        boost = bi.BoostInterface()

        scenarios = [{"scenario" : "free",
                      "model": "adaboost",
                      "n_estimator": 50,
                      "learning_rate": 0.1, 
                      "max_depth": 3},

                      {"scenario" : "x1",
                      "n_estimator": 50,
                      "learning_rate": 0.1,
                      "max_depth": 3},
                      
                      {"scenario" : "x2",
                      "n_estimator": 50,
                      "learning_rate": 0.1,
                      "max_depth": 3},
                      
                      {"scenario" : "x1x2",
                      "n_estimator": 50,
                      "learning_rate": 0.1,
                      "max_depth": 3},]

        print('hasil tanpa parameter tuning untuk adaboost dan xgboost')
        for scenario in scenarios:
            boost = ab.Adaboost(self.data)
            boost.initialize(scenario)
            print(f"ada boost successfully trained for scenario {scenario}")
            self.evaluator = ev.Evaluator(boost.prediction_value(), self.data.get_y_test())
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
            boost = xgb.XGBoost(self.data)
            boost.initialize(scenario)
            print(f"xg boost successfully trained for scenario {scenario}")
            self.evaluator = ev.Evaluator(boost.prediction_value(), self.data.get_y_test())
            # RMSE
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
            param_tuning = 
        pass
        print("hasil menggunakan parameter tuning menggunakan gridCV")


    

if __name__ == "__main__":
    app = main()
    print(app.run())