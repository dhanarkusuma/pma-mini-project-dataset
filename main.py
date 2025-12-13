import data_processing as dp
import ada_boost as ab
import xg_boost as xgb
import evaluator as ev
import boostInterface as bi

class main : 
    def __init__(self):
        self.value = "Hello, World!"
        self.data = dp.DataProcesssing()
        self.ada.initialize_model()
        self.ada = ab.AdaBoost(self.data)
        self.xgboost = xgb.XGBoost(self.data)
        self.evaluator = ev.Evaluator()

    def run(self):

        boost = bi.BoostInterface()

        
        boost = ab.Adaboost(self.data)
        boost.initialize("X1")
        self.value = boost.prediction_value()   

        boost = xgb.XGBoost(self.data)
        boost.initialize("X1")
        self.value = boost.prediction_value()

        
        return self.value
    

if __name__ == "__main__":
    app = main()
    print(app.run())