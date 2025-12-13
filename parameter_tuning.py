# wrapper for grid search CV
from sklearn.model_selection import GridSearchCV

class ParameterTuning:
    def __init__(self, model, param_grid, scoring, cv):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.grid_model = None

    def initialize(self):
        self.grid_model = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            verbose=1,
            n_jobs=-1
        )

        self.grid_model.fit(self.X_train, self.y_train)
    
    def get_optimize_parameter(self):
        return self.grid_model.best_params_