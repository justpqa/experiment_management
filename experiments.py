# Define an object for storing the result of experiment
class ExperimentResult:
    def __init__(self, params, best_params, cv_acc, accuracy):
        self.params = params
        self.best_params = best_params
        self.cv_acc = cv_acc
        self.accuracy = accuracy
        
    def __lt__(self, other):
        return self.accuracy < other.accuracy