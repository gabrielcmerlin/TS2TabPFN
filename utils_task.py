from utils_data import get_data_reg
from utils_data import get_data_clf
from utils_train import train_model_reg, get_results_reg, print_results_reg
from utils_train import train_model_clf, get_results_clf, print_results_clf

def choose_functions(task):
    if task == 'tser':
        return get_data_reg, train_model_reg, get_results_reg, print_results_reg
    elif task == 'tsc':
        return get_data_clf, train_model_clf, get_results_clf, print_results_clf 
    else:
        raise ValueError("Task not supported :(")