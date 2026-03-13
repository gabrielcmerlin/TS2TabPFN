from utils.utils_data import get_data_reg, get_data_extracted_reg
from utils.utils_data import get_data_clf, get_data_extracted_clf
from utils.utils_train import train_test_model_reg, get_results_reg, print_results_reg
from utils.utils_train import train_test_model_clf, get_results_clf, print_results_clf

def choose_functions(task):
    if task == 'tser':
        return get_data_reg, train_test_model_reg, get_results_reg, print_results_reg, get_data_extracted_reg
    elif task == 'tsc':
        return get_data_clf, train_test_model_clf, get_results_clf, print_results_clf, get_data_extracted_clf
    else:
        raise ValueError("Task not supported :(")