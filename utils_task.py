from utils_data import get_data_reg
from utils_data import get_data_clas
from utils_train import train_model_reg, get_results_reg, print_results_reg
from utils_train import train_model_clas, get_results_clas, print_results_clas

def choose_functions(task):
    if task == 'regression':
        return get_data_reg, train_model_reg, get_results_reg, print_results_reg
    elif task == 'classification':
        return get_data_clas, train_model_clas, get_results_clas, print_results_clas 
    else:
        print('Task not supported yet')