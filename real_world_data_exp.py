from SHP import SHP
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score


def get_performance(fitted, real, threshold=0, name="SHP", drop_diag=True):
    fitted = np.abs(fitted)
    if drop_diag:
        fitted = fitted - np.diag(np.diag(fitted))
        real = real - np.diag(np.diag(real))

    f1 = f1_score(y_true=real.ravel(), y_pred=np.array(fitted.ravel() > threshold))
    precision = precision_score(y_true=real.ravel(), y_pred=np.array(fitted.ravel() > threshold))
    recall = recall_score(y_true=real.ravel(), y_pred=np.array(fitted.ravel() > threshold))
    temp_result = np.array((f1, precision, recall, threshold))
    result = pd.DataFrame(columns=['F1', "Precision", "Recall", "threshold"])
    result.loc[0] = temp_result
    result["method"] = name
    return result


def SHP_exp(param_dict, time_interval, seed=0, hill_climb=True):
    df = pd.read_csv("data\\18V_55N_Wireless\\Alarm.csv")
    dag = np.load("data\\18V_55N_Wireless\\DAG.npy")
    event_table = df[['device_id', 'start_timestamp', 'alarm_id']]
    event_table.columns = ["seq_id", "time_stamp", "event_type"]

    SHP_model = SHP(event_table.copy(), seed=seed, **param_dict)
    if hill_climb:
        res = SHP_model.Hill_Climb()
    else:
        res = SHP_model.EM_not_HC(np.ones([SHP_model.n, SHP_model.n]) - np.eye(SHP_model.n, SHP_model.n))

    fited_alpha = res[1]
    SHP_res = get_performance(fited_alpha, dag, name="SHP")
    SHP_res["time_interval"] = time_interval
    SHP_res["seed"] = seed
    return SHP_res


if __name__ == '__main__':
    # real-world data experiment
    time_interval_list = [1, 3, 5, 7, 9, 11, 16, 20]

    res = None
    for time_interval in tqdm(time_interval_list):
        param_dict = {
            "decay": 0.22,
            "reg": 0.3,
            "time_interval": time_interval,
            "penalty": 'BIC',
            "time_instant": True
        }
        res_temp = SHP_exp(param_dict=param_dict, time_interval=time_interval, seed=0)
        res = pd.concat([res, res_temp], axis=0)

