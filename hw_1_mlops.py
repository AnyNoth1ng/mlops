from airflow.models import DAG, Variable
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Any, Dict, Literal

import io
import json
import time
import logging
import pickle
import pandas as pd

import warnings 
warnings.filterwarnings('ignore')

from typing import Any, Dict, Literal
from datetime import timedelta, datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.metrics import r2_score


from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BASE_PATH = 'Polyakov_Egor'
BUCKET = Variable.get("S3_BUCKET")
DEFAULT_ARGS = {
    'owner': 'Polyakov Egor' ,
    'email' : 'enpoliakov@edu.hse.ru',
    'email_on_failure': True,
    'email_on_retry': False,
    'retries' : 3,
    "retry_delay" : timedelta(minutes=1)
}
model_names = ["random_forest", "linear_regression", "desicion_tree"]


models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

X_cols = ['HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
y_cols = ['MedInc']


def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "desicion_tree"]):
    dag = DAG(dag_id=dag_id,
              schedule_interval = "0 1 * * * ",
              start_date = days_ago(2),
              catchup  = False
                      
             )

    ####### DAG STEPS #######

    def init(m_name: Literal["random_forest", "linear_regression", "desicion_tree"]) -> Dict[str, Any]:
        metrics = {}
        metrics['model_name'] = m_name
        metrics['start_time'] = datetime.now().strftime("%Y%m%d %H:%M%S")
        
        _LOG.info(f"начало пайплайна для {metrics['model_name']}")
        
        return metrics
        

    def get_data(**kwargs) -> Dict[str, Any]:
        task_instance = kwargs['task_instance']
        metrics = task_instance.xcom_pull(task_ids='init')

        metrics['start_time'] = datetime.now().strftime("%Y%m%d %H:%M%S")
        model_name = kwargs["model_name"]
        
        _LOG.info(f"Время и дата запуска дага и имя модели:{metrics['start_time']} {metrics['model_name']}")

        time_start = time.time()
        # делаем датасет
        calif = fetch_california_housing(as_frame=True)
        df = pd.concat([calif["data"], pd.DataFrame(calif["target"])], axis=1)
        # делаем коннект к хранилищу
        s3_hook = S3Hook("s3_connection")
        buffer = io.BytesIO()
        df.to_pickle(buffer)
        buffer.seek(0)

    # сохраняем
        s3_hook.load_file_obj(
            file_obj=buffer,
            key=f"{BASE_PATH}/{model_name}/datasets/california_housing.pkl",
            bucket_name=BUCKET,
            replace=True,
        )
        
        time_end = time.time()
        metrics['time_load'] = time_end - time_start
        _LOG.info(f'время загрузки: {metrics["time_load"]}')

        return metrics
        

    def prepare_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="get_data")
        model_name = kwargs["model_name"]
        metrics["prerpoc_start"] = datetime.now().strftime("%Y%m%d %H:%M%S")
        _LOG.info(f"Начало preprocessa: {metrics['prerpoc_start']}")
        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(key=f"{BASE_PATH}/{model_name}/datasets/california_housing.pkl", bucket_name=BUCKET)
        df = pd.read_pickle(file)
        
        # preprocess
        X = df[X_cols]
        y = df[y_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        scal = StandardScaler()

        X_train_scal = scal.fit_transform(X_train)
        X_test_scal = scal.fit_transform(X_test)

        X_train_scal = pd.DataFrame(X_train_scal, columns=X_cols)
        X_test_scal = pd.DataFrame(X_test_scal, columns=X_cols)

        for name, data in zip(["X_train", "X_test", "y_train", "y_test"], [X_train_scal,X_test_scal, y_train, y_test]):
            buffer = io.BytesIO()
            pickle.dump(data, buffer)
            buffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=buffer,
                key=f"{BASE_PATH}/{model_name}/datasets/{name}.pkl",
                bucket_name=BUCKET,
                replace=True,
            )

        metrics["prerpoc_end"] = datetime.now().strftime("%Y%m%d %H:%M%S")
        _LOG.info(f"Конец preprocessa: {metrics['prerpoc_end']}")

        return metrics
        

    def train_model(**kwargs) -> Dict[str, Any]:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="prepare_data")
        model_name = kwargs["model_name"]

        metrics["train_st"] = datetime.now().strftime("%Y%m%d %H:%M%S")
        _LOG.info(f"Начало обучения: {metrics['train_st']}")

        s3_hook = S3Hook("s3_connection")

        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(
                key=f"{BASE_PATH}/{model_name}/datasets/{name}.pkl",
                bucket_name=BUCKET,
            )
            data[name] = pd.read_pickle(file)

        model = models[model_name]
        model.fit(data["X_train"], data["y_train"])
        predict = model.predict(data["X_test"])
    
        result = {}
        result["R2"] = r2_score(data["y_test"], predict)
        result["rmse"] = mean_squared_error(data["y_test"], predict) ** 0.5
        result["mae"] = median_absolute_error(data["y_test"], predict)

        metrics["model_metrics"] = result
        metrics["train_end"] = datetime.now().strftime("%Y%m%d %H:%M%S")
        _LOG.info(f"Конец обучения: {metrics['train_end']}")
        _LOG.info(f"Метрики: {metrics['model_metrics']}")

        return metrics
        
        
        

    def save_results(**kwargs) -> None:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="train_model")

        _LOG.info(f"Метрики сохраненине результатов: {metrics}")

        model_name = metrics["model_name"]

        s3_hook = S3Hook("s3_connection")
        buffer = io.BytesIO()
        buffer.write(json.dumps(metrics).encode())
        buffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=buffer,
            key=f"{BASE_PATH}/{model_name}/results/metrics.json",
            bucket_name=BUCKET,
            replace=True
            )
        
        _LOG.info("Всё сохранилось")
    

    with dag:
        
        task_init = PythonOperator(task_id = "init", python_callable = init, dag=dag, op_kwargs={"m_name": m_name})
        task_get_data = PythonOperator(task_id = "get_data", python_callable = get_data, dag = dag, provide_context = True, op_kwargs= {"model_name": model_name})
        task_prepare_data = PythonOperator(task_id = "prepare_data", python_callable = prepare_data, dag = dag, provide_context = True, op_kwargs= {"model_name": model_name})
        task_train_model = PythonOperator(task_id = "train_model", python_callable = train_model, dag = dag, provide_context = True, op_kwargs={"model_name": model_name})
        task_save_results = PythonOperator(task_id = "save_results", python_callable = save_results, dag = dag)
    

    task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


for model_name in models.keys():
    create_dag(f"Polyakov_Egor_{model_name}", model_name)