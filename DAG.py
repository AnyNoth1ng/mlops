from airflow.models import DAG, Variable
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Any, Dict, Literal
import os
import io
import json
import time
import logging
import pickle
import mlflow
import pandas as pd
import datetime
import numpy as np

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
from mlflow.models import infer_signature


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

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        ]:
            os.environ[key] = Variable.get(key) 

models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

X_cols = ['HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
y_cols = ['MedInc']


dag = DAG(dag_id=f'{BASE_PATH}',
            schedule_interval = "0 1 * * * ",   
            start_date = days_ago(2),
            catchup  = False,
            tags = ["mlops"],
            default_args = DEFAULT_ARGS
            )

    ####### DAG STEPS #######

def init() -> Dict[str, Any]:
    configure_mlflow()
    metrics = {}
    metrics['start_time'] = datetime.now().strftime("%Y%m%d %H:%M%S")
    _LOG.info(f"Начало: {metrics['start_time']}")
    
    # exp_id = mlflow.get_experiment_by_name(BASE_PATH)
    
    try:
        exp_id = mlflow.get_experiment_by_name(BASE_PATH)

        if exp_id is not None:
            exp_id = exp_id.experiment_id
            _LOG.info(f'Эксперимент уже создан')
        else:
            exp_id = mlflow.create_experiment(BASE_PATH, artifact_location=f"s3://{BUCKET}/{BASE_PATH}")
            _LOG.info(f"Создан эксперимент: {exp_id}")

        mlflow.set_experiment(BASE_PATH)

    except Exception as e:
        _LOG.error(f"Ошибка с экспериментом :{e}")
    
    with mlflow.start_run(run_name='@Iam_PrP', experiment_id=exp_id, description='parent_run_id') as parent_run:
        run_id = parent_run.info.run_id

    metrics.update({
        "run_id": run_id,
        "experiment_name": BASE_PATH,
        "experiment_id": exp_id
    })
    
    return metrics
        

def get_data(**kwargs) -> Dict[str, Any]:
    task_instance = kwargs['task_instance']
    metrics = task_instance.xcom_pull(task_ids='init')
    metrics['start_time'] = datetime.now().strftime("%Y%m%d %H:%M%S")
    
    _LOG.info(f"Время и дата запуска дага: {metrics['start_time']}")
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
        key=f"{BASE_PATH}/datasets/california_housing.pkl",
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
    metrics["prerpoc_start"] = datetime.now().strftime("%Y%m%d %H:%M%S")
    _LOG.info(f"Начало preprocessa: {metrics['prerpoc_start']}")
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key=f"{BASE_PATH}/datasets/california_housing.pkl", bucket_name=BUCKET)
    df = pd.read_pickle(file)
    
    # preprocess
    X = df[X_cols]
    y = df[y_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scal = StandardScaler()
    X_train_scal = scal.fit_transform(X_train)
    X_test_scal = scal.fit_transform(X_test)
    X_train_scal = pd.DataFrame(X_train_scal, columns=X_cols, index=X_train.index)
    X_test_scal = pd.DataFrame(X_test_scal, columns=X_cols, index=X_test.index)
    for name, data in zip(["X_train", "X_test", "y_train", "y_test"], [X_train_scal,X_test_scal, y_train, y_test]):
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=buffer,
            key=f"{BASE_PATH}/datasets/{name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )
    metrics["prerpoc_end"] = datetime.now().strftime("%Y%m%d %H:%M%S")
    _LOG.info(f"Конец preprocessa: {metrics['prerpoc_end']}")
    return metrics


def train_model(**kwargs) -> Dict[str, Any]:
    configure_mlflow()
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="prepare_data")
    exp_id = kwargs['ti'].xcom_pull(key='experiment_id')
    parent_run_id = kwargs['ti'].xcom_pull(key='parent_run_id')

    st_train = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    model_name = kwargs["model_name"]

    s3_hook = S3Hook("s3_connection")

    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            key=f"{BASE_PATH}/datasets/{name}.pkl",
            bucket_name=BUCKET,
        )
        data[name] = pd.read_pickle(file)

    
    X_val, X_test, y_val, y_test = train_test_split(data['X_test'], data['y_test'], test_size=0.5)
 
    _LOG.info("Начинаем обучать модели")

    model = models[model_name]
    
    with mlflow.start_run(parent_run_id=parent_run_id, run_name=model_name, experiment_id=exp_id, nested=True) as child_run:
        model.fit(data["X_train"], data["y_train"])
        fin_train = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        predict = model.predict(X_val)

        val_data = X_val.copy()
        val_data['target']= y_val
        # _LOG.info(f'{model_name} -обучение ')
        # _LOG.info(f'Форма X_train: {data["X_train"].shape}')
        # _LOG.info(f'Форма y_train: {data["y_train"].shape}')
        # _LOG.info(f'Форма X_test: {data["X_test"].shape}')
        # _LOG.info(f'Форма y_test: {data["y_test"].shape}')

        signature = infer_signature(X_test, predict)
        model_info = mlflow.sklearn.log_model(model, model_name, signature=signature, registered_model_name=f"{model_name}")
        mlflow.evaluate(
            model=model_info.model_uri,
            data=val_data,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
        )
        _LOG.info(f'{model} -обучилась ')
    
    kwargs["ti"].xcom_push(key=f'start-{model_name}', value=st_train)
    kwargs["ti"].xcom_push(key=f'finish-{model_name}', value=fin_train)


    _LOG.info("Конец обучения")
        
    return metrics

def save_results(**kwargs) -> None:
    
    metrics = {}

    metrics['train_linear_regression'] = kwargs["task_instance"].xcom_pull(task_ids="train_linear_regression")
    metrics['train_random_forest'] = kwargs["task_instance"].xcom_pull(task_ids="train_random_forest")
    metrics['train_desicion_tree'] = kwargs["task_instance"].xcom_pull(task_ids="train_desicion_tree")
    
    s3_hook = S3Hook("s3_connection")
    
    buffer = io.BytesIO()
    buffer.write(json.dumps(metrics).encode())
    buffer.seek(0)
    s3_hook.load_file_obj(
        file_obj=buffer,
        key=f"{BASE_PATH}/results/metrics.json",
        bucket_name=BUCKET,
        replace=True
        )
    
    _LOG.info("Всё сохранилось")

with dag:
    
    task_init = PythonOperator(task_id = "init", python_callable = init, dag=dag)
    task_get_data = PythonOperator(task_id = "get_data", python_callable = get_data, dag = dag)
    task_prepare_data = PythonOperator(task_id = "prepare_data", python_callable = prepare_data, dag = dag)
    task_train_model = [
        PythonOperator(task_id=f"{model_name}", python_callable=train_model, dag=dag, provide_context=True, op_kwargs={"model_name": model_name}) for model_name in models.keys()
        ]
    task_save_results = PythonOperator(task_id = "save_results", python_callable = save_results, dag = dag)

configure_mlflow()


task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


