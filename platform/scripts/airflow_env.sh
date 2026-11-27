# source platform/scripts/airflow_env.sh   -> local Airflow (SQLite, no examples)
_PLATFORM="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AIRFLOW_HOME="$_PLATFORM/data/airflow-home"
export AIRFLOW__CORE__DAGS_FOLDER="$_PLATFORM/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__UNIT_TEST_MODE=False
