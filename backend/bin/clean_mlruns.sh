#!/bin/sh

sql="select run_uuid from runs where lifecycle_stage == 'deleted'"

echo "$sql" \
    | sqlite3 data/mlflow.db \
    | awk '{print "mlruns/1/"$1}' \
    | xargs rm -rf

