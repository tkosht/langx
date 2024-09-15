#!/usr/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

sqlite="sqlite3 data/experiment.db"

sql_uuids="select run_uuid
from runs
where lifecycle_stage = 'deleted'
"

# echo "$sql_uuids" | sqlite3 data/experiment.db | xargs


# sql_artifact_locations="select artifact_location from experiments"
# dir_list=$(echo "$sql_artifact_locations" | sqlite3 data/experiment.db)
# echo $dir_list

# select rn.run_uuid, rn.lifecycle_stage, rn.artifact_uri, ex.artifact_location, ex.name

echo "select ex.artifact_location || '/' || rn.run_uuid as file_path
from runs as rn, experiments as ex
where rn.experiment_id = ex.experiment_id
and rn.lifecycle_stage = 'deleted'
" | sqlite3 data/experiment.db | xargs rm -rf

