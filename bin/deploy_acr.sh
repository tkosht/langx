#!/usr/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../
. ./.env
# echo $ACR_SERVICE_PRINCIPAL_ID
# echo $ACR_SERVICE_PRINCIPAL_PSWD
# pwd; exit 0;

acr_name="cralys"
container_name="aci-tutorial-app"
tag="v1"

acr_login_server=$(az acr show --name $acr_name --query loginServer --output json | jq -r .)



az container create --resource-group rgrg \
    --name $container_name \
    --image $acr_login_server/$container_name:$tag \
    --cpu 1 \
    --memory 1 \
    --registry-username $ACR_SERVICE_PRINCIPAL_ID \
    --registry-password $ACR_SERVICE_PRINCIPAL_PSWD \
    --registry-login-server $acr_login_server \
    --ip-address Public \
    --dns-name-label cralys \
    --ports 80

