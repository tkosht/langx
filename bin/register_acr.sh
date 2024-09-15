#!/usr/bin/sh

container_name="aci-tutorial-app"

acr_login_server=$(az acr show --name cralys --query loginServer --output json | jq -r .)
# echo $acr_login_server


docker tag $container_name $acr_login_server/$container_name:v1
docker push $acr_login_server/$container_name:v1

