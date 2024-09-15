
acr_name="cralys"
container_name="aci-tutorial-app"
az acr repository show-tags --name $acr_name --repository $container_name # --output table
