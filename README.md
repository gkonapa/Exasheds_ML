# Exasheds_ML
This Repository contains all the data and it's corresponding scripts for Konapala, G., Kao, S. C., Painter, S. L., & Lu, D. (2020). Machine learning assisted hybrid models can improve streamflow simulation in diverse catchments across the conterminous US. Environmental Research Letters, 15(10), 104022.  

## Folder description: 

#### DATA:
Contains the streamflow NSE values for SAC-SMA models, LSTM and two hybrid models for CAMELS catchments. 
#### camels_attributes_v2.0: 
Contains the catchment attributes of CAMELS dataset 
#### Scripts: 
Contains the scripts to analyze the data and generate figures as produced in our manuscript. These scripts are self sufficient and do not need any additional data except for the one provided in this repository
#### LSTM_Modelzoo: 
Contains the scripts to build LSTM models for CAMELS catchments. These scripts should be run after downloading CAMELS database available at  https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip
