## Final Project for ESS 569/469  

**Elements**  
**1. Formulation of an oustanding research question**  
*Status*: **Full writeup required**  

Gist:  
The PNSN is charged with providing the definitive cataliog of all seismic events in Washington and Oregon, USA, including a >700 km long stretch of the Cascadia Subduction Zone (CSZ). Due to the geometry of the coastline relative to the CSZ, land-based long-term seismic stations have difficulty accurately constraining the location and source prameters of offshore earthquakes within the current automated earthquake processing pipeline used at the PNSN (AQMS; Hartog et al., 2020). Moreover, the PNSN is also charged with providing near-real-time analysis of potentially damaging earthquakes to aid in earthquake early warning, triggering a number of rapid-response alerts for seismic events with preliminary magnitude estimates $\geq$ 2.95.

The recent paper and codebase published by Pablo Lara and colleagues (2023) present a machine-learning based approach to constrain these parameters from recordings of the first few seconds of earthquake energy (P-waves) observed at a single three-component (3-C) seismic station. This ensemble of random forest models intended to aid Earthquake Early Warning, dubbed `E3WS`, was trained on global (STEAD; Mousavi et al., 2019) and subduction zone (Peru and Japan) seismic datasets, making an appealing argument for direct application to the CSZ and PNSN monitoring operations.  

In this repository we document the curation of a labeled earthquake dataset from the PNSN archives, waveform preprocessing and feature extraction, and all stages of model (re)training, testing, and validation using this new-to-the-model dataset.


**2. Design and deploy a scientific workflow: describe in prose or in a diagram**  
    *Status*: **In Development**  

**3. Design and deploy a data and computing workflow**  
    *Status*: **In Development**  
        - **ID** `PNSN_src.core.predict`  
        - **ID** `PNSN_src.core.train`  
        - **Beta** `PNSN_src.core.preprocess` - needs SAC_PZ deconvolution implementation  
        - **ID** `PNSN_src.driver.split_data`  

**4. Gather/curate data**  
*Status*: **Complete**  
`PNSN_metadata`  - Event   
`PNSN_data` - waveforms, instrument response files, and labeled features  
- `EVID*/` - directories containing event-specific waveform data. Number corresponds to a PNSN catalog Event ID (EVID)  
    - `bulk25tp45.mseed`  - raw waveform data from 25 sec before p-wave arrival to 45 sec after p-wave arrival  
    - `paz/` - directory containing SAC poles-and-zeros text files attachable to traces  
    - `station.xml` - station XML file containing station information for all stations in `bulk25tp45.mseed` down to the `response` level 
    - `event_mag_phase_nwf.csv` - data labels for each phase associated with the event int he PNSN AQMS database, plus numbers of traces and unique channels each station has represented in `bulk25tp45.mseed` . 

**NOTE**:  
- All `EVID*` directories and their contents are not included in this repository. 
- A temporarily hosted *.zip archive of the dataset is hosted on **FILL IN HERE**  
- The scripts `PNSN_src/drivers/fetch_raw_waveoforms_bulk.py` and `PNSN_src/drivers/get_RESP_inv_from_bulk.py`, along with event/magnitude/phase metadata included in the `PNSN_metadata/Event_Mag_Phase/` directory as inputs, will (re)populate the `PNSN_data` directory's contents.


**5. Develop and deploy and algorithm**

**6. Assess the performance of the algorithm**
*Status*: **In Development**  
    - Step 1: run PNSN data on published model  
    - Step 2: retrain model on a subset of PNSN data  
    - Step 3: re-run predictions on PNSN data witheld for validation  

**7. Ensure the reproducability of results**
    *Status*: **In Progress**
    - This repository  


**REFERENCES**  
Hartog, J. R., Friberg, P. A., Kress, V. C., Bodin, P., & Bhadha, R. (2020). Open-Source ANSS Quake Monitoring System Software. Seismological Research Letters, 91(2A), 677–686. https://doi.org/10.1785/022019021  

Lara, P., Bletery, Q., Ampuero, J., Inza, A., & Tavera, H. (2023). Earthquake Early Warning Starting From 3 s of Records on a Single Station With Machine Learning. Journal of Geophysical Research: Solid Earth, 128(11), e2023JB026575. https://doi.org/10.1029/2023JB026575  

Mousavi, S. M., Sheng, Y., Zhu, W., & Beroza, G. C. (2019). STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI. IEEE Access, 7, 179464–179476. https://doi.org/10.1109/ACCESS.2019.2947848
