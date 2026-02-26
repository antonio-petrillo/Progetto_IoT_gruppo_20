# Project IOT 2025/2026 - Group 20
Repository for the IOT project, academic year 2025/2026.  

| Student             | Matricola | Email                               |
|---------------------|-----------|-------------------------------------|
| Antonio Petrillo    | N97000496 | antonio.petrillo4@studenti.unina.it |
| Alessandro Petrella | DE5000053 | al.petrella@studenti.unina.it       |

# Download
Download the source code from the following repository
```bash
git clone https://github.com/antonio-petrillo/Progetto_IoT_Gruppo_20
```
The only requirements for this step is to have `git` installed on your machine.

# Create Virtual Environment
Once you have downloaded the source code change directory into the root of the project
```bash
cd Progetto_IoT_Gruppo_20
```
Create a python virtual environment:
```bash
python -m venv venv
```

# Activate the virtual environment
- MacOS / Linux / BSD `source ./venv/bin/activate`
- Windows `.\venv\Scripts\activate`

If you are on Windows and the system complains that the current user cannot execute scripts, ensure that your **Execution Policy** is at least setted to **RemoteSigned**.  
If necessary, follow the following [article](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/set-executionpolicy?view=powershell-7.5) from Microsoft or the more concise [guide](https://sentry.io/answers/bypass-and-set-powershell-script-execution-policies/).  
## NOTE
The virtual environment should be activated every time a new shell is opened in the project's root.

# Install the dependencies
Before following this step ensure that your virtual environment is active.  
Run the following command:
```bash
python -m pip install -r requirements.txt
```
This should be executed only one time after the download. 

# Download Dataset
Download the [dataset](https://www.kaggle.com/datasets/mexwell/esports-sensors-dataset) from kaggle.  
The dataset can be downloaded everywhere in the filesystem as long as the **environment variables**  are setted accordingly.  
As a personal suggestion, just download the dataset into the root of the project and don't bother with the environments variables. 

# Set Environment variable
If you need to customize the path to the filesystem, the output directory (where the data & plots are saved) you can use [.env file](https://www.ibm.com/docs/it/aix/7.2.0?topic=files-env-file).  
Withing such file you can set [environment variables](https://en.wikipedia.org/wiki/Environment_variable) that are automatically picked by the script.  

## Customizable Variables
- `DATASET_PATH`
- `OUTPUT_PATH`
- `SENSORS_FILES`
  
The variable `DATASET_PATH` specify the path to the root of the dataset in the host machine.  
Example:
```bash
DATASET_PATH="/Users/username/dataset"
```


The variable `OUTPUT_PATH` specify the root directory where the files produced by the scripts should be saved.  
Example:
```bash
OUTPUT_PATH="/Users/username/Documents/output"
```


The variable `SENSORS_FILES` specify which sensor file should be considered during the analysis, you can specify the values as strings and separated by comma `,` (whitespace, tabs and newline are allowed between each entry).  
Example:
```bash
SENSORS_FILES="eeg_band_power.csv,
eeg_metrics,
emg.csv,
eye_tracker.csv,
facial_skin_temperature.csv,
gsr.csv,
heart_rate.csv,
imu_chair_back.csv,
imu_chair_seat.csv,
imu_head.csv,
imu_left_hand.csv,
imu_right_hand.csv,
keyboard.csv,
mouse.csv,
spo2.csv"
```

## NOTE
There is a file `env_example` that can be used as template, rename that file into `.env` and change the variable as you need.  
Also it is not required to use the `.env` file the default parameters expected by the scripts are:
- `DATASET_PATH`: `eSports_Sensors_Dataset-master`
- `OUTPUT_PATH`: `out`
- `SENSORS_FILES`: the one from the example above (it include every kind of sensor available in the dataset)

# Run the script
Once everything is in place run the following command:
```bash
python main.py
```

## NOTE
If the scripts complain about dependencies remember to activate the [virtual environment](#activate-the-virtual-environment), if the problem persist also ensure that you have installed all the [dependencies](#install-the-dependencies).

