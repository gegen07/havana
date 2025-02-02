# HAVANA

## Description

This repository presents HAVANA: a graph neural network able to semantically enriching Points of Interest (POIs) according to spatiotemporal data.

## Data

<span style="color: red;">**OBS: The Data was removed to not overload the repository.**</span>

## Requirements and Execution

### Requirements

* Python 3
    * [link](https://www.python.org/downloads/)
* virtualenv
    * `$ pip install virtualenv`

### Execution

To execute this project, first you need to create and activate your virtual environment. The step-by-step is:

<span style="color: red;">**OBS: The follow step-by-step was created to use in Linux Operation System.**</span>

1. Create your virtual environment
    * `$ virtualenv myenv`
    * Replace 'myenv' with a name of your choice
    * Execute this command only at the first time

2. Activate the virtual environment
    * `$ source myenv/bin/activate`

3. Update pip
    * `$ pip install --upgrade pip`

4. Install requirements
    * `$ pip install -r requirements.txt`

5. Execute
    * `python main.py --job <job> --state <state> [--model <model>] [--base_line <base_line>] [--base_line_general <base_line_general>]`
    * Arguments
        * --job (required): Specifies the job to be executed. Valid options are preprocess for preprocessing and categorize for categorization.
        * --state (required): The state of the data to be executed.
        * --model (optional): The model to be used for categorization. Required if --job is categorize. Valid options are pgc, havana, havana_arma, havana_gat, havana_no_agg.
        * --base_line (optional): The path to the baseline results for comparison.
        * --base_line_general (optional): The path to the general baseline results for comparison.
    * Example
        * `python main.py --job categorize --state Alabama --model havana --base_line path/to/baseline.csv --base_line_general path/to/baseline_general.csv`
### Info

1. **Preprocessing Job**
   - Before using the categorization job (`--job categorize`), it is necessary to first generate all matrices. You can do this by executing the preprocess job (`--job preprocess`).

2. **Required Input File**
   - Ensure you have a file named `checking.csv` in `gowalla` directory prepared for the preprocessing job. The file should follow this example format:
   
     ```csv
     userid,category,placeid,local_datetime,latitude,longitude,country_name,state_name
     1338,Travel,16772,2010-03-22 15:21:29+00:00,30.6811779503,-88.2442259789,United States,Alabama
     1338,Travel,16772,2010-03-19 23:11:53+00:00,30.6811779503,-88.2442259789,United States,Alabama
     ```

3. **Categorization Job**
   - After successfully completing the preprocessing job and ensuring the presence of all matrices in `gowalla` directory, you can proceed with the categorization job following the instructions provided above.
