# sempervirens
Entropy based reconstruction of phylogenetic trees

```Project structure is modified version of the list give here: https://github.com/dssg/hitchhikers-guide/tree/master/sources/curriculum/0_before_you_start/pipelines-and-project-workflow├── LICENSE
├── README.md <- The top-level README for developers using this project.
├── conf│
├── data
│ ├── 01raw <- Imutable input data
│ ├── 02_intermediate<- Cleaned version of raw
│ ├── 03_processed <- The data used for modelling
│ ├── 04 output <- Model output
│ └── 06reporting <- Reports and input to frontend
│
├── docs <- Space for Sphinx documentation
│
├── notebooks <- Jupyter notebooks. Naming convention is date YYYYMMDD (for ordering),
│ the creator’s initials, and a short - delimited description, e.g.
│ 20190601-jqp-initial-data-exploration.
│
├── references <- Data dictionaries, manuals, and all other explanatory materials.
│
├── results <- Intermediate analysis as HTML, PDF, LaTeX, etc.
│
├── requirements.txt <- The requirements file for reproducing the analysis environment, e.g.
│ generated with pip freeze > requirements.txt
│
├── .gitignore <- Avoids uploading data, credentials, outputs, system files etc
│
└── src <- Source code for use in this project.
 ├── _init.py <- Makes src a Python module
 │
 ├── d00_utils <- Functions used across the project
 │ └── utils.py
 │
 ├── d01_data <- Scripts to reading and writing data etc
 │ └── load_data.py
 │
 ├── d02_processing <- Scripts to manipulating data
 │ └── transform.py
 │
 ├── d03_evaluation <- Scripts that analyse algorithm performance
 │ └── calculate_performance_metrics.py
 │
 ├── d04_reporting <- Scripts to produce reporting tables
 │ └── create_rpt_summary.py
 │
 └── d05_visualisation<- Scripts to create frequently used plots
   └── visualise.py
```