# CS-552 - Open Project Final Submission

Welcome to the final submission for the MNLP open project!

As detailed in the [open project description](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit#heading=h.la508tp94epa), the final submission is your last submission for the project and the main component of your grade. You should have a full project report, individual notebooks and all code files ready for this submission.

## Deliverables
Please refer to the [project description - Part III - Deliverables](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit#heading=h.r9bmkwphevm1) for details.

## Rubric
Please refer to the [project description - Part III - Rubric](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit#heading=h.bjfcgbdr5ka0) for details.

## Repo Structure
The repository is organized as follows:

### [`rcp_support/`](rcp_support)
We provide a step by step guide to help you set up and use the RCP framework effectively in the [rcp_support](rcp_support).
This includes instructions for submitting runai jobs and building your own docker images if needed.
Read [rcp_support documentation](rcp_support/README.md) carefully to ensure you can successfully run your experiments on the cluster.

### [`notebooks/`](notebooks)
As part of the final submission, you are required to submit a notebook per team member. 
Put these notebooks in the [notebooks](notebooks) directory.
A template is provided in [notebooks/individual_notebook_template.ipynb](notebooks/individual_notebook_template.ipynb) that explains the structure and content required for the notebook.

Use this structure for the notebook submissions:

```text
notebooks/
  submit.sh # (one per team, to launch the notebook environment)
  <first_name>_<last_name>_<sciper>.ipynb
  <first_name>_<last_name>_<sciper>.ipynb
  <first_name>_<last_name>_<sciper>.ipynb
  <first_name>_<last_name>_<sciper>.ipynb
```

See [rcp_support/README.md](rcp_support/README.md) for submit script details.

### [`report/`](report)
Put your milestone report in the [report](report) directory. 

### [`code/`](code)
Put all the code files in the [code](code) directory. You are free to organize your code as you see fit in this directory.