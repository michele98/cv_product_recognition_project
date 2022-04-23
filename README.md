## Requirements #####
Python with Anaconda distribution and jupyter.

Create the conda environment using:
```console
conda env create -f cv_product_recognition.yml -p <env_path/cv_product_recognition>
```
where ``<env_path>`` is the folder in which conda installs the environments

Add the environment to ipykernel. Do this from the base environment:
```console
python -m ipykernel install --user --name=cv_product_recognition --display-name="Python 3.7 (cv_product_recognition)"
```

Create the folder ``images/results`` to display the outputs of the notebook.