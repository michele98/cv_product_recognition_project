## Requirements #####
Python with Anaconda distribution and jupyter.

Create the conda environment using:
```console
conda env create -f cv_product_recognition.yml -p <env_path/cv_product_recognition>
```
where ``<env_path>`` is the folder in which conda installs the environments. For windows users use the file ``cv_product_recognition_win.yml``.

Add the environment to ipykernel. Do this after having activated the created environment in a terminal or in an anaconda prompt:
```console
python -m ipykernel install --user --name=cv_product_recognition --display-name="Python 3.9 (cv_product_recognition)"
```
For Windows users change ``Python 3.9`` to ``Python 3.7``.

Create the folder ``images/results`` to display the outputs of the notebook.