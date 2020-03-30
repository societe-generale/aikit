# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:16:43 2020

@author: Lionel Massoulard
"""

import os.path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


if __name__ == "__main__":
    notebooks_to_execute = ("DefaultPipeline", "GraphPipeline", "ModelJson", "NumericalEncoder", "Transformers", "Stacking")
    
    all_errors = []
    for notebook in notebooks_to_execute:
        
        notebook_filename = os.path.join("./notebooks/%s.ipynb" % notebook)
        
        assert os.path.exists(notebook_filename)
        
        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)
        
        print("Start Execution of '%s' ..." % notebook)
    
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            out = ep.preprocess(nb, {'metadata': {'path': "./notebooks"}})
        except CellExecutionError:
            out = None
            msg = 'Error executing the notebook "%s".\n\n' % notebook
            msg += 'See notebook "%s" for the traceback.' % notebook
            print(msg)
            all_errors.append((notebook, msg))
        finally:
            with open(notebook_filename, mode='w', encoding='utf-8') as f:
                nbformat.write(nb, f)
    
    
    if len(all_errors) > 0:
        for nb, msg in all_errors:
            print("error in %s" % nb)
            print(msg)
            print("")
    
        raise ValueError("Error executing the notebooks")
    
        print("... Notebook executed !")