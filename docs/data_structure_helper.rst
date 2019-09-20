.. _data_structure_helper

Data Structure Helper
=====================

Data Types
**********

Aikit helps dealing with the multiple type of data that coexist within the scikit-learn, pandas, numpy and scipy environments. Mainly :

 * pandas DataFrame
 * pandas Sparse DataFrame
 * numpy array
 * scipy sparse array (csc, csr, coo, ...)

The library offers tools to easily convert between each type.


Within aikit.enums there is a DataType enumeration with the following values :

 * 'DataFrame'
 * 'SparseDataFrame'
 * 'Series'
 * 'NumpyArray'
 * 'SparseArray'

This is better use as an enumeration but the values are actual strings so you can use the string directly if needed.
 
The function `aikit.tools.data_structure_helper.get_type` retrieve the type (one element of the element).

Example of use::

    from aikit.tools.data_structure_helper import get_type, DataTypes
    df = pd.DataFrame({"a":[0,1,2],"b":["aaa","bbb","ccc"]})
    mapped_type = get_type(df)
    
    if mapped_type == DataTypes.DataFrame:
        first_column = df.loc[:,"a"]
    else:
        first_column = df[:,0]


Generic Conversion
******************

You can also convert each type to the desired type. This can be useful if a transformer only accepts DataFrames, or doesn't work with a Sparse Array, ...
For that use the function `aikit.tools.data_structure_helper.convert_generic`

.. autofunction:: aikit.tools.data_structure_helper.convert_generic
 
Example::
    
    from aikit.tools.data_structure_helper import convert_generic, DataTypes
    df = pd.DataFrame({"a":[0,1,2],"b":["aaa","bbb","ccc"]})

    arr = convert_generic(df, output_type = DataTypes.NumpyArray)
    

Generic Horizontal Stacking
***************************

You can also horizontally concatenate multiple datas together (assuming they have the same number of rows). You can either specify the output type you want, if that is not the case the algorithm will guess :

 * if same type will use that type
 * if DataFrame and Array use DataFrame
 * if Sparse and Non Sparse : convert to full if not to big otherwise keep Sparse

(See `aikit.tools.data_structure_helper.guess_output_type`)

The function to concatenate everything is `aikit.tools.data_structure_helper.generic_hstack`

Example::

    df1 = pd.DataFrame({"a":list(range(10)),"b":["aaaa","bbbbb","cccc"]*3 + ["ezzzz"]})
    df2 = pd.DataFrame({"c":list(range(10)),"d":["aaaa","bbbbb","cccc"]*3 + ["ezzzz"]})
    
    df12 = generic_hstack((df1,df2))
    
.. autofunction:: aikit.tools.data_structure_helper.generic_hstack


 
Other
*****

Two other functions that can be useful are `aikit.tools.data_structure_helper.make1dimension` and `aikit.tools.data_structure_helper.make2dimensions`. It convert to a 1 dimensional or 2 dimensional object whenever possible.

.. autofunction:: aikit.tools.data_structure_helper.make1dimension

.. autofunction:: aikit.tools.data_structure_helper.make2dimensions

