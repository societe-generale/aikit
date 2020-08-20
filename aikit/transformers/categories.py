# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:13:56 2018

@author: Lionel Massoulard
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import pandas as pd
import numpy as np

try:
    import category_encoders
except ImportError:
    category_encoders = None

from collections import defaultdict


from aikit.enums import DataTypes, TypeOfVariables
from aikit.tools.data_structure_helper import get_type, get_rid_of_categories


from aikit.transformers.model_wrapper import ModelWrapper


class _NumericalEncoder(BaseEstimator, TransformerMixin):
    """ Numerical Encoder of categorical variables
    
    Parameters
    ----------        
    min_modalities_number : int, default = 20
        if less that 'min_modalities_number' modalities no modalities will be filtered
        
    max_modalities_number : int, default = 100,
        the number of modalities kept will never be more than 'max_modalities_number'
        
    max_cum_proba : float, default = 0.95
        if modalities should be filtered, first filter applied is removing modalities that account for less than 1-'max_cum_proba'
        
    min_nb_observations : int, default = 10
        if modalities should be filtered, modalities with less thant 'min_nb_observations' observations will be removed
        
    max_na_percentage : float, default = 0.05
        if more than 'max_na_percentage' percentage of missing value, None will be treated as a special modality named '__null__'
        otherwise, will just put -1 (for encoding_type == 'num') or 0 everywhere (for encoding_type == 'dummy')
        
    encoding_type : 'dummy' or 'num', default = 'dummy'
        type of encoding between a numerical encoding and a dummy encoding

    """

    def __init__(
        self,
        min_modalities_number=20,
        max_modalities_number=100,
        max_cum_proba=0.95,
        min_nb_observations=10,
        max_na_percentage=0.05,
        encoding_type="dummy",
    ):
        self.min_modalities_number = min_modalities_number
        self.max_modalities_number = max_modalities_number

        self.max_cum_proba = max_cum_proba
        self.min_nb_observations = min_nb_observations
        self.max_na_percentage = max_na_percentage

        self.encoding_type = encoding_type

    def modalities_filter(self, input_serie):
        """ take a modality and filter the modalities that will be kept """
        if not isinstance(input_serie, pd.Series):
            raise TypeError("input_serie should be a pd.Series")

        value_count = input_serie.value_counts()
        nb_null = np.asarray(input_serie.isnull().values).sum() # Remark : input_serie.isnull().sum() doesn't work if sparse Serie
        if nb_null > self.max_na_percentage * len(input_serie):
            value_count["__null__"] = nb_null
            value_count.sort_values(ascending=False, inplace=True)
            # Careful : pandas behavior, change order of index with equality ...

        nb_modalities = value_count.shape[0]  # nb of different modalities (__null__ included)

        if self.min_modalities_number is not None and nb_modalities > self.min_modalities_number:
            # In that case I have too many modalities, I'll filter the one I want to keep

            NN = value_count.sum()
            to_keep = pd.Series(True, value_count.index)

            ### Filter 1 => using 'Max Cum Proba' ###
            if self.max_cum_proba is not None:
                cum_proba = value_count.cumsum().shift().fillna(0) / NN
                to_keep = to_keep & (cum_proba < self.max_cum_proba)

            ### Filter2 => using 'Min Nb Of Observations' ###
            if self.min_nb_observations is not None:
                if isinstance(self.min_nb_observations, float) and self.min_nb_observations < 1:
                    min_nb = int(NN * self.min_nb_observations)
                else:
                    min_nb = self.min_nb_observations

                to_keep = to_keep & (value_count >= min_nb)

            modalities_to_keep = value_count[to_keep]

            ### Filter 3 => If I still have too many modalities, keep only the first one ###
            if self.max_modalities_number is not None and modalities_to_keep.shape[0] > self.max_modalities_number:
                modalities_to_keep = modalities_to_keep.iloc[0 : self.max_modalities_number]

        else:
            modalities_to_keep = value_count

        mapping_dico = {
            m: k for k, m in enumerate(modalities_to_keep.index)
        }  # modality that should be kept are flagged 0,1, ...  P-1
        if len(modalities_to_keep) < len(value_count):
            mapping_dico["__default__"] = len(modalities_to_keep)

        return mapping_dico

    def fit(self, X, y=None):

        Xtype = get_type(X)
        if Xtype != DataTypes.DataFrame:
            raise TypeError("X should be a DataFrame")

        Xcolumns = list(X.columns)

        self._columns_to_encode = Xcolumns  # Force to encode everything now

        X = get_rid_of_categories(X)

        # Verif:
        if not isinstance(self._columns_to_encode, list):
            raise TypeError("_columns_to_encode should be a list")

        for c in self._columns_to_encode:
            if c not in Xcolumns:
                raise ValueError("column %s isn't in the DataFrame" % c)

        self.variable_modality_mapping = {col: self.modalities_filter(X[col]) for col in self._columns_to_encode}

        # Rmk : si on veut pas faire un encodage ou les variables sont par ordre croissant, on peut faire un randomization des numbre ici

        if self.encoding_type == "num":
            self._feature_names = self._columns_to_encode

            self.columns_mapping = {c: [c] for c in self._feature_names}

        elif self.encoding_type == "dummy":

            self.columns_mapping = {}

            index_column = {}
            self._variable_shift = {}
            cum_max = 0
            for col in self._columns_to_encode:

                self.columns_mapping[col] = []

                for i, (mod, ind) in enumerate(self.variable_modality_mapping[col].items()):
                    index_column[ind + cum_max] = col + "__" + str(mod)

                    self.columns_mapping[col].append(col + "__" + str(mod))

                self._variable_shift[col] = cum_max
                cum_max += i + 1

            self._dummy_size = cum_max
            self._dummy_feature_names = [index_column[i] for i in range(cum_max)]
            self._feature_names = self._dummy_feature_names

        else:
            raise NotImplementedError("I don't know that type of encoding %s" % self.encoding_type)

        return self

    def get_feature_names(self):
        return self._feature_names

    @staticmethod
    def _get_value(k, mapping, default=-1):
        if pd.isnull(k):
            k = "__null__"

        try:
            res = mapping[k]  # Try in mapping
        except KeyError:
            try:
                res = mapping["__default__"]  # Try in __default__
            except KeyError:
                return default

        return res

        # Rmk : peut etre qu'on peut accelerer un tout petit peu en sauver si il y a default/un null ?

    def transform(self, X):

        if get_type(X) != DataTypes.DataFrame:
            raise TypeError("X should be a DataFrame")

        X = get_rid_of_categories(X)

        result = self._transform_to_encode(X)

        return result

    def _transform_to_encode(self, X):

        all_result_series = []
        for col, mapping in self.variable_modality_mapping.items():
            default_value = -1 if "__default__" not in mapping else mapping["__default__"]
            mapping = defaultdict(lambda: default_value, mapping)
            if "__null__" in mapping:
                mapping[np.nan] = mapping["__null__"]
                mapping[None] = mapping["__null__"]
            all_result_series.append(X[col].map(mapping))

        if self.encoding_type == "num":
            result = pd.concat(all_result_series, axis=1, ignore_index=True, copy=False).astype(np.int32)
            return result

        elif self.encoding_type == "dummy":
            Xres = np.zeros((X.shape[0], self._dummy_size), dtype="int32")

            nn = np.arange(X.shape[0])
            for col, result in zip(self._columns_to_encode, all_result_series):

                resultv = result.values + self._variable_shift[col]

                ii_not_minus_one = result.values != -1

                Xres[nn[ii_not_minus_one], resultv[ii_not_minus_one]] = 1

            return pd.DataFrame(1 * Xres, index=X.index, columns=self._dummy_feature_names)

        else:
            raise NotImplementedError("I don't know that type of encoding %s" % self.encoding_type)


class NumericalEncoder(ModelWrapper):
    """ Numerical Encoder of categorical variables
    
    Parameters
    ----------
    columns_to_use : list of str
        the columns to use
        
    min_modalities_number : int, default = 20
        if less that 'min_modalities_number' modalities no modalities will be filtered
        
    max_modalities_number : int, default = 100,
        the number of modalities kept will never be more than 'max_modalities_number'
        
    max_cum_proba : float, default = 0.95
        if modalities should be filtered, first filter applied is removing modalities that account for less than 1-'max_cum_proba'
        
    min_nb_observations : int, default = 10
        if modalities should be filtered, modalities with less thant 'min_nb_observations' observations will be removed
        
    max_na_percentage : float, default = 0.05
        if more than 'max_na_percentage' percentage of missing value, None will be treated as a special modality named '__null__'
        otherwise, will just put -1 (for encoding_type == 'num') or 0 everywhere (for encoding_type == 'dummy')
        
    encoding_type : 'dummy' or 'num', default = 'dummy'
        type of encoding between a numerical encoding and a dummy encoding
  
    regex_match : boolean, default = False
        if True use regex to match columns
        
    desired_output_type : DataType
        the type of result 

    drop_used_columns : boolean, default=True
        what to do with the ORIGINAL columns that were transformed.
        If False, will keep them in the result (un-transformed)
        If True, only the transformed columns are in the result
        
    drop_unused_columns: boolean, default=True
        what to do with the column that were not used.
        if False, will drop them
        if True, will keep them in the result

    """

    def __init__(
        self,
        columns_to_use=TypeOfVariables.CAT,
        min_modalities_number=20,
        max_modalities_number=100,
        max_cum_proba=0.95,
        min_nb_observations=10,
        max_na_percentage=0.05,
        encoding_type="dummy",
        regex_match=False,
        desired_output_type=DataTypes.DataFrame,
        drop_used_columns=True,
        drop_unused_columns=False,
    ):

        self.min_modalities_number = min_modalities_number
        self.max_modalities_number = max_modalities_number

        self.max_cum_proba = max_cum_proba
        self.min_nb_observations = min_nb_observations
        self.max_na_percentage = max_na_percentage

        self.encoding_type = encoding_type

        super(NumericalEncoder, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=(DataTypes.DataFrame,),
            remove_sparse_serie=True,
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _NumericalEncoder(
            min_modalities_number=self.min_modalities_number,
            max_modalities_number=self.max_modalities_number,
            max_cum_proba=self.max_cum_proba,
            min_nb_observations=self.min_nb_observations,
            max_na_percentage=self.max_na_percentage,
            encoding_type=self.encoding_type,
        )

    @property
    def columns_mapping(self):
        return self.model.columns_mapping


# In[]


# Rmk : Fix of categorical encoder
# class _CategoricalEncoderFixer(object):
#    def fit_transform(self,X,y = None):
#        self.fit(X,y)
#        return self.transform(X,y)
#
# class TargetEncoderFixed(_CategoricalEncoderFixer,category_encoders.TargetEncoder):
#    pass
#
# class LeaveOneOutEncoderFixed(_CategoricalEncoderFixer,category_encoders.LeaveOneOutEncoder):
#    pass


class CategoricalEncoder(ModelWrapper):
    """ Wrapper around categorical encoder package encoder
    
    Parameters
    ----------
    
    columns_to_encode : None or list of str
        the columns to encode (if None will guess)
        
    encoding_type : str, default = 'dummy'
        the type of encoding, possible choices :
            * dummy
            * binary
            * basen
            * hashing
            
    basen_base : int, default = 2
        the base when using encoding_type == 'basen'
        
    hashing_n_components : int, default = 10
        the size of hashing when using encoding_type == 'hashing'
        
    columns_to_use : list of str or None
        the columns to use for that encoder
        
    regex_match : boolean
        if True will use regex to match columns
        
    desired_output_type : list of DataType
        the type of output wanted

    drop_used_columns : boolean, default=True
        what to do with the ORIGINAL columns that were transformed.
        If False, will keep them in the result (un-transformed)
        If True, only the transformed columns are in the result
        
    drop_unused_columns: boolean, default=True
        what to do with the column that were not used.
        if False, will drop them
        if True, will keep them in the result

    """

    def __init__(
        self,
        columns_to_use=TypeOfVariables.CAT,
        encoding_type="dummy",
        basen_base=2,
        hashing_n_components=10,
        regex_match=False,
        desired_output_type=DataTypes.DataFrame,
        drop_used_columns=True,
        drop_unused_columns=False,
    ):

        if category_encoders is None:
            raise ValueError("You need to install 'categorical-encoder' to use this wrapper")

        self.columns_to_use = columns_to_use
        self.encoding_type = encoding_type

        self.basen_base = basen_base
        self.hashing_n_components = hashing_n_components

        # self.columns_to_use = columns_to_use
        # self.regex_match = regex_match
        # self.desired_output_type = desired_output_type

        super(CategoricalEncoder, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
            remove_sparse_serie=True,
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=True,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):

        params = dict(cols=self.columns_to_encode, return_df=self.desired_output_type == DataTypes.DataFrame)

        if self.encoding_type == "dummy":
            return category_encoders.OneHotEncoder(use_cat_names=True, **params)

        elif self.encoding_type == "binary":
            return category_encoders.BinaryEncoder(**params)

        elif self.encoding_type == "basen":
            return category_encoders.BaseNEncoder(base=self.basen_base, **params)

        elif self.encoding_type == "hashing":
            return category_encoders.HashingEncoder(n_components=self.hashing_n_components, **params)

        #        elif self.encoding_type == "helmer":
        #            return category_encoders.HelmertEncoder(**params)
        #
        #        elif self.encoding_type == "polynomial":
        #            return category_encoders.PolynomialEncoder(**params)
        #
        #        elif self.encoding_type == "sum_coding":
        #            return category_encoders.SumEncoder(**params)
        #
        #        elif self.encoding_type == "backward_coding":
        #            return category_encoders.BackwardDifferenceEncoder(**params)
        # don't work well => change time sometimes

        # Rmk : Other categorical not included
        # * Target Encoder
        # * Leave One Out
        # MoreOver : Bug in those encoder : fit_transform does't work correctly
        # Those uses the target and I'd rather know exactly what I'm doing using tailor made classes

        else:
            raise ValueError("Unknown 'encoding_type' : %s" % self.encoding_type)


# In[]

class _OrdinalOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self,
                 categories="auto",
                 dtype=np.int32
                 ):
        self.categories=categories
        self.dtype=dtype
        
        
    def get_feature_names(self, input_features=None):
        
        if input_features is None:
            input_features = self._input_features
            
        if len(input_features) != self._nb_columns:
            raise ValueError("input_features (length = (%d)) should have the same length has the number of columns (%d)" % (len(input_features), self._nb_columns))

        features = []
        for j, f in enumerate(input_features):
            features += [str(f) + "_g_" + str(m) for m in self._all_inv_mapping[j].values[:-1]]
            
        return features

    def fit(self, X, y=None):
        
        if get_type(X) != DataTypes.DataFrame:
            raise TypeError("This transformer only works for DataFrame")
            
        if X.isnull().sum().sum() > 0:
            raise ValueError("This transformer doesn't handle null")

        self._nb_columns = X.shape[1]
            
        is_auto = isinstance(self.categories, str) and self.categories == "auto"
        
        if not is_auto:
            if len(self.categories) != X.shape[1]:
                raise TypeError("categories should be 'auto' or a list the same size as 'X.shape[1]'")

        all_mappings = []
        all_inv_mappings = []
        categories = []
        for j in range(X.shape[1]):

            current_category = None
            if not is_auto: # not automatic
                try:
                    current_category = self.categories[j] # try to find the names
                except KeyError:
                    pass

                if current_category is None:
                    try:
                        current_category = self.categories[X.columns[j]] # try again with name of column
                    except KeyError:
                        pass
                    
            if current_category is None or is_auto or (isinstance(current_category, str) and current_category == "auto"):
                target_classes_j = np.sort(np.unique(X.iloc[:, j].values))
            else:

                target_classes_j = np.array(current_category)
                uy_j = set(list(X.iloc[:, j].values))
                if len(set(list(uy_j)).difference(target_classes_j)) > 0:
                    raise ValueError("I have a categories that doesn't exist, please check")

            integers = np.arange(len(target_classes_j)).astype(self.dtype)
            mapping = pd.Series(integers, index = target_classes_j)
            inv_mapping = pd.Series(target_classes_j, index = integers)
            
            all_mappings.append(mapping)
            all_inv_mappings.append(inv_mapping)
            categories.append(target_classes_j)


        self.categories_ = categories
        self._all_mapping = all_mappings
        self._all_inv_mapping = all_inv_mappings
        
        self._input_features = list(X.columns)
        
        return self
        

    def transform(self, X):

        check_is_fitted(self, "_all_mapping")

        if get_type(X) != DataTypes.DataFrame:
            raise TypeError("This transformer only works for DataFrame")

        if X.shape[1] != self._nb_columns:
            raise ValueError("X doesn't have the correct number of columns")

        all_res = []            
        for j in range(X.shape[1]):
            
            index_line = self._all_mapping[j].loc[X.iloc[:, j]].values
            index_col  = np.arange(len(self._all_mapping[j]) - 1, dtype=np.int32)
            
            assert index_col.ndim == 1
            assert index_col.ndim == 1
            
            res_j = (index_line[:,np.newaxis] > index_col[np.newaxis,:]).astype(self.dtype)
            
            all_res.append(res_j)
            

        result = np.concatenate(all_res, axis=1)
        
        return pd.DataFrame(result, columns = self.get_feature_names(), index=X.index)

    def inverse_transform(self, X):
        
        check_is_fitted(self, "_all_mapping")

        if X.ndim != 2:
            raise TypeError("This transformer expect a two dimensional array")
            
        Xres = []
        start_slice = 0
        for j in range(self._nb_columns):
            end_slice = start_slice + len(self._all_mapping[j]) -1
            
            Xint = X.values[:, start_slice:end_slice].sum(axis=1)
            
            temp = self._all_inv_mapping[j][Xint]
            temp.index = X.index
            Xres.append(temp)
            
            start_slice = end_slice
            
        result = pd.DataFrame({col: serie for col, serie in zip(self._input_features, Xres)}, index=X.index)
        
        return result


class OrdinalOneHotEncoder(ModelWrapper):
    """ Ordinal One Hot Encoder of categorical variables
    
    This encoder create an encoding of variable that respect the natural order of the modalities
    
    For example if the columns 'X' has 3 modalities are present : 'A' < 'B' < 'C' 
    the encoding result will have 2 (= 3 - 1) columns:
        
    'X_g_A' : is X strictly greater than A
    'X_g_B' : is X strictly greater than B
    'X_g_C' : is X strictly greater than C
    
    So the modalities are encoded as follow :
        * 'A' => (0, 0, 0)
        * 'B' => (1, 0, 0)
        * 'C' => (1, 1, 0)
        * 'D' => (1, 1, 1)
    
    Parameters
    ----------
    columns_to_use : list of str
        the columns to use
        
    regex_match : boolean, default = False
        if True use regex to match columns
        
    desired_output_type : DataType
        the type of result 

    drop_used_columns : boolean, default=True
        what to do with the ORIGINAL columns that were transformed.
        If False, will keep them in the result (un-transformed)
        If True, only the transformed columns are in the result
        
    drop_unused_columns: boolean, default=True
        what to do with the column that were not used.
        if False, will drop them
        if True, will keep them in the result
        
    categories : list, dict or "auto"
        the order of the categories for each columns
        
    dtype : numpy type
        the output desired

    """

    def __init__(
        self,
        columns_to_use="all",
        regex_match=False,
        desired_output_type=DataTypes.DataFrame,
        drop_used_columns=True,
        drop_unused_columns=False,
        categories="auto",
        dtype=np.int32
    ):
        self.dtype = dtype
        self.categories=categories

        super(OrdinalOneHotEncoder, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=(DataTypes.DataFrame,),
            remove_sparse_serie=True,
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )


    def _get_model(self, X, y=None):
        return _OrdinalOneHotEncoder(categories=self.categories,
                                     dtype=self.dtype
        )

    @property
    def columns_mapping(self):
        return self.model.columns_mapping
    
    @property
    def categories_(self):
        return self.model.categories_


