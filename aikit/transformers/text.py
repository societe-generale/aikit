# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:45:15 2018

@author: Lionel Massoulard
"""

import logging

logger = logging.getLogger(__name__)
import sklearn.base

import pandas as pd
import numpy as np

import string

try:
    import nltk
except ImportError:
    nltk = None
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.exceptions import NotFittedError

from aikit.tools.helper_functions import unlist
from aikit.enums import DataTypes
from aikit.transformers.model_wrapper import ModelWrapper


try:
    from gensim.models import Word2Vec
except ImportError:
    logger.warning("I wont be able to import Word2Vec")
    Word2Vec = None


class AbstractTextProcessor(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """ This class is to create text preprocessing transformers """

    def __init__(self, concat=False):
        self.concat = concat

    def fit(self, X, y=None):
        """ fit the model """
        # stateless model => no fit
        return self

    def process_one_string(self, string):
        """ function to transform one string """
        raise NotImplementedError("should be implemented in inherited classes")

    def transform(self, X):

        if isinstance(X, list):
            return [self.process_one_string(x) for x in X]

        elif isinstance(X, pd.Series):
            return X.apply(self.process_one_string)

        elif isinstance(X, pd.DataFrame):
            Xc = X.copy()
            for c in Xc.columns:
                Xc[c] = X[c].apply(self.process_one_string)

            if self.concat:
                Xc_concat = Xc.apply(func=lambda x: " ".join(x), axis=1)
                Xc_concat = pd.DataFrame(Xc_concat, columns=["_".join(list(Xc.columns))])
            else:
                Xc_concat = Xc

            return Xc_concat

        else:
            if not hasattr(X, "shape"):
                raise ValueError("I don't know how to transform that")

            if len(X.shape) > 1:
                Xc = X.copy()
                for j in range(X.shape[1]):
                    Xc[:, j] = [self.process_one_string(x) for x in X[:, j]]

                if self.concat:
                    Xc_concat = np.apply_along_axis(func1d=lambda x: " ".join(x), axis=1, arr=Xc)[:, np.newaxis]
                else:
                    Xc_concat = Xc

                return Xc_concat

            else:
                Xc = X.copy()
                Xc[:] = [self.process_one_string(x) for x in X]  # comme ca je garde le même type
                return Xc


# class StrImputer(AbstractTextProcessor):
#    def __init__(self, replace_value = ""):
#        self.replace_value = replace_value
#
#
#    def process_one_string(self, string):
#        if pd.isnull(string):
#            return self.replace_value
#        else:
#            return string


class TextDigitAnonymizer(AbstractTextProcessor):
    """Text transformer to anonymize digits."""

    REGEX = re.compile(r"\d")

    def process_one_string(self, string):
        if pd.isnull(string):
            return None
        try:
            return TextDigitAnonymizer.REGEX.sub("#", string)
        except TypeError:
            logging.error("Invalid data type: {}, val: {}".format(type(string), string))
            raise


class TextNltkProcessing(AbstractTextProcessor):
    """Text transformer using NLKT. It can perform the following steps:

    * put the text in lower case
    * anonymize digits
    * tokenize the words
    * remove every words that doesn't contain any letter
    * remove stopwords
    * stem the rest
    
    Parameters
    ----------
    lower : boolean, default = True
        if True will put the string in lowercase
    
    digit_anonymize : boolean, default = True
        if True will anonymize digits, replacing them with 'digit_character'
        
    digit_character : string, default = '#'
        character to use to replace digits
        
    remove_non_words : boolean, default = True
        if True will remove tokens that are not sequences of letters (aka word)
        
        
    remove_stopwords : boolean, default = True
        if True will remove word that are stop words
        
    
    stem : boolean, default = True
        if True will perform stemming
        
    
    Example
    -------
    >>> texts = ["A stemmer for English operating on the stem cat should identify such strings as cats, catlike, and catty",
    "A stemming algorithm might also reduce the words fishing, fished, and fisher to the stem fish"]
    >>> transformer = TextNltkProcessing()
    >>> transformer.fit_transform(texts)
    >>> ['stemmer english oper stem cat identifi string cat catlik catti',
         'stem algorithm might also reduc word fish fish fisher stem fish']
    """

    DIGIT_REGEX = re.compile(r"\d")
    REGEX = re.compile("^[a-z#]+$")
    # TODO: parameterize stopwords languages
    try:
        STOPOWORDS = set(nltk.corpus.stopwords.words("english") + nltk.corpus.stopwords.words("french"))
    except:
        STOPOWORDS = None
    try:
        STEMMER = nltk.stem.porter.PorterStemmer()
    except:
        STEMMER = None

    def __init__(
        self,
        lower=True,
        digit_anonymize=True,
        digit_character="#",
        remove_non_words=True,
        remove_stopwords=True,
        stem=True,
        concat=False,
    ):
        if nltk is None:
            raise ValueError("Please install NLTK to use this transformer.")

        self.lower = lower
        self.digit_anonymize = digit_anonymize
        self.digit_character = digit_character
        self.remove_non_words = remove_non_words
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.concat = concat

    def process_one_string(self, string):
        """ Processing using nltk. """

        if self.lower:
            string = string.lower()  # Lower

        if self.digit_anonymize:
            string = self.DIGIT_REGEX.sub(self.digit_character, string)

        if self.remove_non_words or self.remove_stopwords or self.stem:
            words = nltk.tokenize.word_tokenize(string)  # tokenize
            if self.remove_non_words:
                words = [word for word in words if self.REGEX.match(word) is not None]

            if self.remove_stopwords:
                if self.STOPOWORDS is None:
                    raise ValueError("I couldn't load NLTK stopswords, please make sure it is available and/or install nltk")
                words = [word for word in words if word not in self.STOPOWORDS]

            if self.stem:
                if self.STEMMER is None:
                    raise ValueError("I couldn't load NLTK stemmer, please make sure it is available and/or install nltk")
                words = [self.STEMMER.stem(word) for word in words]

            return " ".join(words)

        else:
            return string


class TextDefaultProcessing(AbstractTextProcessor):
    """ Default text processing that just put everything in lower case """

    TABLE_TRANS = str.maketrans({key: " " for key in string.punctuation})
    TABLE_TRANS[ord(".")] = " . "
    TABLE_TRANS[ord("?")] = " . "
    TABLE_TRANS[ord("!")] = " . "
    TABLE_TRANS[ord(":")] = " . "
    TABLE_TRANS[ord(";")] = " . "

    REGEX = re.compile(" +")

    def process_one_string(self, string):
        lstring = self.REGEX.sub(" ", string.lower().translate(self.TABLE_TRANS)).strip()
        return lstring
        # return lstring


class CountVectorizerWrapper(ModelWrapper):
    """Wrapper around sklearn :class:`CountVectorizer` with additional capabilities:

    * can select its columns to keep/drop
    * work on more than one columns
    * can return a DataFrame
    * can add a prefix to the name of columns

    Parameters
    ----------
    See sklearn.CountVectorizer for complete list
    
    analyzer : str, default = "word"
        type of analyzer ("char","word","char wb")
        

    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    ngram_range : tuple (min_n, max_n) or int (1, ngram_range)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    max_features : int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.


    tfidf : boolean, default = False
        if True will use a TfIdfVectorizer, otherwise regular CountVectorizer

    columns_to_use : None or list of string
        this parameter will allow the wrapped transformer to select its columns
        
    regex_match : boolean, default = False
        if True will use a regex to match columns otherwise exact match
        
    column_prefix : str or None, default = "BAG"
        prefix of the column

    drop_used_columns : boolean, default=True
        what to do with the ORIGINAL columns that were transformed.
        If False, will keep them in the result (un-transformed)
        If True, only the transformed columns are in the result
        
    drop_unused_columns: boolean, default=True
        what to do with the column that were not used.
        if False, will drop them
        if True, will keep them in the result

    desired_output_type : None or DataType
        specify the desired output type of transformer, a conversion will be made if necesary


    """

    def __init__(
        self,
        analyzer="word",
        max_df=1.0,
        min_df=1,
        ngram_range=1,
        max_features=None,
        vocabulary=None,
        tfidf=False,
        columns_to_use="all",
        regex_match=False,
        desired_output_type=DataTypes.SparseArray,
        column_prefix="BAG",
        drop_used_columns=True,
        drop_unused_columns=True,
        **other_count_vectorizer_arguments
    ):
        self.analyzer = analyzer
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.tfidf = tfidf

        self.columns_to_use = columns_to_use
        self.regex_match = regex_match
        self.desired_output_type = desired_output_type
        self.column_prefix = column_prefix

        self.other_count_vectorizer_arguments = other_count_vectorizer_arguments

        super(CountVectorizerWrapper, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=True,
            all_columns_at_once=False,
            accepted_input_types=(DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.Serie),
            column_prefix=column_prefix,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):

        if not isinstance(self.ngram_range, (tuple, list)):
            ngram_range = (1, self.ngram_range)
        else:
            ngram_range = self.ngram_range

        ngram_range = tuple(ngram_range)

        other_params = {k: v for k, v in self.other_count_vectorizer_arguments.items()}  # shalow copy
        if "dtype" not in other_params:
            other_params["dtype"] = "int32"  # force output to be int32 to limit memory

        if self.tfidf:
            return TfidfVectorizer(
                analyzer=self.analyzer,
                max_df=self.max_df,
                min_df=self.min_df,
                ngram_range=ngram_range,
                max_features=self.max_features,
                vocabulary=self.vocabulary,
                **other_params
            )

        else:
            return CountVectorizer(
                analyzer=self.analyzer,
                max_df=self.max_df,
                min_df=self.min_df,
                ngram_range=ngram_range,
                max_features=self.max_features,
                vocabulary=self.vocabulary,
                **other_params
            )


class _Word2VecVectorizer(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """ wrapper around Word2Vec, learns word2vec and then use average embedding """

    def __init__(
        self,
        size=100,
        window=5,
        digit_anonymize=False,
        text_preprocess="default",
        same_embedding_all_columns=True,
        use_fast_text=False,
        random_state=None,
        min_count=5,
        other_params=None,
    ):

        self.size = size
        self.window = window

        self.text_preprocess = text_preprocess
        self.same_embedding_all_columns = same_embedding_all_columns
        self.use_fast_text = use_fast_text

        self.random_state = random_state

        self.min_count=min_count
        self.other_params = other_params

        self.models = None
        self._digit_anonymizer = None
        self._nltk_processing = None

        if use_fast_text:
            raise NotImplementedError("I didn't code fasttext wrapping yet, please use gensim")
        else:
            if Word2Vec is None:
                raise ValueError("You need to install Word2Vec")

    def _fit_transform(self, X, y=None, do_fit=True, do_transform=True):

        if not isinstance(X, pd.DataFrame):
            raise NotImplementedError("I didn't code that case yet")

        ###############################
        ### 1) Create preprocessing ###
        ###############################

        if do_fit:
            if self.text_preprocess is None:
                self._text_preprocessor = None

            elif self.text_preprocess == "default":
                self._text_preprocessor = TextDefaultProcessing()

            elif self.text_preprocess == "digit":
                self._text_preprocessor = TextDigitAnonymizer()

            elif self.text_preprocess == "nltk":
                self._text_preprocessor = TextNltkProcessing()

        ##############################
        ### 2) Apply preprocessing ###
        ##############################

        if self._text_preprocessor is not None:
            if do_fit:
                newX = self._text_preprocessor.fit_transform(X)
            else:
                newX = self._text_preprocessor.transform(X)
        else:
            newX = X

        ######################
        ### 3) Split text  ###
        ######################
        if do_fit:
            self._nbcols = newX.shape[1]
        else:
            if newX.shape[1] != self._nbcols:
                raise ValueError(
                    "I don't have the correct number of columns %d, expected %d"(newX.shape[1], self._nbcols)
                )

        Xsplitted = [[x.split() for x in newX.iloc[:, j]] for j in range(newX.shape[1])]
        # Here : Xsplitted is a list, one item for each text column
        # each item, is also a list (one item per observations)
        # each item is a list of word

        # => Xsplitted = [list of list of list of words]

        #######################
        ### 4) Fit word2vec ###
        #######################
        if do_fit:
            if self.other_params is None:
                other_params = {}
            else:
                other_params = self.other_params

            if self.use_fast_text:
                raise NotImplementedError("")

            if self.same_embedding_all_columns:
                ##############################################
                ### One embedding for ALL the text columns ###
                ##############################################

                Xsplitted_all = []
                for Xs in Xsplitted:
                    Xsplitted_all += Xs
                # Unlist everything

                model = Word2Vec(size=self.size, window=self.window, seed=self.random_state, workers=1, min_count=self.min_count, **other_params)
                model.build_vocab(Xsplitted_all)
                if not model.wv.vocab:
                    raise ValueError("Empty vocabulary, please change 'min_count'")
                model.train(Xsplitted_all, total_examples=model.corpus_count, epochs=model.epochs)

                self.models = [model for j in range(self._nbcols)]  # j time the same model, model train on everything

            else:
                ######################################
                ### One embedding PER text columns ###
                ######################################
                self.models = []
                for jj, Xs in enumerate(Xsplitted):
                    seed = self.random_state + jj if self.random_state else None
                    model = Word2Vec(size=self.size, window=self.window, seed=seed, workers=1, min_count=self.min_count, **other_params)
                    model.build_vocab(Xs) # For some reason Word2Vec doesn't with few sample ....
                    if not model.wv.vocab:
                        raise ValueError(f"Empty vocabulary for column {jj}, please change 'min_count'")
                    model.train(Xs, total_examples=model.corpus_count, epochs=model.epochs)

                    self.models.append(model)

            self._features_names = []
            for j in range(self._nbcols):
                self._features_names += ["%s__EMB__%d" % (X.columns[j], w) for w in range(self.size)]

        if not do_transform:
            return self

        if self.models is None:
            raise NotFittedError("You must fit the model first")

        #########################
        ### 5) Apply Word2Vec ###
        #########################

        # TODO : make an optimized version (with numba maybe... )
        # TODO : allow the return of a tensor
        XXres = np.zeros((X.shape[0], self.size * self._nbcols), dtype=np.float32)
        for j, (modelj, Xs) in enumerate(zip(self.models, Xsplitted)):

            for i, sentence in enumerate(Xs):

                count = 0

                for word in sentence:
                    try:
                        emb = modelj.wv[word]
                    except KeyError:
                        emb = None

                    if emb is not None:
                        count += 1
                        XXres[i, (self.size * j) : (self.size * (j + 1))] += emb

                if count > 0:
                    XXres[i, (self.size * j) : (self.size * (j + 1))] /= count

        return pd.DataFrame(XXres, columns=self._features_names, index=X.index)

    def get_feature_names(self):
        return self._features_names

    def fit(self, X, y=None):
        self._fit_transform(X=X, y=y, do_fit=True, do_transform=False)
        return self

    def fit_transform(self, X, y=None):
        return self._fit_transform(X=X, y=y, do_fit=True, do_transform=True)

    def transform(self, X):
        return self._fit_transform(X, y=None, do_fit=False, do_transform=True)


class Word2VecVectorizer(ModelWrapper):
    """ Word2Vec vectorizer, this model does an average of the embedding of each word :
        it is sometimes called 'Continuous Bag of Word'
        
    Parameter
    ---------
    size : int, default = 100
        the size of the embedding
    
    window : int, default = 5
        the size of the training window of the word2vec model
        
    text_preprocess: string, default = 'default'
        type of text preprocessing to use, possible choices are :
        * 'default' : TextDefaultProcessing : put everything in lower case and remove some punctuation
        * 'digit'   : TextDigitAnonymizer   : anonymize digits
        * 'nltk'    : TextNltkProcessing    : lower, stemming, remove stopwords, ...
        * None      : do nothing 
            
   same_embedding_all_columns : boolean, default = True
       if True will fit ONE embedding for ALL the text columns, otherwise will fit one word2vec PER text column
       
   use_fast_text : boolean, default = False
       if True will use fasttext instead of gensim
       
   random_state : None or int
       state of random generator
       
   other_params : dict or None, default = None
       if not None, additional parameters to be passed to the word2vec model
       
    columns_to_use : list,
        columns to encode
        
    desired_output_type : data type, default = DataFrame
        desired output type
        
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
        size=100,
        window=5,
        min_count=5,
        text_preprocess="default",
        same_embedding_all_columns=True,
        use_fast_text=False,
        random_state=None,
        other_params=None,
        columns_to_use="all",
        desired_output_type=DataTypes.DataFrame,
        regex_match=False,
        drop_used_columns=True,
        drop_unused_columns=True,
    ):

        self.size = size
        self.window = window
        self.min_count=min_count
        self.text_preprocess = text_preprocess
        self.same_embedding_all_columns = same_embedding_all_columns
        self.use_fast_text = use_fast_text
        self.random_state = random_state
        self.other_params = other_params

        self.columns_to_use = columns_to_use
        self.desired_output_type = desired_output_type
        self.regex_match = regex_match

        super(Word2VecVectorizer, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=(DataTypes.DataFrame,),
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):

        return _Word2VecVectorizer(
            size=self.size,
            window=self.window,
            min_count=self.min_count,
            text_preprocess=self.text_preprocess,
            same_embedding_all_columns=self.same_embedding_all_columns,
            use_fast_text=self.use_fast_text,
            random_state=self.random_state,
            other_params=self.other_params,
        )


def _retrieve_all_rolling_string_parts(string, ngram=4):
    res = []
    nb_split = len(string) // ngram
    for k in range(ngram):

        temp_res = [string[(k + (ngram * i)) : (k + (ngram * (i + 1)))] for i in range(nb_split)]
        temp_res = [r for r in temp_res if len(r) == ngram]

        res.append(temp_res)
    return res


class _Char2VecVectorizer(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(
        self,
        size=50,
        window=5,
        ngram=3,
        text_preprocess=None,
        same_embedding_all_columns=True,
        use_fast_text=False,
        random_state=None,
        other_params=None,
    ):

        self.size = size
        self.window = window
        self.ngram = ngram

        self.text_preprocess = text_preprocess
        self.same_embedding_all_columns = same_embedding_all_columns
        self.use_fast_text = use_fast_text

        self.random_state = random_state
        self.other_params = other_params

        self.models = None
        self._text_preprocessor = None

        if use_fast_text:
            raise NotImplementedError("I didn't code fasttext wrapping yet, please use gensim")
        else:
            if Word2Vec is None:
                raise ValueError("You need to install Word2Vec")

    def _fit_transform(self, X, y, do_fit, do_transform):

        ###############################
        ### 1) Create preprocessing ###
        ###############################
        if do_fit:
            if self.text_preprocess is None:
                self._text_preprocessor = None

            elif self.text_preprocess == "default":
                self._text_preprocessor = TextDefaultProcessing()

            elif self.text_preprocess == "digit":
                self._text_preprocessor = TextDigitAnonymizer()

            elif self.text_preprocess == "nltk":
                self._text_preprocessor = TextNltkProcessing()

        ##############################
        ### 2) Apply preprocessing ###
        ##############################

        if self._text_preprocessor is not None:
            if do_fit:
                newX = self._text_preprocessor.fit_transform(X)
            else:
                newX = self._text_preprocessor.transform(X)
        else:
            newX = X

        if do_fit:
            self._nbcols = newX.shape[1]
        else:
            if newX.shape[1] != self._nbcols:
                raise ValueError(
                    "I don't have the correct number of columns %d, expected %d"(newX.shape[1], self._nbcols)
                )

        #######################################################
        ### 2) get all sub string of length 'self.nb_chars' ###
        #######################################################

        Xsplitted = [
            [_retrieve_all_rolling_string_parts(string, ngram=self.ngram) for string in newX.iloc[:, j]]
            for j in range(self._nbcols)
        ]

        #################################
        ### 3) fit Word2Vec embedding ###
        #################################
        if do_fit:
            if self.other_params is None:
                other_params = {}
            else:
                other_params = self.other_params

            if self.same_embedding_all_columns:
                ##############################################
                ### One embedding for ALL the text columns ###
                ##############################################

                Xsplitted_all = []
                for Xs in Xsplitted:
                    Xsplitted_all += unlist(Xs)

                model = Word2Vec(size=self.size, window=self.window, seed=self.random_state, **other_params)
                model.build_vocab(Xsplitted_all)
                model.train(Xsplitted_all, total_examples=model.corpus_count, epochs=model.epochs)

                self.models = [model for j in range(self._nbcols)]  # j time the same model, model train on everything

            else:
                ######################################
                ### One embedding PER text columns ###
                ######################################
                self.models = []
                for jj, Xs in enumerate(Xsplitted):
                    seed = self.random_state + jj if self.random_state else None
                    uXs = unlist(Xs)

                    model = Word2Vec(size=self.size, window=self.window, seed=seed, **other_params)
                    model.build_vocab(uXs)
                    model.train(uXs, total_examples=model.corpus_count, epochs=model.epochs)

                    self.models.append(model)

            self._features_names = []
            for j in range(self._nbcols):
                self._features_names += ["%s__EMB__%d" % (X.columns[j], w) for w in range(self.size)]

        if not do_transform:
            return self

        if self.models is None:
            raise NotFittedError("You must fit the model first")

        #########################
        ### 5) Apply Word2Vec ###
        #########################

        # Rmk : il faudrait refaire ca en vectorialisee... ou peut etre accelerer avec numba
        XXres = np.zeros((X.shape[0], self.size * self._nbcols), dtype=np.float32)
        for j, (modelj, Xs) in enumerate(zip(self.models, Xsplitted)):

            for i, sentence in enumerate(Xs):
                count = 0
                for k, sub_sentence in enumerate(sentence):

                    for word in sub_sentence:
                        try:
                            emb = modelj.wv[word]
                        except KeyError:
                            emb = None

                        if emb is not None:
                            count += 1
                            XXres[i, (self.size * j) : (self.size * (j + 1))] += emb

                if count > 0:
                    XXres[i, (self.size * j) : (self.size * (j + 1))] /= count

        return pd.DataFrame(XXres, columns=self._features_names, index=X.index)

    def get_feature_names(self):
        return self._features_names

    def fit(self, X, y=None):
        self._fit_transform(X=X, y=y, do_fit=True, do_transform=False)
        return self

    def fit_transform(self, X, y=None):
        return self._fit_transform(X=X, y=y, do_fit=True, do_transform=True)

    def transform(self, X):
        return self._fit_transform(X, y=None, do_fit=False, do_transform=True)


class Char2VecVectorizer(ModelWrapper):
    """ Char2Vec vectorizer, this model does an average of the embedding of each ngram :
        it is sometimes called 'Continuous Bag of Word'
        
    Parameter
    ---------
    size : int, default = 50
        the size of the embedding
    
    window : int, default = 5
        the size of the training window of the word2vec model
        
    ngram : int, default = 3
        the size of the ngram on which we will fit embedding
        
    text_preprocess: string, default = 'default'
        type of text preprocessing to use, possible choices are :
        * 'default' : TextDefaultProcessing : put everything in lower case and remove some punctuation
        * 'digit'   : TextDigitAnonymizer   : anonymize digits
        * 'nltk'    : TextNltkProcessing    : lower, stemming, remove stopwords, ...
        * None      : do nothing 
            
   same_embedding_all_columns : boolean, default = True
       if True will fit ONE embedding for ALL the text column, otherwise will fit one word2vec PER text column
       
   random_state : None or int
       state of random generator
       
   other_params : dict or None, default = None
       if not None, additional parameters to be passed to the word2vec model
       
    columns_to_use : list,
        columns to encode
        
    desired_output_type : data type, default = DataFrame
        desired output type
        
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
        size=100,
        window=5,
        ngram=3,
        text_preprocess="default",
        same_embedding_all_columns=True,
        use_fast_text=False,
        random_state=None,
        other_params=None,
        columns_to_use="all",
        desired_output_type=DataTypes.DataFrame,
        regex_match=False,
        drop_used_columns=True,
        drop_unused_columns=True,
    ):

        self.size = size
        self.window = window
        self.ngram = ngram
        self.text_preprocess = text_preprocess
        self.same_embedding_all_columns = same_embedding_all_columns
        self.use_fast_text = use_fast_text
        self.random_state = random_state
        self.other_params = other_params

        self.columns_to_use = columns_to_use
        self.desired_output_type = desired_output_type
        self.regex_match = regex_match

        super(Char2VecVectorizer, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=(DataTypes.DataFrame,),
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):

        return _Char2VecVectorizer(
            size=self.size,
            window=self.window,
            ngram=self.ngram,
            text_preprocess=self.text_preprocess,
            same_embedding_all_columns=self.same_embedding_all_columns,
            use_fast_text=self.use_fast_text,
            random_state=self.random_state,
            other_params=self.other_params,
        )
