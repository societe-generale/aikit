from aikit.transformers import CountVectorizerWrapper, Word2VecVectorizer, Char2VecVectorizer, TextNltkProcessing, \
    TextDefaultProcessing, TextDigitAnonymizer
from ._base import ModelRepresentationBase
from ..hyper_parameters import HyperComposition, HyperRangeInt, HyperChoice, \
    HyperCrossProduct, HyperRangeBetaInt
from .._registry import register
from ...enums import StepCategory, VariableType

try:
    import nltk
except ImportError:
    nltk = None
    print("NLTK not available, AutoML won't run with NLTK transformers")

try:
    import gensim
except ImportError:
    gensim = None
    print("Gensim not available, AutoML won't run with Gensim models")


@register
class CountVectorizerTextEncoder(ModelRepresentationBase):
    klass = CountVectorizerWrapper
    category = StepCategory.TextEncoder
    type_of_variable = VariableType.TEXT
    use_y = False  # It means in 'approx_cv mode' CV won't be done
    use_for_block_search = True

    @classmethod
    def get_hyper_parameter(cls):
        # Specific function to handle the fact that I don't want ngram != 1 IF analyzer = word
        res = HyperComposition(
            [
                (
                    0.5,
                    HyperCrossProduct(
                        {
                            "ngram_range": 1,
                            "analyzer": "word",
                            "min_df": [1, 0.001, 0.01, 0.05],
                            "max_df": [0.999, 0.99, 0.95],
                            "tfidf": [True, False],
                        }
                    ),
                ),
                (
                    0.5,
                    HyperCrossProduct(
                        {
                            "ngram_range": HyperRangeBetaInt(
                                start=1, end=5, alpha=2, beta=1
                            ),  # 1 = 1.5% ; 2 = 12% ; 3 = 25% ; 4 = 37% ; 5 = 24%
                            "analyzer": HyperChoice(("char", "char_wb")),
                            "min_df": [1, 0.001, 0.01, 0.05],
                            "max_df": [0.999, 0.99, 0.95],
                            "tfidf": [True, False],
                        }
                    ),
                ),
            ]
        )
        return res


if gensim is not None:
    @register
    class Word2VecVectorizerTextEncoder(ModelRepresentationBase):
        klass = Word2VecVectorizer
        category = StepCategory.TextEncoder
        type_of_variable = VariableType.TEXT
        custom_hyper = {
            "size": HyperRangeInt(50, 300, step=10),
            "window": [3, 5, 7],
            "same_embedding_all_columns": [True, False],
            "text_preprocess": [None, "default", "digit", "nltk"],
        }
        type_of_model = None
        use_y = False

if gensim is not None:
    @register
    class Char2VecVectorizerTextEncoder(ModelRepresentationBase):
        klass = Char2VecVectorizer
        category = StepCategory.TextEncoder
        type_of_variable = VariableType.TEXT
        custom_hyper = {
            "size": HyperRangeInt(50, 300, step=10),
            "window": [3, 5, 7],
            "ngram": HyperRangeInt(2, 6),
            "same_embedding_all_columns": [True, False],
            "text_preprocess": [None, "default", "digit", "nltk"],
        }
        type_of_model = None
        use_y = False

if nltk is not None:
    @register
    class TextNltkProcessingTextPreprocessor(ModelRepresentationBase):
        klass = TextNltkProcessing
        category = StepCategory.TextPreprocessing
        type_of_variable = VariableType.TEXT
        type_of_model = None
        use_y = False


@register
class TextProcessingDefaultPreprocessor(ModelRepresentationBase):
    klass = TextDefaultProcessing
    category = StepCategory.TextPreprocessing
    type_of_variable = VariableType.TEXT
    type_of_model = None
    use_y = False


@register
class TextProcessingDigitAnonymizer(ModelRepresentationBase):
    klass = TextDigitAnonymizer
    category = StepCategory.TextPreprocessing
    type_of_variable = VariableType.TEXT
    type_of_model = None
    use_y = False
