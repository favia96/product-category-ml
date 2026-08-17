
from shop_ml.preprocess.tokenizer import normalize, tokenize, join_name_brand

def test_normalize_lowercases_and_strips():
    assert normalize(" ÀÉ  ") == " àé"

def test_tokenize_basic():
    assert tokenize("Sony 55\" TV - 4K") == ["sony", "55", "tv", "4k"]

def test_join_name_brand():
    assert join_name_brand("iphone 15", "apple") == "iphone 15 [sep] apple"
