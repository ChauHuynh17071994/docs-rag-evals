import math

import pytest

from rag.embeddings import DenseIndex, HashingEmbedder, cosine, get_embedder


def test_vectors_are_unit_length_and_deterministic():
    emb = HashingEmbedder(dim=256)
    a, b = emb.embed(["hotel rate per night", "hotel rate per night"])
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
    assert a == b


def test_empty_text_gives_zero_vector():
    (vec,) = HashingEmbedder(dim=64).embed(["the"])  # only a stopword
    assert not any(vec)


def test_morphological_variants_are_similar():
    emb = HashingEmbedder()
    q, plural, other = emb.embed(["password length", "Passwords must be long", "hotel rate"])
    assert cosine(q, plural) > cosine(q, other)
    assert cosine(q, plural) > 0.2


def test_dense_index_finds_variant_that_bm25_would_miss():
    docs = ["Passwords must be at least 14 characters", "The daily allowance is 52 euros"]
    index = DenseIndex(docs, HashingEmbedder())
    assert index.search("minimum password", k=1)[0][0] == 0


def test_get_embedder_rejects_unknown_names():
    assert get_embedder("hashing").name == "hashing"
    with pytest.raises(ValueError):
        get_embedder("nope")
