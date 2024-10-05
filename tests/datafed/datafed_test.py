import pytest
from m3util.datafed.datafed import search_for_alias


class MockItem:
    def __init__(self, alias, id):
        self.alias = alias
        self.id = id


class MockListingReply:
    def __init__(self, items):
        self.item = items


@pytest.fixture
def mock_listing_reply():
    items = [
        MockItem(alias="alias1", id="id1"),
        MockItem(alias="alias2", id="id2"),
        MockItem(alias="alias3", id="id3"),
    ]
    return MockListingReply(items)


def test_search_for_alias_found(mock_listing_reply):
    result = search_for_alias(mock_listing_reply, "alias2")
    assert result == "id2", "Expected to find alias2 with id2"


def test_search_for_alias_not_found(mock_listing_reply):
    result = search_for_alias(mock_listing_reply, "alias4")
    assert result is None, "Expected not to find alias4"


def test_search_for_alias_empty_listing():
    empty_listing_reply = MockListingReply([])
    result = search_for_alias(empty_listing_reply, "alias1")
    assert result is None, "Expected not to find alias1 in an empty listing"
