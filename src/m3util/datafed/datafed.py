from datafed.CommandLib import API


def search_for_alias(listing_reply, target_alias):
    """
    Searches for an item with a matching alias within a ListingReply object.

    Args:
        listing_reply (SDMS_Auth_pb2.ListingReply): The ListingReply object containing a list of items to search through.
        target_alias (str): The alias to search for in the ListingReply's items.

    Returns:
        str or None: The ID of the item with the matching alias if found, otherwise None.

    Example:
        ```python
        result = search_for_alias(listing_reply, "my_alias")
        if result:
            print(f"Found item ID: {result}")
        else:
            print("Alias not found.")
        ```

    This function iterates over the `item` field of the provided `listing_reply`,
    which is assumed to contain a list of items. Each item is expected to have an
    `alias` attribute. If an item's `alias` matches the provided `target_alias`,
    the function returns the corresponding item's `id`. If no such item is found,
    the function returns None.
    """
    # Iterate over each item in the ListingReply's 'item' field.
    for item in listing_reply.item:
        # Check if the current item's alias matches the target alias.
        if item.alias == target_alias:  # Assuming each item has an 'alias' field.
            return item.id  # Return the item's ID if a match is found.

    # If no matching alias is found, return None.
    return None
