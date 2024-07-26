from dataclasses import asdict, dataclass


@dataclass
class IsaacItem:
    """
    Dataclass to encapsulate all important info for an Isaac item.
    """

    # an item's name, such as "Guppy's Head"
    name: str

    # item's id, such as "5.100.145"
    item_id: str

    # ex: https://static.wikia.nocookie.net/bindingofisaacre_gamepedia/images/3/35/Collectible_Guppy%27s_Head_icon.png/revision/latest?cb=20210821042544
    img_url: str

    # ex: https://bindingofisaacrebirth.fandom.com/wiki/Guppy%27s_Head
    wiki_url: str

    # this "description" field will store all description sentences as a flattened list of strings.
    description: str

    # item quality can be one of [0-4] discretely. ex: Guppy's head is 2.
    # we'll store these as a string since it can also be empty (no quality / removed item).
    item_quality: str

    # ex: "Reusable fly hive" for Guppy's Head. What you see when in-game you pick it up.
    quote: str

    def to_dict(self) -> dict:
        """Get the dictionary reprentation of the IsaacItem dataclass."""
        return asdict(self)
