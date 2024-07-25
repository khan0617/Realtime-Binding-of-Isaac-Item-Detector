import logging
import os

import requests
from bs4 import BeautifulSoup

from constants import WIKI_ITEMS_HOMEPAGE
from logging_config import configure_logging
from scraping.isaac_item import IsaacItem

configure_logging()
logger = logging.getLogger(__name__)


class Scraper:
    """
    Scraper is responsible for collecting all Isaac item info we need from the Wiki's HTML.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def fetch_page(url: str) -> str | None:
        """
        Fetches the HTML content of a given webpage.

        Args:
            url (str): URL of the webpage to fetch.

        Returns:
            str | None: HTML content of the page else None if the request fails.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            logger.info(f"fetch_page: Successfully got response for {url = }")
            return response.text
        except requests.HTTPError as e:
            logger.error(f"fetch_page: Failed to retrieve page for {url = }, {e}")
            return None

    @staticmethod
    def parse_isaac_items_from_html(html: str) -> list[IsaacItem] | None:
        """
        Parses IsaacItems from the HTML content.

        Args:
            html (str): The HTML content of the webpage.

        Returns:
            list: A list of all IsaacItems from the HTML, None on failure.
        """
        soup = BeautifulSoup(html, "html.parser")

        # the items on the Isaac wiki page are organized in tables.
        # each <tr> in the tables has the class "row-collectible".
        html_tr_elems = soup.find_all("tr", {"class": "row-collectible"})

        # 1st td in the tr: the Item name (with an href to the wiki)
        # so we could access the item name like td.a.text or however that is done (see below)
        # <td data-sort-value="Tooth Picks"><a href="/wiki/Tooth_Picks" title="Tooth Picks">Tooth Picks</a></td>

        # 2nd td (item id) looks like this:
        # <td data-sort-value="183"><span class="ghost">5.100.</span>183</td>

        # 3rd td is for the image link.
        # basically goes td > a > access image link via the <a>'s src attribute.

        # 4th td is for the quote (in the "i" tag):
        # <td><i>Tears + shot speed up</i></td>

        # 5th td is the item description. Example for Tooth Picks:
        # <td>+0.7 <a href="/wiki/Tears" title="Tears">tears</a>, +0.16 <a href="/wiki/Shot_speed" class="mw-redirect" title="Shot speed">shot speed</a>.</td>

        return list()
