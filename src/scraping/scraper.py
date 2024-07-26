import json
import logging
import os
import typing
from pprint import pprint
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

from constants import CACHE_DIR, CACHE_FILE, WIKI_HOMEPAGE_ROOT, WIKI_ITEMS_HOMEPAGE
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
    def fetch_page(url: str, use_cached_results: bool = True) -> str | None:
        """Fetches the HTML content of a given webpage.

        Args:
            url (str): The URL of the webpage to fetch.

            use_cached_results (bool): If False, always make a GET request to the provided URL.
                When True, true to utilize scraper_cache/cache.json, which stores previous
                responses. If the url is found in the cache, do not make another GET request.
                Requests are always cached independent of the value of use_cached_results.

        Returns:
            The HTML content of the page else None if the request fails.
        """
        # create the cache dir and load the cache.
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache: dict[str, str] = json.load(f)
        except FileNotFoundError:
            cache = {}

        if use_cached_results and url in cache:
            logger.info("fetch_page: Cache hit for url: %s", url)
            return cache[url]

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            html_content = response.text

            cache[url] = html_content
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=4)

            logger.info("fetch_page: Successfully got response for url and cached it: %s", url)
            return html_content

        except requests.HTTPError as e:
            logger.error("fetch_page: Failed to retrieve page for url %s, err: %s", url, str(e))
            return None

    @typing.no_type_check  # mypy gets upset with all the html stuff, don't want tons of None checks.
    @staticmethod
    def parse_isaac_items_from_html(html: str) -> list[IsaacItem]:
        """Parses IsaacItem objects from the HTML content.

        Args:
            html (str): The HTML content of the webpage.

        Returns:
            A list of all IsaacItems from the HTML
        """
        soup = BeautifulSoup(html, "html.parser")

        # the items on the Isaac wiki page are organized in tables.
        # each <tr> in the tables has the class "row-collectible".
        html_tr_elems: list[Tag] = soup.find_all("tr", {"class": "row-collectible"})

        isaac_items: list[IsaacItem] = []
        for tr in html_tr_elems:
            # the tables on the Isaac Items wiki page have columns laid out like this:
            # Name | ID | Icon | Quote | Description | Quality
            all_tds: list[Tag] = tr.find_all("td")

            # 1st parse the item name and link to the Wiki page, such as "Guppy's Head"
            name_td = all_tds[0]
            name = name_td.find("a").text.strip()
            relative_wiki_url = name_td.find("a")["href"]

            # parse the item id
            item_id_td = all_tds[1]
            item_id = item_id_td.text.strip()

            # get the img url
            img_td = all_tds[2]
            img_url = img_td.find("img")["data-src"]  # the static content link is stored in the data-src attr

            # get the item pickup quote
            quote_td = all_tds[3]
            quote = quote_td.find("i").text.strip()

            # item description
            desc_td = all_tds[4]
            description = " ".join(desc_td.stripped_strings)  # this deals w/ nested links and multiple parts

            # item quality
            item_quality_td = all_tds[5]
            item_quality = item_quality_td.text.strip()

            isaac_items.append(
                IsaacItem(
                    name=name,
                    item_id=item_id,
                    img_url=img_url,
                    wiki_url=urljoin(WIKI_HOMEPAGE_ROOT, relative_wiki_url),
                    description=description,
                    item_quality=item_quality,
                    quote=quote,
                )
            )

        logger.info(
            "parse_isaac_items_from_html: found %d IsaacItems",
            len(isaac_items),
        )
        return isaac_items


if __name__ == "__main__":
    # let's fetch the HTML, prase it, and see the first and last item.
    # the first item should be "A Pony" and last "Tonsil" as of 7/25/2024 on the wiki.
    fetched_html = Scraper.fetch_page(WIKI_ITEMS_HOMEPAGE)
    parsed_isaac_items = Scraper.parse_isaac_items_from_html(fetched_html)
    pprint(parsed_isaac_items[0].to_dict())
    pprint(parsed_isaac_items[-1].to_dict())
