import json
import logging
import os
import typing
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

from constants import (
    BROKEN_SHOVEL_ACTIVE_ID,
    BROKEN_SHOVEL_PASSIVE_ID,
    CACHE_DIR,
    CACHE_FILE,
    DATA_DIR,
    JSON_DUMP_FILE,
    UNMODIFIED_FILE_NAME,
    WIKI_HOMEPAGE_ROOT,
    WIKI_ITEMS_HOMEPAGE,
)
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
    def get_encoded_name_from_url(url: str) -> str:
        """
        Extracts the URL-encoded name from a wiki URL.

        Args:
            url (str): The URL of the wiki page.

        Returns:
            str: The URL-encoded item name.
        """
        path = urllib.parse.urlparse(url).path
        encoded_name = path.split("/")[-1]
        return encoded_name

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

    @typing.no_type_check  # mypy gets upset with all the html stuff, don't want to add tons of `None` checks.
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

            # this deals w/ nested links and extra spaces in the string
            # maybe not efficient but more readable than regex...
            description = " ".join(desc_td.stripped_strings).replace(" ,", ",").replace(" .", ".").replace(" )", ")")

            # item quality
            item_quality_td = all_tds[5]
            item_quality = item_quality_td.text.strip()

            # get the wiki_url and url_encoded_name from our parsed data
            wiki_url = urllib.parse.urljoin(WIKI_HOMEPAGE_ROOT, relative_wiki_url)
            url_encoded_name = Scraper.get_encoded_name_from_url(wiki_url)

            # EDGE CASE! There are two "Broken Shovel" items. ID 5.100.550 is the "Active" one, the first piece.
            # Item ID 5.100.551 is the 2nd broken shovel piece, the passive collectible.
            # we need to account for these here.
            if item_id == BROKEN_SHOVEL_ACTIVE_ID:
                name = "Broken Shovel (Active)"
            elif item_id == BROKEN_SHOVEL_PASSIVE_ID:
                name = "Broken Shovel (Passive)"

            if item_id in [BROKEN_SHOVEL_PASSIVE_ID, BROKEN_SHOVEL_ACTIVE_ID]:
                # doing this lets us have 2 unique dirs for the broken shovel items
                img_dir = name
            else:
                img_dir = url_encoded_name

            isaac_items.append(
                IsaacItem(
                    name=name,
                    item_id=item_id,
                    img_url=img_url,
                    wiki_url=wiki_url,
                    description=description,
                    item_quality=item_quality,
                    quote=quote,
                    url_encoded_name=url_encoded_name,
                    img_dir=img_dir,
                )
            )

        logger.info(
            "parse_isaac_items_from_html: found %d IsaacItems",
            len(isaac_items),
        )
        return isaac_items

    @staticmethod
    def _download_item_image(isaac_item: IsaacItem, save_dir: str) -> None:
        """Threadpool helper: Downloads the image for this IsaacItem and saves it to the specified directory.

        All item images will be downloaded into {save_dir}/{item_name}/UNMODIFIED_FILE_NAME
        For example, the image for "Guppy's Head" will be like: {save_dir}/Guppy's Head/UNMODIFIED_FILE_NAME

        Args:
            isaac_item (IsaacItem): The IsaacItem we want to download an image for.
            save_dir (str): The root directory where the image will be saved.
        """
        # define the directory and file path using the encoded name
        item_dir = os.path.join(save_dir, isaac_item.img_dir)
        save_path = os.path.join(item_dir, UNMODIFIED_FILE_NAME)

        # make sure the destination directory exists
        os.makedirs(item_dir, exist_ok=True)

        # if the path exists, we already have the image, don't need to download!
        if os.path.exists(save_path):
            logger.debug("Image for %s already exists at %s", isaac_item.name, save_path)
            return

        try:
            # download the image then save it as it streams in as chunks
            response = requests.get(isaac_item.img_url, timeout=10, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logger.debug("Saved image for %s at %s", isaac_item.name, save_path)
        except (requests.RequestException, Exception) as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to download image for %s! %s", isaac_item.name, str(e))

    @staticmethod
    def download_item_images(isaac_items: list[IsaacItem], save_dir: str, parallel: bool = True) -> None:
        """Downloads images for all provided IsaacItems.

        Args:
            isaac_items (List[IsaacItem]): A list of IsaacItems.
            save_dir (str): The directory where images will be saved.
            parallel (bool): If true, parallelize the downloads, else do them sequentially.
        """
        os.makedirs(save_dir, exist_ok=True)
        logger.info("download_item_images: Downloading images, this may take a while...")
        if parallel:
            with ThreadPoolExecutor() as executor:
                executor.map(lambda item: Scraper._download_item_image(item, save_dir), isaac_items)
        else:
            for item in isaac_items:
                Scraper._download_item_image(item, save_dir)
        logger.info("download_item_images: Done downloading images!")

    @staticmethod
    def _find_imgs_that_failed_to_download(isaac_items: list[IsaacItem], data_dir: str) -> list[IsaacItem]:
        """Compare the data_dir and isaac_items and and get a list of which images failed to download.

        This method assumes that data_dir organizes item images as follows: {data_dir}/{item_name}/original_img.png.
        Ex. data/Guppy's Head/original_img.png.

        Args:
            isaac_items (List[IsaacItem]): A list of IsaacItems.
            data_dir (str): Root directory where image folders are stored.

        Returns:
            A list of str. Each str in this list is an IsaacItem which does not have a downloaded img.
            Ex. if we return [IsaacItem("Guppy's Head", ...)] then {data_dir}/Guppy's Head/original_img.png does not exist.
            All other items had their images downloaded successfully.
        """
        files = set(os.listdir(data_dir))
        missing_items = []
        for item in isaac_items:
            if item.name not in files and item.url_encoded_name not in files:
                missing_items.append(item)
        logger.info("find_imgs_that_failed_to_download: found %d missing images!", len(missing_items))
        return missing_items

    @staticmethod
    def dump_item_data_to_json(isaac_items: list[IsaacItem], filename: str) -> None:
        """Write the dictionary representation for each item into the specified filename.

        In the JSON file, each object is represented as follows.
        `<img_dir>: {... the rest of the object ...}`

        Args:
            isaac_items (list[IsaacItem]): The IsaacItems to dump to the file.
            filename (str): Where to save the json file.
        """
        with open(filename, "w", encoding="utf-8") as f:
            data = {}
            for item in isaac_items:
                item_as_dict: dict[str, str] = item.to_dict()
                img_dir = item_as_dict.pop("img_dir")
                data[img_dir] = item_as_dict

            json.dump(data, f, indent=4)
            logger.info("dump_item_data_to_json: Successfully created %s", filename)

    @staticmethod
    def get_isaac_items_from_json(filename: str) -> list[IsaacItem] | None:
        """Attempt to parse IsaacItems from the provided json filename.

        Isaac items should be stored in the following example format (item's img_dir is the main key):

        "Ventricle_Razor": {
            "name": "Ventricle Razor",
            "item_id": "5.100.396",
            "img_url": "https://static.wikia.nocookie.net/bindingofisaacre_gamepedia/images/9/97/Collectible_Ventricle_Razor_icon.png/revision/latest?cb=20210821162403",
            "wiki_url": "https://bindingofisaacrebirth.fandom.com/wiki/Ventricle_Razor",
            "description": "Creates up to two portals that remain even if Isaac leaves the room. Upon entering a portal, Isaac is teleported to the other portal.",
            "item_quality": "1",
            "quote": "Short cutter",
            "url_encoded_name": "Ventricle_Razor"
        },

        Args:
            filename (str): The name of the json file with IsaacItem info.

        Returns:
            A list of IsaacItems. Upon failure, return None.
        """
        try:
            isaac_items = []
            with open(filename, "r", encoding="utf-8") as f:
                data: dict[str, dict[str, str]] = json.load(f)

                # add the img_dir key back into the dict then create an isaac item.
                for img_dir, item_data in data.items():
                    item_data.update({"img_dir": img_dir})
                    isaac_items.append(IsaacItem.from_dict(item_data))

                logger.info("get_isaac_items_from_json: Parsed %d IsaacItems from %s", len(isaac_items), filename)
                return isaac_items
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("get_isaac_items_from_json: Failed to file %s: %s", filename, str(e))
            return None


def main() -> None:  # pylint: disable=missing-function-docstring
    # first, let's fetch the HTML and parse it
    # the first item should be "A Pony" and last "Tonsil" as of 7/25/2024 on the wiki.
    html = Scraper.fetch_page(WIKI_ITEMS_HOMEPAGE)
    isaac_items = Scraper.parse_isaac_items_from_html(html)
    Scraper.download_item_images(isaac_items, DATA_DIR)
    print(f"1st IsaacItem: {isaac_items[0].name}, last IsaacItem: {isaac_items[-1].name}")

    # save the isaac items to a json file like this
    Scraper.dump_item_data_to_json(isaac_items, JSON_DUMP_FILE)

    # since we've dumped to json, we can also load items in from json like this
    isaac_items_from_json = Scraper.get_isaac_items_from_json(JSON_DUMP_FILE)

    # we can even compare the two lists and make sure they're the same after sorting
    if isaac_items_from_json is not None:
        isaac_items_from_json.sort(key=lambda x: x.name)
        isaac_items.sort(key=lambda x: x.name)
        assert isaac_items == isaac_items_from_json  # this works because dataclass implements __eq__ for us.


if __name__ == "__main__":
    main()
