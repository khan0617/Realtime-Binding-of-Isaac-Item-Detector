import json
import logging
import os
import re
import shutil
import typing
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

from constants import CACHE_DIR, CACHE_FILE, DATA_DIR, WIKI_HOMEPAGE_ROOT, WIKI_ITEMS_HOMEPAGE
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
    def sanitize_filename(name: str) -> str:
        """
        Sanitizes the item name to be used as a valid filename.

        Args:
            name (str): The item name to sanitize.

        Returns:
            str: The sanitized filename.
        """
        # URL encode the item name, including '/' and '.' as unsafe
        encoded_name = urllib.parse.quote(name, safe="")

        # replace the periods with %2E explicitly
        encoded_name = encoded_name.replace(".", "%2E")
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
                    wiki_url=urllib.parse.urljoin(WIKI_HOMEPAGE_ROOT, relative_wiki_url),
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

    @staticmethod
    def _download_item_image(isaac_item: IsaacItem, save_dir: str) -> None:
        """Threadpool helper: Downloads the image for this IsaacItem and saves it to the specified directory.

        All item images will be downloaded into {save_dir}/{item_name}/original_img.png
        For example, the image for "Guppy's Head" will be like: {save_dir}/Guppy's Head/original_img.png
        Item names with characters that cannot be in filepaths, such as "?", ".", "/", "<" etc are
        encoded using urllib.quote().

        Args:
            isaac_item (IsaacItem): The IsaacItem we want to download an image for.
            save_dir (str): The root directory where the image will be saved.
        """
        try:
            # try to use the original item name first
            item_name = isaac_item.name
            item_dir = os.path.join(save_dir, item_name)
            save_path = os.path.join(item_dir, "original_img.png")

            # make sure the directory exists
            os.makedirs(item_dir, exist_ok=True)

        except OSError:
            # when OSError occurs, fallback to using URL-encoded name
            logger.warning("Falling back to URL-encoded name for item: %s", isaac_item.name)
            sanitized_name = Scraper.sanitize_filename(isaac_item.name)
            item_dir = os.path.join(save_dir, sanitized_name)
            save_path = os.path.join(item_dir, "original_img.png")
            os.makedirs(item_dir, exist_ok=True)

        # If we the path exists we already have the image, don't need to download!
        if os.path.exists(save_path):
            logger.info("Image for %s already exists at %s", isaac_item.name, save_path)
            return

        try:
            # download the image
            response = requests.get(isaac_item.img_url, timeout=10, stream=True)
            response.raise_for_status()

            # save the image in chunks
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logger.info("Saved image for %s at %s", isaac_item.name, save_path)
        except requests.RequestException as e:
            logger.error("Failed to download image for %s! %s", isaac_item.name, str(e))

    @staticmethod
    def download_item_images(isaac_items: list[IsaacItem], save_dir: str, max_workers: int = 10) -> None:
        """Downloads images for all provided IsaacItems in parallel.

        Args:
            isaac_items (List[IsaacItem]): A list of IsaacItems.
            save_dir (str): The directory where images will be saved.
            max_workers (int): The maximum number of threads to use for parallel downloading.
        """
        os.makedirs(save_dir, exist_ok=True)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(lambda item: Scraper._download_item_image(item, save_dir), isaac_items)

    @staticmethod
    def reorganize_downloaded_images(data_dir: str) -> None:
        """Helper method to cleanup my (incorrectly) downloaded images (lol).

        Basically, instead of downloading images like {data_dir}/Guppy's Head/original_img.jpg,
        I downloaded them as {data_dir}/Guppy's Head.png. This method cleans it up.
        Probably only need to run once.

        Args:
            data_dir (str): The current root dir where all images are stored.
        """
        moved_any_file = False

        for filename in os.listdir(data_dir):
            if filename.endswith(".png"):
                # create a directory for the new item
                item_name = os.path.splitext(filename)[0]
                item_dir = os.path.join(data_dir, item_name)
                os.makedirs(item_dir, exist_ok=True)

                # define the new filepath then move the image
                new_path = os.path.join(item_dir, "original_img.png")
                original_path = os.path.join(data_dir, filename)
                shutil.move(original_path, new_path)
                logger.info("reorganize_downloaded_images: Moved %s to %s", original_path, new_path)
                moved_any_file = True

        if not moved_any_file:
            logger.info("reorganize_downloaded_images: No images to reorganize!")

    @staticmethod
    def find_imgs_that_failed_to_download(isaac_items: list[IsaacItem], data_dir: str) -> list[IsaacItem]:
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
            if item.name not in files and urllib.parse.quote(item.name, safe="") not in files:
                missing_items.append(item)
        logger.info("find_imgs_that_failed_to_download: found %d missing images!", len(missing_items))
        return missing_items

    @staticmethod
    def download_missing_items(missing_items: list[IsaacItem], save_dir: str) -> None:
        """Attempt to download each of the missing items.

        Args:
            missing_items (list[IsaacItem]): The IsaacItems with no downloaded image.
            save_dir: The root directory to downloaded the image in.
        """
        if not missing_items:
            logging.info("download_missing_items: No missing items! All good :)")
        for item in missing_items:
            logger.info("download_missing_items: Trying to download %s", item.name)
            Scraper._download_item_image(item, save_dir)


def main() -> None:
    # let's fetch the HTML, prase it, and see the first and last item.
    # the first item should be "A Pony" and last "Tonsil" as of 7/25/2024 on the wiki.
    fetched_html = Scraper.fetch_page(WIKI_ITEMS_HOMEPAGE)
    parsed_isaac_items = Scraper.parse_isaac_items_from_html(fetched_html)

    # print the first and last IsaacItem.
    pprint(parsed_isaac_items[0].to_dict())
    pprint(parsed_isaac_items[-1].to_dict())

    # download the image for each item
    Scraper.download_item_images(parsed_isaac_items, DATA_DIR)

    # example to check if you have issues downloading some images
    missing_items = Scraper.find_imgs_that_failed_to_download(parsed_isaac_items, DATA_DIR)
    pprint(
        [
            (Scraper.sanitize_filename(item.name), urllib.parse.unquote(Scraper.sanitize_filename(item.name)))
            for item in missing_items
        ]
    )
    # Scraper.download_missing_items(missing_items, DATA_DIR)


if __name__ == "__main__":
    main()
