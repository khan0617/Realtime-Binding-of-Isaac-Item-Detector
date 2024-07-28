import os

# constants for scraper.py
WIKI_HOMEPAGE_ROOT = "https://bindingofisaacrebirth.fandom.com/"
WIKI_ITEMS_HOMEPAGE = "https://bindingofisaacrebirth.fandom.com/wiki/Items"
CACHE_DIR = "scraper_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "cache.json")
DATA_DIR = "data"
UNMODIFIED_FILE_NAME = "original_img.png"
JSON_DUMP_FILE = "dumped_isaac_items.json"
BROKEN_SHOVEL_ACTIVE_ID = "5.100.550"
BROKEN_SHOVEL_PASSIVE_ID = "5.100.551"
