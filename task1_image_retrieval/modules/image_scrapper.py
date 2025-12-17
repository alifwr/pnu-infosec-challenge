"""
Web Scraping Image Collector using Playwright
Scrapes images from Google, Bing, DuckDuckGo, and other sources
No API keys needed!
"""

import asyncio
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime
from io import BytesIO

import requests
import imagehash
from PIL import Image
from playwright.async_api import async_playwright, Page


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]


class ImageScraperBase:
    """Base class for image scraping"""

    def __init__(self, output_dir: str, min_width: int = 640, min_height: int = 480):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_width = min_width
        self.min_height = min_height

        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.metadata_file = self.output_dir / "metadata.json"
        self.metadata = self._load_metadata()

        self.seen_hashes: Set[str] = set()
        self.seen_phashes: Set[str] = set()
        self.seen_urls: Set[str] = set()
        self._load_existing_hashes()

    def _load_metadata(self) -> Dict:
        """Load existing metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"images": [], "stats": {}}

    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _load_existing_hashes(self):
        """Load hashes of existing images for deduplication"""
        for img_path in self.images_dir.glob("*.jpg"):
            try:
                img = Image.open(img_path)
                phash = str(imagehash.phash(img))
                self.seen_phashes.add(phash)
            except Exception as e:
                logger.warning(f"Could not load hash for {img_path}: {e}")

        for img_data in self.metadata.get("images", []):
            if "url" in img_data:
                self.seen_urls.add(img_data["url"])

    def _compute_file_hash(self, content: bytes) -> str:
        """Compute MD5 hash of file content"""
        return hashlib.md5(content).hexdigest()

    def _is_duplicate(self, img: Image.Image, file_hash: str, url: str) -> bool:
        """Check if image is duplicate"""

        if url in self.seen_urls:
            return True

        if file_hash in self.seen_hashes:
            return True

        try:
            phash = str(imagehash.phash(img))
            if phash in self.seen_phashes:
                return True
            return False
        except Exception as e:
            logger.warning(f"Error computing phash: {e}")
            return False

    def _is_valid_image(self, img: Image.Image) -> bool:
        """Check if image meets minimum requirements"""
        width, height = img.size
        return width >= self.min_width and height >= self.min_height

    def download_image(
        self, url: str, query: str, source: str, extra_metadata: Dict = None
    ) -> bool:
        """Download and save image with metadata"""
        try:
            if url in self.seen_urls:
                logger.debug(f"Skipping duplicate URL: {url[:100]}")
                return False

            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Referer": "https://www.google.com/",
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            }

            response = requests.get(
                url, timeout=15, headers=headers, stream=True, allow_redirects=True
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type.lower():
                logger.debug(
                    f"Not an image (Content-Type: {content_type}): {url[:100]}"
                )
                return False

            content = response.content
            if len(content) < 1024:
                logger.debug(f"Image too small ({len(content)} bytes): {url[:100]}")
                return False

            img = Image.open(BytesIO(content))

            if img.mode != "RGB":
                img = img.convert("RGB")

            if not self._is_valid_image(img):
                logger.debug(
                    f"Image too small ({img.size[0]}x{img.size[1]}): {url[:100]}"
                )
                return False

            file_hash = self._compute_file_hash(content)
            if self._is_duplicate(img, file_hash, url):
                logger.debug(f"Duplicate image detected: {url[:100]}")
                return False

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_query = "".join(c if c.isalnum() else "_" for c in query)[:30]
            filename = f"{safe_query}_{source}_{timestamp}.jpg"
            filepath = self.images_dir / filename

            img.save(filepath, "JPEG", quality=95)

            self.seen_hashes.add(file_hash)
            self.seen_urls.add(url)
            phash = str(imagehash.phash(img))
            self.seen_phashes.add(phash)

            metadata_entry = {
                "filename": filename,
                "url": url,
                "query": query,
                "source": source,
                "width": img.size[0],
                "height": img.size[1],
                "downloaded_at": datetime.now().isoformat(),
                "file_hash": file_hash,
                "phash": phash,
            }

            if extra_metadata:
                metadata_entry.update(extra_metadata)

            self.metadata["images"].append(metadata_entry)
            self._save_metadata()

            logger.info(f"✓ Downloaded: {filename} ({img.size[0]}x{img.size[1]})")
            return True

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout downloading: {url[:100]}")
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error for {url[:100]}: {str(e)[:100]}")
            return False
        except Exception as e:
            logger.warning(f"Error processing {url[:100]}: {str(e)[:100]}")
            return False

    def get_random_user_agent(self) -> str:
        """Get random user agent"""
        return random.choice(USER_AGENTS)


class GoogleImageScraper(ImageScraperBase):
    """Scrape images from Google Images"""

    async def scrape_with_browser(
        self, page: Page, query: str, count: int
    ) -> List[str]:
        """Scrape image URLs using Playwright"""
        logger.info(f"Scraping Google Images for: {query}")

        search_url = f"https://www.google.com/search?q={query}&tbm=isch"

        try:
            await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            logger.error(f"Failed to load Google Images page: {e}")
            return []

        try:
            accept_button = page.locator(
                'button:has-text("Accept all"), button:has-text("I agree"), button:has-text("Agree")'
            )
            if await accept_button.count() > 0:
                await accept_button.first.click(timeout=20000)
                await asyncio.sleep(1)
        except Exception as e:
            logger.debug(f"No cookie consent or already accepted: {e}")

        await asyncio.sleep(3)

        try:
            img_count = await page.locator("img").count()
            if img_count == 0:
                logger.error("No images found on page")
                return []
        except Exception as e:
            logger.error(f"Error checking for images: {e}")
            return []

        image_urls = set()
        last_height = 0
        scroll_attempts = 0
        max_scrolls = 10

        while len(image_urls) < count and scroll_attempts < max_scrolls:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(random.uniform(1, 2))

            current_height = await page.evaluate("document.body.scrollHeight")

            images = await page.query_selector_all("img[src], img[data-src]")
            for img in images:
                try:
                    src = await img.get_attribute("src")
                    data_src = await img.get_attribute("data-src")
                    data_iurl = await img.get_attribute("data-iurl")

                    url = data_iurl or data_src or src

                    if url and url.startswith("http") and "base64" not in url:
                        if not any(
                            skip in url
                            for skip in [
                                "gstatic.com",
                                "google.com/images",
                                "encrypted-tbn",
                            ]
                        ):
                            image_urls.add(url)
                            logger.debug(f"Added URL: {url[:100]}...")

                        if len(image_urls) >= count * 2:
                            break
                except Exception as e:
                    continue

            if len(image_urls) < count:
                try:
                    containers = await page.query_selector_all("div[data-id]")
                    for i, container in enumerate(containers[:5]):
                        try:
                            await container.click(timeout=20000)
                            await asyncio.sleep(0.5)

                            large_img = await page.query_selector("img[jsname]")
                            if large_img:
                                large_src = await large_img.get_attribute("src")
                                if (
                                    large_src
                                    and large_src.startswith("http")
                                    and "base64" not in large_src
                                ):
                                    if "encrypted-tbn" not in large_src:
                                        image_urls.add(large_src)
                                        logger.debug(
                                            f"Added large image URL: {large_src[:100]}..."
                                        )
                        except:
                            continue
                except:
                    pass

            if current_height == last_height:
                try:
                    more_button = await page.query_selector(
                        'input[value="Show more results"]'
                    )
                    if more_button:
                        await more_button.click()
                        await asyncio.sleep(2)
                except:
                    pass

                scroll_attempts += 1
            else:
                scroll_attempts = 0

            last_height = current_height

        logger.info(f"Found {len(image_urls)} image URLs from Google")
        return list(image_urls)[: count * 2]

    async def scrape(self, query: str, count: int = 100) -> int:
        """Scrape images from Google"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=self.get_random_user_agent(),
                viewport={"width": 1920, "height": 1080},
            )
            page = await context.new_page()

            try:
                image_urls = await self.scrape_with_browser(page, query, count)

                downloaded = 0
                for url in image_urls:
                    if downloaded >= count:
                        break

                    if self.download_image(url, query, "google"):
                        downloaded += 1

                    await asyncio.sleep(random.uniform(0.3, 0.7))

                logger.info(f"Downloaded {downloaded} images from Google")
                return downloaded

            finally:
                await browser.close()


class BingImageScraper(ImageScraperBase):
    """Scrape images from Bing Images"""

    async def scrape_with_browser(
        self, page: Page, query: str, count: int
    ) -> List[str]:
        """Scrape image URLs using Playwright"""
        logger.info(f"Scraping Bing Images for: {query}")

        search_url = f"https://www.bing.com/images/search?q={query}"
        await page.goto(search_url, wait_until="networkidle")

        await asyncio.sleep(2)

        image_urls = set()
        last_count = 0
        scroll_attempts = 0
        max_scrolls = 10

        while len(image_urls) < count and scroll_attempts < max_scrolls:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(random.uniform(1, 2))

            thumbnails = await page.query_selector_all(".iusc")

            for thumb in thumbnails:
                try:
                    m_attr = await thumb.get_attribute("m")
                    if m_attr:
                        import json as json_lib

                        data = json_lib.loads(m_attr)
                        if "murl" in data:
                            image_urls.add(data["murl"])
                        elif "turl" in data:
                            image_urls.add(data["turl"])
                except Exception as e:
                    continue

            if len(image_urls) == last_count:
                scroll_attempts += 1
            else:
                scroll_attempts = 0

            last_count = len(image_urls)

            if len(image_urls) >= count * 2:
                break

        logger.info(f"Found {len(image_urls)} image URLs from Bing")
        return list(image_urls)[: count * 2]

    async def scrape(self, query: str, count: int = 100) -> int:
        """Scrape images from Bing"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=self.get_random_user_agent(),
                viewport={"width": 1920, "height": 1080},
            )
            page = await context.new_page()

            try:
                image_urls = await self.scrape_with_browser(page, query, count)

                downloaded = 0
                for url in image_urls:
                    if downloaded >= count:
                        break

                    if self.download_image(url, query, "bing"):
                        downloaded += 1

                    await asyncio.sleep(random.uniform(0.3, 0.7))

                logger.info(f"Downloaded {downloaded} images from Bing")
                return downloaded

            finally:
                await browser.close()


class DuckDuckGoImageScraper(ImageScraperBase):
    """Scrape images from DuckDuckGo Images"""

    async def scrape_with_browser(
        self, page: Page, query: str, count: int
    ) -> List[str]:
        """Scrape image URLs using Playwright"""
        logger.info(f"Scraping DuckDuckGo Images for: {query}")

        search_url = f"https://duckduckgo.com/?q={query}&iax=images&ia=images"
        await page.goto(search_url, wait_until="networkidle")

        await asyncio.sleep(2)

        image_urls = set()
        scroll_attempts = 0
        max_scrolls = 10

        while len(image_urls) < count and scroll_attempts < max_scrolls:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(random.uniform(1, 2))

            images = await page.query_selector_all("img.tile--img__img")

            for img in images:
                try:
                    src = await img.get_attribute("src")
                    data_src = await img.get_attribute("data-src")

                    url = data_src or src

                    if url and url.startswith("http"):
                        image_urls.add(url)
                except Exception as e:
                    continue

            scroll_attempts += 1

            if len(image_urls) >= count * 2:
                break

        logger.info(f"Found {len(image_urls)} image URLs from DuckDuckGo")
        return list(image_urls)[: count * 2]

    async def scrape(self, query: str, count: int = 100) -> int:
        """Scrape images from DuckDuckGo"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=self.get_random_user_agent(),
                viewport={"width": 1920, "height": 1080},
            )
            page = await context.new_page()

            try:
                image_urls = await self.scrape_with_browser(page, query, count)

                downloaded = 0
                for url in image_urls:
                    if downloaded >= count:
                        break

                    if self.download_image(url, query, "duckduckgo"):
                        downloaded += 1

                    await asyncio.sleep(random.uniform(0.3, 0.7))

                logger.info(f"Downloaded {downloaded} images from DuckDuckGo")
                return downloaded

            finally:
                await browser.close()


def main():
    """Example usage"""

    config = {
        "output_dir": "scraped_dataset",
        "queries": ["cat", "dog"],
        "images_per_query": 50,
        "min_width": 640,
        "min_height": 480,
    }

    async def collect_all():
        scrapers = [
            GoogleImageScraper(
                config["output_dir"],
                min_width=config["min_width"],
                min_height=config["min_height"],
            ),
            BingImageScraper(
                config["output_dir"],
                min_width=config["min_width"],
                min_height=config["min_height"],
            ),
            DuckDuckGoImageScraper(
                config["output_dir"],
                min_width=config["min_width"],
                min_height=config["min_height"],
            ),
        ]

        for query in config["queries"]:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Collecting images for: {query}")
            logger.info(f"{'=' * 60}")

            total_downloaded = 0
            per_source = config["images_per_query"] // len(scrapers)

            for scraper in scrapers:
                source_name = scraper.__class__.__name__.replace("ImageScraper", "")
                logger.info(f"\nScraping from {source_name}...")

                downloaded = await scraper.scrape(query, per_source)
                total_downloaded += downloaded

            logger.info(f"\n✓ Total downloaded for '{query}': {total_downloaded}")

        logger.info("\n" + "=" * 60)
        logger.info("Scraping complete!")
        logger.info(f"Images saved to: {config['output_dir']}/images")
        logger.info(f"Metadata saved to: {config['output_dir']}/metadata.json")

    asyncio.run(collect_all())


if __name__ == "__main__":
    main()
