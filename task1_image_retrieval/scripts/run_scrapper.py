"""
Web Scraper Runner Script
No API keys needed - just run and collect!
"""

import yaml
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.scraper_collector import (
    GoogleImageScraper,
    BingImageScraper,
    DuckDuckGoImageScraper,
    logger,
)


def load_config(config_path: str = "configs/scraper_config.yaml") -> dict:
    """Load configuration from YAML file"""
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


async def main():
    # Load configuration
    config = load_config()

    logger.info("=" * 60)
    logger.info("WEB SCRAPING IMAGE COLLECTOR")
    logger.info("=" * 60)
    logger.info("No API keys needed!")
    logger.info("")

    # Initialize scrapers based on config
    scrapers = []

    if config["sources"].get("google", True):
        logger.info("✓ Google Images scraper enabled")
        scrapers.append(
            GoogleImageScraper(
                config["output_dir"],
                min_width=config["min_width"],
                min_height=config["min_height"],
            )
        )

    if config["sources"].get("bing", True):
        logger.info("✓ Bing Images scraper enabled")
        scrapers.append(
            BingImageScraper(
                config["output_dir"],
                min_width=config["min_width"],
                min_height=config["min_height"],
            )
        )

    if config["sources"].get("duckduckgo", True):
        logger.info("✓ DuckDuckGo Images scraper enabled")
        scrapers.append(
            DuckDuckGoImageScraper(
                config["output_dir"],
                min_width=config["min_width"],
                min_height=config["min_height"],
            )
        )

    if not scrapers:
        logger.error(
            "❌ No scrapers enabled! Please enable at least one source in config."
        )
        sys.exit(1)

    logger.info(f"\nActive scrapers: {len(scrapers)}")
    logger.info(f"Queries: {len(config['queries'])}")
    logger.info(f"Target images per query: {config['images_per_query']}")
    logger.info("")

    # Collect images for each query
    total_collected = 0

    for i, query in enumerate(config["queries"], 1):
        logger.info(f"{'=' * 60}")
        logger.info(f"Query {i}/{len(config['queries'])}: {query}")
        logger.info(f"{'=' * 60}")

        query_total = 0
        per_source = config["images_per_query"] // len(scrapers)

        for scraper in scrapers:
            source_name = scraper.__class__.__name__.replace("ImageScraper", "")
            logger.info(f"\n🔍 Scraping from {source_name}...")

            try:
                downloaded = await scraper.scrape(query, per_source)
                query_total += downloaded
            except Exception as e:
                logger.error(f"Error scraping from {source_name}: {e}")

        logger.info(f"\n✓ Downloaded {query_total} images for '{query}'")
        total_collected += query_total

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SCRAPING COMPLETE!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total images collected: {total_collected}")
    logger.info(f"Images saved to: {config['output_dir']}/images/")
    logger.info(f"Metadata saved to: {config['output_dir']}/metadata.json")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review images: cd {}/images".format(config["output_dir"]))
    logger.info(
        "  2. Analyze dataset: python analyze_dataset.py {}".format(
            config["output_dir"]
        )
    )
    logger.info("  3. Annotate for object detection")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\n⚠ Scraping interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
