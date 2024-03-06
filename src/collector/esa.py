import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from piyo import Client
from tqdm import tqdm


def setup_logger(name, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_format = logging.Formatter("[%(filename)s][%(levelname)s] %(message)s")
    stream_handler.setFormatter(stream_format)

    logging.basicConfig(level=level, handlers=[stream_handler])
    return logger


logger = setup_logger(__file__)


class EsaCrawlerWriter(ABC):
    @abstractmethod
    def write(self, post: List[Dict]): ...

    @abstractmethod
    def __enter__(self): ...

    @abstractmethod
    def __exit__(self, ex_type, ex_value, trace): ...


class EsaCrawlerJsonLWriter(EsaCrawlerWriter):
    def __init__(self, output_path):
        self.output_path = Path(output_path)

    def __enter__(self):
        self.f = self.output_path.open("w")
        return self

    def __exit__(self, ex_type, ex_value, trace):
        self.f.close()

    def write(self, posts: List[Dict]):
        if not hasattr(self, "f"):
            raise Exception("Use this object within 'with' statement")

        for post in posts:
            line = json.dumps(post, ensure_ascii=False) + "\n"
            self.f.write(line)


class EsaCrawler(object):
    def __init__(self):
        if "ESA_ACCESS_TOKEN" not in os.environ and "ESA_TEAM_NAME" not in os.environ:
            raise Exception(
                "env variable ESA_ACCESS_TOKEN and ESA_TEAM_NAME is needed to be specified."
            )
        self.client = Client(current_team=os.environ["ESA_TEAM_NAME"])
        self.error = None
        self._load_last_crawled_data()

    def _load_last_crawled_data(self, file_path: str = ".last_crawled.json"):
        data_path = Path(file_path)
        last_crawled_data = (
            json.loads(data_path.read_text()) if data_path.exists() else {}
        )
        if last_crawled_data.get("exit_status") != "success":
            logger.warn(
                "The last crawl did not end successfully. Crawl posts from entire period."
            )
            last_crawled_data = {}

        self.last_updated_at = last_crawled_data.get("last_updated_at")
        self.last_executed_at = last_crawled_data.get("exeuted_at")

    def _update_last_crawled_data(self, file_path: str = ".last_crawled.json"):
        data = {
            "exeuted_at": self.exeuted_at.strftime("%Y-%m-%d"),
            "last_updated_at": self.last_post_updated_at,
            "exit_status": self.crawle_status,
        }
        file_path = Path(file_path)
        file_path.write_text(json.dumps(data, ensure_ascii=False))

    def get_total_post(self):
        stats = self.client.stats()
        return stats["posts"]

    def create_search_options(self, options):
        return " ".join([f"{k}: {v}" for k, v in options.items()])

    def crawl_posts(
        self,
        writer_obj: EsaCrawlerWriter,
        interval: int = 13,
    ):
        self.exeuted_at = datetime.now()
        search_options = {"sort": "updated-asc"}
        if self.last_executed_at:
            search_options["updated"] = f">={self.last_executed_at}"

        params = {
            "q": self.create_search_options(search_options),
            "per_page": 100,
            "include": "comments",
        }
        next_page = 1
        total_posts = self.get_total_post()
        try:
            self.crawle_status = "crawling"
            with writer_obj as writer, tqdm(total=total_posts) as pbar:
                while next_page is not None:
                    time.sleep(interval)

                    params["page"] = next_page
                    result = self.client.posts(params=params)
                    posts = result["posts"]

                    updated_at = datetime.fromisoformat(posts[-1]["updated_at"])
                    self.last_post_updated_at = updated_at.strftime("%Y-%m-%d")

                    writer.write(posts)

                    next_page = result["next_page"]
                    pbar.update(len(posts))

            self.crawle_status = "success"

        except Exception as e:
            self.crawle_status = f"error: ({e})"
            logger.error(e)
        finally:
            self._update_last_crawled_data()


if __name__ == "__main__":
    output_path = "esa_posts.jsonl"
    crawler = EsaCrawler()
    writer = EsaCrawlerJsonLWriter(output_path)
    res = crawler.crawl_posts(writer)
