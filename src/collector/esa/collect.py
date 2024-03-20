import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import psycopg2
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


logger = setup_logger(__file__, logging.DEBUG)


def load_dotenv():
    env_path = Path(__file__, "../../../.env")
    if env_path.exists():
        with env_path.open() as f:
            for line in f:
                key, value = line.strip().split("=")
                os.environ[key] = value


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


class EsaCrawlerPostgreWriter(EsaCrawlerWriter):
    def __init__(self, connection_info: Dict):
        self.connection_info = connection_info

    def __enter__(self):
        self.connection = psycopg2.connect(**self.connection_info)
        self.cursor = self.connection.cursor()
        return self

    def __exit__(self, ex_type, ex_value, trace):
        self.cursor.close()
        self.connection.close()

    def _create_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS esa_docs (
                number VARCHAR(10),
                full_name TEXT,
                wip BOOLEAN,
                body_md TEXT,
                body_html TEXT,
                created_at timestamp with time zone,
                updated_at timestamp with time zone,
                url TEXT,
                created_by VARCHAR(20),
                PRIMARY KEY (number)
            )
            """
        )
        self.connection.commit()

    def write(self, posts: List[Dict]):
        if not hasattr(self, "cursor"):
            raise Exception("Use this object within 'with' statement")

        for post in posts:
            number = post["number"]
            full_name = post["full_name"]
            wip = post["wip"]
            body_md = post["body_md"]
            body_html = post["body_html"]
            created_at = post["created_at"]
            updated_at = post["updated_at"]
            url = post["url"]
            created_by = post["created_by"]["screen_name"]

            self.cursor.execute(
                """
                    INSERT INTO esa_docs (
                        number, full_name, wip, body_md, body_html, created_at
                        , updated_at, url, created_by)
                    VALUES (
                        %s, %s, %s, %s, %s, %s
                        , %s, %s, %s)
                    ON CONFLICT (number) DO UPDATE SET
                        full_name = EXCLUDED.full_name,
                        wip = EXCLUDED.wip,
                        body_md = EXCLUDED.body_md,
                        body_html = EXCLUDED.body_html,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at,
                        url = EXCLUDED.url,
                        created_by = EXCLUDED.created_by
                """,
                (
                    number,
                    full_name,
                    wip,
                    body_md,
                    body_html,
                    created_at,
                    updated_at,
                    url,
                    created_by,
                ),
            )

        self.connection.commit()


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
            "last_updated_at": getattr(self, "last_post_updated_at", None),
            "exit_status": self.crawle_status,
        }
        file_path = Path(file_path)
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

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
        current_posts = 0
        logger.info(f"Total posts: {total_posts}")
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
                    current_posts += len(posts)
                    logger.debug(
                        f"{current_posts / total_posts} % posts are collected."
                    )

            self.crawle_status = "success"

        except Exception as e:
            self.crawle_status = f"error: ({e})"
            raise e
            logger.error(e)
        finally:
            self._update_last_crawled_data()


if __name__ == "__main__":
    output_path = "esa_posts.jsonl"
    crawler = EsaCrawler()
    # writer = EsaCrawlerJsonLWriter(output_path)
    writer = EsaCrawlerPostgreWriter(
        {
            "host": os.environ["DB_HOST"],
            "port": os.environ["DB_PORT"],
            "dbname": os.environ["DB_DBNAME"],
            "user": os.environ["DB_USER"],
            "password": os.environ["DB_PASSWORD"],
        }
    )
    res = crawler.crawl_posts(writer)
