import os
from argparse import ArgumentParser, Namespace

from esa.collect import EsaCrawler, EsaCrawlerPostgreWriter


def collect_esa(args: Namespace):
    crawler = EsaCrawler()
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
    print(res)


def collect(args: Namespace):
    collect_esa(args)


if __name__ == "__main__":

    def parse_args():
        parser = ArgumentParser(description='kasyore collector')

        args, other = parser.parse_known_args()

        return args

    args = parse_args()
    collect(args)
