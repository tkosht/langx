# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


import sqlite3
import time

import ulid
from newspy.items import NewsItem
from scrapy.exceptions import DropItem

# useful for handling different item types with a single interface
# from itemadapter import ItemAdapter
from typing_extensions import Self


def build_ulid(prefix: str = "") -> str:
    assert isinstance(prefix, str)
    # NOTE: keep time order
    time.sleep(1 / 1000)
    return prefix + str(ulid.new())


class NewsPipeline:
    def process_item(self, item: NewsItem, spider) -> NewsItem:
        self._init_db()
        try:
            self._store_item(item)
        except sqlite3.IntegrityError as e:
            # possibly for unique constraint
            item["html"] = "(omitted)"  # for more simplification of log output
            raise DropItem(f"{e}: skipped [{item['url']}]")
        finally:
            self._term_db()
        return item

    def _init_db(self) -> Self:
        self.cnn = sqlite3.connect("data/news.db")

        # テーブル作成
        self.csr = self.cnn.cursor()
        self.csr.execute(
            """CREATE TABLE IF NOT EXISTS news_data(
                url TEXT PRIMARY KEY,
                doc_id TEXT UNIQUE NOT NULL,
                html TEXT NOT NULL,
                created_at DATE NOT NULL,
                updated_at DATE
            );
            """
        )

        return self

    def _term_db(self) -> Self:
        if self.csr is not None:
            self.csr.close()
            self.csr = None
        if self.cnn is not None:
            self.cnn.close()
            self.cnn = None
        return self

    def _store_item(self, item: NewsItem):
        doc_id = build_ulid(prefix="Nws")
        values = [item["url"], doc_id, item["html"]]

        self.csr.execute(
            """INSERT INTO news_data (
            url, doc_id, html, created_at
        ) VALUES (?, ?, ?, datetime('now', 'localtime'))
        """,
            values,
        )
        self.cnn.commit()

        return
