import logging
from urllib.parse import urljoin, urlparse

import scrapy
from newspy.items import NewsItem

g_logger = logging.getLogger(__name__)


class NewsSpider(scrapy.Spider):
    name = "news"
    start_urls = [
        # 一般
        # "https://news.yahoo.co.jp/",
        "https://www.nikkei.com/",
        # "https://www3.nhk.or.jp/news/",
        # "https://www.jiji.com/",
        # "https://diamond-rm.net/",
        # IT
        "https://xtech.nikkei.com/top/it/",
        "https://www.itmedia.co.jp/news/",
        "https://japan.zdnet.com/",
        # AI
        "https://ledge.ai/theme/news/",
        "https://ainow.ai/",
        "https://news.mynavi.jp/techplus/tag/artificial_intelligence/",
        "https://ja.stateofaiguides.com/",
        "https://www.itmedia.co.jp/news/subtop/aiplus/",
    ]
    allowed_domains = [urlparse(u).netloc for u in start_urls]

    def parse(self, response):

        item = NewsItem()
        item["url"] = response.url
        item["html"] = response.body.decode(response.encoding)

        yield item

        for ank in response.css("a::attr(href)"):
            url: str = ank.get().strip()
            if not url:
                continue

            if url[0] in ["/"]:
                url = urljoin(response.url, url)

            scheme = "http"
            if url[: len(scheme)] != scheme:
                g_logger.warning(
                    f"Found non {scheme} scheme: skipped [{url=}][parent.url={response.url}]"
                )
                continue

            yield scrapy.Request(url, callback=self.parse)

        return
