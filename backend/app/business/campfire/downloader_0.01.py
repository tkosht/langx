import datetime
import re
import time
from enum import Enum
from inspect import signature
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import typer
from omegaconf import DictConfig
from retry import retry
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from typing_extensions import Annotated

from app.base.component.ulid import build_ulid
from app.business.campfire.graphdb import GraphDb
from app.business.campfire.project import ProjectDetails, ProjectRecord, ReturnBox


class SortBy(str, Enum):
    popular = "popular"
    fresh = "fresh"
    last_spurt = "last_spurt"
    most_funded = "most_funded"
    density = "density"


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def pickup_numbers(text: str, null_value: int = 0) -> int | None:
    n = re.sub(r"\D", "", text)
    if n:
        return int(n)
    return null_value


def pickup_url(text: str) -> str:
    return re.sub(r"url\((.*)\)", r"\1", text)


def _cleanup_url(url: str) -> str:
    """
    Cleans up the given URL by removing any query parameters.

    Args:
        url (str): The URL to be cleaned up.

    Returns:
        str: The cleaned up URL without any query parameters.
    """
    return url.split("?")[0]


class CampfireFetcher(object):
    """
    A class for automating web scraping using Selenium.
    """

    def __init__(self) -> None:
        """
        Initializes the AutomateSelenium class.
        """
        self.driver = self.create_driver()
        self.user_agent = self.driver.execute_script("return navigator.userAgent")

    def create_driver(self) -> webdriver.Chrome:
        """
        Creates a Selenium WebDriver instance with the specified options.
        Returns:
            driver (webdriver.Chrome): The created WebDriver instance.
        """

        options = Options()
        options.binary_location = "/opt/chrome-linux64/chrome"
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-chrome-browser-cloud-management")
        options.add_argument("--enable-javascript")
        service = Service("/usr/local/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
        return driver

    def can_fetch(self, url: str):
        """robots.txt での許可があるかどうかをチェックする"""
        parsed = urlparse(url)
        url_robots = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rbp = RobotFileParser(url=url_robots)
        rbp.read()
        return rbp.can_fetch(self.user_agent, url)

    @retry(TimeoutException, tries=3, delay=2)
    def _fetch(
        self,
        url: str,
        elm_path: str,
        wait_path: str = None,
        max_wait: int = 1,  # secs
        do_scroll: bool = False,
        scroll_size: int = 300,
        scroll_wait: float = 0.10,
        wait_after_scroll: float = 0.10,
    ) -> list[WebElement]:
        """
        Fetches elements from a web page using the specified URL and element paths.
        Args:
            url (str): The URL of the web page to fetch elements from.
            elm_path (str): The XPath of the elements to fetch.
            wait_path (str, optional): The XPath of the element to wait for before fetching. If specified, the method will wait for the element to be present before fetching. Defaults to None.
            max_tries (int, optional): The maximum number of tries to fetch the elements. Defaults to 3.
        Returns:
            elms (list): The fetched elements.
        Raises:
            TimeoutException: If the element to wait for is not found within the specified time.
        """

        if url:
            # robots.txt での許可がない場合、取得をスキップする
            if not self.can_fetch(url):
                print(
                    now(),
                    "[WARNING]",
                    "robots.txt での許可がないため、取得をスキップします。",
                    url,
                )
                raise Exception(
                    f"robots.txt での許可がないため、取得をスキップします。{url}"
                )
            self.driver.get(url)

        # wait for the element to be present
        # # default is None, so that it will wait for element which is specified by elm_path
        wait_path = elm_path if wait_path is None else wait_path
        WebDriverWait(self.driver, max_wait).until(
            expected_conditions.presence_of_element_located((By.XPATH, wait_path))
        )

        if do_scroll:
            # scroll の高さ高さを取得する
            last_height = self.driver.execute_script(
                "return document.body.scrollHeight"
            )

            # 画面の最下部まで少しずつスクロールする
            # # lazy loading で追加される要素を取得するため
            scroll_size = max(last_height // 5, scroll_size)
            for offset in range(0, last_height, scroll_size):
                # lazy loading で追加される要素を取得する
                self.driver.execute_script(
                    f"window.scrollTo({offset}, {offset+scroll_size});"
                )

                # scroll が終わるまで待つ
                time.sleep(scroll_wait)

        time.sleep(wait_after_scroll)
        elms: list[WebElement] = self.driver.find_elements(by=By.XPATH, value=elm_path)
        return elms

    def get_project_url(self, page: int, sortby: SortBy = SortBy.popular):
        project_url = f"https://camp-fire.jp/projects?page={page}&sort={sortby}"
        return project_url

    def fetch_project_list_boxes(
        self, page: int = 1, sortby: SortBy = SortBy.popular
    ) -> list[WebElement]:
        """
        Fetches the project list boxes from Campfire website.

        Args:
            page (int): The page number to fetch (default is 1).
            sortby (str): The sorting option for the projects (default is "popular").

        Returns:
            list: A list of project boxes fetched from the page.
        """

        url = self.get_project_url(page, sortby)
        container: list[WebElement] = self._fetch(
            url=url,
            elm_path='//div[@class="container"]',
            wait_path='//*[@id="fb-root"]',
            do_scroll=True,
        )
        boxes: list[WebElement] = container[0].find_elements(
            by=By.XPATH, value='.//div[@class="box"]'
        )
        assert boxes, f"no project boxes found in {url}"

        return boxes

    def fetch_projects(self, params: DictConfig) -> list[ProjectRecord]:
        """
        Fetches projects from Campfire based on the given parameters.

        Args:
            params (DictConfig): A dictionary-like object containing the parameters for fetching projects.

        Returns:
            list[ProjectRecord]: A list of ProjectRecord objects representing the fetched project records.
        """

        projects = []
        for n_page in range(1, params.max_pages + 1):
            print(f"{now()} Fetching page {n_page} ...")
            project_boxes: list[WebElement] = self.fetch_project_list_boxes(
                page=n_page, sortby=params.sortby
            )
            project_records: list[ProjectRecord] = [
                self.fetch_project(box_idx, n_page, bx)
                for box_idx, bx in enumerate(project_boxes)
            ]
            print(f"{now()} {n_page=} {len(project_records)} projects fetched.")
            projects += project_records

        print(f"{now()} total is {len(projects)} projects fetched.")
        return projects

    def fetch_project(self, box_idx: int, page: int, bx: WebElement) -> ProjectRecord:
        """
        Fetches project information from a web element.

        Args:
            bx (WebElement): The web element containing the project information.

        Returns:
            ProjectRecord: An instance of the ProjectRecord class containing the fetched project information.
        """

        # NOTE: ファンディング情報を取得する
        thumnail_elm: WebElement = bx.find_element(
            By.XPATH, './/div[@class="box-thumbnail"]'
        )
        img_url = thumnail_elm.find_element(By.XPATH, ".//img").get_attribute("src")
        # project_url like 'https://camp-fire.jp/projects/view/736020?list=projects_popular_page1'
        project_url = thumnail_elm.find_elements(By.XPATH, ".//a")[-1].get_attribute(
            "href"
        )  # first or second element in the <a> tag list
        project_url = _cleanup_url(project_url)
        try:
            area: str = bx.find_element(By.XPATH, './/*[@class="area"]').text
        except NoSuchElementException:
            # NOTE: 一部のプロジェクトは、エリア情報がない場合がある
            area = None

        title: str = bx.find_element(By.XPATH, './/*[@class="box-title"]').text
        status: str = "OPEN"
        try:
            meter: str = bx.find_element(By.XPATH, './/*[@class="meter"]').text
        except NoSuchElementException:
            # NOTE: 100% になっている場合があるので、その場合は、success-summary から取得する
            try:
                elm = bx.find_element(By.XPATH, './/*[@class="success-summary"]')
                meter: str = "100%"
                status: str = elm.text
            except NoSuchElementException:
                # 新規プロジェクトの場合、meter がない場合がある
                meter = "0%"
                assert status == "OPEN"

        category: str = bx.find_element(By.XPATH, './/*[@class="category"]').text
        owner: str = bx.find_element(By.XPATH, './/*[@class="owner"]').text
        try:
            current_funding: str = bx.find_element(
                By.XPATH, './/*[@class="total"]'
            ).text
        except NoSuchElementException:
            # 新規プロジェクトの場合、current_funding がない場合がある
            current_funding: str = "0"
        try:
            supporters: str = bx.find_element(By.XPATH, './/*[@class="rest"]').text
        except NoSuchElementException:
            # 新規プロジェクトの場合、supporters がない場合がある
            supporters: str = "0"
        try:
            remaining_days: str = bx.find_element(By.XPATH, './/*[@class="per"]').text
        except NoSuchElementException:
            # 新規プロジェクトの場合、remaining_days がない場合がある
            remaining_days: str = None

        # NOTE: 形式が以下のようになっているので、整形する
        # - meter:  1009% -> 1009: int
        # - current_funding:  現在\n30,270,000円 -> 30270000: int
        # - supporters:  支援者\n450人 -> 450: int
        # - remaining_days:  残り\n7日 -> 7: int
        meter = pickup_numbers(meter)
        current_funding = pickup_numbers(current_funding)
        supporters = pickup_numbers(supporters)
        if remaining_days is not None:
            if "終了" in remaining_days:
                remaining_days = 0
            else:
                remaining_days = pickup_numbers(remaining_days, null_value=None)

        return ProjectRecord(
            box_idx,
            img_url,
            page,
            project_url,
            area,
            title,
            meter,
            category,
            owner,
            current_funding,
            supporters,
            remaining_days,
            status,
        )

    def _fetch_simply(self, elm_path: str) -> list[WebElement]:
        """
        Fetches web elements based on the given element path.

        Args:
            elm_path (str): The path of the web element to fetch.

        Returns:
            list[WebElement]: A list of web elements matching the given path.
        """

        elms: list[WebElement] = self._fetch(
            url=None,  # use opened page
            elm_path=elm_path,
        )
        return elms

    def _fetch_details_community(self, project_url: str) -> ProjectDetails:
        def _get_text(xpath: str) -> str | None:
            try:
                elms: list[WebElement] = self._fetch_simply(xpath)
            except TimeoutException:
                return None
            text: str = elms[0].text if elms else None
            return text

        # プロジェクトが取れない場合、コミュニティのページかもしれないので、コミュニティの情報を取得する
        title: str = _get_text('//*[@class="community-name"]')
        img_url: str = _get_text('//*[@class="flicking-panel"]')
        backer_amount: str = _get_text('//*[@class="category-and-amount"]')
        abstract: str = _get_text('//*[@class="community-description"]')
        elms: list[WebElement] = self._fetch_simply('//section[@id="project-body"]')
        article_text: str = elms[0].text
        article_html: str = elms[0].get_attribute("innerHTML")
        returns: list[WebElement] = self._fetch_simply('//div[@class="reward"]')
        return_boxes = []
        for idx, rwd in enumerate(returns):
            try:
                return_img_url: str = rwd.find_element(
                    By.XPATH, ".//img"
                ).get_attribute("src")
            except NoSuchElementException:
                return_img_url: str = None
            price: str = rwd.find_element(By.XPATH, './/div[@class="price"]').text
            rbx = ReturnBox(
                return_idx=idx,
                return_img_url=return_img_url,
                price=price,
                desc=rwd.text,
            )
            return_boxes.append(rbx)
        user_name: str = _get_text('//*[@class="owner-name"]')
        elms: list[WebElement] = self._fetch_simply(
            '//div[@class="owner-profile"]/*/img'
        )
        icon_url: str = elms[0].get_attribute("src")
        prefecture: str = _get_text(
            '//div[@class="owner-profile"]/div[@class="owner-name"]/p[@class="prefecture"]'
        )
        profile_text: str = _get_text(
            '//div[@class="owner-profile"]/*/*[@class="profile-body"]'
        )
        elms: list[WebElement] = self._fetch_simply('//div[@class="owner-profile"]')
        profile_url: str = elms[0].find_element(By.XPATH, ".//a").get_attribute("href")

        return ProjectDetails(
            type="community",
            project_url=project_url,
            title=title,
            img_url=img_url,
            backer_amount=backer_amount,
            abstract=abstract,
            article_text=article_text,
            article_html=article_html,
            profile_text=profile_text,
            profile_url=profile_url,
            icon_url=icon_url,
            user_name=user_name,
            prefecture=prefecture,
            project_exprience=None,
            readmore=None,
            return_boxes=return_boxes,
        )

    def _fetch_details_others(self, project_url: str) -> ProjectDetails:
        def _get_text(xpath: str) -> str | None:
            try:
                elms: list[WebElement] = self._fetch_simply(xpath)
            except TimeoutException:
                return None
            text: str = elms[0].text if elms else None
            return text

        title: str = _get_text('//h2[@class="header_top__title"]')
        elms: list[WebElement] = self._fetch_simply(
            '//div[contains(@class, "slide-item")]/img'
        )
        img_url: str = elms[0].get_attribute("src")

        elms: list[WebElement] = self._fetch_simply(
            '//div[@class="project_status"]//div[@class="status"]'
        )
        backer_amount: str = elms[0].text
        abstract: str = _get_text('//div[@class="header_bottom__cap"]')

        elms: list[WebElement] = self._fetch_simply('//div[@class="column_main"]')
        article_text: str = elms[0].text
        article_html: str = elms[0].get_attribute("innerHTML")

        profile_text: str = _get_text('//div[@class="column_side"]')
        profile_url: str = None
        elms: list[WebElement] = self._fetch_simply('//div[@class="author__img"]')
        icon_url: str = elms[0].get_attribute("src") if elms else None
        user_name: str = _get_text('//div[@class="author__name"]')
        prefecture: str = _get_text('//span[@class="address"]')

        # リターン情報を取得する
        try:
            elms: list[WebElement] = self._fetch_simply(
                '//div[contains(@class, "return-box") and not(contains(@class, "return-box-inner"))]'
            )
            return_boxes = []
            for return_idx, elm in enumerate(elms):
                try:
                    return_img_url: str = elm.find_element(
                        By.XPATH, ".//img"
                    ).get_attribute("data-src")
                except NoSuchElementException:
                    return_img_url = None

                price: str = elm.find_element(
                    By.XPATH, './/div[contains(@class, "return__price")]'
                ).text
                desc: str = elm.get_attribute("innerHTML")

                rbx = ReturnBox(return_idx, return_img_url, price, desc)
                return_boxes.append(rbx)
        except TimeoutException:
            price: str = _get_text('//div[@class="return__price"]')
            elms: list[WebElement] = self._fetch_simply(
                '//div[contains(@class, "return__img")]'
            )
            return_img_url: str = (
                elms[0].find_element(By.XPATH, ".//img").get_attribute("src")
            )

            elms: list[WebElement] = self._fetch_simply(
                '//p[contains(@class, "return__list") and contains(@class, "readmore")]'
            )
            readmore_link = elms[0].find_element(By.LINK_TEXT, "もっと見る")
            actions = ActionChains(self.driver)
            actions.move_to_element(readmore_link).perform()
            readmore_link.click()
            self.driver.implicitly_wait(3)
            desc: str = _get_text(
                '//p[contains(@class, "return__list") and contains(@class, "readmore")]'
            )
            rbx = ReturnBox(0, return_img_url, price, desc)
            return_boxes = [rbx]

        return ProjectDetails(
            type="others",  # furusato, support, etc.
            project_url=project_url,
            title=title,
            img_url=img_url,
            backer_amount=backer_amount,
            abstract=abstract,
            article_text=article_text,
            article_html=article_html,
            profile_text=profile_text,
            profile_url=profile_url,
            icon_url=icon_url,
            user_name=user_name,
            prefecture=prefecture,
            project_exprience=None,
            readmore=None,
            return_boxes=return_boxes,
        )

    def _fetch_details_project(self, project_url: str) -> ProjectDetails:
        # ヘッダ情報
        # # タイトルを取得する
        title: str = self._fetch_simply('//label[@class="project-name"]')[0].text

        # # サムネイル画像を取得する
        elms: list[WebElement] = self._fetch_simply(
            '//div[contains(@class, "slide-item")]/img'
        )
        img_url: str = elms[0].get_attribute("src")

        # 現在の支援総額を取得する
        elms: list[WebElement] = self._fetch_simply('//div[@class="backer-amount"]')
        backer_amount: str = elms[0].text

        # # 概要を取得する
        elms: list[WebElement] = self._fetch_simply(
            '//section[contains(@class, "caption") and contains(@class, "sp-none")]'
        )
        abstract: str = elms[0].text

        # メイン情報
        # # 記事内容を取得する
        elms: list[WebElement] = self._fetch_simply(
            '//article[contains(@class, "project-body-in") and contains(@class, "fr-view")]'
        )
        article_text: str = elms[0].text
        article_html: str = elms[0].get_attribute("innerHTML")

        # プロフィールを取得する
        elms: list[WebElement] = self._fetch_simply(
            '//section[contains(@class, "profile")]',
        )
        try:
            # 「もっと見る」リンクがある場合は、クリックして全文を取得する
            # 特に、「もっと見る」リンクがない場合はエラーになり、その場合は、そのまま取得する
            readmore_link = elms[0].find_element(By.LINK_TEXT, "もっと見る")
            actions = ActionChains(self.driver)
            actions.move_to_element(readmore_link).perform()
            readmore_link.click()
            self.driver.implicitly_wait(3)
        except NoSuchElementException:
            # raise しないようにキャッチしておく
            pass
        finally:
            profile_text = elms[0].text

        try:
            elms: list[WebElement] = self._fetch_simply(
                '//section[contains(@class, "profile")]/div[@class="icon"]/a'
            )
            profile_url: str = elms[0].get_attribute("href")
        except TimeoutException:
            profile_url: str = None

        try:
            elms: list[WebElement] = self._fetch_simply(
                '//section[contains(@class, "profile")]/div[@class="icon"]/*/img'
            )
            icon_url: str = elms[0].get_attribute("src")
        except TimeoutException:
            icon_url: str = None

        # ユーザ名を取得する
        elms: list[WebElement] = self._fetch_simply('//div[@class="username"]')
        user_name: str = elms[0].text

        # 県名を取得する
        elms: list[WebElement] = elms[0].find_elements(By.XPATH, '//ul[@class="pref"]')
        prefecture: str = elms[0].text if elms else None

        # プロジェクトの経験情報を取得する
        try:
            elms: list[WebElement] = self._fetch_simply(
                '//div[@class="projects-count"]'
            )
            project_exprience: str = elms[0].text
        except TimeoutException:
            project_exprience: str = None

        # プロフィールの追加情報を取得する
        try:
            elms: list[WebElement] = self._fetch_simply('//p[@class="readmore"]')
            readmore: str = elms[0].text
        except TimeoutException:
            readmore = None

        # リターン情報を取得する
        elms: list[WebElement] = self._fetch_simply(
            '//div[contains(@class, "return-box") and not(contains(@class, "return-box-inner"))]'
        )

        return_boxes = []
        for return_idx, elm in enumerate(elms):
            try:
                return_img_url: str = elm.find_element(
                    By.XPATH, ".//img"
                ).get_attribute("data-src")
            except NoSuchElementException:
                return_img_url = None

            price: str = elm.find_element(
                By.XPATH, './/div[contains(@class, "price")]'
            ).text

            try:
                desc: str = elm.find_element(
                    By.XPATH, './/div[@class="abbreviated-description"]'
                ).get_attribute("innerHTML")
            except NoSuchElementException:
                desc: str = None

            rbx = ReturnBox(return_idx, return_img_url, price, desc)
            return_boxes.append(rbx)

        return ProjectDetails(
            type="project",
            project_url=project_url,
            title=title,
            img_url=img_url,
            backer_amount=backer_amount,
            abstract=abstract,
            article_text=article_text,
            article_html=article_html,
            profile_text=profile_text,
            profile_url=profile_url,
            icon_url=icon_url,
            user_name=user_name,
            prefecture=prefecture,
            project_exprience=project_exprience,
            readmore=readmore,
            return_boxes=return_boxes,
        )

    def fetch_project_details(self, project_url: str) -> ProjectDetails:
        """
        Fetches the detailed information of a project from the given URL.

        Args:
            project_url (str): The URL of the project detail page.

        Returns:
            ProjectRecord: An instance of the ProjectRecord class containing the fetched project information.
        """

        # プロジェクト名を取得する
        elms: list[WebElement] = self._fetch(
            url=project_url,
            elm_path='//label[@class="project-name"]',
            wait_path="//*[@id='fb-root']",
            max_wait=10,
            do_scroll=True,
            scroll_size=500,
            scroll_wait=0.15,
            wait_after_scroll=3,
        )

        # 詳細ページでプロジェクト名が取得できた場合は、プロジェクト詳細情報を取得する
        if elms:
            return self._fetch_details_project(project_url)

        # 詳細ページでプロジェクト名が取得できなかった場合は、コミュニティ詳細情報を取得する
        elms: list[WebElement] = self._fetch(
            url=project_url,
            elm_path='//*[@class="community-name"]',
            wait_path="//*[@id='fb-root']",
        )
        # コミュニティ名を取得できたら、コミュニティ詳細情報を取得する
        if elms:
            return self._fetch_details_community(project_url)

        # ふるさと納税等の他の情報を取得する
        return self._fetch_details_others(project_url)

    def quit_driver(self):
        """
        Quits the Selenium WebDriver instance.
        """
        self.driver.quit()
        self.driver = None


def _main(params: DictConfig):
    cfr = CampfireFetcher()

    # Campfire のサイトから クラウドファンディングの情報一覧を取得する
    print(f"{now()} Fetching projects ...")
    project_records: list[ProjectRecord] = cfr.fetch_projects(params)

    # リスト毎にプロジェクトの詳細情報を取得
    print(f"{now()} Fetching project details ...")
    details_records: list[ProjectDetails] = []
    for pr in tqdm(project_records):
        details_record: ProjectDetails = cfr.fetch_project_details(pr.project_url)
        details_records.append(details_record)

    # GraphDb に取得した情報を保存する
    g = GraphDb()

    # 全てのプロジェクトに共通のキー情報を、ループの外で生成・設定する
    fetch_id: str = build_ulid("FCH")
    created_at: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sortby: SortBy = params.sortby
    source_id: str = "Campfire"

    # プロジェクト毎に保存する
    print(f"{now()} Saving projects to the graph database ...")
    try:
        for pr, dr in zip(project_records, details_records):
            project_id: str = build_ulid("PRJ")

            # # URL をプロジェクトルートノードとして保存する
            g.merge_node(
                label="ProjectRoot",
                source_id=source_id,
                project_url=pr.project_url,
            )
            g.create_index("ProjectRoot", keys=["project_url"])

            # # プロジェクト情報をノードとして保存する
            g.add_node(
                label="Project",
                project_id=project_id,
                fetch_id=fetch_id,
                source_id=source_id,
                sortby=sortby,
                **pr._asdict(),
                created_at=created_at,
            )
            g.create_index("Project", keys=["project_id"])
            g.create_index("Project", keys=["fetch_id"])
            g.create_index("Project", keys=["sortby"])
            g.create_index("Project", keys=["created_at", "sortby"])

            # # プロジェクトルートとプロジェクト情報を紐付ける
            g.merge_edge(
                label="Project",
                node_keys_src={"label": "ProjectRoot", "project_url": pr.project_url},
                node_keys_trg={"label": "Project", "project_id": project_id},
            )

            # # プロジェクト詳細情報をノードとして保存する
            details: dict = dr._asdict()
            return_boxes: list[ReturnBox] = details.pop("return_boxes")
            g.add_node(
                label="ProjectDetails",
                project_id=project_id,
                fetch_id=fetch_id,
                source_id=source_id,
                sortby=sortby,
                **details,
                created_at=created_at,
            )
            g.create_index("ProjectDetails", keys=["project_id"])
            g.create_index("ProjectDetails", keys=["fetch_id", "type"])
            g.create_index("ProjectDetails", keys=["sortby"])
            g.create_index("ProjectDetails", keys=["created_at", "sortby", "type"])

            # # プロジェクト情報とプロジェクト詳細情報を紐付ける
            g.add_edge(
                label="Details",
                node_keys_src={"label": "Project", "project_id": project_id},
                node_keys_trg={"label": "ProjectDetails", "project_id": project_id},
            )

            # # リターン情報毎に保存する
            for rbx in return_boxes:
                # # # リターン情報をノードとして保存する
                g.add_node(
                    label="ReturnBox",
                    project_id=project_id,
                    fetch_id=fetch_id,
                    source_id=source_id,
                    sortby=sortby,
                    **rbx._asdict(),
                    created_at=created_at,
                )
                # # # プロジェクト詳細情報とリターン情報を紐付ける
                g.add_edge(
                    label="ReturnBox",
                    node_keys_src={"label": "ProjectDetails", "project_id": project_id},
                    node_keys_trg={
                        "label": "ReturnBox",
                        "project_id": project_id,
                        "return_idx": rbx.return_idx,
                    },
                )
            g.create_index("ReturnBox", keys=["project_id", "return_idx"])
            g.create_index("ReturnBox", keys=["fetch_id"])
            g.create_index("ReturnBox", keys=["sortby"])
            g.create_index("ReturnBox", keys=["created_at", "sortby"])

        print(f"{now()} {len(project_records)} projects saved to the graph database.")
    finally:
        g.close()
        cfr.quit_driver()
        print(now(), "done!")


def config():
    cfg = DictConfig(dict(is_experiment=True, do_share=False))
    return cfg


def main(
    max_pages: int = 1,
    sortby: Annotated[SortBy, typer.Option(case_sensitive=False)] = SortBy.fresh,
):
    s = signature(main)
    kwargs = {}
    for k in list(s.parameters):
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    params = config()  # use as default
    params.update(kwargs)
    return _main(params)


if __name__ == "__main__":
    typer.run(main)
