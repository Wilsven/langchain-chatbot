import re
import urllib.request

from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse


# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute,
        # add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])


def get_hyperlinks(url: str) -> list[str]:
    """Function to get the hyperlinks from a URL.

    Parameters
    ----------
    url: str
        URL to get the hyperlinks from.

    Returns
    -------
    list[str]
        Hyperlinks from the URL.
    """
    # Try to open the URL and read the HTML
    try:
        request = urllib.request.Request(url=url, headers={"User-Agent": "Mozilla/5.0"})
        # Open the URL and read the HTML
        with urllib.request.urlopen(request) as response:
            # If the response is not HTML, return an empty list
            if not response.info().get("Content-Type").startswith("text/html"):
                return []

            # Decode the HTML
            html = response.read().decode("utf-8")
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks


def get_domain_hyperlinks(
    local_domain: str, url: str, http_url_pattern: str
) -> list[str]:
    """Function to get the hyperlinks from a URL that are within the same domain.

    Parameters
    ----------
    local_domain: str
        Domain.

    url: str
        URL to get the hyperlinks from.

    Returns
    -------
    list[str]
        Hyperlinks within the same domain as the URL.
    """
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # Regex pattern to match a URL
        HTTP_URL_PATTERN = http_url_pattern

        # If link matches any strings below, continue
        if (
            re.search(r"email", link)
            or re.search(r"people-directory", link)
            or re.search(r"login", link)
            or re.search(r"profile", link)
            or re.search(r"register", link)
            or re.search(r"password", link)
            or re.search(r"javascript", link)
        ):
            continue
        # If the link is a URL, check if it is within the same domain
        elif re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif (
                link.startswith("#")
                or link.startswith("mailto:")
                or re.search(r"tel:", link)
            ):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            # if clean_link.endswith("/"):
            #     clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))


def crawl(url: str, http_url_pattern: str = r"^http[s]*://.+") -> set[str]:
    """Crawl the given domain URL to get all the hyperlinks.

    Parameters
    ----------
    url: str
        URL to get the hyperlinks from.

    Returns
    -------
    list[str]
            Hyperlinks crawled from root URL.
    """
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # While the queue is not empty, continue crawling
    while queue:
        # Get the next URL from the queue
        url = queue.pop()
        print(url)  # for debugging and to see the progress

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url, http_url_pattern):
            if link not in seen:
                queue.append(link)
                seen.add(link)

    return seen


def clean(
    urls: set[str], exclude: list[str] = ["png", "jpg", "jpeg", "mp4"]
) -> list[str]:
    """Remove URLs with suffix to exclude.

    Parameters
    ----------
    urls: set[str]
        Hyperlinks crawled from root URL.

    exclude: list[str]
        List of suffix to exclude.

    Returns
    -------
    list[str]
        Hyperlinks with excluded suffix removed.
    """
    temp_urls = list(urls).copy()
    for url in temp_urls:
        for suffix in exclude:
            if url.endswith(suffix):
                urls.remove(url)

    return urls


def strip_content(page_content: str) -> str:
    """
    Remove white spaces, new lines and tab
    lines from page content.

    Parameters
    ----------
    page_content: str
        Page content in Document module.

    Returns
    -------
    new_content: str
        New page content
    """
    new_content = re.sub("\s+", " ", page_content)
    return new_content
