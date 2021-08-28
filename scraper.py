import requests
import urllib.request as urllib2
from bs4 import BeautifulSoup

root_url = 'https://www.jmlr.org'
page = urllib2.urlopen(root_url)

soup = BeautifulSoup(page, 'html.parser')
link_tags = soup.find_all("a", string="pdf")
i = 0
for link_tag in link_tags:
    i = i + 1
    partial_url = link_tag.get('href')

    file_url = "".join([root_url, partial_url])

    r = requests.get(file_url, stream=True)

    with open(f"./download/{i}.pdf", "wb") as pdf:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                pdf.write(chunk)
