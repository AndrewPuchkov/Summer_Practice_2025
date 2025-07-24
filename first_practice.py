from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

driver = webdriver.Chrome()


def parse(adress, cr, count):
    driver.get(adress)
    js_content = driver.page_source
    soup = BeautifulSoup(js_content, "html.parser")
    table = soup.find("table", {"id": "apple_quality"})
    columns_tags = table.find("thead")
    columns_massive = [th.get_text(strip=True) for th in columns_tags.find_all("th")]
    table_body_tag = table.find("tbody")
    rows_tags = table_body_tag.find_all("tr")
    for row_tag in rows_tags:
        rows_items_tags = row_tag.find_all("td")
        rows_items = [tag_row_item.text for tag_row_item in rows_items_tags]
        cr.append(rows_items)
    if count == 0:
        df = pd.DataFrame(columns=columns_massive)
        df.to_csv('quotes.csv', index=False, encoding='cp1251')
    return cr


def get_adresses(site):
    driver.get(site)
    urls_content = driver.page_source
    site_w_adresses = BeautifulSoup(urls_content, "html.parser")
    adresses = [tag.text for tag in site_w_adresses.find_all("li")]
    return adresses


adresses = get_adresses("https://ratcatcher.ru/media/summer_prac/parcing/17/index.html")
print(adresses)
common_rows = []
count = 0
for adress in adresses:
    common_rows = parse(adress, common_rows, count)
    count += 1
df1 = pd.DataFrame(columns=common_rows)
df1.to_csv('quotes.csv', mode='a', index=False, encoding='cp1251')
