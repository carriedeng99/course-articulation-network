# Scraper for NJTransfer: Develop scraping: https://njtransfer.org/artweb/chgri.cgi?8612971757946984

# Date Created: 01/16/2026
# Last Edited: 01/24/2026
# Created By: CD

#%%
# Import packages
from importlib import reload
import re
import os
import glob
import csv
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
import pandas as pd

#%%
# ============================================================
# =============== Scrape All Course-pair urls ================
# ============================================================

# Helper: initialize a clean session with cookies
def new_session():
    """Initializes New NJTransfer Session"""
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    
    s.get("https://njtransfer.org/artweb/crs-srch.cgi?S")
    return s

# Load change-institution page
session = new_session()
html = session.get("https://njtransfer.org/artweb/chgri.cgi").text
soup = BeautifulSoup(html, "html.parser")

# Helper: acquire lists for sending and receiving institutions
def extract_hidden_mapping(soup, input_name):
    inp = soup.find("input", {"name": input_name})
    if not inp:
        raise RuntimeError(f"{input_name} not found")

    raw = inp["value"]
    mapping = {}

    for chunk in raw.split("!"):
        if "=" in chunk:
            code, name = chunk.split("=", 1)
            code = code.strip()
            name = name.strip()
            if code and code != "All":
                mapping[code] = name

    return mapping

# Define institution pairs
sending_institutions = extract_hidden_mapping(soup, "si_string") # 19 sending institutions
receiving_institutions = extract_hidden_mapping(soup, "ri_string") # 41 receiving institutions

pairs = [(item1, item2) for item1 in sending_institutions for item2 in receiving_institutions] # 779 pairs

# Loop through letters and 0-9 as keywords
keyword_cat = list(map(chr, range(97, 123))) + list(range(10))

# Collect course-pair urls
for si, ri in tqdm(pairs):
    
    course_links = []
    df = pd.DataFrame()

    #Continue if we have this already
    if os.path.exists(f"../data/urls/{si}_{ri}.csv"):
        continue

    for _,x in enumerate(keyword_cat):

        session = new_session()
        
        url = "https://njtransfer.org/artweb/kwicsrch.cgi"
        
        data = {
                # control flags
                "SG": "Go",
                "flipto": "kwicsrch.cgi",
                "From_where": "crs-srch.cgi?S",

                # TO PLUG IN looped inputs
                "SI": si,
                "RI": ri,
                "RRI": ri,
                "Keyword": x,

                # include inactive courses
                "InActive": "on",

                # required empty fields
                "SIGED": "",
                "RIGED": "",
                "KeyCourse": "",
                "SKeyCourse": "",
                "CrsId": "",
            }
        
        headers = {
            "Referer": "https://njtransfer.org/artweb/crs-srch.cgi?S",
            }

        r = session.post(url, data=data, headers=headers)
        r.raise_for_status()

        html = r.text
        
        BASE = "https://njtransfer.org/artweb/"
        soup = BeautifulSoup(html, "html.parser")

        links_pair_siged = []

        for row in soup.select("table tr"):
            cells = row.find_all("td")
            if not cells:
                continue

            status_text = cells[-1].get_text(" ", strip=True)

            # Skip non-transferable and EC rows
            if status_text == "Not Transferable" or status_text == "EC":
                continue

            a = row.select_one("a[href^='crs-srch.cgi']")
            if not a:
                continue

            links_pair_siged.append(urljoin(BASE, a["href"]))

        if links_pair_siged:
            orig = set(course_links)
            new = set(links_pair_siged)
            comb = orig | new
            course_links = list(comb)
        
        time.sleep(3)
    
    df["url"] = course_links
    df.to_csv(f"../data/urls/{si}_{ri}.csv", index=False)

urls_folder = r"../data/urls"
files = os.path.join(urls_folder, "*.csv")
combined_df = pd.concat((pd.read_csv(f) for f in glob.glob(files)), ignore_index=True)
all_urls = combined_df.drop_duplicates().reset_index(drop = True)

#Construct unique ids from urls so that we can save records
expand = all_urls.url.str.split("~", expand = True)
all_urls['record_id'] = expand[1]
all_urls['sending_institution'] = expand[2]
all_urls['receiving_institution'] = expand[3]
all_urls['course'] = expand[4]

all_urls['start_date'] = expand[5]
all_urls['end_date'] = expand[6]

#Create start and end date var so we can scrape most recent courses first
for date in ['start', 'end']:
    print("Cleaning", date, "dates")
    all_urls[[f'{date}_term', 
              f'{date}_year']] = (all_urls[f'{date}_date'].
                                            str.split("*", expand = True))

    recode_terms = {"FALL": "09",
                    "SPRING": "04",
                    "WINTER": "01",
                    "SUMMER": "06"}

    if date == "end":
        #Fifteen end date urls are missing
        #Drop these as they are negligable in dataset
        assert((all_urls['end_date'] == "*").sum() == 15)
        all_urls = all_urls.query("end_date != '*'").copy()
        assert((all_urls['end_date'] == "*").sum() == 0)


    all_urls[f'{date}_date_int'] = (all_urls[f'{date}_year'] +
                                    all_urls[f'{date}_term'].
                                    replace(recode_terms)
                                    ).astype(int)
    
all_urls['course_id'] = (all_urls['sending_institution'] + "_" +
                         all_urls['receiving_institution'] + "_" +
                         all_urls['course'] + "_" + 
                         all_urls['end_date_int'].astype(str) + "_" +
                         all_urls['record_id'])

#These are duplicates where we have two start dates and one end date
#Keep only most recent start date
num_duplicates = all_urls['course_id'].duplicated().sum()
print("Dropping duplicates by course_id", num_duplicates)
assert(num_duplicates == 13530)
all_urls.sort_values(by = ['course_id',
                           'end_date_int',
                           'start_date_int'],
                           ascending=False,
                           inplace = True)

all_urls.drop_duplicates(subset = 'course_id',
                         inplace = True)

df = all_urls.copy()

df = df.sort_values("start_date_int", ascending=False)\
       .drop_duplicates(subset="course", keep="first")

# ============================================================
# ===========-===== Parse Course-pair pages ==================
# ============================================================
# Helper Functions
# Helper: save raw text for auditing
def textraw(lines):
    """Return text block containing all raw info"""
    try:
        start = lines.index("Transfer from")
        end = lines.index("Previous")
        return lines[start:end]
    
    except (ValueError, IndexError):
        return None

# Helper: parse course codes
def split_course_code(course_code: str):
    """
    Splits a course code like 'HI101', 'BIO203', 'MATH-221', 'CS 50'
    into (subject, number)
    """
    match = re.search(r"([A-Z]+)\s*-?\s*([0-9A-Z]+)", course_code.upper())
    if not match:
        return None, course_code
    return match.group(1), match.group(2)

# Helper: check existence of labels in soup
def safe_next(lines, label):
    """Return the line after a label, or None."""
    try:
        idx = lines.index(label)
        return lines[idx + 1]
    except (ValueError, IndexError):
        return None


def safe_all_next(lines, label):
    """Return all lines following a repeated label."""
    results = []
    for i, item in enumerate(lines):
        if item == label and i + 1 < len(lines):
            results.append(lines[i + 1])
    return results

# Helper: handle multi-line values
def safe_single_equivalency(lines):
    """
    Returns a single reconstructed equivalency string, or None.
    """
    try:
        start = lines.index("Equivalency:") + 1
    except ValueError:
        return None

    block = []

    for line in lines[start:]:
        if line == "Minimum Grade:":
            break
        if line.strip():
            block.append(line.strip())

    if not block:
        return None

    return " ".join(l.strip('"').strip() for l in block)

def safe_notes(lines):
    """
    Returns a single reconstructed nootes string, or None.
    """
    try:
        start = lines.index("Foot Note:") + 1
    except ValueError:
        return None

    block = []

    for line in lines[start:]:
        if line == "Previous":
            break
        if line.strip():
            block.append(line.strip())

    if not block:
        return None

    return " ".join(l.strip('"').strip() for l in block)

# Helper: if exists, save sending course syllabus - njtransfer does not host receiving course syllabus
def get_syllabus_url(soup):
    """
    Returns syllabus URL if present, else None
    """
    link = soup.find("a", string=lambda s: s and "VIEW SYLLABUS" in s)
    if link and link.get("href"):
        return link["href"]
    return None

# Main function
def parse_course_pair_page(soup, lines, url):
    '''Returns parsed record given an NJTransfer course-pair url'''
    
    raw_block = textraw(lines)
    rawtext = " ".join(l.strip('"').strip() for l in raw_block)

    if "elective" in rawtext:
        return None

    sending_inst = safe_next(raw_block, "Transfer from")

    sending_course_r = safe_next(raw_block, "Course ID:")
    sending_sub_pref, sending_course = (
        split_course_code(sending_course_r) if sending_course_r else (None, None)
        )

    sending_course_title = safe_next(raw_block, "Title:")

    course_areas = safe_all_next(raw_block, "GenEd Area(s):")
    sending_course_area = course_areas[0] if len(course_areas) >= 1 else None
    sending_course_syllabus = get_syllabus_url(soup)

    receiving_inst = safe_next(raw_block, "Transfer to")
    
    receiving_course_r = safe_single_equivalency(raw_block)
    if receiving_course_r:
        course_r  = receiving_course_r.split()[0]
        receiving_sub_pref, receiving_course = split_course_code(course_r)
        if "(" in receiving_course_r and ")" in receiving_course_r:
            receiving_course_title = (
                receiving_course_r.split("(", 1)[1].split(")", 1)[0].strip()
            )
        else:
            receiving_course_title = None
    else:
        receiving_sub_pref = None
        receiving_course = None
        receiving_course_title = None

    receiving_course_area = course_areas[1] if len(course_areas) >= 2 else None
    notes = safe_notes(raw_block)

    record = {
        "sending_institution": sending_inst,
        "sending_sub_pref": sending_sub_pref,
        "sending_course": sending_course,
        "sending_course_title": sending_course_title,
        "sending_course_area": sending_course_area,
        "sending_course_syllabus": sending_course_syllabus,
        "receiving_institution": receiving_inst,
        "receiving_sub_pref": receiving_sub_pref,
        "receiving_course": receiving_course,
        "receiving_course_title": receiving_course_title,
        "receiving_course_area": receiving_course_area,
        "receiving_course_syllabus": None,
        "notes": notes,
        "url": url,
        "raw": rawtext
        }

    return record

all_urls['record_id'] = all_urls.url.str.split("~")

# Scrape and save to json
from tqdm.contrib import tzip
os.makedirs("data/jsons/new", exist_ok=True)
os.makedirs("data/error/new", exist_ok=True)
session = requests.Session()
HEADERS = {"User-Agent": "Mozilla/5.0"}

for x,y in tzip(df['url'], df['course_id']):
    try:
        file = f"data/jsons/new/{y}.json"

        # Check if we have a file for this record and continue if not
        if os.path.exists(file):
            continue

        resp = session.get(x, headers=HEADERS, timeout=100, verify=False)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        text = soup.get_text("\n", strip=True)

        lines = [l for l in text.split("\n") if l.strip()]

        record = parse_course_pair_page(soup=soup,
        lines=lines,
        url=x
            )

        if record is None:
            continue   # skip non-transferable course

        with open(file, "w") as json_file:
            json.dump(record, json_file, indent=4) 
        time.sleep(3)
    
    except Exception as e:

        print(str(e))

        data = {"url": x,
                "error": str(e)
                }
        with open(f"data/error/new/{y}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        time.sleep(3)

folder = r"data/jsons"
jsons = glob.glob(os.path.join(folder, '*.json'))

records = []

for j in tqdm(jsons):
    with open(j, 'r', encoding='utf-8') as f:
        record = json.load(f)
    records.append(record)

df_all = pd.DataFrame(records)

df_all['receiving_course_title'] = df_all['receiving_course_title'].str.strip()
df_all["sending_course_area"] = df_all["sending_course_area"].str.strip()
df_all["receiving_course_area"] = df_all["receiving_course_area"].str.strip()

# Filter out electives
df_all['keep'] = (~df_all['receiving_course_title']
              .str.contains(r'elective|gen\s*ed|general\s*education|\belec\b',
                            case=False, na=True)
             ).astype(int)

df_all[["sending_course_area", "receiving_course_area"]] = (
    df_all[["sending_course_area", "receiving_course_area"]]
    .replace(r"(?i).*?(course\s*status|foot\s*note).*", None, regex=True))

df_all = df_all[df_all['keep'] == 1].drop(columns=['keep'])

df_all.to_csv("data/panel.csv", index=False)