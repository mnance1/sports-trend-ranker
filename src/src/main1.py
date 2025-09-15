import os, time, json, re, math, datetime as dt
from collections import defaultdict, Counter
import requests, feedparser, pandas as pd

USE_TRENDS = os.getenv("USE_TRENDS", "0") == "1"
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "0") == "1"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------- Sources ----------
SUBS = ["nfl","nba","soccer","boxing","MMA"]
NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=site:espn.com+OR+site:espn.go.com&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=site:bleacherreport.com&hl=en-US&gl=US&ceid=US:en",
]

SPORT_TERMS = ["NFL","NBA","UFC","Boxing","Soccer"]

def now_ms():
    return int(time.time()*1000)

def hours_ago(ts_ms):
    return (now_ms() - ts_ms)/3_600_000

def fetch_reddit_top_day(sub):
    url = f"https://www.reddit.com/r/{sub}/top.json?t=day&limit=25"
    r = requests.get(url, headers={"User-Agent":"trend-checker/1.0"})
    r.raise_for_status()
    data = r.json()
    out=[]
    for c in data.get("data",{}).get("children",[]):
        d=c["data"]
        out.append({
            "source":"reddit",
            "subreddit":sub,
            "title":d.get("title","").strip(),
            "url":"https://reddit.com"+d.get("permalink",""),
            "ts": int(d.get("created_utc", time.time()))*1000,
            "ups": d.get("ups",0),
            "comments": d.get("num_comments",0)
        })
    return out

def fetch_news(feed_url):
    feed = feedparser.parse(feed_url)
    items=[]
    for e in feed.entries:
        ts = None
        for key in ["published_parsed","updated_parsed"]:
            if getattr(e,key,None):
                ts = int(time.mktime(getattr(e,key)))
                break
        ts_ms = (ts or int(time.time()))*1000
        items.append({
            "source":"news",
            "title": e.title.strip(),
            "url": e.link,
            "ts": ts_ms,
            "ups":0,
            "comments":0
        })
    # keep last 24h
    items = [i for i in items if hours_ago(i["ts"])<=24]
    return items

# ---------- Entities (spaCy) ----------
import spacy
nlp = spacy.load("en_core_web_sm")
def extract_entities(text):
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON","ORG","GPE","NORP","EVENT","WORK_OF_ART"}:
            ents.append(ent.text.strip())
    # also capture all-caps tokens like NFL, NBA, UFC
    caps = re.findall(r"\b([A-Z]{2,})\b", text)
    ents.extend(caps)
    # de-dupe, lowercase for matching, but keep display in join
    uniq = []
    seen = set()
    for e in ents:
        k = e.lower()
        if k not in seen:
            seen.add(k); uniq.append(e)
    return uniq

# ---------- Trends (optional) ----------
def get_trend_slopes(terms):
    if not USE_TRENDS:
        return {}
    from pytrends.request import TrendReq
    pytrends = TrendReq(hl='en-US', tz=360)  # CST/CDT-ish
    slopes={}
    for term in terms:
        try:
            pytrends.build_payload([term], timeframe="now 1-d", geo="US")
            df = pytrends.interest_over_time()
            if df.empty: 
                slopes[term]=0.0; continue
            y = df[term].values.tolist()
            if len(y)<3: 
                slopes[term]=0.0; continue
            # simple slope: end - start normalized by 100
            slope = (y[-1]-y[0])/100.0
            slopes[term]=slope
        except Exception:
            slopes[term]=0.0
    return slopes

# ---------- Scoring ----------
def normalize(val, lo, hi):
    if hi==lo: return 0.0
    x=(val-lo)/(hi-lo)
    return max(0.0,min(1.0,x))

def score_items(items):
    # per-Reddit normalization
    r = [i for i in items if i["source"]=="reddit"]
    min_up = min([x["ups"] for x in r], default=0)
    max_up = max([x["ups"] for x in r], default=1)
    min_com= min([x["comments"] for x in r], default=0)
    max_com= max([x["comments"] for x in r], default=1)

    # entity frequency in last 12h
    twelve_ms = now_ms() - 12*3600*1000
    freq=Counter()
    for it in items:
        if it["ts"]>=twelve_ms:
            for e in it.get("entities",[]):
                freq[e.lower()] += 1
    max_freq = max(freq.values(), default=1)

    def entity_velocity(ents):
        if not ents: return 0.0
        v=0
        for e in ents:
            v += freq[e.lower()]
        return v/max_freq

    # cross-source flag
    def cross_source(ents):
        if not ents: return 0
        s=set([e.lower() for e in ents])
        has_r = any(i for i in items if i["source"]=="reddit" and s.intersection({e.lower() for e in i.get("entities",[])}))
        has_n = any(i for i in items if i["source"]=="news" and s.intersection({e.lower() for e in i.get("entities",[])}))
        return 1 if (has_r and has_n) else 0

    # optional trend slopes
    trend_slopes = get_trend_slopes(SPORT_TERMS)
    def trend_signal(ents):
        # if item mentions a core sport term, use that slope max
        val = 0.0
        for core in SPORT_TERMS:
            if core.lower() in " ".join(ents).lower():
                val = max(val, trend_slopes.get(core,0.0))
        # clamp to [0,1] assuming slope in [-1,1]
        return max(0.0, min(1.0, (val+1)/2))

    out=[]
    for it in items:
        R_up = normalize(it["ups"], min_up, max_up) if it["source"]=="reddit" else 0.0
        R_com= normalize(it["comments"], min_com, max_com) if it["source"]=="reddit" else 0.0
        Fresh= max(0.0, 1 - hours_ago(it["ts"])/24)
        NewsVel = entity_velocity(it.get("entities",[]))
        Cross = cross_source(it.get("entities",[]))
        Trend = trend_signal(it.get("entities",[])) if USE_TRENDS else 0.0

        score = (0.30*R_up + 0.20*R_com + 0.15*Fresh + 0.20*NewsVel + 0.10*Cross + 0.05*Trend)
        it["R_up"], it["R_com"], it["Fresh"], it["NewsVel"], it["Cross"], it["TrendSlope"] = R_up, R_com, Fresh, NewsVel, Cross, Trend
        it["ViralityScore"] = round(score,4)
        out.append(it)
    # dedupe by URL/title, keep highest
    seen={}
    for it in out:
        key = it["url"] if it["url"] else it["title"].lower()
        if key not in seen or it["ViralityScore"]>seen[key]["ViralityScore"]:
            seen[key]=it
    ranked = sorted(seen.values(), key=lambda x: x["ViralityScore"], reverse=True)
    return ranked

def suggest_hook(title):
    # 9-word spicy hook generator (cheap heuristic)
    verbs = ["exposed","changes","breaks","proves","ends","reveals","stuns","shocks","explodes","erases"]
    words = re.findall(r"[A-Za-z0-9']+", title)
    subject = " ".join(words[:5]) if words else "This"
    return f"{subject} {verbs[hash(title)%len(verbs)]} everything you thought."

def main():
    all_items=[]
    # Reddit
    for sub in SUBS:
        all_items += fetch_reddit_top_day(sub)
    # News
    for feed in NEWS_FEEDS:
        all_items += fetch_news(feed)
    # Entities
    for it in all_items:
        it["entities"] = extract_entities(it["title"])
    ranked = score_items(all_items)

    # write CSV
    day = dt.datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(ranked)
    df.to_csv(f"data/daily_{day}.csv", index=False)

    # update README snippet
    top = ranked[:15]
    lines = ["| Rank | Score | Source | Title | Link |",
             "|---:|---:|:--:|---|---|"]
    for i, it in enumerate(top, start=1):
        lines.append(f"| {i} | {it['ViralityScore']:.2f} | {it['source']} | {it['title'].replace('|','-')} | [open]({it['url']}) |")
    table = "\n".join(lines)

    # replace section in README
    with open("README.md","r",encoding="utf-8") as f:
        readme = f.read()
    new_block = f"<!-- DAILY_START -->\n## Today’s Top 15 (auto)\n\n{table}\n\n<!-- DAILY_END -->"
    if "<!-- DAILY_START -->" in readme and "<!-- DAILY_END -->" in readme:
        readme = re.sub(r"<!-- DAILY_START -->.*?<!-- DAILY_END -->", new_block, readme, flags=re.S)
    else:
        readme += "\n\n"+new_block
    with open("README.md","w",encoding="utf-8") as f:
        f.write(readme)

    # Telegram (optional)
    if SEND_TELEGRAM and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        import telegram
        bot = telegram.Bot(token=TELEGRAM_TOKEN)
        msg_lines = ["Top 10 Sports Topics Today:"]
        for i, it in enumerate(ranked[:10], start=1):
            hook = suggest_hook(it["title"])
            msg_lines.append(f"{i}) {it['title']}  [{it['ViralityScore']:.2f}]\n{hook}\n{it['url']}")
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="\n\n".join(msg_lines))
    print("DONE")

if __name__=="__main__":
    main()
