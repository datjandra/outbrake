#!/usr/bin/env python
"""Extract named entities from the WHO Disease Outbreak News RSS feed.

Loads the model saved by ``train.py``, fetches the feed, tags every token of
each entry's title and description and prints the entities found.

Example
-------
    python infer_rss.py --artifacts artifacts
    python infer_rss.py --feed-url https://www.who.int/feeds/entity/csr/don/en/rss.xml --limit 5
"""

from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence TensorFlow start-up chatter

import feedparser
from bs4 import BeautifulSoup

from outbrake_ner.inference import WordTagger, format_entities, group_entities
from outbrake_ner.persistence import load_artifacts

DEFAULT_FEED_URL = "https://www.who.int/feeds/entity/csr/don/en/rss.xml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts", default="artifacts", help="directory written by train.py")
    p.add_argument("--feed-url", default=DEFAULT_FEED_URL, help="RSS feed URL (or a local .xml file path)")
    p.add_argument("--limit", type=int, default=0, help="only process the first N entries (0 = all)")
    p.add_argument("--quiet", action="store_true", help="do not echo the entry summary text")
    return p.parse_args()


def html_to_text(html: str) -> str:
    return BeautifulSoup(html or "", features="html.parser").get_text()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    config, encoder, decoder, inp_tokenizer, targ_tokenizer = load_artifacts(args.artifacts)
    tagger = WordTagger(config, encoder, decoder, inp_tokenizer, targ_tokenizer)

    feed = feedparser.parse(args.feed_url)
    if feed.bozo and not feed.entries:
        raise SystemExit(f"Could not read feed {args.feed_url!r}: {feed.bozo_exception}")

    entries = feed.entries[: args.limit] if args.limit else feed.entries
    print(f"Processing {len(entries)} entries from {args.feed_url}\n")

    for entry in entries:
        title = html_to_text(entry.get("title", ""))
        description = html_to_text(entry.get("description", "") or entry.get("summary", ""))

        print(title)
        print(entry.get("link", ""))
        if not args.quiet:
            print(description)

        word_tags = tagger.tag_tokens(title.split()) + tagger.tag_tokens(description.split())
        entities = group_entities(word_tags)

        lines = format_entities(entities)
        if lines:
            for line in lines:
                print(line)
        else:
            print("(no entities found)")
        print()


if __name__ == "__main__":
    main()
