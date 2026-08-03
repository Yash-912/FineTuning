from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("review_queue")


class ReviewQueue:
    def __init__(self, db_path: str = "data/review_queue.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS reviews (
                request_id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                production_pred INTEGER NOT NULL,
                production_conf REAL NOT NULL,
                shadow_pred INTEGER,
                shadow_conf REAL,
                human_label INTEGER,
                human_labeled_at TEXT,
                reviewed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reviews_unreviewed ON reviews(reviewed);
        """)
        self._conn.commit()

    def add(
        self,
        request_id: int,
        text: str,
        production_pred: int,
        production_conf: float,
        shadow_pred: Optional[int] = None,
        shadow_conf: Optional[float] = None,
    ):
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO reviews
                (request_id, text, production_pred, production_conf,
                 shadow_pred, shadow_conf, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (request_id, text, production_pred, production_conf, shadow_pred, shadow_conf, now),
        )
        self._conn.commit()

    def pull_unreviewed(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM reviews WHERE reviewed = 0 ORDER BY request_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        columns = [d[0] for d in self._conn.execute("PRAGMA table_info(reviews)").fetchall()]
        return [dict(zip(columns, row)) for row in rows]

    def label(self, request_id: int, human_label: int):
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE reviews SET human_label = ?, human_labeled_at = ?, reviewed = 1 WHERE request_id = ?",
            (human_label, now, request_id),
        )
        self._conn.commit()

    def export_labeled(self, output_path: str = "data/reviewed_labels.jsonl") -> int:
        rows = self._conn.execute(
            "SELECT text, human_label FROM reviews WHERE reviewed = 1 AND human_label IS NOT NULL"
        ).fetchall()
        with open(output_path, "w", encoding="utf-8") as f:
            for text, label in rows:
                f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
        logger.info("Exported %d labeled examples to %s", len(rows), output_path)
        return len(rows)

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        reviewed = self._conn.execute("SELECT COUNT(*) FROM reviews WHERE reviewed = 1").fetchone()[0]
        injection_hits = self._conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE production_pred = 1 AND reviewed = 1"
        ).fetchone()[0]
        return {
            "total_requests": total,
            "reviewed": reviewed,
            "pending": total - reviewed,
            "injection_hits_labeled": injection_hits,
        }

    def close(self):
        self._conn.close()


def cli_main():
    import argparse

    parser = argparse.ArgumentParser("review_queue")
    parser.add_argument("--db", default="data/review_queue.db")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--export", type=str, help="Export labeled examples to JSONL")
    parser.add_argument("--stats", action="store_true", help="Show queue statistics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    q = ReviewQueue(args.db)

    if args.stats:
        s = q.stats()
        for k, v in s.items():
            print(f"  {k}: {v}")
        return

    if args.export:
        n = q.export_labeled(args.export)
        print(f"Exported {n} records to {args.export}")
        return

    items = q.pull_unreviewed(args.limit)
    if not items:
        print("No unreviewed items.")
        return

    print(f"\n{'='*60}")
    print(f"  {len(items)} items pending review")
    print(f"{'='*60}\n")

    for item in items:
        print(f"Request #{item['request_id']}  |  created: {item['created_at'][:19]}")
        print(f"  Production: {'INJECTION' if item['production_pred'] == 1 else 'BENIGN'} "
              f"(conf={item['production_conf']:.4f})")
        if item['shadow_pred'] is not None:
            shadow_label = "INJECTION" if item['shadow_pred'] == 1 else "BENIGN"
            print(f"  Shadow:     {shadow_label} (conf={item['shadow_conf']:.4f})")
        print(f"  Text: {item['text'][:200]}")
        print()
        while True:
            ans = input("  Label (i=injection, b=benign, s=skip, q=quit): ").strip().lower()
            if ans == "q":
                q.close()
                return
            if ans == "s":
                print()
                break
            if ans == "i":
                q.label(item["request_id"], 1)
                print("  -> Labeled INJECTION\n")
                break
            if ans == "b":
                q.label(item["request_id"], 0)
                print("  -> Labeled BENIGN\n")
                break


if __name__ == "__main__":
    cli_main()
