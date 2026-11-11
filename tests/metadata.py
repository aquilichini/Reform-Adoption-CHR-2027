from pathlib import Path
import pandas as pd

# ============================================================
# PARAMÈTRES
# ============================================================

CORPUS_DIR = Path("/Volumes/Extreme_SSD/conllu_corpus")

OUTPUT_DIR = Path("/Volumes/Extreme_SSD/corpus_statistics")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# PARCOURS DU CORPUS
# ============================================================

rows = []

print("Scanning corpus...")

for file in CORPUS_DIR.rglob("*.conllu"):

    stem = file.stem

    # Vérifie le format AAAAMMJJ_nomdujournal
    if len(stem) < 10:
        continue

    if not stem[:8].isdigit():
        continue

    date = stem[:8]
    journal = stem[9:]

    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    decade = (year // 10) * 10

    rows.append(
        {
            "filename": file.name,
            "filepath": str(file),
            "journal": journal,
            "date": date,
            "year": year,
            "month": month,
            "day": day,
            "decade": decade,
        }
    )

metadata = pd.DataFrame(rows)

metadata = metadata.sort_values(["journal", "date"])

metadata.to_csv(
    OUTPUT_DIR / "metadata.csv",
    index=False,
    encoding="utf-8",
)

print(f"{len(metadata):,} fichiers trouvés.")

# ============================================================
# STATISTIQUES PAR JOURNAL
# ============================================================

journal_stats = (
    metadata.groupby("journal")
    .agg(
        files=("filename", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        years_covered=("year", "nunique"),
    )
    .sort_values("files", ascending=False)
)

journal_stats.to_csv(
    OUTPUT_DIR / "journal_statistics.csv",
    encoding="utf-8",
)

# ============================================================
# STATISTIQUES PAR ANNÉE
# ============================================================

year_stats = (
    metadata.groupby("year")
    .agg(
        files=("filename", "count"),
        journals=("journal", "nunique"),
    )
    .sort_index()
)

year_stats.to_csv(
    OUTPUT_DIR / "year_statistics.csv",
    encoding="utf-8",
)

# ============================================================
# STATISTIQUES PAR DÉCENNIE
# ============================================================

decade_stats = (
    metadata.groupby("decade")
    .agg(
        files=("filename", "count"),
        journals=("journal", "nunique"),
    )
    .sort_index()
)

decade_stats.to_csv(
    OUTPUT_DIR / "decade_statistics.csv",
    encoding="utf-8",
)

# ============================================================
# RÉSUMÉ GLOBAL
# ============================================================

summary = []

summary.append("CORPUS SUMMARY")
summary.append("=" * 60)
summary.append("")

summary.append(f"Total files           : {len(metadata):,}")
summary.append(f"Distinct journals     : {metadata['journal'].nunique():,}")
summary.append(f"First date            : {metadata['date'].min()}")
summary.append(f"Last date             : {metadata['date'].max()}")
summary.append(f"First year            : {metadata['year'].min()}")
summary.append(f"Last year             : {metadata['year'].max()}")
summary.append(f"Years represented     : {metadata['year'].nunique()}")
summary.append(f"Decades represented   : {metadata['decade'].nunique()}")

summary.append("")
summary.append("FILES PER YEAR")
summary.append("-" * 60)

summary.append(
    f"Minimum : {year_stats['files'].min():,}"
)
summary.append(
    f"Maximum : {year_stats['files'].max():,}"
)
summary.append(
    f"Mean    : {year_stats['files'].mean():.1f}"
)
summary.append(
    f"Median  : {year_stats['files'].median():.1f}"
)

summary.append("")
summary.append("FILES PER JOURNAL")
summary.append("-" * 60)

summary.append(
    f"Minimum : {journal_stats['files'].min():,}"
)
summary.append(
    f"Maximum : {journal_stats['files'].max():,}"
)
summary.append(
    f"Mean    : {journal_stats['files'].mean():.1f}"
)
summary.append(
    f"Median  : {journal_stats['files'].median():.1f}"
)

summary.append("")
summary.append("TOP 20 JOURNALS")
summary.append("-" * 60)

for journal, row in journal_stats.head(20).iterrows():
    summary.append(
        f"{journal:<40} {row['files']:>6}"
    )

with open(
    OUTPUT_DIR / "corpus_summary.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(summary))

print()
print("Done.")
print(f"Results written to {OUTPUT_DIR}")
