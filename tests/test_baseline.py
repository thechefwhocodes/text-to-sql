from src.baseline import extract_sql


def test_extract_sql_from_plain_text():
    assert extract_sql("SELECT 1") == "SELECT 1"


def test_extract_sql_from_fenced_block_with_language_tag():
    text = "Here you go:\n```sql\nSELECT * FROM items\n```"
    assert extract_sql(text) == "SELECT * FROM items"


def test_extract_sql_from_fenced_block_without_language_tag():
    text = "```\nSELECT * FROM items\n```"
    assert extract_sql(text) == "SELECT * FROM items"
