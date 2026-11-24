from telco_churn.download_data import download_dataset


def test_download_dataset_replaces_existing_file_atomically(tmp_path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "data.csv"
    source.write_text("customerID,Churn\n1,No\n", encoding="utf-8")
    output.write_text("old data", encoding="utf-8")

    downloaded = download_dataset(source.as_uri(), output)

    assert downloaded == output
    assert output.read_text(encoding="utf-8") == "customerID,Churn\n1,No\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "data.csv",
        "source.csv",
    ]
