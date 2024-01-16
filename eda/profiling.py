import webbrowser

import fire
from ydata_profiling import ProfileReport

from preprocessing.data_processing import load_cleaned_data

"Build profiling script"


def profile(src: str = "data/cleaned.csv", dest: str = "./eda/report.html"):
    df = load_cleaned_data(src=src)
    profile = ProfileReport(df)
    profile.to_file(dest)
    webbrowser.open(dest, new=0, autoraise=True)


if __name__ == "__main__":
    fire.Fire(profile)
