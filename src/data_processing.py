"""
data_processing.py - Dataset loading, augmentation with adversarial examples,
and Bengali-font setup for matplotlib.
"""

import os
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm
import subprocess

from src.config import CSV_FILENAME, AUGMENTED_CSV, resolve_dataset_path

def setup_bengali_font():
    """
    Downloads (if needed) and registers a Bengali-compatible font for matplotlib.
    Returns a FontProperties object, or None if no font is found.
    """
    try:
        _font_dir = os.path.expanduser("~/.local/share/fonts")
        os.makedirs(_font_dir, exist_ok=True)
        ttf_path = os.path.join(_font_dir, "NotoSansBengali-Regular.ttf")
        if not os.path.exists(ttf_path):
            subprocess.run(
                ["wget", "-q", "-O", ttf_path,
                 "https://github.com/google/fonts/raw/main/ofl/notosansbengali/"
                 "NotoSansBengali%5Bwdth%2Cwght%5D.ttf"],
                check=False
            )
            subprocess.run(["fc-cache", "-f", "-v"], capture_output=True, check=False)
    except Exception:
        pass

    cache_dir = matplotlib.get_cachedir()
    for f in os.listdir(cache_dir):
        if f.startswith("fontlist") and f.endswith(".json"):
            os.remove(os.path.join(cache_dir, f))
    fm._load_fontmanager(try_read_cache=False)

    bengali_font = None
    for path in fm.findSystemFonts():
        fname = os.path.basename(path).lower()
        if "notosansbengali" in fname or ("noto" in fname and "bengali" in fname):
            bengali_font = path
            break
        if "lohit" in fname and "bengali" in fname:
            bengali_font = path

    if bengali_font is None:
        for candidate in [
            os.path.expanduser("~/.local/share/fonts/NotoSansBengali-Regular.ttf"),
            "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
            "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf",
        ]:
            if os.path.exists(candidate):
                bengali_font = candidate
                break

    if bengali_font:
        fp = fm.FontProperties(fname=bengali_font)
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = fp.get_name()
        fm.fontManager.addfont(bengali_font)
        print(f"✓ Bengali font configured: {os.path.basename(bengali_font)}")
        return fp
    else:
        print("⚠ No Bengali font found. Graph labels may not render correctly.")
        return None

ADVERSARIAL_SAMPLES = [
    {"question": "কে রাবণকে বধ করেছিলেন?", "reference_answer": "রাম রাবণকে বধ করেছিলেন।", "student_answer": "রাবণ রামকে বধ করেছিলেন।", "label": "incorrect", "human_score": 0, "subject": "ইতিহাস"},
    {"question": "কে বাঘ শিকার করেছিল?", "reference_answer": "শিকারী বাঘ শিকার করেছিল।", "student_answer": "বাঘ শিকারীকে শিকার করেছিল।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে চিঠি লিখেছিল?", "reference_answer": "রহিম চিঠি লিখেছিল।", "student_answer": "চিঠি রহিমকে লিখেছিল।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "পুলিশ কাকে ধরেছে?", "reference_answer": "পুলিশ চোরকে ধরেছে।", "student_answer": "চোর পুলিশকে ধরেছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে ভাত খাচ্ছে?", "reference_answer": "ছেলেটি ভাত খাচ্ছে।", "student_answer": "ভাত ছেলেটিকে খাচ্ছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে কাকে সাহায্য করেছে?", "reference_answer": "ডাক্তার রোগীকে সাহায্য করেছে।", "student_answer": "রোগী ডাক্তারকে সাহায্য করেছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে ইঁদুর ধরেছে?", "reference_answer": "বিড়াল ইঁদুর ধরেছে।", "student_answer": "ইঁদুর বিড়ালকে ধরেছে।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "কে সাপ মেরেছে?", "reference_answer": "কৃষক সাপ মেরেছে।", "student_answer": "সাপ কৃষককে মেরেছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে বই পড়ছে?", "reference_answer": "ছাত্রটি বই পড়ছে।", "student_answer": "বইটি ছাত্রটিকে পড়ছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে শিক্ষককে প্রশ্ন করল?", "reference_answer": "ছাত্রটি শিক্ষককে প্রশ্ন করল।", "student_answer": "শিক্ষক ছাত্রটিকে প্রশ্ন করল।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে ছবি এঁকেছে?", "reference_answer": "শিল্পী ছবিটি এঁকেছে।", "student_answer": "ছবিটি শিল্পীকে এঁকেছে।", "label": "incorrect", "human_score": 0, "subject": "শিল্প"},
    {"question": "বোলার কাকে আউট করেছে?", "reference_answer": "বোলার ব্যাটসম্যানকে আউট করেছে।", "student_answer": "ব্যাটসম্যান বোলারকে আউট করেছে।", "label": "incorrect", "human_score": 0, "subject": "খেলা"},
    {"question": "কে গাড়ি চালাচ্ছে?", "reference_answer": "চালক গাড়িটি চালাচ্ছে।", "student_answer": "গাড়িটি চালককে চালাচ্ছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "মা কাকে বকা দিয়েছে?", "reference_answer": "মা ছেলেকে বকা দিয়েছে।", "student_answer": "ছেলে মাকে বকা দিয়েছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে কাকে পরাজিত করেছে?", "reference_answer": "আলেকজান্ডার পোরাসকে পরাজিত করেছেন।", "student_answer": "পোরাস আলেকজান্ডারকে পরাজিত করেছেন।", "label": "incorrect", "human_score": 0, "subject": "ইতিহাস"},
    {"question": "কে কাকে কামড়েছে?", "reference_answer": "কুকুরটি মানুষটিকে কামড়েছে।", "student_answer": "মানুষটি কুকুরটিকে কামড়েছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে ফুল তুলেছে?", "reference_answer": "মেয়েটি ফুল তুলেছে।", "student_answer": "ফুলটি মেয়েটিকে তুলেছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "সূর্য কোথায় ওঠে?", "reference_answer": "সূর্য পূর্বদিকে ওঠে।", "student_answer": "সূর্য পূর্বদিকে ওঠে না।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "পৃথিবী কীসের চারদিকে ঘোরে?", "reference_answer": "পৃথিবী সূর্যের চারদিকে ঘোরে।", "student_answer": "পৃথিবী সূর্যের চারদিকে ঘোরে না।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "জলীয়বাষ্প থেকে কী তৈরি হয়?", "reference_answer": "জলীয়বাষ্প ঘনীভূত হয়ে মেঘ তৈরি করে।", "student_answer": "জলীয়বাষ্প ঘনীভূত হয়ে মেঘ তৈরি করে না।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "মানুষের শরীরে কয়টি হাড় থাকে?", "reference_answer": "মানুষের শরীরে ২০৬টি হাড় থাকে।", "student_answer": "মানুষের শরীরে ২০৬টি হাড় থাকে না।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "গাছ কী তৈরি করে?", "reference_answer": "গাছ অক্সিজেন তৈরি করে।", "student_answer": "গাছ অক্সিজেন তৈরি করে না।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "বরফ গলে কী হয়?", "reference_answer": "বরফ গলে জল হয়।", "student_answer": "বরফ গলে জল হয় না।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "কে ভারতের প্রথম প্রধানমন্ত্রী ছিলেন?", "reference_answer": "জওহরলাল নেহরু ভারতের প্রথম প্রধানমন্ত্রী ছিলেন।", "student_answer": "জওহরলাল নেহরু ভারতের প্রথম প্রধানমন্ত্রী ছিলেন না।", "label": "incorrect", "human_score": 0, "subject": "ইতিহাস"},
    {"question": "বাংলাদেশ কবে স্বাধীন হয়েছিল?", "reference_answer": "বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়েছিল।", "student_answer": "বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়নি।", "label": "incorrect", "human_score": 0, "subject": "ইতিহাস"},
    {"question": "সিপাহি বিদ্রোহ কবে হয়েছিল?", "reference_answer": "সিপাহি বিদ্রোহ ১৮৫৭ সালে হয়েছিল।", "student_answer": "সিপাহি বিদ্রোহ ১৮৫৭ সালে হয়নি।", "label": "incorrect", "human_score": 0, "subject": "ইতিহাস"},
    {"question": "চাঁদ কীভাবে আলো দেয়?", "reference_answer": "চাঁদ সূর্যের আলো প্রতিফলিত করে আলো দেয়।", "student_answer": "চাঁদ সূর্যের আলো প্রতিফলিত করে আলো দেয় না।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "সর্বোচ্চ পর্বতশৃঙ্গ কোনটি?", "reference_answer": "মাউন্ট এভারেস্ট সর্বোচ্চ পর্বতশৃঙ্গ।", "student_answer": "মাউন্ট এভারেস্ট সর্বোচ্চ পর্বতশৃঙ্গ নয়।", "label": "incorrect", "human_score": 0, "subject": "ভূগোল"},
    {"question": "সবচেয়ে বড় গ্রহ কোনটি?", "reference_answer": "সৌরজগতের সবচেয়ে বড় গ্রহ বৃহস্পতি।", "student_answer": "সৌরজগতের সবচেয়ে বড় গ্রহ বৃহস্পতি নয়।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "শব্দের গতি কত?", "reference_answer": "শব্দের গতি প্রতি সেকেন্ডে ৩৪৩ মিটার।", "student_answer": "শব্দের গতি প্রতি সেকেন্ডে ৩৪৩ মিটার নয়।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "রক্তের রং কী?", "reference_answer": "রক্তের রং লাল।", "student_answer": "রক্তের রং লাল নয়।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "গাছের পাতার রং কী?", "reference_answer": "গাছের পাতার রং সবুজ।", "student_answer": "গাছের পাতার রং সবুজ নয়।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "কে আমেরিকা আবিষ্কার করেছিলেন?", "reference_answer": "ক্রিস্টোফার কলম্বাস আমেরিকা আবিষ্কার করেছিলেন।", "student_answer": "ক্রিস্টোফার কলম্বাস আমেরিকা আবিষ্কার করেননি।", "label": "incorrect", "human_score": 0, "subject": "ইতিহাস"},
    {"question": "ভাষা আন্দোলন কবে হয়েছিল?", "reference_answer": "ভাষা আন্দোলন ১৯৫২ সালে হয়েছিল।", "student_answer": "ভাষা আন্দোলন ১৯৫২ সালে হয়নি।", "label": "incorrect", "human_score": 0, "subject": "ইতিহাস"},
    {"question": "কে রাবণকে বধ করেছিলেন?", "reference_answer": "রাম রাবণকে বধ করেছিলেন।", "student_answer": "রাবণের দ্বারা রাম নিহত হয়েছেন।", "label": "incorrect", "human_score": 0, "subject": "ইতিহাস"},
    {"question": "কে বাঘ শিকার করেছিল?", "reference_answer": "শিকারী বাঘ শিকার করেছিল।", "student_answer": "বাঘের দ্বারা শিকারী নিহত হয়েছিল।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "পুলিশ কাকে ধরেছে?", "reference_answer": "পুলিশ চোরকে ধরেছে।", "student_answer": "চোরের দ্বারা পুলিশ ধৃত হয়েছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "বিড়াল কাকে ধরেছে?", "reference_answer": "বিড়াল ইঁদুর ধরেছে।", "student_answer": "ইঁদুরের দ্বারা বিড়াল ধরা পড়েছে।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "কে সাপ মেরেছে?", "reference_answer": "কৃষক সাপ মেরেছে।", "student_answer": "সাপের দ্বারা কৃষক মারা গেছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "মা কাকে বকা দিয়েছে?", "reference_answer": "মা ছেলেকে বকা দিয়েছে।", "student_answer": "ছেলের দ্বারা মা বকা খেয়েছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে কাকে পরাজিত করেছে?", "reference_answer": "আলেকজান্ডার পোরাসকে পরাজিত করেছেন।", "student_answer": "পোরাসের দ্বারা আলেকজান্ডার পরাজিত হয়েছেন।", "label": "incorrect", "human_score": 0, "subject": "ইতিহাস"},
    {"question": "বোলার কাকে আউট করেছে?", "reference_answer": "বোলার ব্যাটসম্যানকে আউট করেছে।", "student_answer": "ব্যাটসম্যানের দ্বারা বোলার আউট হয়েছে।", "label": "incorrect", "human_score": 0, "subject": "খেলা"},
    {"question": "কে কাকে কামড়েছে?", "reference_answer": "কুকুরটি মানুষটিকে কামড়েছে।", "student_answer": "মানুষটির দ্বারা কুকুরটি কামড় খেয়েছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে ইঁদুর খেয়েছে?", "reference_answer": "সাপটি ইঁদুর খেয়েছে।", "student_answer": "ইঁদুরের দ্বারা সাপটি খাওয়া হয়েছে।", "label": "incorrect", "human_score": 0, "subject": "বিজ্ঞান"},
    {"question": "কে চিঠি লিখেছে?", "reference_answer": "রহিম চিঠি লিখেছে।", "student_answer": "চিঠির দ্বারা রহিম লেখা হয়েছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে বই পড়ছে?", "reference_answer": "ছাত্রটি বই পড়ছে।", "student_answer": "বইটির দ্বারা ছাত্রটি পঠিত হচ্ছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে ছবি এঁকেছে?", "reference_answer": "শিল্পী ছবিটি এঁকেছে।", "student_answer": "ছবিটির দ্বারা শিল্পী আঁকা হয়েছে।", "label": "incorrect", "human_score": 0, "subject": "শিল্প"},
    {"question": "কে গাড়ি চালাচ্ছে?", "reference_answer": "চালক গাড়িটি চালাচ্ছে।", "student_answer": "গাড়িটির দ্বারা চালক চালিত হচ্ছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে মাছ ধরেছে?", "reference_answer": "জেলে মাছ ধরেছে।", "student_answer": "মাছের দ্বারা জেলে ধরা পড়েছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
    {"question": "কে চা বানিয়েছে?", "reference_answer": "মা চা বানিয়েছে।", "student_answer": "চায়ের দ্বারা মা বানানো হয়েছে।", "label": "incorrect", "human_score": 0, "subject": "সাধারণ জ্ঞান"},
]

# Mapping from raw CSV column names to internal names
_COL_RENAME = {
    "teacher_ans": "reference_answer",
    "student_ans": "student_answer",
    "reference_ans": "reference_answer",   # alternate naming
}

def load_and_augment_dataset(filepath: str, output_filepath: str) -> pd.DataFrame:
    """Loads the original dataset and appends 50 adversarial examples."""
    print(f"Loading dataset from {filepath}...")
    with open(filepath, encoding="utf-8") as f:
        first_line = f.readline()
    sep = "\t" if first_line.count("\t") >= first_line.count(",") else ","
    df = pd.read_csv(filepath, sep=sep)

    df.rename(columns=_COL_RENAME, inplace=True)
    if "subject" not in df.columns:
        df["subject"] = "সাধারণ"

    df_adv = pd.DataFrame(ADVERSARIAL_SAMPLES)
    start_id = int(df["id"].max()) + 1
    df_adv["id"] = range(start_id, start_id + len(df_adv))
    df_adv = df_adv[["id", "question", "reference_answer", "student_answer", "label", "human_score", "subject"]]

    df_augmented = pd.concat([df, df_adv], ignore_index=True)
    df_augmented.to_csv(output_filepath, index=False)
    print(f"Added {len(df_adv)} adversarial examples. Total samples: {len(df_augmented)}")
    print(f"Augmented dataset saved to {output_filepath}")
    return df_augmented

def get_augmented_dataset() -> pd.DataFrame:
    """Resolve paths, augment if necessary, and return the full DataFrame."""
    original_csv = resolve_dataset_path()
    if not os.path.exists(AUGMENTED_CSV):
        df = load_and_augment_dataset(original_csv, AUGMENTED_CSV)
    else:
        df = pd.read_csv(AUGMENTED_CSV)
        print(f"Augmented dataset already exists. Loaded {len(df)} samples.")

    print(f"\nDataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    return df
