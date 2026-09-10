import soundfile as sf
for f in ["data/raw/test_voice.wav", "data/processed/naive_shifted.wav"]:
    info = sf.info(f)
    print(f, "-> duration:", info.duration, "sec, subtype:", info.subtype)