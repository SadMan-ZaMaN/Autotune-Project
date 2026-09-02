from src.autotune.config import AutoTuneConfig
from src.autotune.io_utils import load_audio, save_audio
from src.autotune.framing import frame_signal, overlap_add

def test_ola_reconstruction():

    config = AutoTuneConfig()

    audio, sr = load_audio("data/raw/test_voice.wav", target_sr=config.sample_rate)
    frames, pad_len = frame_signal(audio, config)
    print(f"Loaded {len(audio)} samples -> {frames.shape[0]} frames of size {frames.shape[1]}")

    reconstructed = overlap_add(frames, config, pad_len)
    print(f"reconstructed length:  {len(reconstructed)} (original: {len(audio)})")

    save_audio("data/processed/reconstructed_test.wav", reconstructed, sr)
    print("Saved data/processed/reconstructed_test.wav - listen and compare to the original")

if __name__ == "__main__":
    test_ola_reconstruction()