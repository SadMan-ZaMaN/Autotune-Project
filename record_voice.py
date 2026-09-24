import time
import sounddevice as sd
from src.autotune.io_utils import save_audio

def record_user_voice(filename="data/raw/my_voice.wav", duration=5, sample_rate=44100):
    print(f"🎤 Get ready! Recording will start in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print(f"🔴 RECORDING NOW for {duration} seconds... Sing or speak a sustained pitch!")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording finishes
    print("✅ Recording complete!")

    # Flatten stereo/mono array to 1D
    audio_signal = recording.flatten()

    # Save audio file using your project's io_utils
    save_audio(filename, audio_signal, sample_rate)
    print(f"💾 Audio saved successfully to: {filename}")

if __name__ == "__main__":
    record_user_voice()
