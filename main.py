from beat_this.inference import File2Beats
file2beats = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)

audio_path = "resources/sample3.mp3"
beats, downbeats = file2beats(audio_path)

print(beats)
print(downbeats)

from beat_this.utils import save_beat_tsv
outpath = "outputs/sample3.beats"
save_beat_tsv(beats, downbeats, outpath)