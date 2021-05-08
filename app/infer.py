from fastai.vision.all import *
import librosa
import torchaudio
import gdown

export_file_name = 'resnet-lung.pkl'
export_file_url = 'https://drive.google.com/uc?export=download&id=1H9ueZQzL57EwauPfJNhDCTg--MgJBpUh'

path = Path(__file__).parent

# configuration for audio processing
n_fft=1024
hop_length=256
target_rate=44100
num_samples=int(target_rate)

## Helper method to tranform audio array to Spectrogram
au2spec = torchaudio.transforms.MelSpectrogram(sample_rate=target_rate,n_fft=n_fft, hop_length=hop_length, n_mels=256)
ampli2db = torchaudio.transforms.AmplitudeToDB()

## Method for labelling sample (Healthy/Unhealthy)
def get_y(path): 
    desease = p_diag[p_diag[0] == int(path.stem[:3])][1].values[0]
    if desease == "Healthy":
        return "Healthy"
    else : 
        return "Unhealthy"

def get_x(path, target_rate=target_rate, num_samples=num_samples*2):
    x, rate = torchaudio.load_wav(path)
    if rate != target_rate: 
        x = torchaudio.transforms.Resample(orig_freq=rate, new_freq=target_rate, resampling_method='sinc_interpolation')(x)
    x = x[0] / 32768
    x = x.numpy()
    sample_total = x.shape[0]
    randstart = random.randint(target_rate, sample_total-target_rate*3)
    x = x[randstart:num_samples+randstart]
    x = librosa.util.fix_length(x, num_samples)
    torch_x = torch.tensor(x)
    spec = au2spec(torch_x)
    spec_db = ampli2db(spec)
    spec_db = spec_db.data.squeeze(0).numpy()
    spec_db = spec_db - spec_db.min()
    spec_db = spec_db/spec_db.max()*255
    return spec_db

if not (path / export_file_name).exists():
    gdown.download(export_file_url, str(path / export_file_name), quiet=False)

learn = load_learner(path / export_file_name)
classes = learn.dls.vocab