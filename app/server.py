# uvicorn imports
import aiohttp
import asyncio
import uvicorn
import asyncio
import aiofiles
from statistics import mode


# starlette imports
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from io import BytesIO

# fastai
from fastai.vision.all import *
import gdown
import torchaudio
import soundfile as sf
import librosa
import time

# Any custom imports should be done here, for example:
# from lib.utilities import *
# lib.utilities contains custom functions used during training that pickle is expecting


random.seed(23)
# export_file_url = YOUR_GDRIVE_LINK_HERE
export_file_name = 'resnet-lung.pkl'
export_file_url = 'https://drive.google.com/uc?export=download&id=1JsVaEKIBowY4Oqh6K3AQZr8w257K6XZ_'

path = Path(__file__).parent
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

if not (path / export_file_name).exists():
    gdown.download(export_file_url, str(path / export_file_name), quiet=False)

def get_y(): pass

# configuration for audio processing
n_fft=1024
hop_length=256
target_rate=44100
num_samples=int(target_rate)

## Helper method to tranform audio array to Spectrogram
au2spec = torchaudio.transforms.MelSpectrogram(sample_rate=target_rate,n_fft=n_fft, hop_length=hop_length, n_mels=256)
ampli2db = torchaudio.transforms.AmplitudeToDB()

def get_x(path, target_rate=target_rate, num_samples=num_samples*2):
    x, rate = torchaudio.load_wav(path)
    if rate != target_rate: 
        x = torchaudio.transforms.Resample(orig_freq=rate, new_freq=target_rate, resampling_method='sinc_interpolation')(x)
    x = x[0] / 32768
    x = x.numpy()
    sample_total = x.shape[0]
    # randstart = random.randint(target_rate, sample_total-target_rate*3)
    randstart = int(target_rate*1.5)
    x = x[randstart:num_samples+randstart]
    x = librosa.util.fix_length(x, num_samples)
    torch_x = torch.tensor(x)
    spec = au2spec(torch_x)
    spec_db = ampli2db(spec)
    spec_db = spec_db.data.squeeze(0).numpy()
    spec_db = spec_db - spec_db.min()
    spec_db = spec_db/spec_db.max()*255
    return spec_db

learn = load_learner(path / export_file_name)
classes = learn.dls.vocab

@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())

    name = f'./audio-files/{time.time()}.wav'
    async with aiofiles.open(name, mode='bx') as f:
        await f.write(img_bytes)

    img_np = get_x(name)
    print(name)
    # img_np = np.array(Image.open(BytesIO(img_bytes)))
    # pred = mode([learn.predict(img_np)[0] for i in range(5)])
    pred = learn.predict(img_np)
    print(pred)
    return JSONResponse({
        'result': str(pred[0])
    })

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
