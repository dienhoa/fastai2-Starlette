# uvicorn imports
import aiohttp
import asyncio
import uvicorn

# starlette imports
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from io import BytesIO
# fastai
from .infer import *

export_file_name = 'resnet-lung.pkl'
export_file_url = 'https://drive.google.com/uc?export=download&id=1H9ueZQzL57EwauPfJNhDCTg--MgJBpUh'

path = Path(__file__).parent
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

if not (path / export_file_name).exists():
    gdown.download(export_file_url, str(path / export_file_name), quiet=False)

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
  img_np = np.array(Image.open(BytesIO(img_bytes)))
  pred = learn.predict(img_np)
  return JSONResponse({
      'result': str(pred[0])
  })

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
