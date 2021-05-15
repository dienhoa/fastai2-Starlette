# Starter for deploying Lung_Sound

This repo can be used as a starting point to deploy [fast.ai](https://github.com/fastai/fastai) models with Starlette.

See the Deployment notebook for hints

To run locally:
`python3 app/server.py`

# Docker related

- Build Image
```
    docker build -t lung_sound .
```
- Run Image

Locally:

```
    docker run -p 5000:5000 lung_sound
```
Or you can pull from docker-hub and run the prebuilt image