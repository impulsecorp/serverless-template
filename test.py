# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = {'image': 'https://i1.sndcdn.com/artworks-000482938665-czesz7-t500x500.jpg'}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())