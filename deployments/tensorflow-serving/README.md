## Private Detector on Tensorflow Serving

Tensorflow serving image could be pulled via:

```shell
docker pull <image_name>
```

Run: 

```shell
docker run -p 8501:8501 -e MODEL_NAME=private_detector -t <image_name>
```

Ports exposed:
* REST API: **8501**
* GRPC: **8502**

Calling a model:

```python
import base64
import json
import requests

image_string = open("./test.jpeg", "rb").read()
endpoint = "http://localhost:8501/v1/models/private_detector:predict"
jpeg_bytes = base64.b64encode(image_string).decode('utf-8')
predict_request = {"instances" : [{"b64": jpeg_bytes}]}

response = requests.post(endpoint, json.dumps(predict_request))
print(response.json())

>>> {'predictions': [[0.0535223149, 0.946477652]]}
```