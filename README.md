# MLOps PyTorch CIFAR-10 Pipeline

This project implements an end-to-end MLOps pipeline for training and serving a
CIFAR-10 image classification model using PyTorch, Docker, and Kubernetes.

The model uses a ResNet-18 architecture and provides a FastAPI-based inference
service with health checking and horizontal pod autoscaling.

## Architecture

```text
                         ┌──────────────────────┐
                         │     CIFAR-10 Data    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Kubernetes Training │
                         │        Job           │
                         │   mlops-train:v1     │
                         └──────────┬───────────┘
                                    │
                                    │ model.pt
                                    ▼
                         ┌──────────────────────┐
                         │ Checkpoint PVC       │
                         │ mlops-checkpoints-pvc│
                         └──────────┬───────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │        Kubernetes Serving Layer         │
              │                                         │
              │  ┌──────────────┐  ┌──────────────┐    │
              │  │ Serving Pod  │  │ Serving Pod  │    │
              │  │  ResNet-18   │  │  ResNet-18   │    │
              │  └──────┬───────┘  └──────┬───────┘    │
              │         └──────────┬───────┘            │
              │                    │                    │
              │             ClusterIP Service           │
              │                    │                    │
              │             Port 80 → 8080             │
              └────────────────────┼────────────────────┘
                                   │
                                   ▼
                            POST /predict
                                   │
                                   ▼
                              Image → Class

                         ┌──────────────────────┐
                         │        HPA           │
                         │ CPU target: 70%      │
                         │ Min: 2 pods           │
                         │ Max: 4 pods           │
                         └──────────────────────┘
```

## Repository Structure
.
├── configs/
│   ├── sanity_config.yaml
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── data-pvc.yaml
│   ├── pvc.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── serve.py
└── tests/
    └── test_model.py


## Model

The pipeline trains a ResNet-18 image classifier on CIFAR-10.

The training pipeline supports:

Configurable hyperparameters
Training and validation metrics
JSON-line metric logging
Configurable checkpoint output
Early stopping

The trained model checkpoint is stored on a Kubernetes PersistentVolumeClaim
so that the serving pods can load the trained model.

## Docker
Build the training image:
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
Build the serving image:
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

The serving image runs the FastAPI application on port 8080 as a non-root
user and includes a container health check.

## Kubernetes Deployment

Create the namespace and supporting resources:

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/data-pvc.yaml
kubectl apply -f k8s/pvc.yaml

Run the training Job:

kubectl apply -f k8s/training-job.yaml

Check the training Job:

kubectl get jobs -n mlops
kubectl get pods -n mlops

After training completes, deploy the serving layer:

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

Verify the deployment:

kubectl get pods -n mlops
kubectl get deployment -n mlops
kubectl get service -n mlops
kubectl get hpa -n mlops


## Model Serving API

The FastAPI service exposes:

Health check
GET /health

Example response:

{
  "status": "ok",
  "model": "cifar10-resnet18"
}

Prediction
POST /predict

The endpoint accepts an image using multipart form data and returns the
predicted CIFAR-10 class, class index, and class probabilities.

Example:

curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"


## Port Forwarding

For local Kubernetes testing:

kubectl port-forward svc/cifar10-serving 8080:80 -n mlops

Then test:

curl http://localhost:8080/health

and:

curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"


## Horizontal Pod Autoscaling

The serving deployment uses a Kubernetes HorizontalPodAutoscaler.

Configuration:

Minimum replicas: 2
Maximum replicas: 4
CPU target:       70%

The HPA uses Kubernetes resource metrics to increase or decrease the number of
serving replicas based on CPU utilization.

Metrics Server is required for CPU-based HPA metrics.

## Validation

The Kubernetes serving deployment was validated using:

kubectl get pods -n mlops
kubectl get deployment -n mlops
kubectl get service -n mlops
kubectl get hpa -n mlops

The serving pods reached Running and Ready state.

The health endpoint returned:

{
  "status": "ok",
  "model": "cifar10-resnet18"
}

The prediction endpoint successfully returned a CIFAR-10 prediction with class
probabilities.

CPU metrics were also successfully obtained through Metrics Server and the HPA
reported the current CPU utilization.


## Testing

A model unit test verifies that the ResNet-18 model produces the expected
10-class output for CIFAR-10 input images.

Run the test with:

```bash
python -m pytest tests/test_model.py
```

## Technologies
Python
PyTorch
Torchvision
FastAPI
Docker
Kubernetes
PersistentVolumeClaims
Kubernetes Jobs
Horizontal Pod Autoscaling
Metrics Server