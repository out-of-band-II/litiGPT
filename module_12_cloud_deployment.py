"""
Module 12: Cloud Deployment Scripts
Scripts for deploying to various cloud platforms
"""

# =============================================================================
# deploy_runpod.sh - Deploy training to RunPod
# =============================================================================
"""
#!/bin/bash
# deploy_runpod.sh - Deploy training container to RunPod

set -e

echo "Deploying Reddit Chatbot Training to RunPod..."

# Build and push to registry
echo "Building training container..."
docker build -f Dockerfile.training -t reddit-chatbot-training:latest .

# Tag for registry (use your registry)
docker tag reddit-chatbot-training:latest YOUR_REGISTRY/reddit-chatbot-training:latest

# Push to registry
echo "Pushing to container registry..."
docker push YOUR_REGISTRY/reddit-chatbot-training:latest

echo "Container ready for RunPod deployment!"
echo ""
echo "RunPod Setup Instructions:"
echo "1. Go to https://www.runpod.io/console/pods"
echo "2. Select GPU (RTX 4090 recommended)"
echo "3. Choose 'Custom Container'"
echo "4. Enter: YOUR_REGISTRY/reddit-chatbot-training:latest"
echo "5. Mount volumes:"
echo "   - /workspace/data -> Your data bucket"
echo "   - /workspace/models -> Your models bucket"
echo "6. Set environment variables from .env"
echo "7. Start pod and SSH in"
echo ""
echo "Run training with:"
echo "  python module_3_training.py"
"""

# =============================================================================
# deploy_aws.py - Deploy bot to AWS ECS
# =============================================================================

import boto3
import json
from typing import Dict

class AWSDeployer:
    """Deploy Reddit bot to AWS ECS"""
    
    def __init__(self, region: str = "us-east-1"):
        self.ecs = boto3.client('ecs', region_name=region)
        self.ecr = boto3.client('ecr', region_name=region)
        self.logs = boto3.client('logs', region_name=region)
        self.region = region
    
    def create_ecr_repository(self, repo_name: str = "reddit-chatbot-bot"):
        """Create ECR repository for Docker images"""
        
        try:
            response = self.ecr.create_repository(
                repositoryName=repo_name,
                imageScanningConfiguration={'scanOnPush': True},
                encryptionConfiguration={'encryptionType': 'AES256'}
            )
            print(f"Created ECR repository: {response['repository']['repositoryUri']}")
            return response['repository']['repositoryUri']
        
        except self.ecr.exceptions.RepositoryAlreadyExistsException:
            response = self.ecr.describe_repositories(repositoryNames=[repo_name])
            return response['repositories'][0]['repositoryUri']
    
    def create_task_definition(self, 
                               image_uri: str,
                               task_name: str = "reddit-bot-task",
                               cpu: str = "256",
                               memory: str = "512") -> str:
        """Create ECS task definition"""
        
        task_def = {
            "family": task_name,
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": cpu,
            "memory": memory,
            "containerDefinitions": [
                {
                    "name": "reddit-bot",
                    "image": image_uri,
                    "essential": True,
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/{task_name}",
                            "awslogs-region": self.region,
                            "awslogs-stream-prefix": "bot"
                        }
                    },
                    "environment": [
                        {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow:5000"}
                    ],
                    "secrets": [
                        {
                            "name": "REDDIT_CLIENT_ID",
                            "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:reddit-bot"
                        }
                    ]
                }
            ]
        }
        
        # Create CloudWatch log group
        try:
            self.logs.create_log_group(logGroupName=f"/ecs/{task_name}")
        except self.logs.exceptions.ResourceAlreadyExistsException:
            pass
        
        # Register task definition
        response = self.ecs.register_task_definition(**task_def)
        
        task_def_arn = response['taskDefinition']['taskDefinitionArn']
        print(f"Created task definition: {task_def_arn}")
        return task_def_arn
    
    def create_service(self,
                      cluster_name: str,
                      service_name: str,
                      task_definition: str,
                      subnet_ids: list,
                      security_group_ids: list,
                      desired_count: int = 1):
        """Create ECS service"""
        
        response = self.ecs.create_service(
            cluster=cluster_name,
            serviceName=service_name,
            taskDefinition=task_definition,
            desiredCount=desired_count,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': subnet_ids,
                    'securityGroups': security_group_ids,
                    'assignPublicIp': 'ENABLED'
                }
            }
        )
        
        print(f"Created service: {response['service']['serviceName']}")
        return response['service']

# =============================================================================
# deploy_gcp.sh - Deploy to Google Cloud Run
# =============================================================================
"""
#!/bin/bash
# deploy_gcp.sh - Deploy bot to Google Cloud Run

set -e

PROJECT_ID="your-gcp-project"
REGION="us-central1"
SERVICE_NAME="reddit-bot"

echo "Deploying to Google Cloud Run..."

# Build and push to GCR
echo "Building container..."
docker build -f Dockerfile.bot -t gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest .

echo "Pushing to Google Container Registry..."
docker push gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --no-allow-unauthenticated \
  --set-env-vars MLFLOW_TRACKING_URI=http://mlflow:5000 \
  --set-secrets REDDIT_CLIENT_ID=reddit-bot-secrets:latest \
  --set-secrets REDDIT_CLIENT_SECRET=reddit-bot-secrets:latest

echo "Deployment complete!"
gcloud run services describe ${SERVICE_NAME} --region ${REGION}
"""

# =============================================================================
# deploy_lambda.py - Deploy to AWS Lambda (for lightweight bots)
# =============================================================================

class LambdaDeployer:
    """Deploy bot as AWS Lambda function (for infrequent responses)"""
    
    def __init__(self, region: str = "us-east-1"):
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.iam = boto3.client('iam', region_name=region)
    
    def create_lambda_function(self,
                              function_name: str,
                              image_uri: str,
                              timeout: int = 300,
                              memory: int = 2048):
        """Create Lambda function from container image"""
        
        # Create IAM role
        role_arn = self._create_lambda_role(f"{function_name}-role")
        
        response = self.lambda_client.create_function(
            FunctionName=function_name,
            PackageType='Image',
            Code={'ImageUri': image_uri},
            Role=role_arn,
            Timeout=timeout,
            MemorySize=memory,
            Environment={
                'Variables': {
                    'MLFLOW_TRACKING_URI': 'http://mlflow:5000'
                }
            }
        )
        
        print(f"Created Lambda function: {response['FunctionArn']}")
        return response['FunctionArn']
    
    def _create_lambda_role(self, role_name: str) -> str:
        """Create IAM role for Lambda"""
        
        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }
        
        try:
            response = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy)
            )
            
            # Attach basic execution policy
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            
            return response['Role']['Arn']
        
        except self.iam.exceptions.EntityAlreadyExistsException:
            response = self.iam.get_role(RoleName=role_name)
            return response['Role']['Arn']

# =============================================================================
# Kubernetes deployment (k8s.yaml)
# =============================================================================
"""
# k8s-bot-deployment.yaml
# Deploy Reddit bot to Kubernetes cluster

apiVersion: v1
kind: Namespace
metadata:
  name: reddit-bot

---
apiVersion: v1
kind: Secret
metadata:
  name: reddit-credentials
  namespace: reddit-bot
type: Opaque
stringData:
  client-id: your_client_id
  client-secret: your_client_secret
  username: your_username
  password: your_password

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: bot-config
  namespace: reddit-bot
data:
  config.yaml: |
    bot:
      subreddit: "test"
      reply_probability: 0.2
      cooldown_seconds: 120

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reddit-bot
  namespace: reddit-bot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: reddit-bot
  template:
    metadata:
      labels:
        app: reddit-bot
    spec:
      containers:
      - name: bot
        image: your-registry/reddit-chatbot-bot:latest
        env:
        - name: REDDIT_CLIENT_ID
          valueFrom:
            secretKeyRef:
              name: reddit-credentials
              key: client-id
        - name: REDDIT_CLIENT_SECRET
          valueFrom:
            secretKeyRef:
              name: reddit-credentials
              key: client-secret
        volumeMounts:
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
        - name: models
          mountPath: /app/models
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: config
        configMap:
          name: bot-config
      - name: models
        persistentVolumeClaim:
          claimName: model-storage

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
  namespace: reddit-bot
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi

# Deploy with:
# kubectl apply -f k8s-bot-deployment.yaml
#
# View logs:
# kubectl logs -f -n reddit-bot deployment/reddit-bot
#
# Scale:
# kubectl scale -n reddit-bot deployment/reddit-bot --replicas=3
"""

if __name__ == "__main__":
    print("Cloud deployment scripts loaded.")
    print("\nAvailable deployment options:")
    print("1. RunPod (GPU training): ./deploy_runpod.sh")
    print("2. AWS ECS (bot): python deploy_aws.py")
    print("3. Google Cloud Run (bot): ./deploy_gcp.sh")
    print("4. AWS Lambda (lightweight): python deploy_lambda.py")
    print("5. Kubernetes: kubectl apply -f k8s-bot-deployment.yaml")
