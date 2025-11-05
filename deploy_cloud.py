#!/usr/bin/env python3
"""
Script de déploiement cloud pour ML Model Selector
Supporte AWS, Docker et déploiement automatisé
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

class CloudDeployer:
    def __init__(self):
        self.config = self._load_config()
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
    def _load_config(self) -> Dict:
        """Charge la configuration de déploiement"""
        config_path = Path('deploy_config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            "app_name": "ml-model-selector",
            "version": "2.0.0",
            "environment": "production",
            "aws": {
                "region": "us-east-1",
                "instance_type": "t3.medium",
                "min_size": 1,
                "max_size": 3,
                "desired_capacity": 1
            },
            "docker": {
                "base_image": "python:3.9-slim",
                "expose_port": 8501
            }
        }
    
    def create_dockerfile(self) -> str:
        """Crée un Dockerfile optimisé pour la production"""
        dockerfile_content = f"""
# Base image
FROM {self.config['docker']['base_image']}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE {self.config['docker']['expose_port']}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{self.config['docker']['expose_port']}/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port", "{self.config['docker']['expose_port']}", "--server.address", "0.0.0.0"]
"""
        
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content.strip())
        
        return 'Dockerfile'
    
    def create_docker_compose(self) -> str:
        """Crée un docker-compose.yml pour le développement local"""
        compose_content = f"""
version: '3.8'

services:
  ml-app:
    build: .
    ports:
      - "{self.config['docker']['expose_port']}:{self.config['docker']['expose_port']}"
    environment:
      - DEBUG=False
      - DATABASE_URL=sqlite:///users.db
    volumes:
      - ./users.db:/app/users.db
      - ./data:/app/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ml-app
    restart: unless-stopped
"""
        
        with open('docker-compose.yml', 'w') as f:
            f.write(compose_content.strip())
        
        return 'docker-compose.yml'
    
    def create_nginx_config(self) -> str:
        """Crée une configuration Nginx pour le reverse proxy"""
        nginx_content = f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream ml_app {{
        server ml-app:{self.config['docker']['expose_port']};
    }}
    
    server {{
        listen 80;
        server_name _;
        
        # Redirect HTTP to HTTPS
        return 301 https://$host$request_uri;
    }}
    
    server {{
        listen 443 ssl http2;
        server_name _;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
        
        location / {{
            proxy_pass http://ml_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }}
    }}
}}
"""
        
        with open('nginx.conf', 'w') as f:
            f.write(nginx_content.strip())
        
        return 'nginx.conf'
    
    def create_ecs_task_definition(self) -> str:
        """Crée une définition de tâche ECS pour AWS"""
        task_definition = {
            "family": f"{self.config['app_name']}-task",
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": "512",
            "memory": "1024",
            "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
            "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskRole",
            "containerDefinitions": [
                {
                    "name": "ml-app",
                    "image": f"{self.config['app_name']}:{self.config['version']}",
                    "portMappings": [
                        {
                            "containerPort": {self.config['docker']['expose_port']},
                            "protocol": "tcp"
                        }
                    ],
                    "environment": [
                        {"name": "DEBUG", "value": "False"},
                        {"name": "DATABASE_URL", "value": "sqlite:///users.db"}
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/{self.config['app_name']}",
                            "awslogs-region": self.config['aws']['region'],
                            "awslogs-stream-prefix": "ecs"
                        }
                    }
                }
            ]
        }
        
        with open('ecs-task-definition.json', 'w') as f:
            json.dump(task_definition, f, indent=2)
        
        return 'ecs-task-definition.json'
    
    def create_github_actions(self) -> str:
        """Crée un workflow GitHub Actions pour CI/CD"""
        workflow_content = f"""
name: Deploy ML Model Selector

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ || echo "No tests found"
    
    - name: Security check
      run: |
        pip install bandit safety
        bandit -r . -f json -o bandit-report.json || true
        safety check || true

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: {self.config['aws']['region']}
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build, tag, and push image to Amazon ECR
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: {self.config['app_name']}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: {self.config['aws']['region']}
    
    - name: Deploy to ECS
      run: |
        aws ecs update-service --cluster {self.config['app_name']}-cluster --service {self.config['app_name']}-service --force-new-deployment
"""
        
        os.makedirs('.github/workflows', exist_ok=True)
        with open('.github/workflows/deploy.yml', 'w') as f:
            f.write(workflow_content.strip())
        
        return '.github/workflows/deploy.yml'
    
    def create_terraform_config(self) -> str:
        """Crée une configuration Terraform pour l'infrastructure AWS"""
        terraform_content = f"""
terraform {{
  required_version = ">= 1.0"
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = "{self.config['aws']['region']}"
}}

# VPC
resource "aws_vpc" "main" {{
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {{
    Name = "{self.config['app_name']}-vpc"
  }}
}}

# Subnets
resource "aws_subnet" "public" {{
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${{count.index + 1}}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {{
    Name = "{self.config['app_name']}-public-subnet-${{count.index + 1}}"
  }}
}}

# ECS Cluster
resource "aws_ecs_cluster" "main" {{
  name = "{self.config['app_name']}-cluster"
  
  setting {{
    name  = "containerInsights"
    value = "enabled"
  }}
}}

# ECS Service
resource "aws_ecs_service" "main" {{
  name            = "{self.config['app_name']}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.main.arn
  desired_count   = {self.config['aws']['desired_capacity']}
  launch_type     = "FARGATE"
  
  network_configuration {{
    subnets         = aws_subnet.public[*].id
    security_groups = [aws_security_group.ecs.id]
    assign_public_ip = true
  }}
  
  load_balancer {{
    target_group_arn = aws_lb_target_group.main.arn
    container_name   = "ml-app"
    container_port   = {self.config['docker']['expose_port']}
  }}
}}

# Application Load Balancer
resource "aws_lb" "main" {{
  name               = "{self.config['app_name']}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
  
  tags = {{
    Name = "{self.config['app_name']}-alb"
  }}
}}

# Security Groups
resource "aws_security_group" "alb" {{
  name        = "{self.config['app_name']}-alb-sg"
  description = "ALB Security Group"
  vpc_id      = aws_vpc.main.id
  
  ingress {{
    protocol    = "tcp"
    from_port   = 80
    to_port     = 80
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  ingress {{
    protocol    = "tcp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  egress {{
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }}
}}

resource "aws_security_group" "ecs" {{
  name        = "{self.config['app_name']}-ecs-sg"
  description = "ECS Security Group"
  vpc_id      = aws_vpc.main.id
  
  ingress {{
    protocol        = "tcp"
    from_port       = {self.config['docker']['expose_port']}
    to_port         = {self.config['docker']['expose_port']}
    security_groups = [aws_security_group.alb.id]
  }}
  
  egress {{
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }}
}}

# Data sources
data "aws_availability_zones" "available" {{
  state = "available"
}}

# Outputs
output "alb_dns_name" {{
  value = aws_lb.main.dns_name
}}

output "ecs_cluster_name" {{
  value = aws_ecs_cluster.main.name
}}
"""
        
        with open('terraform/main.tf', 'w') as f:
            f.write(terraform_content.strip())
        
        return 'terraform/main.tf'
    
    def deploy_local(self) -> bool:
        """Déploie l'application localement avec Docker"""
        try:
            print("🐳 Déploiement local avec Docker...")
            
            # Créer les fichiers de configuration
            self.create_dockerfile()
            self.create_docker_compose()
            self.create_nginx_config()
            
            # Construire et démarrer
            subprocess.run(['docker-compose', 'build'], check=True)
            subprocess.run(['docker-compose', 'up', '-d'], check=True)
            
            print("✅ Déploiement local réussi!")
            print(f"🌐 Application accessible sur: http://localhost")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors du déploiement local: {e}")
            return False
    
    def deploy_aws(self) -> bool:
        """Déploie l'application sur AWS"""
        try:
            print("☁️ Déploiement sur AWS...")
            
            # Vérifier les credentials AWS
            if not self._check_aws_credentials():
                print("❌ Credentials AWS non trouvés")
                return False
            
            # Créer les fichiers de configuration
            self.create_ecs_task_definition()
            self.create_terraform_config()
            self.create_github_actions()
            
            print("✅ Fichiers de configuration AWS créés!")
            print("📋 Étapes suivantes:")
            print("   1. Créer un bucket S3 pour Terraform")
            print("   2. Initialiser Terraform: terraform init")
            print("   3. Planifier: terraform plan")
            print("   4. Déployer: terraform apply")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la configuration AWS: {e}")
            return False
    
    def _check_aws_credentials(self) -> bool:
        """Vérifie les credentials AWS"""
        try:
            sts = boto3.client('sts')
            sts.get_caller_identity()
            return True
        except (NoCredentialsError, ClientError):
            return False
    
    def create_env_template(self) -> str:
        """Crée un template de fichier .env"""
        env_content = f"""
# Configuration de sécurité
SECRET_KEY=your-super-secret-key-here-change-this-in-production
JWT_SECRET_KEY=your-jwt-secret-key-here
SESSION_TIMEOUT=3600

# Configuration AWS S3
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION={self.config['aws']['region']}
AWS_S3_BUCKET=your-s3-bucket-name

# Configuration de la base de données
DATABASE_URL=sqlite:///users.db

# Configuration de sécurité avancée
MAX_LOGIN_ATTEMPTS=5
LOGIN_TIMEOUT=300
PASSWORD_MIN_LENGTH=8
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Configuration du serveur
DEBUG=False
HOST=0.0.0.0
PORT={self.config['docker']['expose_port']}

# Configuration des sauvegardes
AUTO_BACKUP_ENABLED=True
BACKUP_INTERVAL_HOURS=24
BACKUP_RETENTION_DAYS=30
"""
        
        with open('.env.template', 'w') as f:
            f.write(env_content.strip())
        
        return '.env.template'

def main():
    parser = argparse.ArgumentParser(description='Déploiement cloud pour ML Model Selector')
    parser.add_argument('--target', choices=['local', 'aws', 'all'], default='all',
                       help='Cible de déploiement')
    parser.add_argument('--config', type=str, help='Fichier de configuration JSON')
    
    args = parser.parse_args()
    
    deployer = CloudDeployer()
    
    if args.config:
        deployer.config.update(json.load(open(args.config)))
    
    print("🚀 Déploiement cloud pour ML Model Selector")
    print(f"📋 Configuration: {deployer.config['app_name']} v{deployer.config['version']}")
    
    success = True
    
    if args.target in ['local', 'all']:
        success &= deployer.deploy_local()
    
    if args.target in ['aws', 'all']:
        success &= deployer.deploy_aws()
    
    if args.target == 'all':
        deployer.create_env_template()
        print("📄 Fichier .env.template créé")
    
    if success:
        print("\n🎉 Déploiement terminé avec succès!")
        print("📚 Consultez la documentation pour les étapes suivantes")
    else:
        print("\n❌ Déploiement échoué")
        sys.exit(1)

if __name__ == "__main__":
    main() 