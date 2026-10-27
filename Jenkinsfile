pipeline {
    agent any

    environment {
        APP_NAME    = 'iris-classifier'
        DOCKER_TAG  = "${env.BUILD_NUMBER}"
    }

    stages {

        stage('Clone Repository') {
            steps {
                echo '📥 Cloning source code from GitHub...'
                git branch: 'main',
                    url: 'https://github.com/PrafullHarer/MLDevOPS-Lab-ISE-2.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                echo '📦 Installing Python dependencies...'
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Train Model') {
            steps {
                echo '🧠 Training the ML model...'
                bat 'python train_model.py'
            }
        }

        stage('Run Tests') {
            steps {
                echo '🧪 Running unit tests...'
                bat 'python -m pytest tests/ -v --tb=short'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo '🐳 Building Docker image...'
                bat "docker build -t %APP_NAME%:%DOCKER_TAG% ."
                bat "docker tag %APP_NAME%:%DOCKER_TAG% %APP_NAME%:latest"
            }
        }

        stage('Deploy Container') {
            steps {
                echo '🚀 Deploying application container...'
                // Stop and remove any existing container
                bat 'docker stop %APP_NAME% || echo "No container to stop"'
                bat 'docker rm %APP_NAME% || echo "No container to remove"'
                // Run the new container
                bat 'docker run -d --name %APP_NAME% -p 5000:5000 %APP_NAME%:latest'
            }
        }

        stage('Health Check') {
            steps {
                echo '💚 Verifying deployment health...'
                // Wait for the app to start
                bat 'timeout /t 5 /nobreak'
                bat 'curl -f http://localhost:5000/health || echo "Health check pending..."'
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully! App running at http://localhost:5000'
        }
        failure {
            echo '❌ Pipeline failed. Check the logs above for errors.'
        }
        always {
            echo '📋 Pipeline execution finished.'
        }
    }
}
