pipeline {
    agent { label 'gpu' } // Assumes you have a node labeled 'gpu'

    environment {
        // Define any environment variables needed for training
        DATA_PATH = '/path/to/data'
        MODEL_SAVE_PATH = '/path/to/save/model'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                sh '''
                conda create --name myenv python=3.8 -y
                source activate myenv
                pip install -r requirements.txt
                # Ensure DVC is installed
                pip install dvc
                '''
            }
        }
        
        stage('Pull Data with DVC') {
            steps {
                script {
                    // Pull data and model files needed for training from DVC remote
                    sh 'dvc pull'
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    // Replace with your actual training script command
                    sh "python train_model.py --data ${DATA_PATH} --save_path ${MODEL_SAVE_PATH}"
                    // After training, add model to DVC tracking
                    sh "dvc add ${MODEL_SAVE_PATH}/model.pt"
                }
            }
        }

        stage('Push Changes with DVC') {
            steps {
                script {
                    // Commit DVC changes
                    sh 'git add .dvc/config ${MODEL_SAVE_PATH}/model.pt.dvc'
                    sh 'git commit -m "Update model and DVC config"'
                    // Push DVC changes to remote
                    sh 'dvc push'
                    // Ensure to push Git changes as well, including .dvc files and .gitignore
                    sh 'git push origin master'
                }
            }
        }

        stage('Evaluate Model') {
            steps {
                script {
                    // Replace with your actual evaluation script command
                    sh "python evaluate_model.py --model ${MODEL_SAVE_PATH}/model.pt"
                }
            }
        }
    }

    post {
        always {
            echo 'Cleaning up...'
            // Add any cleanup steps if necessary
        }
        success {
            echo 'Pipeline succeeded!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
