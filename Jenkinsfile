// TODO: Integrate DVC for data versioning
// TODO: Use Docker here
pipeline {
    agent any
    environment {
        PROJECT_NAME = 'usl_for_embedding'
        CONDA_HOME = '/home/ubuntu/anaconda3'
    }
    stages {
        stage('Setup Environment') {
            steps {
                // Assuming you've set up Miniconda or similar in your Jenkins environment
                sh '#!/bin/bash'
                sh 'source ${CONDA_HOME}/etc/profile.d/conda.sh'
                sh 'make create_environment'
                sh 'source activate ${PROJECT_NAME}'
            }
        }
        stage('Run Make Commands') {
            steps {
                // Example of running a make command
                sh 'make test_environment'
            }
        }
    }
    post {
        always {
            // Clean up or additional steps
            sh 'make clean'
        }
    }
}
