pipeline {
    agent any

    options{
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    environment{
        registryCredential = 'dockerhub'
    }

    stages{
        stage('Install'){
            agent{
                docker{
                    image 'python:3.9.2-slim-buster'
                }
            }
            steps{
                echo 'Installing libraries...'
                sh 'pip install isort ruff'
            }
        }

        stage('Check code quality'){
            steps{
                echo 'Check sorting using isort'
                sh 'isort --atomic .'
                echo 'Check format and linting using ruff'
                sh 'ruff format .'
                sh 'ruff check --fix .'
            }
        }
    }
}