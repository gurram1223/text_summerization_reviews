# Model Deployed on GCP

## Prerequisites

You must have PyTorch and Flask (for API) installed.

## Project Structure

Text_Summerization.ipynb - This is the file where we trained the model and stored the weights.

main.py - This contains Flask APIs that receives text data through GUI or API calls, summerizes the text and outputs it.

app.yaml - You configure your App Engine app's settings in the app.yaml file. Each service in your app has its own app.yaml file, which acts as a descriptor for its deployment.

templates - This folder contains the HTML template to allow user to enter text data and displays the summerized text.

requirements.txt - This file is used for specifying what python packages are required to run the project.

## Running the project (Local Environment)

Ensure that you are in the project home directory.

Run main.py using below command to start Flask API - python main.py

By default, flask will run on port 5000.

Navigate to URL http://localhost:5000

You should be able to view the homepage as below :

![alt text](https://github.com/gurram1223/text_summerization_reviews/blob/master/Image%20files/summerizer.PNG)

Enter the text and click on predict. Please find the below results.

![alt text](https://github.com/gurram1223/text_summerization_reviews/blob/master/Image%20files/summerizer_output.PNG)

## Google Cloud Compute(GCP) - Using App Engine

1. Open Google cloud compute platform (https://console.cloud.google.com/)

2. IAM & Admin -> Manage Resources -> Create new project

3. open command prompt ->type gcloud init -> It asks you to login and connect to the project you have created in the cloud.

4. Deploy - gcloud app deploy app.yaml --project {project_name}.

5. URL: https://summerization-reviews.el.r.appspot.com/ - It has limitation on the requests because we have large sized weights file and dependecies which consumes all the default memory in the cloud.


