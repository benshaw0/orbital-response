from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root():
    return {"Message": "Today there was no attack. Use endpoint /predict to ...."}

# This is a decorator from a web framework like FastAPI
# It tells the application:
# ➜ “When someone sends a GET request to /, run the function right below this line.”
# The "/" refers to the root path of a web application. 
# Basically, when a user goes to the base URL like http://localhost:8000/.
@app.get('/predict')
def predict(attack=10000, no_attack=287.738):
    params = {
        "attack": attack,
        "no_attack": no_attack
    }
    damage = float(params["attack"]) - float(params["no_attack"])
    return {"message": f"The Damage from the latest attacks is: {damage}"}


# First, we write a Dockerfile for our application
# The Dockerfile is a blueprint that describes the steps required to create a Docker image.

# Then, from the Dockerfile we build a Docker image
# The Docker image is a mold from which containers are created. The Docker image bundles together the code of our application, its environment, and the platform required to run it.


