import torch
from concentric_circles import Model
#first what we do is load in the saved weights and biases from our previously trained model
state_dicts = torch.load("/Users/nealdamireddy/mlp/circle_model.pth", weights_only = True)
#we then create another model object that has the architecture of the original model. Remember, we only have the weights
model = Model(2, 1, 16)
#we then upload the saved weights onto the model
model.load_state_dict(state_dicts)
#move into evaluate mode
model.eval()
#create example point, convert it to tensor
point = torch.tensor([0.5,0.5]).float()
#run the prediciton with torch.no_grad() so that this is not saved as a value
with torch.no_grad():
    logits = model(point) #run the model with the given points with tensor point
    probabilities = torch.sigmoid(logits) #save that and label it as a probability using sigmoid function
    print(probabilities)
#create prediction function that takes in 2 values(floats) and returns a dictionary
def predict(x1, x2):
    tensor = torch.tensor([x1,x2]).float() 
    with torch.no_grad():
        logits = model(tensor).float()
        probability = torch.sigmoid(logits)
        prediction = int((probability >= 0.5).item())
        return {"prediciton": prediction, "probability" : probability.item()}
print(predict(0.5, 0.5))
print(predict(0.0, 0.0))
print(predict(1.0, 1.0))
