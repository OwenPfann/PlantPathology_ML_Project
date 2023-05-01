import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

table = pd.read_csv(r'C:\Users\17814\Desktop\Plant_Pathology\PlantPathology_ML_Project\ML Final Project\train.csv')
trainTable = pd.DataFrame(table, columns=['image', 'labels'], index=range(100)) # limited to 100 rows
testTableY = ['placeholder'] # change w/ images

trainTableX = trainTable.drop(columns=['labels'])
trainTableY = trainTable.drop(columns=['image'])
testTableX = ['placeholder']
testTableY = ['placeholder'] # change w/ images

plantArrayTrainX = trainTableX.to_numpy()
plantArrayTrainY = trainTableY.to_numpy()
# plantArrayTestX = testTableX.to_numpy()
# plantArrayTestY = testTableY.to_numpy()

def findModel(x, y):#, xTest, yTest): # x = features, y = labels
    # weights = np.polyfit(x, y, 1)
    # model = np.poly1d(weights)
    model = LinearRegression().fit(x, y)
    predicted = model.predict(x)
    rmse = np.sqrt(mean_squared_error(y, predicted))
    r2 = r2_score(y, predicted)
    print(f"Train rmse: {rmse}")
    print(f"Train r2: {r2}")

    ##### Testing Data #####

    # print("x testing: ")
    # print(xTest)
    # print("y testing: ")
    # print(yTest)

    # weights = np.polyfit(x, y, 1)
    # model = np.poly1d(weights)
    # predicted = model.predict(xTest)
    # rmse = np.sqrt(mean_squared_error(yTest, predicted))
    # r2 = r2_score(yTest, predicted)
    # print(f"Test rmse: {rmse}")
    # print(f"Test r2: {r2}")

    # r_sq = model.score(x, y)
    # print(f"coefficient of determination: {r_sq}")
    # print(f"intercept: {model.intercept_}")
    # print(f"coefficients: {model.coef_}")

print("Plant model:")
findModel(plantArrayTrainX, plantArrayTrainY)#, plantArrayTestX, plantArrayTestY)