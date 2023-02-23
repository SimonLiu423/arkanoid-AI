import pickle
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import math

path = ".\log"
allFile = os.listdir(path)
data_set = []
for file in allFile:
    with open(os.path.join(path, file), "rb") as f:
        data_set.append(pickle.load(f))

# feature
Ball_x = []
Ball_y = []
Speed_x = []
Speed_y = []
Direction = []

for data in data_set:
    for i, sceneInfo in enumerate(data['1P']["scene_info"][2:-3]):
        Ball_x.append(data['1P']['scene_info'][i + 1]["ball"][0])
        Ball_y.append(data['1P']['scene_info'][i + 1]["ball"][1])
        Speed_x.append(data['1P']['scene_info'][i + 1]["ball"][0] - data['1P']['scene_info'][i]["ball"][0])
        Speed_y.append(data['1P']['scene_info'][i + 1]["ball"][1] - data['1P']['scene_info'][i]["ball"][1])
        if Speed_x[-1] > 0:
            if Speed_y[-1] > 0:
                Direction.append(0)
            else:
                Direction.append(1)
        else:
            if Speed_y[-1] > 0:
                Direction.append(2)
            else:
                Direction.append(3)

X = np.array([0, 0, 0, 0, 0])
for i in range(len(Ball_x)):
    X = np.vstack((X, [Ball_x[i], Ball_y[i], Speed_x[i], Speed_y[i], Direction[i]]))
X = X[1::]

# label
Position_pred = []
platform_position_y = 400
ball_speed_y = 7
platform_width = 200
for i in range(len(Ball_x)):
    pred = Ball_x[i] + ((platform_position_y - Ball_y[i]) // ball_speed_y) * Speed_x[i]

    section = (pred // platform_width)
    if (section % 2 == 0):
        pred = abs(pred - platform_width * section)
    else:
        pred = platform_width - abs(pred - platform_width * section)

    Position_pred.append(pred)

Position_pred = np.array(Position_pred)
Y = Position_pred

# training
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = DecisionTreeRegressor(criterion='squared_error', max_depth=8000, splitter='best')
model.fit(x_train, y_train)

# evaluation
y_predict = model.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
print(mse)
rmse = math.sqrt(mse)
print("RMSE=%.2f" % rmse)

# save model
if not os.path.exists(os.path.dirname(__file__) + "/save"):
    os.makedirs(os.path.dirname(__file__) + "/save")
with open(os.path.join(os.path.dirname(__file__), 'save', "model.pickle"), 'wb') as f:
    pickle.dump(model, f)
