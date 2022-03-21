import pytest
from src import prediction
from src.prediction import form_response, api_response
import warnings
warnings.filterwarnings('ignore')

input_data = {
    "incorrect_range":
    {"Week_Day": 7897897,
    "Week": 555,
    "Hour": 99,
    "Minutes": 99,
    "Seconds": 120,
    "Average_Speed": 789,
    "Clouds": 120,
    "Temp": 220,
    "Wind_deg": 365,
    "Wind_speed": 9,
    "Rain_1h": 2,
    "Rain_3h": 5,
    "Snow_1h": 6,
    "Snow_3h": 7,
    },

    "correct_range":
    {"Week_Day": 5,
    "Week": 42,
    "Hour": 2,
    "Minutes": 20,
    "Seconds": 30,
    "Average_Speed": 20,
    "Clouds": 80,
    "Temp": 280,
    "Wind_deg": 320,
    "Wind_speed": 0.2,
    "Rain_1h": 0.5,
    "Rain_3h": 0.8,
    "Snow_1h": 0,
    "Snow_3h": 1,
    },
    "incorrect_col":
    {"fixed acidity": 5,
    "volatile acidity": 1,
    "citric acid": 0.5,
    "residual sugar": 10,
    "chlorides": 0.5,
    "free sulfur dioxide": 3,
    "total_sulfur dioxide": 75,
    "density": 1,
    "pH": 3,
    "sulphates": 1,
    "alcohol": 9
    }
}

TARGET_range = {
    "min": 3.0,
    "max": 20.0
}

def test_form_response_correct_range(data=input_data["correct_range"]):
    res = form_response(data)
    assert  TARGET_range["min"] <= res <= TARGET_range["max"]

def test_api_response_correct_range(data=input_data["correct_range"]):
    res = api_response(data)
    assert  TARGET_range["min"] <= res["response"] <= TARGET_range["max"]

def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction.NotInRange):
        res = form_response(data)

def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction.NotInRange().message

def test_api_response_incorrect_col(data=input_data["incorrect_col"]):
    res = api_response(data)
    assert res["response"] == prediction.NotInCols().message