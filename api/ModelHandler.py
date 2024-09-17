import pandas as pd
import numpy as np
import tensorflow as tf
import os
from Forms import StockForm, SalesForm, EmployeeAttritionForm, BankruptcyForm
from sklearn.preprocessing import MinMaxScaler
from Visualizations import DataWindow

class Handler:
    def parse_date(date: str) -> pd.DataFrame:
        year = int(date[0:4])
        month = int(date[5:7])
        day = int(date[9:10])
        print(year, month, day)
        return pd.DataFrame(
            {
                "Low": 0,
                "Open": 0,
                "Volume": 0,
                "High": 0,
                "Close": 0,
                "Adjusted Close": 0,
                "Month": month,
                "Year": year,
                "Day": day,
            },
            index=[0],
        )

    def normalize_month(month: int) -> int:
        return month / 12

    def normalize_day(day: int) -> int:
        return day / 31

    def predictStocks(stockInput: StockForm) -> int:
        # date = Handler.parse_date(stockInput.date)
        # data = pd.DataFrame(pd.read_csv("../models/stocks/IBM_train.csv"))
        # year, month, day = date["Year"][0], date["Month"][0], date["Day"][0]
        # if year == 2022:
        #     year = 1.4054
        # elif year == 2023:
        #     year = 1.44
        # month = Handler.normalize_month(month)
        # day = Handler.normalize_day(day)
        # print(year, month, day)

        # model = tf.keras.models.load_model("../models/stocks/IBM_CNN.h5")
        # val = pd.DataFrame(pd.read_csv("../models/stocks/IBM_val.csv"))
        last_inputs = pd.DataFrame(pd.read_csv("../models/stocks/IBM_test.csv"))
        prediction_inputs = pd.DataFrame(last_inputs.tail(1))

        # prediction_inputs["Year"] = year
        # prediction_inputs["Month"] = month
        # prediction_inputs["Day"] = day
        # print(prediction_inputs)
        # # prediction_inputs = prediction_inputs.drop(columns=["Close"])
        # print(prediction_inputs)

        # KERNEL_WIDTH = 3
        # LABEL_WIDTH = 21
        # INPUT_WIDTH = LABEL_WIDTH + KERNEL_WIDTH - 1

        # cnn_multi_window = DataWindow(
        #     input_width=1,
        #     label_width=1,
        #     shift=1,
        #     train_df=data,
        #     val_df=val,
        #     test_df=prediction_inputs,
        #     label_columns=["Close"],
        # )

        # prediction = model.predict(cnn_multi_window.test)
        # print(prediction)
        return last_inputs["Close"][0]

    def predictSales(salesInput: SalesForm):
        print(salesInput)
        sales = np.random.randint(1000, 100000)
        print(sales)
        return sales

    def predictEmployeeAttrition(employeeAttritionInput: EmployeeAttritionForm):
        print(employeeAttritionInput)
        leave = "False"
        if employeeAttritionInput.monthlyIncome < 5000:
            leave = "True"
        elif employeeAttritionInput.businessTravel is "Travel_Frequently":
            leave = "True"
        return leave
    
    def predictBankruptcy(bankruptcyInput: BankruptcyForm):
        print(bankruptcyInput)
        bankruptcy = "False"
        if bankruptcyInput.debtRatio > 0.5:
            bankruptcy = "True"
        return bankruptcy
