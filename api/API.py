from typing import Union

from pydantic import BaseModel
from requests import request

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ModelHandler import Handler
from Forms import StockForm, SalesForm, EmployeeAttritionForm, BankruptcyForm

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class Test(BaseModel):
    title: str
    body: str
    id: int


class TestOut(BaseModel):
    response: str



@app.post("/api/stocks")
def read_stock_post(stockInput: StockForm):
    print(stockInput)
    close = Handler.predictStocks(stockInput)
    return {"Close": close}


@app.post("/api/sales")
def read_sales_post(saleInput: SalesForm):
    print(saleInput)
    sales = Handler.predictSales(saleInput)
    return {"Estimated sales": sales}


@app.post("/api/employee-attrition")
def read_employee_atrittion_post(employeeAttritionInput: EmployeeAttritionForm):
    print(employeeAttritionInput)
    leave = Handler.predictEmployeeAttrition(employeeAttritionInput)
    print(leave)
    return {"EmployeeAttrition": leave}


@app.post("/api/bankruptcy")
def read_bankruptcy_post(bankruptcyInput: BankruptcyForm):
    print(bankruptcyInput)
    bankruptcy = Handler.predictBankruptcy(bankruptcyInput)
    print(bankruptcy)
    return {"Bankruptcy": bankruptcy}
