from pydantic import BaseModel


class StockForm(BaseModel):
    stockTicker: str
    date: str


class SalesForm(BaseModel):
    gender: str
    merchantName: str
    category: str
    age: int
    month: int
    year: int


class EmployeeAttritionForm(BaseModel):
    age: int
    businessTravel: str
    department: str
    maritalStatus: str
    monthlyIncome: int
    yearsAtCompany: int


class BankruptcyForm(BaseModel):
    currentRatio: float
    operatingCashFlow: float
    debtRatio: float
