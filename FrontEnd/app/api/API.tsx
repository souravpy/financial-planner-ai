import { stockFormData } from "../types";

const API_URL = "http://127.0.0.1:8000/api";

export async function createStockPost(stockFormData: stockFormData) {
    return fetch(`${API_URL}/stocks`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(stockFormData),
    })
        .then((response) => response.json())
        .then((response) => {
            console.log(response);
            return response;
        })
        .catch((error) => {
            console.log("Error: ", error);
            return {};
        });
}

export async function createSalePost(saleFormData: any) {
    return fetch(`${API_URL}/sales/`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(saleFormData),
    })
        .then((response) => response.json())
        .then((response) => {
            console.log(response);
            return response;
        })
        .catch((error) => {
            console.log("Error: ", error);
            return {};
        });
}

export async function createEmployeeAtritionPost(employeeAtritionFormData: any) {
    return fetch(`${API_URL}/employee-attrition/`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(employeeAtritionFormData),
    })
        .then((response) => response.json())
        .then((response) => {
            console.log(response);
            return response;
        })
        .catch((error) => {
            console.log("Error: ", error);
            return {};
        });
}

export async function createBankruptcyPost(bankruptcyFormData: any) {
    return fetch(`${API_URL}/bankruptcy/`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(bankruptcyFormData),
    })
        .then((response) => response.json())
        .then((response) => {
            console.log(response);
            return response;
        })
        .catch((error) => {
            console.log("Error: ", error);
            return {};
        });
}

export async function createTestPost(testFormData: any) {
    return fetch(`${API_URL}/test/`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(testFormData),
    })
        .then((response) => response.json())
        .then((response) => {
            console.log(response);
            return response;
        })
        .catch((error) => {
            console.log("Error: ", error);
            return {};
        });
}
