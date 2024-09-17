"use client";
import React from "react";
import { Button, DatePicker, Form, Input, InputNumber, Select } from "antd";

import {
    stockFormData,
    saleFormData,
    bankruptcyFormData,
    employeeAttritionFormData,
} from "../types";

type Props = {
    id: string;
    name: string;
    onChange: (e: any) => void;
};

const FormTemplate = (props: Props) => {
    const [form] = Form.useForm();
    const handleChange = (e: any) => {
        props.onChange(e.currentTarget.value);
    };

    const onFinish = (values: any) => {
        console.log(values.dateRange);
    };

    const onStockSubmit = (values: stockFormData) => {
        console.log(values);
        props.onChange(values);
    };

    const onSaleSubmit = (values: saleFormData) => {
        console.log(values);
        props.onChange(values);
    };

    const onBankruptcySubmit = (values: bankruptcyFormData) => {
        console.log(values);
        props.onChange(values);
    };

    const onEmployeeAtritionSubmit = (values: employeeAttritionFormData) => {
        console.log(values);
        props.onChange(values);
    };

    switch (props.name) {
        case "Stocks":
            return (
                <div id={props.id} className="w-full">
                    <Form
                        labelCol={{ span: 11 }}
                        wrapperCol={{ span: 12 }}
                        layout="horizontal"
                        size="middle"
                        className="my-16 shadow-lg p-4 rounded-xl backdrop-filter backdrop-blur-lg bg-opacity-70 bg-primary-off-white"
                        onFinish={onStockSubmit}
                        form={form}
                    >
                        <h2 className="text-3xl font-bold text-center mb-8 text-primary-purple">
                            {props.name}
                        </h2>
                        <div className="grid grid-flow-row grid-cols-2 gap-4 mx-16">
                            <Form.Item label="Stock Ticker" name="stockTicker">
                                <Input />
                            </Form.Item>
                            <Form.Item label="Date" name="date">
                                <DatePicker />
                            </Form.Item>
                        </div>
                        <Form.Item className="flex justify-center">
                            <Button
                                className="ml-4 bg-primary-purple"
                                htmlType="submit"
                                type="primary"
                            >
                                Submit
                            </Button>
                        </Form.Item>
                    </Form>
                </div>
            );
        case "Sales Info":
            return (
                <div id={props.id} className="w-full">
                    <Form
                        labelCol={{ span: 15 }}
                        wrapperCol={{ span: 18 }}
                        layout="horizontal"
                        size="large"
                        className="w-full my-16 shadow-lg p-8 rounded-xl backdrop-filter backdrop-blur-lg bg-opacity-70 bg-primary-off-white"
                        onFinish={onSaleSubmit}
                        form={form}
                    >
                        <h2 className="text-3xl font-bold text-center mb-8 text-primary-purple">
                            {props.name}
                        </h2>
                        <div className="grid grid-flow-row grid-cols-2 gap-4 my-4">
                            <Form.Item label="Gender" name="gender">
                                <Select>
                                    <Select.Option value="M">
                                        Male
                                    </Select.Option>
                                    <Select.Option value="F">
                                        Female
                                    </Select.Option>
                                    <Select.Option value="O">
                                        Other
                                    </Select.Option>
                                </Select>
                            </Form.Item>
                            <Form.Item label="Age" name="age">
                                <InputNumber />
                            </Form.Item>
                            <Form.Item label="Category" name="category">
                                <Input />
                            </Form.Item>
                            <Form.Item
                                label="Merchant Name"
                                name="merchantName"
                            >
                                <Input />
                            </Form.Item>
                            <Form.Item label="Month" name="month">
                                <InputNumber />
                            </Form.Item>
                            <Form.Item label="Year" name="year">
                                <InputNumber />
                            </Form.Item>
                        </div>
                        <Form.Item className="flex justify-center">
                            <Button
                                className="ml-4 bg-primary-purple"
                                htmlType="submit"
                                type="primary"
                            >
                                Submit
                            </Button>
                        </Form.Item>
                    </Form>
                </div>
            );
        case "Bankruptcy":
            return (
                <div id={props.id} className="w-full">
                    <Form
                        labelCol={{ span: 17 }}
                        wrapperCol={{ span: 12 }}
                        layout="horizontal"
                        size="middle"
                        className="my-16 shadow-lg p-4 rounded-xl backdrop-filter backdrop-blur-lg bg-opacity-70 bg-primary-off-white"
                        onFinish={onBankruptcySubmit}
                        form={form}
                    >
                        <h2 className="text-3xl font-bold text-center mb-8 text-primary-purple">
                            {props.name}
                        </h2>
                        <div className="grid grid-flow-row grid-cols-2 gap-4 mx-16">
                            <Form.Item
                                label="Current Ratio"
                                name="currentRatio"
                            >
                                <InputNumber placeholder="0.25" />
                            </Form.Item>
                            <Form.Item
                                label="Operating Cash Flow"
                                name="operatingCashFlow"
                            >
                                <InputNumber />
                            </Form.Item>
                            <Form.Item label="Debt Ratio" name="debtRatio">
                                <InputNumber placeholder="0.25" />
                            </Form.Item>
                        </div>
                        <Form.Item className="flex justify-center">
                            <Button
                                className="ml-4 bg-primary-purple"
                                htmlType="submit"
                                type="primary"
                            >
                                Submit
                            </Button>
                        </Form.Item>
                    </Form>
                </div>
            );

        case "Employee Attrition":
            return (
                <div id={props.id} className="w-full">
                    <Form
                        labelCol={{ span: 14 }}
                        wrapperCol={{ span: 18 }}
                        layout="horizontal"
                        size="large"
                        className="w-full my-16 shadow-lg p-8 rounded-xl backdrop-filter backdrop-blur-lg bg-opacity-70 bg-primary-off-white"
                        onFinish={onEmployeeAtritionSubmit}
                        form={form}
                    >
                        <h2 className="text-3xl font-bold text-center mb-8 text-primary-purple">
                            {props.name}
                        </h2>
                        <div className="grid grid-flow-row grid-cols-2 gap-4 my-4">
                            <Form.Item label="Age" name="age">
                                <InputNumber />
                            </Form.Item>
                            <Form.Item
                                label="Business Travel"
                                name="businessTravel"
                            >
                                <Select>
                                    <Select.Option value="travel_frequently">
                                        Travel Frequently
                                    </Select.Option>
                                    <Select.Option value="travel_rarely">
                                        Travel Rarely
                                    </Select.Option>
                                    <Select.Option value="non_travel">
                                        Non-Travel
                                    </Select.Option>
                                </Select>
                            </Form.Item>
                            <Form.Item label="Department" name="department">
                                <Select>
                                    <Select.Option value="sales">
                                        Sales
                                    </Select.Option>
                                    <Select.Option value="research_development">
                                        Research and Development
                                    </Select.Option>
                                    <Select.Option value="human_resources">
                                        Human Resources
                                    </Select.Option>
                                </Select>
                            </Form.Item>
                            <Form.Item
                                label="Marital Status"
                                name="maritalStatus"
                            >
                                <Select>
                                    <Select.Option value="single">
                                        Single
                                    </Select.Option>
                                    <Select.Option value="married">
                                        Married
                                    </Select.Option>
                                    <Select.Option value="divorced">
                                        Divorced
                                    </Select.Option>
                                </Select>
                            </Form.Item>
                            <Form.Item
                                label="Monthly Income"
                                name="monthlyIncome"
                            >
                                <InputNumber />
                            </Form.Item>
                            <Form.Item
                                label="Years at Company"
                                name="yearsAtCompany"
                            >
                                <InputNumber />
                            </Form.Item>
                        </div>
                        <Form.Item className="flex justify-center">
                            <Button
                                className="ml-4 bg-primary-purple"
                                htmlType="submit"
                                type="primary"
                            >
                                Submit
                            </Button>
                        </Form.Item>
                    </Form>
                </div>
            );
    }
};

export default FormTemplate;
