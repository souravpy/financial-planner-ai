import React from "react";
import { Inter } from "next/font/google";
import { ConfigProvider } from "antd";
import Navbar from "./(Shared-Components)/Navbar";

import StyledComponentsRegistry from "../lib/AntdRegistry";

import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
    title: "FinAdvisor",
    description: "Official website for FinAdvisor.",
};

const antDCustomizeTheme = {
    token: {},
    components: {
        Menu: {
            horizontalLineHeight: "0px",
            // darkItemBg: "#0F0F0F",
            darkItemBg: "transparent",
            darkItemHoverColor: "#7743DB",
            darkItemColor: "#0F0F0F"
        },
    },
};

const RootLayout = ({ children }: React.PropsWithChildren) => (
    <html lang="en">
        <body
            className={`${inter.className} bg-gray-50 text-gray-950 relative`}
        >
            <div className="bg-[#7843db2f] absolute top-[-6rem] right-[20rem] h-[31.25rem] w-[31.25rem] -z-10 rounded-full blur-[10rem] sm:w[68.75rem]"></div>
            <div className="bg-[#f7efe58a] absolute top-[-1rem] left-[-30rem] h-[31.25rem] w-[50rem] -z-10 rounded-full blur-[10rem] sm:w[68.75rem] md:left-[-22rem] lg:left-[-15rem] xl:left-[-5rem] 2xl:left-[5rem]"></div>
            <StyledComponentsRegistry>
                <ConfigProvider theme={antDCustomizeTheme}>
                    <Navbar />

                    {children}
                </ConfigProvider>
            </StyledComponentsRegistry>
        </body>
    </html>
);

export default RootLayout;
