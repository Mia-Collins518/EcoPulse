import type { Metadata } from "next";
import { Lato } from "next/font/google";
import "./globals.css";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/ui/app-sidebar"
import Heading from "@/components/heading";

const lato = Lato({
  subsets: ["latin"],
  weight: ["300", "400", "700"],
  style: ['normal', 'italic'],
  variable: "--font-lato",
});

export const metadata: Metadata = {
  title: "EcoPulse",
  description: "Sustainable AI Technologies",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${lato.variable} font-sans`} >
        <SidebarProvider>
          <AppSidebar />
          <SidebarInset>  
            <Heading />
            {children}
          </SidebarInset>
        </SidebarProvider>
      </body>
    </html>
  );
}
