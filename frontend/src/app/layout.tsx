import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "True Markets | Crypto Prediction Engine",
  description: "Independent ensemble ML predictions for crypto price targets",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-tm-bg text-tm-text antialiased">
        {children}
      </body>
    </html>
  );
}
