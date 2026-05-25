import "./globals.css";
import BottomNav from "@/components/BottomNav";
import Header from "@/components/Header";
import { LanguageProvider } from "@/contexts/LanguageContext";

export const metadata = {
  title: "KGU Smart Assistant",
  description: "KGU Smart Assistant Frontend",
};

export default function RootLayout({ children }) {
  return (
    <html lang="ko">
      <body
        style={{
          margin: 0,
          backgroundColor: "black",
          color: "white",
        }}
      >
        <div
          style={{
            width: "100%",
            maxWidth: "430px",
            minHeight: "100vh",
            margin: "0 auto",
            position: "relative",
            paddingBottom: "80px",
            backgroundColor: "black",
          }}
        >
          <LanguageProvider>
            <Header />
            {children}
            <BottomNav />
          </LanguageProvider>
        </div>
      </body>
    </html>
  );
}